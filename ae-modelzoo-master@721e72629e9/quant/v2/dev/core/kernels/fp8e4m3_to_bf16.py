# triton_fp8_dequant.py
import torch
import triton
import triton.language as tl
from typing import Tuple


# ----------------------------
# Triton kernel: dequant one block tile (BLOCK x BLOCK)
# ----------------------------
@triton.jit
def _dequant_block_kernel(
    x_ptr,         # pointer to int8/uint8 input padded matrix flattened
    s_ptr,         # pointer to float32 scales [M1 * N1]
    out_ptr,       # pointer to float32 output padded matrix flattened
    target_N,      # total columns in padded matrix (int)
    M1,            # number of block-rows
    N1,            # number of block-cols
    BLOCK: tl.constexpr
):
    pid_m = tl.program_id(0)  # block row index (0..M1-1)
    pid_n = tl.program_id(1)  # block col index (0..N1-1)

    # block row/col start (in element coords)
    m_start = pid_m * BLOCK
    n_start = pid_n * BLOCK

    # row/col indices inside global matrix for this tile
    rows = tl.arange(0, BLOCK)
    cols = tl.arange(0, BLOCK)

    # create 2D grid of indices (BLOCK x BLOCK)
    row_idx = m_start + rows[:, None]   # shape [BLOCK, 1]
    col_idx = n_start + cols[None, :]   # shape [1, BLOCK]

    # flatten indices into 1D offsets for load/store:
    # idx = row_idx * target_N + col_idx  -> shape [BLOCK, BLOCK]
    idx = row_idx * target_N + col_idx

    # load block (int8) with mask; safe since padded array supplies zeros for out-of-bound
    mask = tl.where(idx < (M1 * BLOCK) * target_N, True, False)  # always True for valid padded region
    # Note: We rely on padded input being at least (M1*BLOCK, N1*BLOCK). Mask still required for correct loads.
    vals_i8 = tl.load(x_ptr + idx, mask=mask, other=0)  # returns int8 values

    # load scale for this block
    s_idx = pid_m * N1 + pid_n
    s_val = tl.load(s_ptr + s_idx)  # float32 scalar

    # convert to float32 and multiply
    vals_f32 = tl.cast(vals_i8, tl.float32) * s_val

    # store results to output (float32)
    tl.store(out_ptr + idx, vals_f32, mask=mask)


# ----------------------------
# Python wrapper
# ----------------------------
def weight_dequant_fp8e4m3_to_bf16(
    x: torch.Tensor,        # int8/uint8 tensor shape [M, N]
    s: torch.Tensor,        # float32 tensor shape [M1, N1]
    block_size: int = 128,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Dequantize FP8-block-packed integer matrix `x` with per-block scales `s` to BF16.

    Returns: torch.Tensor of shape [M, N] and dtype torch.bfloat16 (on same device as inputs).

    Implementation notes:
      - Kernel writes float32 outputs; we convert to bfloat16 in PyTorch afterwards
      - `x` must be on CUDA device (Triton runs on CUDA)
      - `s` must be float32 on same device
      - This function pads `x` to full block grid using `pad_value`
    """
    if not x.is_cuda:
        raise ValueError("triton_fp8_dequant_to_bf16 requires CUDA tensors (x must be on CUDA).")
    if not s.is_cuda:
        raise ValueError("s must be on CUDA device too.")
    if x.dim() != 2 or s.dim() != 2:
        raise ValueError("x must be 2D, s must be 2D (block-grid).")

    device = x.device
    M, N = x.shape
    M1, N1 = s.shape
    target_M = M1 * block_size
    target_N = N1 * block_size
    if M > target_M or N > target_N:
        raise ValueError(f"Input x shape {(M,N)} is larger than block grid implied by s: {(target_M, target_N)}")

    # ensure x dtype is int8 or uint8. If not, cast (we assume FP8 mantissa stored as int8)
    if x.dtype not in (torch.int8, torch.uint8):
        x_int8 = x.to(torch.int8)
    else:
        x_int8 = x

    # prepare padded input (int8) - use zeros to avoid uninitialized values
    if M == target_M and N == target_N and x_int8.is_contiguous():
        padded_x = x_int8
    else:
        padded_x = torch.full((target_M, target_N), fill_value=int(pad_value), dtype=torch.int8, device=device)
        padded_x[:M, :N] = x_int8

    # allocate output padded float32 buffer
    out_f32 = torch.empty((target_M, target_N), dtype=torch.float32, device=device)

    # pointers
    x_ptr = padded_x
    s_ptr = s.to(device=device, dtype=torch.float32)  # ensure float32 and on device
    out_ptr = out_f32

    # launch grid: each program handles one block tile
    grid = (M1, N1)

    # compute num warps heuristics: bigger blocks -> fewer warps; set default to 8
    num_warps = 8

    _dequant_block_kernel[grid](x_ptr, s_ptr, out_ptr, target_N, M1, N1, BLOCK=block_size, num_warps=num_warps)

    # crop back to original shape and convert to bfloat16
    res_bf16 = out_f32[:M, :N].to(dtype=torch.bfloat16)

    return res_bf16


# ----------------------------
# Quick sanity test (only run if this file executed directly)
# ----------------------------
if __name__ == "__main__":
    # small example to sanity-check on GPU
    if not torch.cuda.is_available():
        print("CUDA not available; cannot run Triton kernel.")
    else:
        device = torch.device("cuda")
        M, N = 300, 500
        block_size = 128
        M1 = (M + block_size - 1) // block_size
        N1 = (N + block_size - 1) // block_size

        # create small int8-like data and small scales
        x = torch.randint(-10, 11, (M, N), dtype=torch.int8, device=device)
        s = torch.rand((M1, N1), dtype=torch.float32, device=device) * 0.05 + 0.01

        out = triton_fp8_dequant_to_bf16(x, s, block_size=block_size)
        print("out.shape=", out.shape, " dtype=", out.dtype, " device=", out.device)
        # quick numeric check against naive CPU implementation
        # naive:
        padded = torch.zeros((M1*block_size, N1*block_size), dtype=torch.int8, device=device)
        padded[:M, :N] = x
        blocks = padded.view(M1, block_size, N1, block_size).permute(0,2,1,3)
        s_expanded = s.unsqueeze(-1).unsqueeze(-1).to(torch.float32)
        naive = (blocks.to(torch.float32) * s_expanded).permute(0,2,1,3).reshape(M1*block_size, N1*block_size)[:M, :N].to(torch.bfloat16)
        diff = (naive.to(torch.float32) - out.to(torch.float32)).abs().max()
        print("max abs diff (float32):", diff.item())
