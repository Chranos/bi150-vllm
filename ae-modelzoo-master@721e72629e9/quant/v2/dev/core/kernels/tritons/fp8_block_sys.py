import triton
import triton.language as tl
import torch


@triton.jit
def fp8_block_dequant_kernel(
    qweight_ptr, scale_ptr, dequant_ptr,
    M, N, block_size, num_blocks,
    SCALE_ROWWISE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    row_mask = row_offsets < M
    col_mask = col_offsets < N

    offs = row_offsets[:, None] * N + col_offsets[None, :]
    x = tl.load(qweight_ptr + offs, mask=row_mask[:, None] & col_mask[None, :], other=0)
    x = x.to(tl.float32)

    block_id = col_offsets // block_size

    if SCALE_ROWWISE:
        # scale: [M, num_blocks] -> load per row
        scale_offs = row_offsets[:, None] * num_blocks + block_id[None, :]
        s = tl.load(scale_ptr + scale_offs, mask=row_mask[:, None] & col_mask[None, :], other=1.0)
    else:
        # scale: [num_blocks] -> broadcast row
        s = tl.load(scale_ptr + block_id[None, :])
        s = s[None, :]  # broadcast to rows

    dequant = x * s
    tl.store(dequant_ptr + offs, dequant, mask=row_mask[:, None] & col_mask[None, :])


def dequant_fp8_block_sym_triton(artifacts, rq):
    qweight = artifacts.qweight
    scale = artifacts.scale
    M, N = qweight.shape

    # block size
    if rq.block_structure:
        block_size = rq.block_structure[1]
    elif rq.group_size:
        block_size = rq.group_size
    else:
        block_size = N

    num_blocks = (N + block_size - 1) // block_size

    # 判断 scale 是否 row-wise
    SCALE_ROWWISE = scale.ndim == 2
    if SCALE_ROWWISE:
        assert scale.shape[0] == M and scale.shape[1] == num_blocks, \
            f"Row-wise scale shape must be [M, num_blocks], got {scale.shape}"
    else:
        # scale 1D
        assert scale.shape[0] == num_blocks, \
            f"Scale shape must be [num_blocks], got {scale.shape}"

    dequant = torch.empty((M, N), dtype=torch.float32, device=qweight.device)

    BLOCK_M = 64
    BLOCK_N = 128
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)

    fp8_block_dequant_kernel[grid](
        qweight, scale, dequant,
        M, N, block_size, num_blocks,
        SCALE_ROWWISE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )

    if rq.dtype.lower() == "bf16":
        dequant = dequant.to(torch.bfloat16)

    return dequant