import torch
import numpy as np

from typing import List
from torch.multiprocessing import Pool


# @csabakecskemeti
# def unpack_int4_from_int32(packed_tensor, original_shape):
#     """
#     Unpack INT4 values from INT32 storage.

#     Each INT32 contains 8 INT4 values:
#     INT32: [31..28][27..24][23..20][19..16][15..12][11..8][7..4][3..0]
#            │  val7 ││  val6 ││  val5 ││  val4 ││  val3 ││ val2││val1││val0│

#     Args:
#         packed_tensor: torch.Tensor with dtype int32 containing packed INT4 values
#         original_shape: tuple, the original shape of the weight before packing

#     Returns:
#         torch.Tensor with unpacked INT4 values as float32
#     """
#     # Convert to numpy for reliable bit operations
#     packed_np = packed_tensor.detach().cpu().numpy().astype(np.uint32)

#     # Extract 8 INT4 values from each UINT32
#     unpacked_list = []
#     for i in range(8):
#         shift = i * 4
#         mask = 0xF  # 4-bit mask (0b1111)
#         int4_val = (packed_np >> shift) & mask
#         # Convert from unsigned [0,15] to signed [-8,7]
#         int4_signed = int4_val.astype(np.int8) - 8
#         unpacked_list.append(int4_signed)

#     # Stack along new axis: [..., 8]
#     unpacked_np = np.stack(unpacked_list, axis=-1)

#     # Flatten the last dimension: [..., 8] -> [..., 8*elements]
#     flat_shape = list(unpacked_np.shape[:-1]) + [-1]
#     unpacked_flat = unpacked_np.reshape(flat_shape)

#     # Calculate expected total elements
#     total_elements = 1
#     for dim in original_shape:
#         total_elements *= dim

#     # Handle potential padding
#     unpacked_1d = unpacked_flat.flatten()
#     if len(unpacked_1d) > total_elements:
#         unpacked_1d = unpacked_1d[:total_elements]
#     elif len(unpacked_1d) < total_elements:
#         # Pad if needed (shouldn't happen with correct packing)
#         padding = total_elements - len(unpacked_1d)
#         unpacked_1d = np.concatenate([unpacked_1d, np.zeros(padding, dtype=np.int8)])

#     # Reshape to original shape
#     result = unpacked_1d.reshape(original_shape)
#     return torch.from_numpy(result).float()

# def unpack_int4_from_int32(packed_tensor: torch.Tensor, original_shape):
#     """
#     High-performance INT4 unpack using pure PyTorch tensor ops.
#     packed_tensor: int32 tensor containing 8 x INT4 packed
#     original_shape: target shape (H, W)
#     return: float32 tensor with unpacked int4 values
#     """

#     # packed_tensor: [N] int32
#     device = packed_tensor.device
#     x = packed_tensor.to(torch.int32)

#     # Prepare shifts for extracting 8 nibbles
#     # shift = [0, 4, 8, 12, 16, 20, 24, 28]
#     shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)

#     # Extract 8 nibble values: [(x >> s) & 0xF for s in shifts]
#     # Expand: x[..., None] shape => [N, 1]
#     # Broadcast: shifts => [1, 8]
#     # Result => [N, 8]
#     unpacked = (x.unsqueeze(-1) >> shifts) & 0xF

#     # Convert unsigned [0..15] → signed [-8..7]
#     unpacked = unpacked.to(torch.int8) - 8

#     # Final reshape
#     total = 1
#     for d in original_shape:
#         total *= d

#     unpacked = unpacked.reshape(-1)[:total]
#     return unpacked.reshape(original_shape).float()


import triton
import triton.language as tl
import torch


@triton.jit
def kernel_unpack_int4(
    packed_ptr, out_ptr, N, OUT_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK

    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < N  # Triton mask，0/1 即可

    # load int32 packed data
    x = tl.load(packed_ptr + offsets, mask=mask, other=0)
    x = x.to(tl.int32)

    # unpack 8 int4 per int32
    shifts = tl.arange(0, 8) * 4
    x = (x[:, None] >> shifts[None, :]) & 0xF
    x = x.to(tl.int8) - 8

    # flatten
    x_flat = tl.reshape(x, (BLOCK * 8,))

    # write to output
    out_offsets = block_start * 8 + tl.arange(0, BLOCK * 8)
    mask_out = out_offsets < OUT_SIZE  # 0/1 mask
    tl.store(out_ptr + out_offsets, x_flat.to(tl.float32), mask=mask_out)


def unpack_int4_from_int32(packed: torch.Tensor, original_shape, out_dtype=torch.float32):
    device = packed.device
    packed = packed.flatten().to(torch.int32)
    N = packed.numel()

    total = int(torch.prod(torch.tensor(original_shape)))
    out = torch.empty(total, dtype=out_dtype, device=device)

    BLOCK = 1024
    grid = lambda meta: ((N + BLOCK - 1) // BLOCK,)

    kernel_unpack_int4[grid](
        packed_ptr=packed,
        out_ptr=out,
        N=N,
        OUT_SIZE=total,
        BLOCK=BLOCK,
    )

    return out.reshape(original_shape)


def apply_group_scaling(unpacked, scale_tensor, group_size=32):
    """
    Apply group-wise scaling to unpacked INT4 values.

    Args:
        unpacked: torch.Tensor, shape [out_features, in_features]
        scale_tensor: torch.Tensor, shape [out_features, in_features//group_size]
        group_size: int, number of elements per group (default 32 for Kimi)

    Returns:
        torch.Tensor with scaled values
    """
    if scale_tensor.numel() == 1:
        # Single scale value
        return unpacked * scale_tensor.item()

    out_features, in_features = unpacked.shape
    scale_out, scale_in = scale_tensor.shape

    if scale_out == out_features and scale_in * group_size == in_features:
        # Standard group-wise scaling
        # Reshape: [out_features, in_features] -> [out_features, scale_in, group_size]
        weight_grouped = unpacked.view(out_features, scale_in, group_size)
        # Expand scale: [out_features, scale_in] -> [out_features, scale_in, 1]
        scale_expanded = scale_tensor.view(out_features, scale_in, 1)
        # Apply scaling
        scaled_grouped = weight_grouped * scale_expanded.float()
        # Reshape back: [out_features, scale_in, group_size] -> [out_features, in_features]
        return scaled_grouped.view(out_features, in_features)

    elif scale_out == out_features:
        # Try to handle irregular group sizes
        actual_group_size = in_features // scale_in
        if actual_group_size > 0 and in_features % scale_in == 0:
            weight_grouped = unpacked.view(out_features, scale_in, actual_group_size)
            scale_expanded = scale_tensor.view(out_features, scale_in, 1)
            scaled_grouped = weight_grouped * scale_expanded.float()
            return scaled_grouped.view(out_features, in_features)
        else:
            # Fallback: repeat scales to match dimensions
            scale_repeated = scale_tensor.repeat_interleave(
                (in_features + scale_in - 1) // scale_in, dim=1
            )[:, :in_features]
            return unpacked * scale_repeated.float()

    else:
        # Last resort: try direct broadcasting
        try:
            return unpacked * scale_tensor.float()
        except RuntimeError:
            # Use mean scale as fallback
            return unpacked * scale_tensor.mean().item()


def weight_dequant_int4(packed_tensor, scale_tensor, shape_tensor):
    """
    Dequantize INT4 weights to BF16.

    This is the INT4 equivalent of weight_dequant() in the FP8 version.

    Args:
        packed_tensor: torch.Tensor, INT32 tensor with packed INT4 values
        scale_tensor: torch.Tensor, BF16 tensor with group-wise scales
        shape_tensor: torch.Tensor, INT32 tensor with original shape [H, W]

    Returns:
        torch.Tensor in BF16 format with original shape
    """
    # Get original shape
    original_shape = tuple(shape_tensor.tolist())

    # Step 1: Unpack INT4 values from INT32 storage
    unpacked = unpack_int4_from_int32(packed_tensor, original_shape).to(scale_tensor.device)

    # Step 2: Apply group-wise scaling
    scaled = apply_group_scaling(unpacked, scale_tensor, group_size=32)

    # Step 3: Convert to BF16
    return scaled.to(torch.bfloat16)


def _dequant_one_prefix(args):
    """
    子进程函数：对一个 prefix（layer）进行反量化
    """
    prefix_name, suffix_map, state_dict, device = args

    packed_tensor = state_dict[suffix_map[".weight_packed"]].to(device, non_blocking=True)
    scale_tensor  = state_dict[suffix_map[".weight_scale"]].to(device, non_blocking=True)
    shape_tensor  = state_dict[suffix_map[".weight_shape"]].to(device, non_blocking=True)

    with torch.no_grad():
        new_weight = weight_dequant_int4(
            packed_tensor,
            scale_tensor,
            shape_tensor
        ).cpu()

    # 避免显存泄漏
    del packed_tensor, scale_tensor, shape_tensor
    torch.cuda.empty_cache()

    return prefix_name, new_weight


def weight_dequant_int4_wrapper(state_dict, suffixes: list[str], device: str = "cuda", batch_size: int = 128):
    """
    批量堆叠反量化版本：
    - 将多个 prefix 的 tensor 堆叠到 GPU
    - 一次性反量化，提高显存和计算利用率
    """
    suffix_set = set(suffixes)
    prefix_map: dict[str, dict[str, str]] = {}
    new_sd = {}
    weight_names = list(state_dict.keys())

    for w in weight_names:
        for suf in suffixes:
            if w.endswith(suf):
                prefix = w.removesuffix(suf)
                prefix_map.setdefault(prefix, {})[suf] = w
                break
        else:
            new_sd[w] = state_dict[w]

    for prefix_name, sufmap in prefix_map.items():
        missing = suffix_set - set(sufmap.keys())
        if missing:
            raise ValueError(f"[反量化失败] prefix={prefix_name} 缺少 {missing}，无法反量化。")

    prefixes = list(prefix_map.items())
    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i:i+batch_size]

        packed_tensors = []
        scale_tensors = []
        shape_tensors = []
        batch_names = []

        for prefix_name, sufmap in batch:
            packed_tensors.append(state_dict[sufmap[".weight_packed"]])
            scale_tensors.append(state_dict[sufmap[".weight_scale"]])
            shape_tensors.append(state_dict[sufmap[".weight_shape"]])
            batch_names.append(prefix_name)

        # 堆叠 tensor（这里使用 torch.stack，如果 shape 不一致，可保留 list）
        # 如果 shape 不一致，weight_dequant_int4 需支持 list 输入
        packed_tensors = [t.to(device, non_blocking=True) for t in packed_tensors]
        scale_tensors = [t.to(device, non_blocking=True) for t in scale_tensors]
        shape_tensors = [t.to(device, non_blocking=True) for t in shape_tensors]

        # 批量反量化
        with torch.no_grad():
            new_weights = []
            for pt, st, sht in zip(packed_tensors, scale_tensors, shape_tensors):
                w = weight_dequant_int4(pt, st, sht).cpu()
                new_weights.append(w)

        # 保存结果
        for name, w in zip(batch_names, new_weights):
            new_sd[name + ".weight"] = w

        # 显存管理
        del packed_tensors, scale_tensors, shape_tensors, new_weights
        # torch.cuda.empty_cache()  # 可选

    return new_sd

def tensor_memory_bytes(t: torch.Tensor):
    return t.numel() * t.element_size()

def state_dict_memory_bytes(sd: dict):
    return sum(tensor_memory_bytes(v) for v in sd.values())

def bytes_to_human(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


if __name__ == "__main__":
    from glob import glob
    from safetensors.torch import load_file

    all_files = glob("/home/mashun/vllm_project/checkpoints/Kimi-K2-Thinking/*.safetensors")
    f = all_files[-1]

    sd = load_file(f)

    suffixes = ["weight_packed", "weight_scale", "weight_shape"]

    nsd = weight_dequant_int4_wrapper(sd, suffixes, "cuda:0")

    total = state_dict_memory_bytes(nsd)
    print("显存占用:", bytes_to_human(total))


