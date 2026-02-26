import torch

def unpack_int4_from_int32(
    packed: torch.Tensor,
    original_shape: tuple,
    *,
    out_dtype=torch.float32,
):
    """
    Unpack INT4 weights packed in INT32.

    Layout per int32 (little-endian nibble order):
    [ 3..0 | 7..4 | 11..8 | ... | 31..28 ]  -> 8 x INT4

    Args:
        packed: int32 tensor, arbitrary shape
        original_shape: target shape after unpack
        out_dtype: output dtype (default float32)

    Returns:
        Tensor of shape original_shape, dtype = out_dtype
    """
    assert packed.dtype == torch.int32
    device = packed.device

    # [N] -> [N, 1]
    x = packed.view(-1, 1)

    # shifts = [0,4,8,...,28]
    shifts = torch.arange(
        0, 32, 4,
        device=device,
        dtype=torch.int32
    ).view(1, 8)

    # [N, 8] uint4 in [0, 15]
    vals = (x >> shifts) & 0xF

    # uint4 -> int4 : [0,15] → [-8,7]
    vals = vals.to(torch.int8).sub_(8)

    # flatten & trim padding
    numel = torch.prod(
        torch.tensor(original_shape, device=device)
    ).item()

    vals = vals.view(-1)[:numel]

    return vals.view(original_shape).to(out_dtype)


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


def weight_dequant_int4(packed_tensor, scale_tensor, shape_tensor, group_size):
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
    scaled = apply_group_scaling(unpacked, scale_tensor, group_size=group_size)

    # Step 3: Convert to BF16
    return scaled.to(torch.bfloat16)


def dequant_int4_group_sym(
    artifacts,
    rq,
    output_dtype=torch.bfloat16,
    device: torch.device = torch.device("cuda")
):
    qweight = artifacts.qweight.to(device, non_blocking=True)
    scale = artifacts.scale.to(device, non_blocking=True)
    shape = artifacts.shape.to(device, non_blocking=True)

    group_size = rq.group_size

    output = weight_dequant_int4(qweight, scale, shape, group_size)

    return output.to(output_dtype)