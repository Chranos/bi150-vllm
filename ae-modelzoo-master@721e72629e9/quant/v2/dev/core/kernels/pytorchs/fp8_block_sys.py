import torch
import torch.nn.functional as F

def dequant_fp8_block_sym(
    artifacts,
    rq,
    output_dtype=torch.bfloat16,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Fully vectorized FP8 block-wise symmetric dequantization on GPU.
    Avoids Python loops for large tensors.
    """
    qweight = artifacts.qweight.to(device, non_blocking=True)
    scale = artifacts.scale.to(device, non_blocking=True)
    out_features, in_features = qweight.shape
    block_size = rq.block_structure[1] if rq.block_structure else rq.group_size or in_features
    num_blocks = (in_features + block_size - 1) // block_size
    pad_size = num_blocks * block_size - in_features

    # pad qweight if necessary
    if pad_size > 0:
        qweight = F.pad(qweight, (0, pad_size))

    # reshape为 [out_features, num_blocks, block_size]
    qweight_blocks = qweight.view(out_features, num_blocks, block_size)

    # 处理 scale
    if scale.ndim == 1 and scale.shape[0] == num_blocks:
        scale_broadcast = scale.view(1, num_blocks, 1)
    elif scale.ndim == 2:
        if scale.shape == (out_features, num_blocks):
            scale_broadcast = scale.unsqueeze(-1)
        else:
            # block-wise scale [num_row_blocks, num_col_blocks]
            num_row_blocks, num_col_blocks = scale.shape
            if num_row_blocks * block_size != out_features or num_col_blocks != num_blocks:
                raise ValueError(f"Unsupported block-wise scale {scale.shape}, {qweight.shape}, {block_size}")
            scale_broadcast = scale.repeat_interleave(block_size, dim=0).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported scale shape {scale.shape}")

    # 向量化反量化
    dequant_blocks = qweight_blocks.float() * scale_broadcast

    # reshape回原始形状
    dequant = dequant_blocks.view(out_features, num_blocks * block_size)
    if pad_size > 0:
        dequant = dequant[:, :in_features]

    output = dequant.to(output_dtype)

    return output
