import torch
import ixformer.inference.functions as ixfop


def weight_quant_bf16_to_int4pack8(
    v: torch.Tensor,            # [B, R, C]
    block_size: int = 128,
    group_size: int = -1,
    format: str = "TN",
    symmetric: bool = True,
    version: int = 2,
):
    """
    Batch 版本 INT4 量化 + 打包。

    Args:
        v: [batch, rows, cols], float Tensor

    Returns:
        i4_weights: [batch, rows, packed_cols]
        scale:      [batch, rows, 1]
        i8scales:   来自 ixfop.quant_repack_int4
        i8zeros:    来自 ixfop.quant_repack_int4
    """
    assert v.dim() == 3, f"expected [batch, rows, cols], got {v.shape}"

    B, R, C = v.shape

    qmax = 127.0

    # abs_max: [B, R, 1]
    abs_max = torch.abs(v).amax(dim=2, keepdim=True)
    scale = abs_max / qmax     # [B, R, 1]

    # quantized: [B, R, C]
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax).to(torch.int8)

    # ixfop.quant_repack_int4 需要 [batch, rows, cols]
    # 它本来就是 batch-first，可直接送进去
    # 返回形状一般是:
    #   i4_weights: [B, R, packed_C]
    #   i8scales:   [B, R, groups]
    #   i8zeros:    [B, R, groups]
    i4_weights, i8scales, i8zeros = ixfop.quant_repack_int4(
        quantized,          # 不需要 unsqueeze，因为本来就是 [B, R, C]
        group_size,
        version,
        format,
        symmetric,
    )

    return (
        i4_weights,          # [B, R, packed_C]
        scale.to(torch.float32),  # [B, R, 1]
        i8scales,            # 来自 repack
        i8zeros
    )
