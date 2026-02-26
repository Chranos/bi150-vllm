import torch
import ixformer.inference.functions as ixfop


def weight_quant_int8(v: torch.Tensor, **kwargs):
    assert v.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (v.shape[0], 1)
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32), None, None


def weight_quant_moe_int4_pack8(
    v: torch.Tensor,
    block_size: int = 128,
    group_size=-1,
    format="TN",
    symmetric=False,
    version=2,
):
    assert v.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (v.shape[0], 1)
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    quantized = quantized.to(torch.int8)
    i4_weights, i8scales, i8zeros = ixfop.quant_repack_int4(
        quantized.to(torch.int8).unsqueeze_(0),
        group_size,
        version,
        format,
        not symmetric,
    )
    return i4_weights.squeeze(0), scale.to(torch.float32), i8scales, i8zeros


weight_quant_int4_pack8 = weight_quant_moe_int4_pack8