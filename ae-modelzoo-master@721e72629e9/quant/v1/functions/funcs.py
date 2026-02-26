from .kernels.ixformers_k import *


def process(inputs, is_transpose: bool):
    if inputs.dim() != 2:
        return inputs
    if is_transpose:
        return inputs.transpose(0, 1)
    return inputs


def postprocess(inputs, quant_dtype, version, is_transpose: bool, is_expert: bool):
    quant_weight, scale, i8scales, i8zeros = inputs

    if is_expert:
        scale = scale.contiguous().view(1, -1) if version == 2 else scale.contiguous()
        if is_transpose:
            quant_weight = quant_weight.transpose(0, 1)
            if quant_dtype == "int4":
                scale = scale.transpose(0, 1)
        else:
            if quant_dtype == "int8":
                scale = scale.T
        
        if i8scales is not None:
            i8scales.squeeze_(0)
            assert i8scales.dim() == 2
        
        if i8zeros is not None:
            i8zeros.squeeze_(0)
            assert i8zeros.dim() == 2
    
    return quant_weight, scale, i8scales, i8zeros


def quantize_weight(
    weight: torch.Tensor,
    quant_dtype: str,
    version: int,
    symmetric: bool, 
    is_transpose: bool = False,
    is_expert: bool = False
):
    """
    quant_dtype: 支持int4-pack8, int8
    is_transpose: 用于处理qwen3-vl-moe相似权重
    """
    if quant_dtype in ['bf16', "fp16", "fp32"]:
        return weight, None, None, None

    weight = process(weight, is_transpose)

    quant_fn = weight_quant_int4_pack8 if quant_dtype == "int4" else weight_quant_int8

    outputs = quant_fn(weight)

    outputs = postprocess(outputs, quant_dtype, version, is_transpose, is_expert)

    return outputs
