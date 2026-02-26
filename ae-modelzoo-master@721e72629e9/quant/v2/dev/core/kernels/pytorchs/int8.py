import torch


def weight_quant_bf16_to_int8(inputs: torch.Tensor):
    assert inputs.dim() == 3, f"inputs shape is [batch, output_dim, input_dim], but got {inputs.dim()}"

    qmax = 127.0
    abs_max = torch.abs(inputs).max(dim=2, keepdim=True)[0]
    scale = abs_max / qmax

    assert scale.shape == (*inputs.shape[:2], 1)

    quantized = torch.round(inputs / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)

