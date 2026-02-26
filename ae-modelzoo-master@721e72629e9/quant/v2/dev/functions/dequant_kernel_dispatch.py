from core import weight_dequant_fp8e4m3_to_bf16

import torch


def fake_kernel(*args, **kwargs):
    return args


def dispatch_kernel(dtype):
    if dtype in [torch.bfloat16, torch.float16, torch.half, torch.float, torch.float32]:
        return fake_kernel
    elif dtype == torch.float8_e4m3fn:
        return weight_dequant_fp8e4m3_to_bf16
    else:
        raise NotImplemented