# 反量化代码
from .kernels.int4_to_bf16 import weight_dequant_int4_wrapper


def fake_func(state_dict, *args, **kwargs):
    return state_dict


def dispatch_low_to_high_func(dtype: str):
    if dtype in ["bf16", "fp16", "fp32"]:
        return fake_func
    elif dtype == "int4":
        return weight_dequant_int4_wrapper
    else:
        raise NotImplemented