import re
import os
import json
import torch

from safetensors.torch import load_file

from .specs import *
from .utils import compile_ignore, get_ignore, estimate_gpu_bytes
from .dequant_register import DEQUANT_KERNEL_TABLE
from .config_builder import create_quant_config_builder, resolve_quant_artifacts, to_dequant_spec


def dequantize_(
    tensors: dict[str, torch.Tensor],
    qconfig: dict,
    custom_dequant_config,
    output_dtype=torch.bfloat16,
    device: torch.device = "cpu",
    **kwargs
):
    ignore_exact, ignore_regex = compile_ignore(
        get_ignore(qconfig),
        get_ignore(custom_dequant_config),
    )

    builder = create_quant_config_builder(
        qconfig,
        custom_dequant_config,
        model_type=kwargs.get("model_type")
    )

    outputs = {}
    gpu_cache = {}
    gpu_cache_size = 0
    max_gpu_cache_size = 16 * (1024**3)

    for weight_name in tensors.keys():
        rq = builder.resolve(weight_name)

        if rq is None:  # 包含scale的一定不会在这里
            assert "scale" not in weight_name, f"{weight_name}"
            outputs[weight_name] = tensors[weight_name]
            continue

        assert not rq.dynamic, "Dynamic quantization not supported in offline dequant"

        artifacts = resolve_quant_artifacts(weight_name, tensors, rq)

        if artifacts is None:
            # 跳过scale, shape等
            continue

        spec = to_dequant_spec(rq)
        if spec not in DEQUANT_KERNEL_TABLE:
            raise RuntimeError(f"No dequant kernel registered for {spec}")

        kernel = DEQUANT_KERNEL_TABLE[spec]

        gpu_cache_size += estimate_gpu_bytes(artifacts, rq)

        try:
            output = kernel(
                artifacts=artifacts,
                rq=rq,
                output_dtype=output_dtype,
                device=device,
            )
        except Exception as e:
            print(weight_name)
            raise e
        # 处理名称
        weight_name = weight_name.replace("weight_packed", "weight").replace("qweight", "weight")
        gpu_cache[weight_name] = output

        if gpu_cache_size >= max_gpu_cache_size:
            for k, v in gpu_cache.items():
                outputs[k] = v.cpu()
            gpu_cache.clear()
            gpu_cache_size = 0
    
    for k, v in gpu_cache.items():
        outputs[k] = v.cpu()
        
    gpu_cache.clear()
    gpu_cache_size = 0

    return outputs


def dequant(file_name: str,
            quant_config,
            custom_dequant_config,
            device: torch.device,
            model_type):

    tensors = load_file(file_name)

    return dequantize_(tensors, quant_config, custom_dequant_config, device=device, model_type=model_type)


if __name__ == "__main__":
    ...
