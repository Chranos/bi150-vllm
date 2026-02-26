import os
import json
import torch
import ixformer.inference.functions as ixfop

from typing import List, Optional


def weight_quant_int8(v: torch.Tensor):
    assert v.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(v).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (v.shape[0], 1)
    quantized = torch.round(v / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def weight_quant_moe_int4_pack8(
    v: torch.Tensor,
    block_size: int = 128,
    group_size=-1,
    format="TN",
    symmetric=True,
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


def generate_compression_config(config, ignore: Optional[List] = None):
    config["compression_config"] = {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "block_structure": None,
                    "dynamic": True,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "memoryless",
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": True,
                    "type": "int",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": None,
                    "num_bits": 8,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": True,
                    "type": "int",
                },
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": 1.0,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen",
    }

    return config


def process_config_weight_map(input_path, output_path, ignore):
    config_path = os.path.join(input_path, "config.json")
    config_save_path = os.path.join(output_path, "config.json")

    with open(config_path, "r") as f_open:
        config = json.load(f_open)

    # 移除原量化字段（如果存在）
    config.pop("quantization_config", None)

    # 增加 int8 量化配置
    config = generate_compression_config(config, ignore)

    with open(config_save_path, "w") as f_save:
        json.dump(config, f_save, indent=4)


# [
#                 "model.visual.blocks.0.attn.qkv",
#                 "model.visual.blocks.0.attn.proj",
#                 "model.visual.blocks.0.mlp.linear_fc1",
#                 "model.visual.blocks.0.mlp.linear_fc2",
#                 "model.visual.blocks.1.attn.qkv",
#                 "model.visual.blocks.1.attn.proj",
#                 "model.visual.blocks.1.mlp.linear_fc1",
#                 "model.visual.blocks.1.mlp.linear_fc2",
#                 "model.visual.blocks.2.attn.qkv",
#                 "model.visual.blocks.2.attn.proj",
#                 "model.visual.blocks.2.mlp.linear_fc1",
#                 "model.visual.blocks.2.mlp.linear_fc2",
#                 "model.visual.blocks.3.attn.qkv",
#                 "model.visual.blocks.3.attn.proj",
#                 "model.visual.blocks.3.mlp.linear_fc1",
#                 "model.visual.blocks.3.mlp.linear_fc2",
#                 "model.visual.blocks.4.attn.qkv",
#                 "model.visual.blocks.4.attn.proj",
#                 "model.visual.blocks.4.mlp.linear_fc1",
#                 "model.visual.blocks.4.mlp.linear_fc2",
#                 "model.visual.blocks.5.attn.qkv",
#                 "model.visual.blocks.5.attn.proj",
#                 "model.visual.blocks.5.mlp.linear_fc1",
#                 "model.visual.blocks.5.mlp.linear_fc2",
#                 "model.visual.blocks.6.attn.qkv",
#                 "model.visual.blocks.6.attn.proj",
#                 "model.visual.blocks.6.mlp.linear_fc1",
#                 "model.visual.blocks.6.mlp.linear_fc2",
#                 "model.visual.blocks.7.attn.qkv",
#                 "model.visual.blocks.7.attn.proj",
#                 "model.visual.blocks.7.mlp.linear_fc1",
#                 "model.visual.blocks.7.mlp.linear_fc2",
#                 "model.visual.blocks.8.attn.qkv",
#                 "model.visual.blocks.8.attn.proj",
#                 "model.visual.blocks.8.mlp.linear_fc1",
#                 "model.visual.blocks.8.mlp.linear_fc2",
#                 "model.visual.blocks.9.attn.qkv",
#                 "model.visual.blocks.9.attn.proj",
#                 "model.visual.blocks.9.mlp.linear_fc1",
#                 "model.visual.blocks.9.mlp.linear_fc2",
#                 "model.visual.blocks.10.attn.qkv",
#                 "model.visual.blocks.10.attn.proj",
#                 "model.visual.blocks.10.mlp.linear_fc1",
#                 "model.visual.blocks.10.mlp.linear_fc2",
#                 "model.visual.blocks.11.attn.qkv",
#                 "model.visual.blocks.11.attn.proj",
#                 "model.visual.blocks.11.mlp.linear_fc1",
#                 "model.visual.blocks.11.mlp.linear_fc2",
#                 "model.visual.blocks.12.attn.qkv",
#                 "model.visual.blocks.12.attn.proj",
#                 "model.visual.blocks.12.mlp.linear_fc1",
#                 "model.visual.blocks.12.mlp.linear_fc2",
#                 "model.visual.blocks.13.attn.qkv",
#                 "model.visual.blocks.13.attn.proj",
#                 "model.visual.blocks.13.mlp.linear_fc1",
#                 "model.visual.blocks.13.mlp.linear_fc2",
#                 "model.visual.blocks.14.attn.qkv",
#                 "model.visual.blocks.14.attn.proj",
#                 "model.visual.blocks.14.mlp.linear_fc1",
#                 "model.visual.blocks.14.mlp.linear_fc2",
#                 "model.visual.blocks.15.attn.qkv",
#                 "model.visual.blocks.15.attn.proj",
#                 "model.visual.blocks.15.mlp.linear_fc1",
#                 "model.visual.blocks.15.mlp.linear_fc2",
#                 "model.visual.blocks.16.attn.qkv",
#                 "model.visual.blocks.16.attn.proj",
#                 "model.visual.blocks.16.mlp.linear_fc1",
#                 "model.visual.blocks.16.mlp.linear_fc2",
#                 "model.visual.blocks.17.attn.qkv",
#                 "model.visual.blocks.17.attn.proj",
#                 "model.visual.blocks.17.mlp.linear_fc1",
#                 "model.visual.blocks.17.mlp.linear_fc2",
#                 "model.visual.blocks.18.attn.qkv",
#                 "model.visual.blocks.18.attn.proj",
#                 "model.visual.blocks.18.mlp.linear_fc1",
#                 "model.visual.blocks.18.mlp.linear_fc2",
#                 "model.visual.blocks.19.attn.qkv",
#                 "model.visual.blocks.19.attn.proj",
#                 "model.visual.blocks.19.mlp.linear_fc1",
#                 "model.visual.blocks.19.mlp.linear_fc2",
#                 "model.visual.blocks.20.attn.qkv",
#                 "model.visual.blocks.20.attn.proj",
#                 "model.visual.blocks.20.mlp.linear_fc1",
#                 "model.visual.blocks.20.mlp.linear_fc2",
#                 "model.visual.blocks.21.attn.qkv",
#                 "model.visual.blocks.21.attn.proj",
#                 "model.visual.blocks.21.mlp.linear_fc1",
#                 "model.visual.blocks.21.mlp.linear_fc2",
#                 "model.visual.blocks.22.attn.qkv",
#                 "model.visual.blocks.22.attn.proj",
#                 "model.visual.blocks.22.mlp.linear_fc1",
#                 "model.visual.blocks.22.mlp.linear_fc2",
#                 "model.visual.blocks.23.attn.qkv",
#                 "model.visual.blocks.23.attn.proj",
#                 "model.visual.blocks.23.mlp.linear_fc1",
#                 "model.visual.blocks.23.mlp.linear_fc2",
#                 "model.visual.blocks.24.attn.qkv",
#                 "model.visual.blocks.24.attn.proj",
#                 "model.visual.blocks.24.mlp.linear_fc1",
#                 "model.visual.blocks.24.mlp.linear_fc2",
#                 "model.visual.blocks.25.attn.qkv",
#                 "model.visual.blocks.25.attn.proj",
#                 "model.visual.blocks.25.mlp.linear_fc1",
#                 "model.visual.blocks.25.mlp.linear_fc2",
#                 "model.visual.blocks.26.attn.qkv",
#                 "model.visual.blocks.26.attn.proj",
#                 "model.visual.blocks.26.mlp.linear_fc1",
#                 "model.visual.blocks.26.mlp.linear_fc2",
#                 "model.visual.merger.linear_fc1",
#                 "model.visual.merger.linear_fc2",
#                 "model.visual.deepstack_merger_list.0.linear_fc1",
#                 "model.visual.deepstack_merger_list.0.linear_fc2",
#                 "model.visual.deepstack_merger_list.1.linear_fc1",
#                 "model.visual.deepstack_merger_list.1.linear_fc2",
#                 "model.visual.deepstack_merger_list.2.linear_fc1",
#                 "model.visual.deepstack_merger_list.2.linear_fc2",
#                 "model.language_model.layers.0.mlp.gate",
#                 "model.language_model.layers.1.mlp.gate",
#                 "model.language_model.layers.2.mlp.gate",
#                 "model.language_model.layers.3.mlp.gate",
#                 "model.language_model.layers.4.mlp.gate",
#                 "model.language_model.layers.5.mlp.gate",
#                 "model.language_model.layers.6.mlp.gate",
#                 "model.language_model.layers.7.mlp.gate",
#                 "model.language_model.layers.8.mlp.gate",
#                 "model.language_model.layers.9.mlp.gate",
#                 "model.language_model.layers.10.mlp.gate",
#                 "model.language_model.layers.11.mlp.gate",
#                 "model.language_model.layers.12.mlp.gate",
#                 "model.language_model.layers.13.mlp.gate",
#                 "model.language_model.layers.14.mlp.gate",
#                 "model.language_model.layers.15.mlp.gate",
#                 "model.language_model.layers.16.mlp.gate",
#                 "model.language_model.layers.17.mlp.gate",
#                 "model.language_model.layers.18.mlp.gate",
#                 "model.language_model.layers.19.mlp.gate",
#                 "model.language_model.layers.20.mlp.gate",
#                 "model.language_model.layers.21.mlp.gate",
#                 "model.language_model.layers.22.mlp.gate",
#                 "model.language_model.layers.23.mlp.gate",
#                 "model.language_model.layers.24.mlp.gate",
#                 "model.language_model.layers.25.mlp.gate",
#                 "model.language_model.layers.26.mlp.gate",
#                 "model.language_model.layers.27.mlp.gate",
#                 "model.language_model.layers.28.mlp.gate",
#                 "model.language_model.layers.29.mlp.gate",
#                 "model.language_model.layers.30.mlp.gate",
#                 "model.language_model.layers.31.mlp.gate",
#                 "model.language_model.layers.32.mlp.gate",
#                 "model.language_model.layers.33.mlp.gate",
#                 "model.language_model.layers.34.mlp.gate",
#                 "model.language_model.layers.35.mlp.gate",
#                 "model.language_model.layers.36.mlp.gate",
#                 "model.language_model.layers.37.mlp.gate",
#                 "model.language_model.layers.38.mlp.gate",
#                 "model.language_model.layers.39.mlp.gate",
#                 "model.language_model.layers.40.mlp.gate",
#                 "model.language_model.layers.41.mlp.gate",
#                 "model.language_model.layers.42.mlp.gate",
#                 "model.language_model.layers.43.mlp.gate",
#                 "model.language_model.layers.44.mlp.gate",
#                 "model.language_model.layers.45.mlp.gate",
#                 "model.language_model.layers.46.mlp.gate",
#                 "model.language_model.layers.47.mlp.gate",
#                 "lm_head",
#                 "model.language_model.embed_tokens"
#             ],
