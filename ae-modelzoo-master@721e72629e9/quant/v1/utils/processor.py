import os
import json

from typing import List, Optional


def generate_compression_config(config, symmetric, ignore: Optional[List] = None):
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
                    "symmetric": bool(symmetric),
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


def process_config_weight_map(input_path, output_path, symmetric, ignore):
    config_path = os.path.join(input_path, "config.json")
    config_save_path = os.path.join(output_path, "config.json")

    with open(config_path, "r") as f_open:
        config = json.load(f_open)

    # 移除原量化字段（如果存在）
    config.pop("quantization_config", None)

    # 增加 int8 量化配置
    config = generate_compression_config(config, symmetric, ignore)

    with open(config_save_path, "w") as f_save:
        json.dump(config, f_save, indent=4)