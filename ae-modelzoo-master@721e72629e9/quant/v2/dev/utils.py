import os
import json
import shutil


def load_config(input_dir: str) -> dict:
    for fname in ("config.json", "params.json"):
        path = os.path.join(input_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f), fname
    raise ValueError(f"Cannot find quantization config in {input_dir}")


def load_quant_config(input_dir: str) -> dict:
    return load_config(input_dir)[0]["quantization_config"]


def get_ignore(quantization_config: dict):
    ignore = []

    ignore.extend(quantization_config.get("modules_to_not_convert", []))
    ignore.extend(quantization_config.get("ignore", []))

    return ignore


def copy_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        src_path = os.path.join(input_dir, file)

        if os.path.isdir(src_path):
            continue

        if file.endswith(".safetensors"):
            continue

        dst_path = os.path.join(output_dir, file)
        shutil.copy(src_path, dst_path)


def process_and_save_info(
    input_dir,
    output_dir,
    weight_map: dict,
    quant: bool = False,
    symmetric: bool = True,
    ignore: list | None = None,
):
    # 保存weight_map
    candidate_suffixes = [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ]

    for cs in candidate_suffixes:
        input_index_path = os.path.join(input_dir, cs)
        output_index_path = os.path.join(output_dir, cs)
        if os.path.exists(input_index_path):
            break
    else:
        raise ValueError(f"Can not find index file, {input_dir}")

    with open(input_index_path, "r") as f:
        ori_model_index = json.load(f)

    ori_model_index["weight_map"] = weight_map

    with open(output_index_path, "w") as f:
        json.dump(ori_model_index, f, indent=2, sort_keys=True)

    config, fname = load_config(output_dir)
    file_name = os.path.join(output_dir, fname)

    key = ""

    try:
        config.pop("quantization_config")
    except Exception as e:
        key = "compression_config"
        pass
    
    try:
        config.pop("compression_config")
    except Exception as e:
        key = "quantization_config"
        pass

    if quant:
        config[key] = {
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
    
    with open(file_name, "w") as f:
        json.dump(config, f, indent=2)
