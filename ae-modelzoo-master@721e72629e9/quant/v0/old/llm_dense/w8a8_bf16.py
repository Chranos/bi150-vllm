import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import shutil
import torch
from safetensors.torch import load_file, save_file


def weight_dequant(x: torch.Tensor, dummy_scale: torch.Tensor = None):
    """
    简单将 BF16 权重线性量化到 INT8，并返回 (int8_tensor, scale)
    按行（out_channel）计算 scale。
    """
    assert x.dim() == 2, "Only 2D weights are supported for quantization."
    qmax = 127.0

    abs_max = torch.abs(x).max(dim=1, keepdim=True)[0] + 1e-8  # 防止除零
    scale = abs_max / qmax
    quantized = torch.round(x / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def process_config_weight_map(fp8_path, int8_path):
    """修改 config.json，添加 int8 压缩配置"""
    config_path = os.path.join(fp8_path, "config.json")
    config_save_path = os.path.join(int8_path, "config.json")

    with open(config_path, "r") as f_open:
        config = json.load(f_open)

    # 移除原量化字段（如果存在）
    config.pop("quantization_config", None)

    # 增加 int8 量化配置
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
                    "type": "int"
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
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": 1.0,
        "ignore": ["lm_head", "embed_tokens"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "frozen"
    }

    with open(config_save_path, "w") as f_save:
        json.dump(config, f_save, indent=4)


def main(bf16_path, int8_path):
    """
    将 BF16 权重转换为 INT8（仅量化 Linear 层），其余保持 BF16。
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)

    # 载入模型索引
    model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    loaded_files = {}
    new_weight_map = {}

    # 拷贝非权重文件（tokenizer, config 等）
    for file in os.listdir(bf16_path):
        if not os.path.isdir(os.path.join(bf16_path, file)) and not file.endswith(".safetensors"):
            shutil.copy(os.path.join(bf16_path, file), os.path.join(int8_path, file))

    # 处理 config.json
    process_config_weight_map(bf16_path, int8_path)

    safetensor_files = sorted(glob(os.path.join(bf16_path, "*.safetensors")))

    for safetensor_file in tqdm(safetensor_files, desc="Converting to INT8"):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():
            # 不量化 embedding / lm_head / norm / bias
            if any(x in weight_name for x in ["embed_tokens", "lm_head", "norm", "bias"]):
                new_state_dict[weight_name] = weight.to(torch.bfloat16)
                new_weight_map[weight_name] = file_name
                continue

            # 仅量化 Linear 层权重
            if weight.ndim == 2 and "weight" in weight_name:
                int8_v, scale = weight_dequant(weight)
                new_state_dict[weight_name] = int8_v
                new_state_dict[f"{weight_name}_scale"] = scale
                new_weight_map[weight_name] = file_name
                new_weight_map[f"{weight_name}_scale"] = file_name
            else:
                # 其他保持原精度
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        # 保存新的 safetensors 文件
        save_file(new_state_dict, os.path.join(int8_path, file_name))

        # 清理缓存节省显存
        del current_state_dict
        torch.cuda.empty_cache()

    # 更新模型索引
    model_index["weight_map"] = new_weight_map
    new_model_index_file = os.path.join(int8_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"✅ INT8 模型转换完成，结果已保存到: {int8_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-bf16-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_bf16_hf_path, args.output_int8_hf_path)
