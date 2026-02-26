import os
import re
import json
from glob import glob
from argparse import ArgumentParser
from datetime import datetime
from natsort import natsorted

VALID_DTYPES = {"bf16", "fp16", "fp32", "int4", "int8"}


def validate_dtype(dtype: str):
    if dtype not in VALID_DTYPES:
        raise ValueError(f"Invalid dtype '{dtype}'. Valid options: {VALID_DTYPES}")


def match_ignore(name: str, rules):
    """支持 substring & 正则两种匹配方式"""
    for rule in rules:
        # 子串匹配
        if rule in name:
            return True
        # 正则匹配
        try:
            if re.search(rule, name):
                return True
        except re.error:
            pass  # 正则错误则跳过
    return False


def load_weight_map(input_dir: str):
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing model index file: {index_path}")

    with open(index_path, "r") as f:
        idx = json.load(f)
    return idx.get("weight_map", {})


def build_output_path(model_name):
    out_dir = os.path.join("quant_configs", model_name)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(out_dir, f"quant_mapping_{timestamp}.json")


def main(
    input_dir,
    expert_quant_dtype="int8",
    other_quant_dtype="int8",
    default_dtype="bf16",
    ignore_rules=None,
):

    # ================
    # 参数合法性检查
    # ================
    for dt in [expert_quant_dtype, other_quant_dtype, default_dtype]:
        validate_dtype(dt)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input dir does not exist: {input_dir}")

    # ======================
    # 获取 model 名字 + 输出路径
    # ======================
    model_name = os.path.basename(input_dir.rstrip("/"))
    output_path = build_output_path(model_name)

    # ================
    # Load weight_map
    # ================
    weight_map = load_weight_map(input_dir)

    ignore_rules = ignore_rules or [
        "lm_head", "gate.weight", "norm", "bias", "embed_tokens", "vision", "visual", "rotary_emb", "shared_experts"
    ]
    quant_map = {}
    stats = {"expert": 0, "other": 0, "ignored": 0}

    # =====================
    # 逐个权重分配量化 dtype
    # =====================
    for name in natsorted(weight_map.keys()):
        if match_ignore(name, ignore_rules):
            quant_map[name] = default_dtype
            stats["ignored"] += 1
            continue

        if "expert" in name:
            quant_map[name] = expert_quant_dtype
            stats["expert"] += 1
        else:
            quant_map[name] = other_quant_dtype
            stats["other"] += 1

    # =====================
    # 保存 JSON
    # =====================
    with open(output_path, "w") as f:
        json.dump(quant_map, f, indent=2)

    # =====================
    # 显示 summary
    # =====================
    print("\n✔ Quant mapping created:")
    print("  Path:", output_path)
    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k:8}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--expert-quant-dtype", type=str, default="int4")
    parser.add_argument("--other-quant-dtype", type=str, default="int8")
    parser.add_argument("--default-dtype", type=str, default="bf16")
    parser.add_argument("--ignore-rule", action="append",
                        help="Add custom ignore rules (substring or regex). Can be used multiple times.")

    args = parser.parse_args()

    main(
        args.input_dir,
        args.expert_quant_dtype,
        args.other_quant_dtype,
        args.default_dtype,
        args.ignore_rule,
    )
