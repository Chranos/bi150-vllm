# 过滤明确量化/非量化的layer_name
import re
import os
import json

from collections import OrderedDict


def filter(input_dir: str, filter_regex_list: list | None = None):

    if filter_regex_list is None:
        filter_regex_list = []

    candidate_suffixes = [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ]

    for cs in candidate_suffixes:
        input_index_path = os.path.join(input_dir, cs)
        if os.path.exists(input_index_path):
            break
    else:
        raise ValueError(f"Can not find index file, {input_dir}")
    
    with open(input_index_path, "r") as f:
        weight_map = json.load(f)['weight_map']
    
    deduped = deduplicate_weight_map(weight_map)

    kept_weight_names = {}
    
    for w, sf in deduped.items():
        if any(re.match(p, w) for p in filter_regex_list):
            continue
        kept_weight_names[w] = sf
    
    with open("temp.json", "w") as f:
        json.dump(kept_weight_names, f, indent=2)


def structure_key(weight_name: str) -> str:
    """
    将 layers.<num>. 归一化为 layers.*.，用于结构级去重
    """
    return re.sub(r"model.layers\.\d+\.", "model.layers.*.", weight_name)


def deduplicate_weight_map(weight_map: dict):
    """
    对 weight_map 进行结构去重，只保留第一个出现的结构
    """
    seen = set()
    deduped = OrderedDict()

    for name, file in weight_map.items():
        key = structure_key(name)
        if key in seen:
            continue
        seen.add(key)
        deduped[name] = file

    return deduped


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--input-dir", type=str)

    args = parser.parse_args()

    filter(args.input_dir)



