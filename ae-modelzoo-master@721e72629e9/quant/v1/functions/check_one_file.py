from typing import List, Dict, Set
from collections import defaultdict


def check_completeness(weight_names: List[str], suffixes: List[str]):
    """
    检查权重是否完整。
    
    Parameters:
        weight_names: 所有权重名字列表
        suffixes: 每个 layer 必须包含的尾缀，如 ["weight_packed", "weight_scale", "weight_shape"]

    Returns:
        complete_dict:   {prefix: set_of_found_suffix}
        incomplete_dict: {prefix: missing_suffixes}
    """

    suffix_set = set(suffixes)
    prefix_map: Dict[str, Set[str]] = {}

    # 1. 遍历所有权重名字，找出suffix对应的prefix
    for w in weight_names:
        for suf in suffixes:
            if w.endswith(suf):
                # 截掉 '.suffix' 得到 prefix
                prefix = w[: -(len(suf) + 1)]  # +1 是因为有一个点 '.'
                prefix_map.setdefault(prefix, set()).add(suf)

    # 2. 检查每个 prefix 是否包含全部 suffix
    complete_dict = defaultdict()    # prefix -> 已找到 suffix
    incomplete_dict = defaultdict()  # prefix -> 缺少的 suffix

    for prefix, found in prefix_map.items():
        if found == suffix_set:
            complete_dict[prefix] = found
        else:
            missing = suffix_set - found
            incomplete_dict[prefix] = missing

    assert len(incomplete_dict) == 0, f"incomplete weight_name: {incomplete_dict.keys()}"


if __name__ == "__main__":
    # weight_names = [
    #     "layers.0.ffn.w1.weight_packed",
    #     "layers.0.ffn.w1.weight_scale",
    #     "layers.0.ffn.w1.weight_shape",

    #     "layers.1.attn.q_proj.weight_packed",
    #     "layers.1.attn.q_proj.weight_scale",
    # ]
    from glob import glob
    from safetensors.torch import load_file

    all_files = glob("/home/mashun/vllm_project/checkpoints/Kimi-K2-Thinking/*.safetensors")

    for f in all_files:
        weight_names = load_file(f).keys()

        suffixes = ["weight_packed", "weight_scale", "weight_shape"]

        check_completeness(weight_names, suffixes)

