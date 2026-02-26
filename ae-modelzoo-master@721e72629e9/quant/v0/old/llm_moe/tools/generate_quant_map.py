import os
import json

from glob import glob
from natsort import natsorted
from argparse import ArgumentParser


def main(
    input_dir,
    expert_quant_dtype: str = "int8",
    other_quant_dtype: str = "int8",
    default_dtype: str = "bfloat16"
):
    custom_ignores = ['lm_head', 'gate.weight', 'norm', 'bias', 'embed_tokens']

    model_name = input_dir.rsplit(os.sep, 1)[-1]

    output_dir = os.path.join("quant_configs", model_name)
    os.makedirs(output_dir, exist_ok=True)
    # print(output_dir)
    # import pdb
    # pdb.set_trace()
    idx = len(glob(f"{output_dir}/*.json")) + 1
    
    output_path = os.path.join("quant_configs", model_name, f"quant_mapping_{idx}.json")

    model_index_file = os.path.join(input_dir, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    quant_map = {}

    for weight_name, _ in weight_map.items():
        if any(x in weight_name for x in custom_ignores):
            quant_map[weight_name] = default_dtype
            continue
        if "expert" in weight_name:
            quant_map[weight_name] = expert_quant_dtype
        else:
            quant_map[weight_name] = other_quant_dtype
    
    sorted_dict = {k: quant_map[k] for k in natsorted(quant_map)}

    with open(output_path, "w") as f:
        json.dump(sorted_dict, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--expert-quant-dtype", type=str, default="int4")
    parser.add_argument("--other-quant-dtype", type=str, default="int8")

    args = parser.parse_args()

    main(args.input_dir, args.expert_quant_dtype, args.other_quant_dtype)
