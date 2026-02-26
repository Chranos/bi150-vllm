import os
import json
import shutil
import torch
import argparse
from glob import glob
from multiprocessing import Process, Manager, Queue, set_start_method

from safetensors.torch import load_file, save_file
from typing import Dict, List

from utils import process_config_weight_map
from functions import quantize_weight, dispatch_low_to_high_func

from collections import defaultdict

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def update_quant_map_after_dequant(
    quant_map_old: Dict[str, str],
    state_dict_new: Dict[str, torch.Tensor],
    quanted_suffixes: List[str]
):
    new_quant_map = {}
    new_keys = set(state_dict_new.keys())
    # 1. 完全匹配：旧 key 在新模型中还存在，直接继承
    for old_key, old_dtype in quant_map_old.items():
        if old_key in new_keys:
            new_quant_map[old_key] = old_dtype

    if set(new_quant_map.keys()) == set(state_dict_new.keys()):
        return new_quant_map

    prefix_map = {}
    for w, dtype in quant_map_old.items():
        for suf in quanted_suffixes:
            if w.endswith(suf):
                prefix = w.split(suf)[0]
                prefix_map.setdefault(prefix, {})[suf] = dtype
                break

    for prefix, suffix_map in prefix_map.items():
        assert len(set(suffix_map.values())) == 1
        dtype = list(suffix_map.values())[0]
        new_name = prefix + ".weight"
        if new_name in state_dict_new.keys():
            new_quant_map[new_name] = dtype

    assert set(new_quant_map.keys()) == set(state_dict_new.keys())

    return new_quant_map


def maybe_ignore(mapping):
    groups = defaultdict(list)
    for k, v in mapping.items():
        prefix = k.rsplit(".", 1)[0]  # 取前缀（. 之前的部分）
        groups[prefix].append((k, v))

    result = {}
    for prefix, items in groups.items():
        values = {v for _, v in items}
        if len(values) == 1:
            for k, v in items:
                result[k] = v

    return result


def update_state_dict(new_sd, weight_map, fname, wname, quant_w, scale, i8s, i8z):

    scale_name = f"{wname}_scale"
    i8s_name = f"{wname}_i8_weight_scale"
    i8z_name = f"{wname}_i8_weight_zero"

    new_sd[wname] = quant_w.contiguous()

    if scale is not None:
        new_sd[scale_name] = scale.contiguous()
    if i8s is not None:
        new_sd[i8s_name] = i8s.contiguous()
    if i8z is not None:
        new_sd[i8z_name] = i8z.contiguous()

    # 更新权重映射
    weight_map[wname] = fname
    if scale is not None:
        weight_map[scale_name] = fname
    if i8s is not None:
        weight_map[i8s_name] = fname
    if i8z is not None:
        weight_map[i8z_name] = fname


#########################
# Worker Process
#########################
def worker(
    gpu_id,
    task_queue,
    output_dir,
    version,
    symmetric,
    quant_map,
    weight_map_shared,
    model_quanted_dtype,
    quanted_suffixes
):
    device = f"cuda:{gpu_id}"

    convert_func = dispatch_low_to_high_func(model_quanted_dtype)

    while True:
        try:
            input_file = task_queue.get_nowait()
        except:
            break

        fname = os.path.basename(input_file)
        print(f"[GPU {gpu_id}] Processing {fname}")

        sd = load_file(input_file)
        # convert weight.dtype

        sd = convert_func(sd, quanted_suffixes, device)
        new_quant_map = update_quant_map_after_dequant(quant_map, sd, quanted_suffixes)
        new_sd = {}

        for wname, weight in sd.items():
            target_dtype = new_quant_map[wname]

            weight = weight.to(device)

            # Qwen3 MoE expert shard
            if weight.dim() == 3 and "expert" in wname:
                temp_w, temp_s, temp_i8_s, temp_i8_z = [], [], [], []
                for i in range(len(weight)):
                    w_i = weight[i]
                    quant_w, scale, i8s, i8z = quantize_weight(
                        w_i, target_dtype, version, symmetric, True, True
                    )
                    temp_w.append(quant_w)
                    if scale is not None:
                        temp_s.append(scale)
                    if i8s is not None:
                        temp_i8_s.append(i8s)
                    if i8z is not None:
                        temp_i8_z.append(i8z)

                quant_w = torch.stack(temp_w)
                scale = torch.stack(temp_s) if temp_s else None
                i8s = torch.stack(temp_i8_s) if temp_i8_s else None
                i8z = torch.stack(temp_i8_z) if temp_i8_z else None

            else:
                try:
                    quant_w, scale, i8s, i8z = quantize_weight(
                        weight,
                        target_dtype,
                        version,
                        symmetric,
                        False,
                        "expert" in wname,
                    )
                except Exception as e:
                    print(wname)
                    raise e

            update_state_dict(
                new_sd,
                weight_map_shared,
                fname,
                wname,
                quant_w,
                scale,
                i8s,
                i8z,
            )

        save_path = os.path.join(output_dir, fname)
        save_file(new_sd, save_path)
        print(f"[GPU {gpu_id}] Finished {fname} -> {save_path}")


#########################
# Main entry
#########################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quant-map-json", required=True)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--symmetric", type=bool, default=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--model-quanted-dtype", type=str, default="bf16")
    parser.add_argument("--quanted-suffixes", type=str, default=":", help="使用方法, --quanted-suffixes x1:x2:x3")
    args = parser.parse_args()
    quanted_suffixes = args.quanted_suffixes.split(":")

    os.makedirs(args.output_dir, exist_ok=True)
    for file in os.listdir(args.input_dir):
        src_path = os.path.join(args.input_dir, file)

        if os.path.isdir(src_path):
            continue

        if file.endswith(".safetensors"):
            continue

        dst_path = os.path.join(args.output_dir, file)
        shutil.copy(src_path, dst_path)

    with open(args.quant_map_json, "r") as f:
        quant_map = json.load(f)

    tensor_files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    assert tensor_files, "No .safetensors files found"

    manager = Manager()
    weight_map_shared = manager.dict()

    task_queue = manager.Queue()
    for f in tensor_files:
        task_queue.put(f)

    processes = []
    for gpu_id in range(args.num_gpus):
        p = Process(
            target=worker,
            args=(
                gpu_id,
                task_queue,
                args.output_dir,
                args.version,
                args.symmetric,
                quant_map,
                weight_map_shared,
                args.model_quanted_dtype,
                quanted_suffixes
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 写回 index.json
    index_path = os.path.join(args.input_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)

    model_index["weight_map"] = dict(weight_map_shared)

    has_bias = "bias" in ",".join(model_index["weight_map"].keys())

    new_index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(new_index_path, "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)

    maybe_ignore_l = maybe_ignore(quant_map)
    ignore = [
        w.rsplit(".", 1)[0]
        for w, t in maybe_ignore_l.items()
        if t in ["bf16", "fp16", "fp32"]
    ]
    process_config_weight_map(args.input_dir, args.output_dir, args.symmetric, ignore)

    print("\n=== All tasks completed! ===")
    print(f"Saved weight map to: {new_index_path}")


if __name__ == "__main__":
    main()
