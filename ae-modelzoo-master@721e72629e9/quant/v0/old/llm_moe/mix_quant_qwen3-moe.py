import os
import json
import shutil
import torch
import argparse
from glob import glob
from multiprocessing import Process, Manager, Queue

from safetensors.torch import load_file, save_file
from typing import Dict

from quant_utils import (
    weight_quant_int4_pack8,
    weight_quant_moe_int4_pack8,
    weight_quant_int8,
    process_config_weight_map
)


#########################
# Quantization Utilities
#########################


def quantize_weight(weight, quant_dtype, is_expert, version):
    if quant_dtype == "int4":
        quant_fn = weight_quant_moe_int4_pack8 if is_expert else weight_quant_int4_pack8
        quant_weights, scale, i8scales, i8zeros = quant_fn(weight)
    elif quant_dtype == "int8":
        quant_weights, scale = weight_quant_int8(weight)
        i8scales, i8zeros = None, None
    else:
        raise NotImplementedError(f"Unsupported quant dtype: {quant_dtype}")

    if scale is not None:
        scale = scale.contiguous().view(1, -1) if version == 2 else scale.contiguous()

    if i8scales is not None:
        i8scales = i8scales.squeeze_(0)
    if i8zeros is not None:
        i8zeros = i8zeros.squeeze_(0)

    return quant_weights.contiguous(), scale, i8scales, i8zeros


def update_state_dict(
    new_sd, weight_map, fname, wname, quant_w, scale, i8s, i8z, quant_dtype
):
    scale_name = f"{wname}_scale"
    i8s_name = f"{wname}_i8_weight_scale"
    i8z_name = f"{wname}_i8_weight_zero"

    if (scale is not None) and quant_dtype == "int8":
        scale = scale.T.contiguous()

    new_sd[wname] = quant_w

    if scale is not None:
        new_sd[scale_name] = scale
    if i8s is not None:
        new_sd[i8s_name] = i8s
    if i8z is not None:
        new_sd[i8z_name] = i8z

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


def worker(gpu_id, task_queue, output_dir, version, quant_map, weight_map_shared, default_dtype):
    device = f"cuda:{gpu_id}"

    while not task_queue.empty():
        try:
            input_file = task_queue.get_nowait()
        except:
            break

        fname = os.path.basename(input_file)
        print(f"[GPU {gpu_id}] Processing {fname}")

        sd = load_file(input_file, device=device)
        new_sd = {}

        for wname, weight in sd.items():
            target_dtype = quant_map[wname]
            if target_dtype == default_dtype:
                quant_w, scale, i8s, i8z = weight, None, None, None
            else:
                quant_w, scale, i8s, i8z = quantize_weight(
                    weight, target_dtype, ("expert" in wname), version
                )

            update_state_dict(
                new_sd,
                weight_map_shared,
                fname,
                wname,
                quant_w,
                scale,
                i8s,
                i8z,
                target_dtype,
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
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--default-dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_index_file = os.path.join(args.input_dir, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)

    for file in os.listdir(args.input_dir):
        if not os.path.isdir(os.path.join(args.input_dir, file)) and not file.endswith(".safetensors"):
            shutil.copy(os.path.join(args.input_dir, file), os.path.join(args.output_dir, file))

    with open(args.quant_map_json, "r") as f:
        quant_map = json.load(f)

    tensor_files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    assert len(tensor_files) > 0

    manager = Manager()
    weight_map_shared = manager.dict()

    # Task Queue
    task_queue = manager.Queue()
    for f in tensor_files:
        task_queue.put(f)

    procs = []
    gpu_list = list(range(args.num_gpus))

    for gpu_id in gpu_list:
        p = Process(
            target=worker,
            args=(
                gpu_id,
                task_queue,
                args.output_dir,
                args.version,
                quant_map,
                weight_map_shared,
                args.default_dtype
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Save mapping
    map_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    model_index["weight_map"] = dict(weight_map_shared)
    with open(map_path, "w") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)

    ignore = []
    # 处理config.json
    for w, wdtype in quant_map.items():
        if wdtype == args.default_dtype:
            ignore.append(w.rsplit(".", 1)[0])
    process_config_weight_map(args.input_dir, args.output_dir, ignore)    

    print("\n=== All tasks completed! ===")
    print(f"Saved weight map to: {map_path}")


if __name__ == "__main__":
    main()
