import os
import torch
import json
import shutil
import multiprocessing as mp

from glob import glob
from multiprocessing import Manager

from worker import worker
from functions.dequant import dequant
from utils import (
    load_quant_config,
    load_config,
    copy_files,
    get_ignore,
    process_and_save_info,
)

from natsort import natsorted


def main(
    input_dir,
    output_dir,
    model_type,
    mode,
    num_workers,
    custom_dequant_config,
    custom_quant_config
):
    print(f"Make output_dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Copy files from {input_dir} to {output_dir}")
    copy_files(input_dir, output_dir)

    safetensor_files = glob(os.path.join(input_dir, "*.safetensors"))
    safetensor_files = natsorted(list(safetensor_files))
    assert len(safetensor_files) > 0

    num_gpus = torch.cuda.device_count()
    num_workers = min(num_workers, min(num_gpus, len(safetensor_files)))
    assert (
        num_workers > 0
    ), f"No CUDA devices found or num_workers incorrect: {num_workers}"

    print(f"{num_workers} GPUs will be used")
    print(f"Found {len(safetensor_files)} safetensor files")

    task_queue = mp.Queue()
    for sf in safetensor_files:
        task_queue.put(sf)

    processes = []
    manager = Manager()
    weight_map_shared = manager.dict()
    quant_map_shared = manager.dict()

    try:
        quant_config = load_quant_config(input_dir)
    except Exception as e:
        quant_config = None

    for gpu_id in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                task_queue,
                gpu_id,
                weight_map_shared,
                quant_map_shared,
                output_dir,
                mode,
                model_type,
                quant_config,
                custom_dequant_config,
                custom_quant_config,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    process_and_save_info(
        input_dir,
        output_dir,
        dict(weight_map_shared),
        mode == "quant" or mode == "dequant+quant",
        custom_quant_config.get("symmetric"),
        ignore=None if mode == "dequant" else get_ignore(custom_quant_config)
    )

    with open("quant_map.json", "w") as f:
        json.dump(dict(quant_map_shared), f, indent=2)


if __name__ == "__main__":
    import os
    import yaml
    from argparse import ArgumentParser

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("--config", type=str)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(
        cfg['paths']['checkpoint_dir'],
        cfg['paths']['output_dir'],
        cfg['model']['model_type'],
        cfg['runtime']['mode'],
        cfg['runtime']['num_workers'],
        cfg['dequant'],
        cfg['quant']
    )
