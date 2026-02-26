import os
import json
import argparse
from glob import glob
from multiprocessing import Process, Manager, Queue

from safetensors.torch import load_file
import torch


# -------------------------
# dtype normalization table
# -------------------------
TORCH_DTYPE_MAP = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: ["int8", "int4"],
    torch.uint8: "uint8",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int64: "int64"
}


def normalize_torch_dtype(dtype):
    """Convert torch dtype to readable string."""
    return TORCH_DTYPE_MAP.get(dtype, str(dtype))


# -------------------------
# Worker
# -------------------------
def worker_task(gpu_id, task_queue, quant_map, mismatch_list, use_gpu=True):
    device = f"cuda:{gpu_id}" if use_gpu else "cpu"

    while not task_queue.empty():
        try:
            file_path = task_queue.get_nowait()
        except:
            break

        file_name = os.path.basename(file_path)
        print(f"[Worker {gpu_id}] Checking {file_name}")

        state_dict = load_file(file_path, device=device)

        for key, tensor in state_dict.items():
            real_dtype = normalize_torch_dtype(tensor.dtype)
            target_dtype = quant_map.get(key, None)

            if "scale" in key:
                continue

            if target_dtype is None:
                mismatch_list.append({
                    "weight": key,
                    "file": file_name,
                    "status": "missing_mapping",
                    "got": real_dtype,
                    "expect": None
                })
                continue

            if not(target_dtype in real_dtype):
                mismatch_list.append({
                    "weight": key,
                    "file": file_name,
                    "status": "dtype_mismatch",
                    "got": real_dtype,
                    "expect": target_dtype
                })

        print(f"[Worker {gpu_id}] Finished {file_name}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", required=True)
    parser.add_argument("--quant-map-json", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", default="dtype_check_result.json")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Load mapping
    with open(args.quant_map_json, "r") as f:
        quant_map = json.load(f)

    # Tasks
    tensor_files = sorted(glob(os.path.join(args.tensor_dir, "*.safetensors")))
    assert tensor_files, "No .safetensors found!"

    manager = Manager()
    mismatch_list = manager.list()
    task_queue = manager.Queue()

    for fp in tensor_files:
        task_queue.put(fp)

    # Launch workers
    procs = []
    for worker_id in range(args.num_workers):
        p = Process(
            target=worker_task,
            args=(worker_id, task_queue, quant_map, mismatch_list, not args.cpu)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Save results
    mismatch_list = list(mismatch_list)
    with open(args.output_json, "w") as f:
        json.dump(mismatch_list, f, indent=2)

    print("\n======== RESULT SUMMARY ========")
    print(f"Total mismatches: {len(mismatch_list)}")
    print(f"Saved to: {args.output_json}")

    if mismatch_list:
        print("First few:")
        for item in mismatch_list[:10]:
            print(item)


if __name__ == "__main__":
    main()
