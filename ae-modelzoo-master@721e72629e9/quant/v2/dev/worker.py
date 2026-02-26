import os
import json
import torch
import multiprocessing as mp
import traceback

from safetensors.torch import save_file


# TODO: 分开加载，不覆盖

def worker(
    task_queue: mp.Queue,
    gpu_id: int,
    weight_map_shared,
    quant_map_shared,
    output_dir: str,
    mode: str,  # dequant, quant, dequant+quant
    model_type: str | None = None,
    quant_config: dict | None = None,  # from original config.json
    custom_dequant_config: dict | None = None,  # from .yml
    custom_quant_config: dict | None = None,  # from .yml
):
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Worker {gpu_id}] Started on {device}")

        from functions.dequant import dequant
        from functions.quant import quant

        while True:
            try:
                sf = task_queue.get_nowait()
            except Exception:
                print(f"[Worker {gpu_id}] Task queue empty, exiting")
                break

            try:
                print(f"[Worker {gpu_id}] Processing {sf}")

                output = None

                if "dequant" in mode:
                    output = dequant(
                        file_name=sf,
                        quant_config=quant_config,
                        custom_dequant_config=custom_dequant_config,
                        device=device,
                        model_type=model_type,
                    )

                    if "+quant" in mode:
                        output, quant_map = quant(
                            state_dict=output,
                            custom_quant_config=custom_quant_config,
                            device=device,
                            model_type=model_type
                        )
                else:
                    if "quant" in mode:
                        output, quant_map = quant(
                            file_name=sf,
                            custom_quant_config=custom_quant_config,
                            device=device,
                            model_type=model_type
                        )
                    else:
                        raise ValueError(f"only support quant, dequant, and dequant+quant, get {mode}")

                file_name = os.path.basename(sf)

                for wname, _ in output.items():
                    weight_map_shared[wname] = file_name
                
                try:
                    for wname, dtype in quant_map.items():
                        quant_map_shared[wname] = dtype
                except Exception as e:
                    pass

                save_path = os.path.join(output_dir, file_name)
                save_file(output, save_path)

                # 主动释放显存
                del output
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[Worker {gpu_id}] Error processing {sf}")
                traceback.print_exc()

    except Exception:
        traceback.print_exc()
