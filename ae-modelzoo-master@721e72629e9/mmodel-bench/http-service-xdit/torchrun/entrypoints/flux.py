import os
import time
import torch
import torch.distributed as dist
import pickle
import base64
import logging
import torch.multiprocessing as mp
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from xfuser import (
    xFuserFluxPipeline,
    xFuserArgs
)

from transformers import T5EncoderModel

from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
)
from typing import Union, List, Optional

# FastAPI initialization
app = FastAPI()

# Environment setup for NCCL
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# Global variables
pipe = None
engine_config = None
input_config = None
local_rank = None
logger = None
initialized = False


def setup_logger():
    global logger
    rank = dist.get_rank()
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)


@app.get("/initialize")
async def check_initialize():
    global initialized
    if initialized:
        return {"status": "initialized"}
    else:
        return {"status": "initializing"}, 202


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized, model_name
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    setup_logger()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    model_name = engine_config.model_config.model.split("/")[-1]

    runtime_dtype = torch.bfloat16
    engine_config.runtime_config.dtype = runtime_dtype
    text_encoder_2 = T5EncoderModel.from_pretrained(
        engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=runtime_dtype)

    pipe = xFuserFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=runtime_dtype,
        text_encoder_2=text_encoder_2,
    ).to(f"cuda:{local_rank}")


    if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
        from w8a8_linear import apply_quant_linear_i8w8o16
        pipe.transformer = apply_quant_linear_i8w8o16(pipe.transformer)

    pipe.prepare_run(input_config)
    logger.info("Model initialization completed")
    initialized = True


def generate_image_parallel(
    prompt, num_inference_steps, height, width, seed, cfg, save_disk_path=None
):
    global pipe, local_rank, input_config
    logger.info(f"Starting image generation with prompt: {prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output = pipe(
        height=height,
        width=width,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        output_type="pil",
        generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(seed),
        guidance_scale=cfg,
        max_sequence_length=input_config.max_sequence_length
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    # logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if save_disk_path is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        file_path = os.path.join(save_disk_path, filename)
        if is_dp_last_group():
            os.makedirs(save_disk_path, exist_ok=True)
            output.images[0].save(file_path)
            logger.info(f"Image saved to: {file_path}")
        output = file_path
    else:
        if is_dp_last_group():
            output_bytes = pickle.dumps(output)
            dist.send(
                torch.tensor(len(output_bytes), device=f"cuda:{local_rank}"), dst=0
            )
            dist.send(
                torch.ByteTensor(list(output_bytes)).to(f"cuda:{local_rank}"), dst=0
            )
            logger.info(f"Output sent to rank 0")

        if dist.get_rank() == 0:
            size = torch.tensor(0, device=f"cuda:{local_rank}")
            dist.recv(size, src=dist.get_world_size() - 1)
            output_bytes = torch.ByteTensor(size.item()).to(f"cuda:{local_rank}")
            dist.recv(output_bytes, src=dist.get_world_size() - 1)
            output = pickle.loads(output_bytes.cpu().numpy().tobytes())

    return output, elapsed_time



# Define request model
class GenerateRequest(BaseModel):
    prompt: str = ""
    num_inference_steps: Optional[int] = 50
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    seed: Optional[int] = 42
    cfg: Optional[float] = 3.5
    save_disk_path: Optional[str] = None



@app.post("/generate")
async def generate_image(request: GenerateRequest):
    logger.info("Received POST request for image generation")
    prompt = request.prompt
    num_inference_steps = request.num_inference_steps
    height = request.height
    width = request.width
    seed = request.seed
    cfg = request.cfg
    save_disk_path = request.save_disk_path

    if save_disk_path and not os.path.isdir(save_disk_path):
        default_path = os.path.join(os.path.expanduser("~"), "tacodit_output")
        os.makedirs(default_path, exist_ok=True)
        logger.warning(f"Invalid save_disk_path. Using default path: {default_path}")
        save_disk_path = default_path

    params = [prompt, num_inference_steps, height, width, seed, cfg, save_disk_path]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    if save_disk_path:
        output_base64 = ""
        image_path = save_disk_path
    else:
        if output and hasattr(output, "images") and output.images:
            output_base64 = base64.b64encode(output.images[0].tobytes()).decode("utf-8")
        else:
            output_base64 = ""
        image_path = ""

    response = {
        "message": "Image generated successfully",
        "elapsed_time": f"{elapsed_time:.2f} sec",
        "output": output_base64 if not save_disk_path else output,
        "save_to_disk": save_disk_path is not None,
    }
    return response


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting FastAPI host on rank 0")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 7
            logger.info(f"Rank {dist.get_rank()} waiting for tasks")
            dist.broadcast_object_list(params, src=0)
            if params[0] is None:
                logger.info("Received exit signal, shutting down")
                break
            logger.info(f"Received task with parameters: {params}")
            generate_image_parallel(*params)


if __name__ == "__main__":
    initialize()
    logger.info(
        f"Process initialized. Rank: {dist.get_rank()}, Local Rank: {os.environ.get('LOCAL_RANK', 'Not Set')}"
    )
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    run_host()