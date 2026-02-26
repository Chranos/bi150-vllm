import os
import time
import torch
import torch.distributed as dist
import pickle
import base64
import logging
import torch.multiprocessing as mp
import PIL.Image
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from xfuser import (
    xFuserWanImageToVideoPipeline,
    xFuserWanPipeline,
    xFuserArgs
)

from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    is_dp_last_group,
    get_data_parallel_world_size,
    get_runtime_state,
)
from typing import Union, List, Optional
from diffusers.utils import export_to_video

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


# Define request model
class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    num_inference_steps: Optional[int] = 40
    seed: Optional[int] = 42
    num_frames: Optional[int] = 81
    cfg: Optional[float] = 5.0
    save_disk_path: Optional[str] = None
    height: Optional[int] = 720
    width: Optional[int] = 1280

## TODO: support online data
image = PIL.Image.open("wan_i2v_input.JPG")    
image = image.resize((GenerateRequest().width, GenerateRequest().height))


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized, model_name, cache_kwargs, I2V_Flag
    mp.set_start_method("spawn", force=True)

    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    setup_logger()

    cache_kwargs = {
        "use_easycache":True,
        "cache_thresh":0.02,  #easy eacch thresh
       }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    model_name = engine_config.model_config.model.split("/")[-1]

    runtime_dtype = torch.bfloat16
    engine_config.runtime_config.dtype = runtime_dtype

    if "I2V" in engine_config.model_config.model:
        I2V_Flag = True
    else:
        I2V_Flag = False
        
    if I2V_Flag:
        pipe = xFuserWanImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=runtime_dtype,
        ).to(f"cuda:{local_rank}")

        if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
            from w8a8_linear import apply_quant_linear_i8w8o16
            pipe.transformer = apply_quant_linear_i8w8o16(pipe.transformer)
            # pipe.transformer_2 = apply_quant_linear_i8w8o16(pipe.transformer_2)
        # else:
        #     pipe.vae.enable_tiling()
        #     pipe.vae.enable_slicing()
            
        pipe(
            image=image,
            height=GenerateRequest().height,
            width=GenerateRequest().width,
            num_frames=GenerateRequest().num_frames,
            prompt=GenerateRequest().prompt,
            negative_prompt = GenerateRequest().negative_prompt,
            num_inference_steps=1,
            guidance_scale=GenerateRequest().cfg,
            generator=torch.Generator(device=f"cuda").manual_seed(GenerateRequest().seed),
        ).frames
    else:
        pipe = xFuserWanPipeline.from_pretrained(
            pretrained_model_name_or_path=engine_config.model_config.model,
            engine_config=engine_config,
            torch_dtype=runtime_dtype,
        ).to(f"cuda:{local_rank}")

        if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
            from w8a8_linear import apply_quant_linear_i8w8o16
            pipe.transformer = apply_quant_linear_i8w8o16(pipe.transformer)
        # else:
        #     pipe.vae.enable_tiling()
        #     pipe.vae.enable_slicing()
            
        pipe(
            height=GenerateRequest().height,
            width=GenerateRequest().width,
            num_frames=GenerateRequest().num_frames,
            prompt=GenerateRequest().prompt,
            negative_prompt = GenerateRequest().negative_prompt,
            num_inference_steps=1,
            guidance_scale=GenerateRequest().cfg,
            generator=torch.Generator(device=f"cuda").manual_seed(GenerateRequest().seed),
        ).frames
        
    torch.cuda.reset_peak_memory_stats()
    
    logger.info("Model initialization completed")
    initialized = True


def generate_image_parallel(
    prompt, num_frames, num_inference_steps, height, width, seed, cfg, save_disk_path=None
):
    global pipe, local_rank, input_config, cache_kwargs, I2V_Flag
    
    logger.info(f"Starting image generation with prompt: {prompt}")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    if I2V_Flag:           
        output = pipe(
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            prompt= prompt,
            negative_prompt = GenerateRequest().negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            cache_kwargs = cache_kwargs
        )
    else:
        output = pipe(
            height=height,
            width=width,
            num_frames=num_frames,
            prompt= prompt,
            negative_prompt = GenerateRequest().negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            cache_kwargs = cache_kwargs
        )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if save_disk_path is not None:
        timestamp = time.strftime("%Y%m%d-%H%M")
        filename = f"generated_video_{timestamp}.mp4"
        file_path = os.path.join(save_disk_path, filename)

        if is_dp_last_group():
            # Create the directory if it doesn't exist
            os.makedirs(save_disk_path, exist_ok=True)
            for i, frames in enumerate(output.frames):
                export_to_video(frames, file_path, fps=16)
            
        output = file_path

    return output, elapsed_time

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    logger.info("Received POST request for image generation")
    prompt = request.prompt
    num_inference_steps = request.num_inference_steps
    num_frames = request.num_frames
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

    params = [prompt, num_frames, num_inference_steps, height, width, seed, cfg, save_disk_path]
    dist.broadcast_object_list(params, src=0)
    logger.info("Parameters broadcasted to all processes")

    output, elapsed_time = generate_image_parallel(*params)

    # Process output results
    if save_disk_path:
        # output is disk path
        response = {
            "message": "Video generated successfully",
            "elapsed_time": f"{elapsed_time:.2f} sec",
            "output": output,  # This is the file path
            "save_to_disk": True
        }
    else:
        response = {
            "message": "Video generated successfully",
            "elapsed_time": f"{elapsed_time:.2f} sec",
            "save_to_disk": False
        }
    return response


def run_host():
    if dist.get_rank() == 0:
        logger.info("Starting FastAPI host on rank 0")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=6000)
    else:
        while True:
            params = [None] * 8
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