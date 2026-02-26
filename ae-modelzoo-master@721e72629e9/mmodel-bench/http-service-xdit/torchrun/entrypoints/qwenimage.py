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

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    is_dp_last_group,
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
)
from xfuser.model_executor.pipelines import xFuserQwenImagePipeline

from typing import Union, List, Optional
import cv2

# FastAPI initialization
app = FastAPI()

# Environment setup for NCCL
#os.environ["NCCL_BLOCKING_WAIT"] = "1"
#os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

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
    negative_prompt: str = ""
    #negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    num_inference_steps: Optional[int] = 40
    seed: Optional[int] = 42
    #num_frames: Optional[int] = 81
    cfg: Optional[float] = 5.0
    save_disk_path: Optional[str] = None
    height: Optional[int] = 720
    width: Optional[int] = 1280
    
## TODO: support online data
#image = PIL.Image.open("wan_i2v_input.JPG")    
#image = image.resize((GenerateRequest().width, GenerateRequest().height))


def initialize():
    global pipe, engine_config, input_config, local_rank, initialized, model_name, cache_kwargs
    mp.set_start_method("spawn", force=True)

    torch.backends.cudnn.benchmark = False
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    engine_config.runtime_config.dtype = torch.bfloat16
    local_rank = get_world_group().local_rank
    torch.cuda.set_device(local_rank)
    setup_logger()

    # cache_kwargs = {
    #     "use_easycache":True,
    #     "cache_thresh":0.02,  #easy eacch thresh
    #    }

    cache_kwargs = {
        "use_teacache": engine_args.use_teacache,
        "use_fbcache": engine_args.use_fbcache,
        "rel_l1_thresh": 0.21,
        "return_hidden_states_first": False,
        "num_steps": input_config.num_inference_steps,
    }

    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #torch.cuda.set_device(local_rank)
    
    logger.info(f"Initializing model on GPU: {torch.cuda.current_device()}")

    model_name = engine_config.model_config.model.split("/")[-1]

    #print("\n\n\nengine_config:", engine_config, "\n\n\n")
    pipe = xFuserQwenImagePipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        cache_args=cache_kwargs,
        torch_dtype=engine_config.runtime_config.dtype,
    )
    
    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        pipe = pipe.to(local_rank)
        
    pipe.vae.enable_tiling()
    #pipe.vae.disable_tiling()
    
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    pipe.prepare_run(input_config, steps=input_config.num_inference_steps, sync_steps=1, warmup_prompt=input_config.prompt)
   
    torch.cuda.reset_peak_memory_stats()
    #print("\n\n\ninput_config:", input_config, "\n\n\n")
    pipe(
        height=GenerateRequest().height,
        width=GenerateRequest().width,
        #num_frames=GenerateRequest().num_frames,
        prompt=GenerateRequest().prompt,
        negative_prompt = GenerateRequest().negative_prompt,
        num_inference_steps=7,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=GenerateRequest().cfg,
        generator=torch.Generator(device=f"cuda").manual_seed(GenerateRequest().seed),
    )  
    
    #torch.cuda.reset_peak_memory_stats()
    
    logger.info("Model initialization completed")
    initialized = True
    
def generate_image_parallel(
    prompt, num_inference_steps, height, width, seed, cfg, save_disk_path=None
):
    global pipe, local_rank, input_config, cache_kwargs
    
    logger.info(f"Starting image generation with prompt: {prompt}")
    #torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    #diffusers needs height and width to be multiple of 16
    input_height = height % 16 + height
    input_width = width % 16 + width
    
    output = pipe(
        height=input_height,
        width=input_width,
        #num_frames=num_frames,
        prompt= prompt,
        negative_prompt = GenerateRequest().negative_prompt,
        num_inference_steps=num_inference_steps,
        output_type=input_config.output_type,
        max_sequence_length=input_config.max_sequence_length,
        guidance_scale=cfg,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        #cache_kwargs = cache_kwargs
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    # logger.info(f"Image generation completed in {elapsed_time:.2f} seconds")

    if save_disk_path is not None and input_config.output_type == "pil":
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        file_path = os.path.join(save_disk_path, filename)
        if pipe.is_dp_last_group():
            os.makedirs(save_disk_path, exist_ok=True)
            output.images[0].save(file_path)
            logger.info(f"Image saved to: {file_path}")
        output = file_path
        # os.makedirs("./results", exist_ok=True)
        # for i, image in enumerate(output.images):
        #     image_name = f"qwenimage_result_{i}.png"
        #     image.resize((width, height)).save(f"./results/{image_name}")
        #     print(f"image {i} saved to ./results/{image_name}")
    else:
        if pipe.is_dp_last_group():
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

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    logger.info("Received POST request for image generation")
    prompt = request.prompt
    num_inference_steps = request.num_inference_steps
    #num_frames = request.num_frames
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
    
    # Process output results
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
    # if save_disk_path:
    #     # output is disk path  
    #     base64_img=''
    #     #print(os.getcwd())
    #     with open('./results/qwenimage_result_0.png', 'rb') as image_file:
    #         base64_img = base64.b64encode(image_file.read())
    #     response = {
    #         "message": "Image generated successfully",
    #         "elapsed_time": f"{elapsed_time:.2f} sec",
    #         "output_base64": base64_img,
    #         "save_to_disk": True
    #     }
    # else:
    #     response = {
    #         "message": "Image generated successfully",
    #         "elapsed_time": f"{elapsed_time:.2f} sec",
    #         "output_base64": base64.b64encode(output.images[0].tobytes()).decode("utf-8") if output and hasattr(output, "images") and output.images else "",
    #         "save_to_disk": False
    #     }
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