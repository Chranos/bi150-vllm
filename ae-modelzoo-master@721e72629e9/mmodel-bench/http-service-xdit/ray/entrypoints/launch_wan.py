import os
import time
import torch
import ray
import io
import logging
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import argparse
import PIL.Image

from xfuser import (
    xFuserWanImageToVideoPipeline,
    xFuserWanPipeline,
    xFuserArgs,
)

from diffusers.utils import export_to_video

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

    # Add input validation
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "a beautiful landscape",
                "num_frames": 81,
                "num_inference_steps": 40,
                "seed": 42,
                "cfg": 5.0,
                "height": 720,
                "width": 1280
            }
        }

app = FastAPI()

@ray.remote(num_gpus=1)
class ImageGenerator:
    def __init__(self, xfuser_args: xFuserArgs, rank: int, world_size: int):
        # Set PyTorch distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        
        self.rank = rank
        self.setup_logger()
        self.initialize_model(xfuser_args)

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)

    def initialize_model(self, xfuser_args : xFuserArgs):

        # init distributed environment in create_config
        self.engine_config, self.input_config = xfuser_args.create_config()
        
        self.model_name = self.engine_config.model_config.model.split("/")[-1]
        
        self.logger.info(f"Initializing model {self.model_name} from {xfuser_args.model}")

        ## TODO support online data
        image = PIL.Image.open("wan_i2v_input.JPG")    
        self.image = image.resize((GenerateRequest().width, GenerateRequest().height))

        runtime_dtype = torch.bfloat16
        self.engine_config.runtime_config.dtype = runtime_dtype


        if "I2V" in self.model_name:
            self.pipe = xFuserWanImageToVideoPipeline.from_pretrained(
                pretrained_model_name_or_path=xfuser_args.model,
                engine_config=self.engine_config,
                torch_dtype=runtime_dtype,
            ).to(f"cuda")
        
            if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
                from w8a8_linear import apply_quant_linear_i8w8o16
                self.pipe.transformer = apply_quant_linear_i8w8o16(self.pipe.transformer)
            else:
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
            
            self.pipe(
                    image=self.image,
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
            self.pipe = xFuserWanPipeline.from_pretrained(
                pretrained_model_name_or_path=xfuser_args.model,
                engine_config=self.engine_config,
                torch_dtype=runtime_dtype,
            ).to(f"cuda")
        
            if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
                from w8a8_linear import apply_quant_linear_i8w8o16
                self.pipe.transformer = apply_quant_linear_i8w8o16(self.pipe.transformer)
            else:
                self.pipe.vae.enable_tiling()
                self.pipe.vae.enable_slicing()
                
            self.pipe(
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
        
        self.logger.info("Model initialization completed")

    def generate(self, request: GenerateRequest):
        try:
            start_time = time.time()
            if "I2V" in self.model_name:           
                output = self.pipe(
                    image=self.image,
                    height=request.height,
                    width=request.width,
                    num_frames=request.num_frames,
                    prompt=request.prompt,
                    negative_prompt = request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.cfg,
                    generator=torch.Generator(device="cuda").manual_seed(request.seed),
                )
            else:
                output = self.pipe(
                    height=request.height,
                    width=request.width,
                    num_frames=request.num_frames,
                    prompt=request.prompt,
                    negative_prompt = request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.cfg,
                    generator=torch.Generator(device="cuda").manual_seed(request.seed),
                )
            elapsed_time = time.time() - start_time

            if self.pipe.is_dp_last_group():
                if request.save_disk_path:
                    timestamp = time.strftime("%Y%m%d-%H%M")
                    filename = f"generated_video_{self.model_name}_{timestamp}.mp4"
                    file_path = os.path.join(request.save_disk_path, filename)
                    os.makedirs(request.save_disk_path, exist_ok=True)

                    for i, frames in enumerate(output.frames):
                        export_to_video(frames, file_path, fps=16)
                        
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "output": file_path,
                        "save_to_disk": True
                    }
                else:
                    return {
                        "message": "Video generated successfully",
                        "elapsed_time": f"{elapsed_time:.2f} sec",
                        "save_to_disk": False
                    }
            return None

        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

class Engine:
    def __init__(self, world_size: int, xfuser_args: xFuserArgs):
        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init(num_gpus=world_size)
        
        num_workers = world_size
        self.workers = [
            ImageGenerator.remote(xfuser_args, rank=rank, world_size=world_size)
            for rank in range(num_workers)
        ]
        
    async def generate(self, request: GenerateRequest):
        results = ray.get([
            worker.generate.remote(request)
            for worker in self.workers
        ])

        return next(path for path in results if path is not None) 



@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # Add input validation
        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if request.height <= 0 or request.width <= 0:
            raise HTTPException(status_code=400, detail="Height and width must be positive")
        if request.num_inference_steps <= 0:
            raise HTTPException(status_code=400, detail="num_inference_steps must be positive")
            
        result = await engine.generate(request)
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xDiT HTTP Service')
    parser.add_argument('--model_path', type=str, help='Path to the model', required=True)
    parser.add_argument('--world_size', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--pipefusion_parallel_degree', type=int, default=1, help='Degree of pipeline fusion parallelism')
    parser.add_argument('--ulysses_parallel_degree', type=int, default=1, help='Degree of Ulysses parallelism')
    parser.add_argument('--tensor_parallel_degree', type=int, default=1, help='Degree of tensor parallelism')
    parser.add_argument('--ring_degree', type=int, default=1, help='Degree of ring parallelism')
    parser.add_argument('--save_disk_path', type=str, default='output', help='Path to save generated images')
    parser.add_argument('--use_cfg_parallel', action='store_true', help='Whether to use CFG parallel')
    args = parser.parse_args()

    xfuser_args = xFuserArgs(
        model=args.model_path,
        trust_remote_code=True,
        warmup_steps=1,
        use_parallel_vae=False,
        use_torch_compile=False,
        ulysses_degree=args.ulysses_parallel_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        tensor_parallel_degree=args.tensor_parallel_degree,
        dit_parallel_size=0,
    )
    
    engine = Engine(
        world_size=args.world_size,
        xfuser_args=xfuser_args
    )
    
    # Start the server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)