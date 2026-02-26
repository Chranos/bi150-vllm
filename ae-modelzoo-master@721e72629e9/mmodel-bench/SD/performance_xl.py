import torch
from diffusers import DDPMScheduler  
import numpy as np
import random
import os
import stat
import csv
import time
import json
import argparse
import ixformer

os.environ["ENABLE_IXFORMER_INFERENCE"] = "1" 
os.environ["ENABLE_INFERENCE_TIME"] = "1" 
# os.environ["ENABLE_IXFORMER_LINEAR_W8A8"] = "0"
def ixformer_accelerate(pipe):
    pipe.unet.fuse_qkv_projections()    
 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = False     
     torch.backends.cudnn.benchmark = True
# from torchinfo import summary     
# 设置随机数种子
setup_seed(20)
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bs",
        "--batch_size", 
        type=int, 
        default=1, 
        help="Batch size."
    )

    return parser.parse_args()
args = parse_arguments()
DEEPCACHE=0
if DEEPCACHE:#DEEPCACHE 只支持StableDiffusionPipeline StableDiffusionXLPipeline StableDiffusionXLImg2ImgPipeline TextToVideoZeroPipeline
     from ixformer.contrib.DeepCache import StableDiffusionXLPipeline 
else:     
     from diffusers import StableDiffusionXLPipeline,DiffusionPipeline,DDPMScheduler  

pipe = StableDiffusionXLPipeline.from_pretrained(f"/data3t/ckpt/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
# pipe.set_progress_bar_config(disable=True)
# pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
ixformer_accelerate(pipe)

prompt = "a TV	Artifacts	Basic	 "
batch_size=args.batch_size
num_inference_steps =50
prompts=[prompt for i in range(batch_size)]

# resolution=[(1024,1024),(512,512)]
resolution=[(512,512),(768,768),(960,960),(1024,1024)]
resolution=[(448,488),(1344,1792),(1280,768)]
for width,height in resolution:
     #warm up
     print(f"************************************************************************************************************")
     print(f"[info] width {width} height {height} warm up ----------------------------------------------------------------------")
     if DEEPCACHE:  
               image = pipe(
               prompts, 
               num_inference_steps=num_inference_steps,
               cache_interval=3, cache_layer_id=0, cache_block_id=0,
               output_type='pt', return_dict=True,width=width,height=height
               )
     else:
          pipe(prompts,num_images_per_prompt=1,num_inference_steps=num_inference_steps,height=height,width=width)
     num_repeat=1
     print(f"[info] width {width} height {height} start----------------------------------------------------------------------")
     torch.cuda.synchronize()
     start_time = time.perf_counter()
     torch.cuda.profiler.start()
     for i in range(num_repeat):
          if DEEPCACHE:
               images = pipe(
               prompts, 
               num_inference_steps=num_inference_steps,
               cache_interval=3, cache_layer_id=0, cache_block_id=0,
               output_type='pt', return_dict=True,width=width,height=height
               ).images
          else:
               images= pipe(prompts,output_type='pt',num_inference_steps=num_inference_steps,num_images_per_prompt=1,height=height,width=width).images
     torch.cuda.profiler.stop()

     torch.cuda.synchronize()
     use_time = time.perf_counter() - start_time
     infer_num=batch_size*num_repeat
     print(f"[info] infer number: {infer_num}; use time: {use_time:.3f}s; "
          f"Throughput: {infer_num/use_time:.3f}")
     print(f"[info] width {width} height {height} end----------------------------------------------------------------------")
     from torchvision.utils import save_image
     def save_images(images):
          for index in range(batch_size):
               image_file=f"demo_{index}_{width}_{height}.png"
               # image.save(image_file)
               save_image(images[index],image_file)
     save_images(images)
