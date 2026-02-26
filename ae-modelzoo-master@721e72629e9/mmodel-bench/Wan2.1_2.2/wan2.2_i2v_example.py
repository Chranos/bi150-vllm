import torch
import numpy as np
import os
from diffusers.utils import export_to_video, load_image
import PIL.Image
import logging
import time
import torch
import torch.distributed
from diffusers import AutoencoderKLTemporalDecoder
from xfuser import xFuserWanImageToVideoPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_world_group
)


from xfuser.core.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from diffusers.utils import export_to_video


def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)

    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank


    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for Wan"
    assert engine_args.ring_degree == 1, "This script cannot support ring_degree > 1."

    pipe = xFuserWanImageToVideoPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    image = PIL.Image.open("./wan_i2v_input.JPG")    
    image = image.resize((input_config.width, input_config.height))

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    if os.environ.get("ENABLE_IXFORMER_W8A8LINEAR", "0") == "1":
        from w8a8_linear import apply_quant_linear_i8w8o16
        pipe.transformer = apply_quant_linear_i8w8o16(pipe.transformer)

    if args.enable_tiling:
        pipe.vae.enable_tiling()

    if args.enable_slicing:
        pipe.vae.enable_slicing()

    # warmup
    output = pipe(
        image=image,
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        guidance_scale=3.5,
        num_inference_steps=1,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    ).frames

    if args.use_easycache:
        cache_kwargs = {
            "use_easycache":True,
            "cache_thresh":0.02,  #easy eacch thresh
        }
    else:
        cache_kwargs = None  

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    output = pipe(
        image=image,
        prompt=input_config.prompt,
        negative_prompt=input_config.negative_prompt,
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        guidance_scale=3.5,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        cache_kwargs=cache_kwargs
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_reserved(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{engine_args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{engine_args.ulysses_degree}_ring{engine_args.ring_degree}_"
        f"tp{engine_args.tensor_parallel_degree}_"
        f"pp{engine_args.pipefusion_parallel_degree}_patch{engine_args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        resolution = f"{input_config.width}x{input_config.height}"
        for i, frames in enumerate(output.frames):
            output_filename = f"results/wan2.2_i2v_{i}_{parallel_info}_{resolution}.mp4"
            export_to_video(frames, output_filename, fps=16)
            print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destroy_distributed_env()


if __name__ == "__main__":
    main()



"""

model_id = "/nvmedata/data/nlp/Wan2.2-I2V-A14B-Diffusers/"
dtype = torch.bfloat16
device = "cuda"

pipe = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe.to(device)


image = PIL.Image.open("wan_i2v_input.JPG")
max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))
prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
generator = torch.Generator(device=device).manual_seed(0)
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=3.5,
    num_inference_steps=40,
    generator=generator,
).frames[0]
export_to_video(output, "i2v_output.mp4", fps=16)
"""
