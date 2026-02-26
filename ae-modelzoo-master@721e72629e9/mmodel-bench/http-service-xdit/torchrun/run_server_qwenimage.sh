export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PYTHONPATH=$PWD:$PYTHONPATH


export ENABLE_IXFORMER_INFERENCE=1
export ENABLE_IXFORMER_SAGEATTN=0
export ENABLE_IXFORMER_W8A8LINEAR=0

MODEL_ID=/data/weights/Qwen-Image
INFERENCE_STEP=28


mkdir -p ./results

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning --guidance_scale 4.0 --seed 42"

N_GPUS=4
PARALLEL_ARGS="--pipefusion_parallel_degree 4 --ulysses_degree 1 --ring_degree 1 --tensor_parallel_degree 1"


torchrun --nproc_per_node=$N_GPUS --master-port 29521 ./entrypoints/qwenimage.py \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
#--num_inference_steps 1 \
#--warmup_steps 0 \
#--prompt "brown dog laying on the ground with a metal bowl in front of him., Ultra HD, 4K, cinematic composition." \
#$CFG_ARGS \
#$PARALLLEL_VAE \
#$COMPILE_FLAG \
#$QUANTIZE_FLAG \
#$CACHE_ARGS \
