set -ex

export TORCH_COMPILE_DISABLE=1
export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export ENABLE_IXFORMER_INFERENCE=1
export ENABLE_IXFORMER_SAGEATTN=1
export ENABLE_IXFORMER_W8A8LINEAR=1

python3 entrypoints/launch_flux.py --world_size 2 --pipefusion_parallel_degree 2 \
--model_path /data1/FLUX.1-dev
