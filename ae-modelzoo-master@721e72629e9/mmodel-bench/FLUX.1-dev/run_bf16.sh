mkdir -p results


export TORCH_COMPILE_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

export ENABLE_FLASH_ATTENTION_WITH_HEAD_DIM_PADDING=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
export PYTHONPATH=$PWD:$PYTHONPATH

#多ring 没提升
# export NCCL_USE_HIGHPRIORITYWARP=1

export ENABLE_IXFORMER_INFERENCE=1

export ENABLE_IXFORMER_SAGEATTN=1
export ENABLE_IXFORMER_W8A8LINEAR=0

#/data3t/ckpt/black-forest-labs/FLUX.1-dev
MODEL_PATH=$1
torchrun --nproc_per_node=2     \
	 --master_addr=127.0.0.1     \
	 --master_port=29500     \
	 examples/flux_example.py     \
	 --model $MODEL_PATH     \
	 --pipefusion_parallel_degree 2     \
	 --ulysses_degree 1     \
	 --ring_degree 1     \
	 --num_inference_steps 50     \
	 --height 1024 \
	 --width  1024 \
	 --prompt "A futuristic city, 4K, ultra detailed"
