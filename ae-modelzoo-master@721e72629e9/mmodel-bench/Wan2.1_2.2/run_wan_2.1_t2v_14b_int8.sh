#!/bin/bash
set -x
export WORD_RANK_SUPPORT_TP=1
export ENABLE_IXFORMER_INFERENCE=1
export ENABLE_IXFORMER_SAGEATTN=1
export TOKENIZERS_PARALLELISM=true
export ENABLE_IXFORMER_W8A8LINEAR=1
export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="wan2.1_t2v_example.py"
MODEL_ID=/data1/Wan-AI/Wan2.1-T2V-14B-Diffusers
INFERENCE_STEP=40

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 720 --width 1280 --num_frames 81 --seed 42 "

# CogVideoX parallel configuration
N_GPUS=16
PARALLEL_ARGS="--ulysses_degree 4 --ring_degree 1 --tensor_parallel_degree  2"
CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
# ENABLE_TILING="--enable_tiling"
# MODEL_OFFLOAD="--enable_model_cpu_offload"
# COMPILE_FLAG="--use_torch_compile"
ENABLE_CACHE="--use_easycache"

torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "一个虎虎生威的老虎" \
--negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG \
$ENABLE_CACHE \
$CFG_ARGS
