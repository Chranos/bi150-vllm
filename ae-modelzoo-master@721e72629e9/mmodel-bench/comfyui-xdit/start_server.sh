#!/bin/bash

# xDiT Flux HTTP Server 启动脚本
# 传入参数
MODEL_PATH=${1:-"/data3t/ckpt/black-forest-labs/FLUX.1-dev"}
PORT=${2:-6000}
NUM_GPUS=${3:-2}

# 并行参数
PIPEFUSION_DEGREE=2
ULYSSES_DEGREE=1
RING_DEGREE=1

# 启动服务器
python3 server.py \
    --model_path $MODEL_PATH \
    --world_size $NUM_GPUS \
    --pipefusion_parallel_degree $PIPEFUSION_DEGREE \
    --ulysses_parallel_degree $ULYSSES_DEGREE \
    --ring_degree $RING_DEGREE \
    --port $PORT



