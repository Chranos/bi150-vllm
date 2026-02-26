
MODEL_PATH=$1
#/data3t/ckpt/Qwen/Qwen2.5-VL-7B-Instruct

vllm serve $MODEL_PATH \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 32768  \
    --max-num-seqs 512 \
    --host 0.0.0.0 \
    --port 8000 \
    --enforce-eager \
    --trust-remote-code
