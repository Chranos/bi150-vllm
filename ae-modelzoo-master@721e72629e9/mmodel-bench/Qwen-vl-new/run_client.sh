MODEL_PATH=$1
#/data3t/ckpt/Qwen/Qwen2.5-VL-7B-Instruct

python3 benchmark_serving_vl.py \
    --backend openai-chat \
    --model $MODEL_PATH \
    --dataset-name qwenvl \
    --vl-image-path 448_488.png \
    --vl-input-len 30 \
    --vl-output-len 300 \
    --num-prompts 78 \
    --request-rate 78 \
    --max-concurrency 78 \
    --base-url http://localhost:8000 \
    --endpoint /v1/chat/completions \
    --ignore-eos
