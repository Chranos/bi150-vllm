#!/bin/bash

spl=0
ql=6144
osl=1
p_con=1000
rate=4

python3 bench_serving.py \
    --model /data00/tianshu/tianshu/pd-deepseek/DeepSeek-R1-W4A8-NN-V2/ \
    --backend sglang \
    --host 0.0.0.0 \
    --port 12116 \
    --dataset-name generated-shared-prefix \
    --gsp-system-prompt-len 0 \
    --gsp-question-len $ql \
    --gsp-output-len $osl \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group $p_con \
    --random-range-ratio 1 \
    --num-prompts $p_con \
    --random-input $ql \
    --random-output $osl \
    --request-rate $rate \
    --max-concurrency 4 \
    --max-micro-batch-size 1
