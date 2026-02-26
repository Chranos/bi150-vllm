export QUANT_LAYERS=$(echo {0..62})

python3 w4a8_bf16.py  \
    --input-fp8-hf-path /home/mashun/vllm_project/checkpoints/Qwen3-32B \
    --output-int8-hf-path /home/mashun/vllm_project/checkpoints/Qwen3-32B-w4a8 \
    --group-size -1 \
    --format TN \
    --symmetric True \
    --version 2 \
    --quant-layers $QUANT_LAYERS \
    --gemm-quant-method w4a8