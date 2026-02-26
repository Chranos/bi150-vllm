#export UMD_SUBMITINTERVAL=1
#export UMD_KMTDIRECTACCESS=1
export ENABLE_MOE_GROUP_GEMV=0
export SGLANG_CHUNK_SIZE=2048
export SGLANG_W4A8_FORMAT=NN
export SGLANG_PP_LAYER_PARTITION="8,8,8,8,8,7,7,7"
export MODEL_PATH=/data00/tianshu/tianshu/pd-deepseek/DeepSeek-R1-W4A8-NN-V2/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/site-packages/.nixl.mesonpy.libs/plugins:/usr/local/ucx/lib/
export SGLANG_ATTENTION_INT8=1

python3 -m sglang.launch_server --model-path $MODEL_PATH \
    --disable-custom-all-reduce \
    --disable-cuda-graph \
    --sampling-backend pytorch  \
    --tp-size 2 \
    --pp-size 8 \
    --context-length 8192 \
    --chunked-prefill-size $SGLANG_CHUNK_SIZE \
    --attention-backend fa3 \
    --page-size 64 \
    --disable-radix-cache \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --kv-cache-dtype fp8_e4m3 \
    --port 12117 \
    --host 192.168.0.39 \
    --disaggregation-bootstrap-port 8117 \
    --mem-fraction-static 0.8
