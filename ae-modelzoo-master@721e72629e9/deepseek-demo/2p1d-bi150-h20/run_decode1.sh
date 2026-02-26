export UCX_NET_DEVICES=mlx5_0:1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/python3.10/dist-packages/.nixl.mesonpy.libs/plugins:/usr/local/ucx/lib/

python3 -m sglang.launch_server --model-path  /data00/pd-deepseek/DeepSeek-R1/ \
    --base-gpu-id 0 \
    --tensor-parallel-size 8 \
    --page-size 64 \
    --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --host 192.168.0.42 \
    --port 12350 \
    --mem-fraction-static 0.9 \
    --max-running-requests 32 \
    --cuda-graph-bs 1 8 16 32
