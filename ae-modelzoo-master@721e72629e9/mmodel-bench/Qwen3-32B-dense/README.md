### Qwen3-32B-dense 测试步骤

```
W8A8 weights
sdk0910
replace vllm,ixformer,ixinfer

vllm:vllm-0.10.1+corex.4.4.0.1102-py3-none-any.whl 7af1411220e51f40b8352de5240ee7a400fcf598
ixformer:ixformer-0.6.0+corex.4.4.0.1103-cp310-cp310-linux_x86_64.whl 4a12c4697c38c866759d9f17cabdc7d56ff1a8f1
ixinfer:./lib/* 720949b9b0e9a5eb458807c62500b27a519a85c6

cd mmodel-bench/Qwen3-32B-dense

# 启动服务
bash start_qwen3_vllm0.10.1.sh /data3t/ckpt/Qwen/Qwen2.5-VL-7B-Instruct

# 启动测试脚本
bash run_client.sh /data3t/ckpt/Qwen/Qwen2.5-VL-7B-Instruct
```
PS:
1)BF16 weights
线量化就把 RE_QUANT_LAYERS 打开，VLLM_USE_MIX_MHA 打开对应 --block-size=32，不然是16

2)qihang.zhang:--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "level": 0, "max_capture_size": 8,"cudagraph_capture_sizes": [1,2,3,4,5,6,7,8]}'

启动server可以试试加一下这个参数，sizes控制到最大并发数，可以节省显存，FULL_DECODE_ONLY 性能有提升.
