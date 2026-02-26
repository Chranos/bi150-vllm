# AE-Compressor

### 支持列表

(w/o attn): without attn

|model|dequant|quant|模型结构|
|:---:|:---:|:---:|:---:|
|Qwen3-Coder-30B-A3B|FP8_block|w4a8 \| w8a8|qwen3 moe|
|Qwen3-30B-A3B||w4a8 \| w8a8|qwen3 moe|
|Kimi-K2-Thinking|int4-pack32_group| w4a8|kimik2|
|MiniMax-M2.1|FP8_block|w4a8|minimax_m2|
|GLM-4.7|FP8_channel|w4a8|glm4_moe|
|Qwen3-32B||w4a8 \| w8a8|qwen3-dense|
|Qwen3-VL-32B|| w4a8 \| w8a8|qwen3-vl|
|Qwen3-VL-30B-A3B||w4a8 \| w8a8|qwen3-vl-moe|
|GLM-4.7-Flash||w4a8|glm4_moe_lite|
|Step3-VL-10B||w4a8|qwen3-dense|

说明：

- 上述为测试过的模型列表
- 一般情况下相同模型结构都可使用

新增特性：

- 支持MoE+Dense w4a8量化（混合； w/o mla）使用 expoert VLLM_W8A8_LINEAR_USE_W4A8=1 启用

### 使用方法

注：脚本在dev目录下

```python
python3 main.py --config configs/xxx.yaml
```

使用前需仔细编写相应的.yaml文件，或者提需求

#### yaml文件编写说明

可参考已有的.yaml文件，需要注意的是ignore及int4,int8部分的编写。

- dequant部分ignore需要覆盖所有不需要反量化的层
- quant部分int4+int8+ignore需要覆盖所有层

### 辅助工具

#### filter_weight_map

该函数用于提取具体layer_name，可以根据生成的`temp.json`编写yaml文件.

```python
python3 tools/filter_weight_map.py --input-dir /home/to/model/dir/
```
