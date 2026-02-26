## 模型量化

可以控制每一层的量化精度。

### 支持

|model|w4a8|w8a8|
|:---:|:---:|:---:|
|Qwen3-30B-A3B|True|True|
|Qwen3-VL-30B-A3B|True|True|
|SeedOSS-36B|False|True|

### 数据结构

```json
{
    "layer_name1": "bf16",
    "layer_name2": "int4",
    ...
}
```

## 使用方法

1、生成量化文件

具体见相关脚本

```python
python3 utils/generate_quant_map.py --input-dir ...
```

该步骤用于生成常见量化映射文件，可通过手动修改其中值实现更准确的量化。

2、量化模型

```python
python3 parallel_runner.py --input-dir ... --output-dir ... --quant-map-json ... --num-gpus 8
```