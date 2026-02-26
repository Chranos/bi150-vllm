## 混合量化配置文件

可以控制每一层的量化精度。

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

```python
python3 tools/generate_quant_map.py --input-dir ...
```

2、量化模型

```python
python3 mix_quant_qwen-moe.py --input-dir ... --output-dir ... --quant-map-json ... --num-gpus 8
```