import torch

from .qwen3moe import Qwen3MoEQuantConfigBuilder
from .default import DefaultQuantConfigBuilder
from .kimik2 import KimiK2QuantConfigBuilder
from .glm4_7 import GLM4_7QuantConfigBuilder
from ..specs import QuantArtifacts, ResolvedQuantConfig, DequantSpec


def create_quant_config_builder(qconfig, custom_dequant_config, model_type):

    if model_type in ["qwen3_moe", "minimax_m2", "minimax"]:
        return Qwen3MoEQuantConfigBuilder(qconfig, custom_dequant_config)
    elif model_type == "kimik2":
        return KimiK2QuantConfigBuilder(qconfig, custom_dequant_config)
    elif model_type == "glm4_moe":
        return GLM4_7QuantConfigBuilder(qconfig, custom_dequant_config)

    return DefaultQuantConfigBuilder(
        qconfig,
        custom_dequant_config
    )


def derive_quant_key(weight_name: str, key_type: str, tensors: dict[str, torch.Tensor]) -> str | None:
    """
    根据 weight_name 推导量化辅助 tensor 名称。
    """
    # 定义所有可能的原始 weight 后缀
    ORIG_WEIGHT_SUFFIXES = QWEIGHT_SUFFIXES

    # 选择目标后缀表
    if key_type == "scale":
        target_suffixes = SCALE_SUFFIXES
    elif key_type == "zero_point":
        target_suffixes = ZERO_POINT_SUFFIXES
    elif key_type == "qweight":
        target_suffixes = QWEIGHT_SUFFIXES
    elif key_type == "shape":
        target_suffixes = SHAPE_SUFFIXES
    else:
        raise ValueError(f"Unknown key_type {key_type}")

    candidates = []
    for orig_suffix in ORIG_WEIGHT_SUFFIXES:
        if weight_name.endswith(orig_suffix):
            prefix = weight_name[: -len(orig_suffix)]
            candidates.extend([prefix + s for s in target_suffixes])
            break
    else:
        # weight_name 不匹配任何已知 weight 后缀，直接加 target_suffix
        candidates.extend([weight_name + s for s in target_suffixes])

    # 返回第一个存在于 tensors 的 key
    for k in candidates:
        if k in tensors:
            return k
    return None


SCALE_SUFFIXES = [
    ".weight_scale",
    ".weight_scale_inv",
    ".weight_block_scale",
    "_scale",
    "_scale_inv",
    "_block_scale",
]

ZERO_POINT_SUFFIXES = [
    ".weight_zero_point",
    ".weight_zp",
    "_zero_point",
    "_zp",
]

QWEIGHT_SUFFIXES = [
    ".weight",
    ".weight_packed",
    ".weight_q",
    "_q",
    "_packed",
]

SHAPE_SUFFIXES = [
    ".weight_shape",
    ".weight_block_shape",
    "_shape",
    "_block_shape",
]


def resolve_quant_artifacts(
    weight_name: str,
    tensors: dict[str, torch.Tensor],
    rq: ResolvedQuantConfig,
) -> QuantArtifacts:

    if weight_name not in tensors:
        raise KeyError(f"Missing quantized weight: {weight_name}")

    if any(
        weight_name.endswith(suffix)
        for suffix in SCALE_SUFFIXES + ZERO_POINT_SUFFIXES + SHAPE_SUFFIXES
    ):
        return None

    qweight_key = derive_quant_key(weight_name, "qweight", tensors)
    scale_key = derive_quant_key(weight_name, "scale", tensors)
    zero_point_key = derive_quant_key(weight_name, "zero_point", tensors)
    shape_key = derive_quant_key(weight_name, "shape", tensors)

    if qweight_key is None:
        raise KeyError(f"Cannot find qweight for {weight_name}")

    qweight = tensors[qweight_key]

    if not scale_key:
        raise KeyError(f"Missing scale tensor: {weight_name}")

    scale = tensors[scale_key]

    zero_point = None
    if not rq.symmetric:
        if zero_point_key is None:
            raise KeyError(f"Cannot find zero_point tensor for {weight_name}")
        zero_point = tensors[zero_point_key]

    shape = tensors.get(shape_key, None)

    return QuantArtifacts(
        qweight=qweight, scale=scale, zero_point=zero_point, shape=shape
    )


def to_dequant_spec(rq: ResolvedQuantConfig) -> DequantSpec:
    return DequantSpec(
        dtype=rq.dtype,
        num_bits=rq.num_bits,
        packed=rq.packed,
        strategy=rq.strategy,
        symmetric=rq.symmetric,
    )