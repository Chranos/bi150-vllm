# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from copy import deepcopy
from typing import Any, Optional

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE

import vllm.model_executor.layers.fused_moe  # noqa
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearMethodBase, set_weight_attrs
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kernels.mixed_precision import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    get_linear_quant_method,
    override_config,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_marlin_supported,
    check_moe_marlin_supports_layer,
    marlin_make_workspace_new,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    marlin_repeat_scales_on_all_ranks,
    verify_marlin_supported,
)
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.transformers_utils.config import get_safetensors_params_metadata
from vllm.utils.collection_utils import is_list_of
import ixformer.inference.functions as ixfops

logger = init_logger(__name__)

#[B,K//8,N] ->[B,K,N]
# less memmory
def unpack_k_batch_opt(packed_w: torch.Tensor, num_bits: int = 4, chunk_size: int = 2) -> torch.Tensor:
    """
    Memory-efficient unpacking for 3D tensor.
    Converts [B, K // pack_factor, N] int32 tensor → [B, K, N] int8 tensor,
    without broadcasting huge intermediate tensors (avoids OOM).

    Args:
        packed_w: torch.int32 tensor of shape [B, K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4 or 2).
        chunk_size: How many bit groups to unpack at once (tradeoff between speed and memory).

    Returns:
        unpacked: torch.int8 tensor of shape [B, K, N].
    """
    B, k_packed, N = packed_w.shape
    pack_factor = 32 // num_bits
    K = k_packed * pack_factor
    mask = (1 << num_bits) - 1

    # Allocate output tensor once
    unpacked = torch.empty((B, K, N), dtype=torch.int8, device=packed_w.device)

    # Process bit chunks iteratively to save memory
    for i in range(0, pack_factor, chunk_size):
        # Precompute shifts for this chunk
        shift_vals = num_bits * torch.arange(i, min(i + chunk_size, pack_factor), device=packed_w.device)
        # [chunk_size, 1, 1, 1]
        shifts = shift_vals.view(-1, 1, 1, 1)
        # Compute small chunk only
        chunk = ((packed_w.unsqueeze(0) >> shifts) & mask).to(torch.int8)

        # chunk: [chunk_size, B, k_packed, N]
        # write into output
        for j in range(chunk.shape[0]):
            unpacked[:, (i + j)::pack_factor, :] = chunk[j]

        del chunk  # release memory early

    return unpacked

# more memmory
def unpack_k_batch(packed_w: torch.Tensor, num_bits: int = 4) -> torch.Tensor:
    """
    Efficient vectorized unpacking for 3D tensor (batch version).
    Converts [B, K // pack_factor, N] int32 tensor → [B, K, N] int8 tensor.

    Args:
        packed_w: torch.int32 tensor of shape [B, K // pack_factor, N].
        num_bits: Number of bits per packed element (e.g., 4).

    Returns:
        unpacked: torch.int8 tensor of shape [B, K, N].
    """
    B, k_packed, n = packed_w.shape
    pack_factor = 32 // num_bits
    k = k_packed * pack_factor

    mask = (1 << num_bits) - 1

    # [pack_factor, 1, 1, 1]
    shifts = (num_bits * torch.arange(pack_factor, device=packed_w.device)).view(-1, 1, 1, 1)

    # [1, B, k_packed, N]
    packed_expanded = packed_w.unsqueeze(0)

    # Extract each group of num_bits using bitwise ops
    unpacked_groups = ((packed_expanded >> shifts) & mask).to(torch.int8)

    # [pack_factor, B, k_packed, N] → [B, K, N]
    unpacked = unpacked_groups.permute(1, 2, 0, 3).reshape(B, k, n)

    return unpacked


#[B,K,N] ->[B,K,N//8]
# less memmory
def pack_n_batch_opt(x: torch.Tensor, pack_num: int = 8, order_map=None, chunk_size: int = 2) -> torch.Tensor:
    """
    Memory-efficient batch packing with correct bit order.
    [B, K, N] int4 -> [B, K, N//pack_num] int32.
    """
    B, K, N = x.shape
    assert N % pack_num == 0, "N must be divisible by pack_num"
    cols = N // pack_num
    unit = 32 // pack_num

    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    shifts = unit * torch.arange(pack_num, device=x.device)  # always 0..unit*(pack_num-1)
    packed = torch.zeros((B, K, cols), dtype=torch.int32, device=x.device)
    x_reshape = x.view(B, K, cols, pack_num) & 0xF

    # process in chunks for memory efficiency
    for start in range(0, pack_num, chunk_size):
        end = min(start + chunk_size, pack_num)
        idx_chunk = order_map[start:end]
        shift_chunk = shifts[start:end]

        vals = torch.gather(x_reshape, 3, idx_chunk.view(1,1,1,-1).expand(B,K,cols,-1)).to(torch.int32)
        for j in range(vals.shape[-1]):
            packed.add_(vals[..., j] << shift_chunk[j])

    return packed

## more memmory
def pack_n_batch(x: torch.Tensor, pack_num: int = 8, order_map=None) -> torch.Tensor:
    """
    Efficient vectorized batch packing: [B, K, N] int4 -> [B, K, N//pack_num] int32.

    Args:
        x: torch.int32 tensor of shape [B, K, N], each element 0-15 (int4).
        pack_num: Number of 4-bit elements per packed int32 (default=8).
        order_map: Optional order of elements within each packed int32.

    Returns:
        torch.int32 tensor of shape [B, K, N//pack_num].
    """
    
    B, K, N = x.shape
    assert N % pack_num == 0, "N must be divisible by pack_num"
    cols = N // pack_num

    if order_map is None:
        order_map = list(range(pack_num))
    order_map = torch.tensor(order_map, device=x.device)

    unit = 32 // pack_num  # number of bits per element

    # reshape to [B, K, cols, pack_num]
    pack_num_int = int(pack_num)

    x_reshape = x.view(B, K, cols, pack_num_int)

    # reorder according to order_map
    x_reorder = torch.gather(
        x_reshape, 3, order_map.view(1, 1, 1, -1).expand(B, K, cols, -1)
    )

    # mask low 4 bits
    x_reorder = x_reorder & 0xF

    # bit shifts [pack_num] -> [1,1,1,pack_num] broadcastable
    shifts = (unit * torch.arange(pack_num_int, device=x.device)).view(1, 1, 1, -1)

    # shift and sum along last dimension to combine bits
    packed = (x_reorder << shifts).sum(dim=-1).to(torch.int32)

    return packed



def get_moe_quant_method(
    config: "GPTQMarlinConfig",
    layer: torch.nn.Module,
    prefix: str,
    moe_method_cls: type,
):
    cloned_config = deepcopy(config)

    if isinstance(layer, FusedMoE):
        # False = skip module, None = no override, else = Positive match
        if (
            get_dynamic_override(  # noqa: E712
                cloned_config,  # noqa: E712
                layer_name=prefix,
            )
            == False
        ):  # noqa: E712
            return UnquantizedFusedMoEMethod(layer.moe_config)

        if prefix:
            # Dynamic per module/layer rules may override base config
            override_config(cloned_config, prefix=prefix)

        return moe_method_cls(cloned_config, layer.moe_config)
    return None


class GPTQMarlinConfig(QuantizationConfig):
    """Config class for GPTQ Marlin"""

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        is_sym: bool,
        lm_head_quantized: bool,
        dynamic: dict[str, dict[str, int | bool]],
        full_config: dict[str, Any],
        modules_in_block_to_quantize: list[str] | None = None,
    ) -> None:
        super().__init__()
        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        # GPTQModel use `dynamic` config property to allow per module
        # quantization config so each module can be individually optimized.
        # Format is dict[str, dict] where key is a regex string that can
        # perform both positive ("+:" prefixed) or negative ("-:" prefixed)
        # matching of a module.
        # Default to positive match, override base quant config mode, if no
        # prefix is used. Value is in dict format of field key and override
        # value.
        # Negative matching will skip quantization init for this module
        # entirely:
        # non-quantized inference. More details and quantization examples can be
        # found at: https://github.com/ModelCloud/GPTQModel
        # Example:
        #  # last 1/2 of the layers 10-21 has 8bit vs 4bit for 0-9
        #  # last 1/4 of the layers 16-21 has 8bit and group_size 64
        # dynamic = {
        #  #`.*\.` matches the layers_node prefix
        #  # positive match layer 10-15
        #  r"+:.*\.(?:1[0-5])\..*": {"bits": 8,},
        #  # positive match layer 16-21
        #  r"+:.*\.(?:1[6-9]|20|21)\..*": {"bits": 8, "group_size": 64,},
        #  r"-:.*\.moe\..*": {}, # negative match (skip) all `moe` layers
        # }
        self.dynamic = dynamic

        self.weight_bits = weight_bits
        self.is_sym = is_sym

        self.pack_factor = 32 // weight_bits  # packed into int32
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.full_config = full_config

        if (weight_bits, is_sym) not in self.TYPE_MAP:
            raise ValueError(
                f"Unsupported quantization config: bits={weight_bits}, sym={is_sym}"
            )

        self.quant_type = self.TYPE_MAP[(weight_bits, is_sym)]

        self.modules_in_block_to_quantize = modules_in_block_to_quantize or []
        # used to identify GPTQ model quantized by autoround
        self.autoround_version = full_config.get("autoround_version", "")

    def __repr__(self) -> str:
        return (
            f"GPTQMarlinConfig(quant_type={self.quant_type}, "
            f"group_size={self.group_size}, "
            f"desc_act={self.desc_act}, "
            f"lm_head_quantized={self.lm_head_quantized}, "
            f"dynamic={self.dynamic}, "
            f"modules_in_block_to_quantize={self.modules_in_block_to_quantize})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "gptq_marlin"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GPTQMarlinConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic

        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        is_sym = cls.get_from_keys(config, ["sym"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        modules_in_block_to_quantize = cls.get_from_keys_or(
            config, ["modules_in_block_to_quantize"], default=None
        )
        return cls(
            weight_bits,
            group_size,
            desc_act,
            is_sym,
            lm_head_quantized,
            dynamic,
            config,
            modules_in_block_to_quantize,
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        can_convert = cls.is_gptq_marlin_compatible(hf_quant_cfg)

        is_valid_user_quant = (
            user_quant is None or user_quant == "marlin" or user_quant == "gptq_marlin"
        )

        if can_convert and is_valid_user_quant:
            msg = (
                "The model is convertible to {} during runtime."
                " Using {} kernel.".format(cls.get_name(), cls.get_name())
            )
            logger.info(msg)
            return cls.get_name()

        if can_convert and user_quant == "gptq":
            logger.info(
                "Detected that the model can run with gptq_marlin"
                ", however you specified quantization=gptq explicitly,"
                " so forcing gptq. Use quantization=gptq_marlin for"
                " faster inference"
            )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, FusedMoE):
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            if not check_moe_marlin_supports_layer(layer, self.group_size):
                logger.warning_once(
                    f"Layer '{prefix}' is not supported by GPTQMoeMarlin. "
                    "Falling back to Moe WNA16 kernels."
                )
                return MoeWNA16Config.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            return get_moe_quant_method(self, layer, prefix, GPTQMarlinMoEMethod)
        return get_linear_quant_method(self, layer, prefix, GPTQMarlinLinearMethod)

    @classmethod
    def is_gptq_marlin_compatible(cls, quant_config: dict[str, Any]):
        quant_method = quant_config.get("quant_method", "").lower()
        num_bits = quant_config.get("bits")
        group_size = quant_config.get("group_size")
        sym = quant_config.get("sym")
        desc_act = quant_config.get("desc_act")

        if not current_platform.is_cuda():
            return False

        if quant_method != "gptq":
            return False

        # Marlin conversion is only valid if required properties are found
        if num_bits is None or group_size is None or sym is None or desc_act is None:
            return False

        if (num_bits, sym) not in cls.TYPE_MAP:
            return False

        return check_marlin_supported(
            quant_type=cls.TYPE_MAP[(num_bits, sym)], group_size=group_size
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.modules_in_block_to_quantize is not None:
            self.modules_in_block_to_quantize = hf_to_vllm_mapper.apply_list(
                self.modules_in_block_to_quantize
            )

    def maybe_update_config(self, model_name: str, revision: str | None = None):
        if self.modules_in_block_to_quantize:
            if is_list_of(self.modules_in_block_to_quantize, list):
                # original modules_in_block_to_quantize: list[list[str]]
                # flatten original modules_in_block_to_quantize
                self.modules_in_block_to_quantize = [
                    item
                    for sublist in self.modules_in_block_to_quantize
                    for item in sublist
                ]
            return

        unquant_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        metadata = get_safetensors_params_metadata(model_name, revision=revision)
        quant_layers: set[str] = {
            param_name.rsplit(".", 1)[0]
            for param_name, info in metadata.items()
            if (dtype := info.get("dtype", None))
            and _SAFETENSORS_TO_TORCH_DTYPE[dtype] not in unquant_dtypes
        }
        self.modules_in_block_to_quantize = list(quant_layers)


class GPTQMarlinLinearMethod(LinearMethodBase):
    """Linear method for GPTQ Marlin.

    Args:
        quant_config: The GPTQ Marlin quantization config.
    """

    _kernel_backends_being_used: set[str] = set()

    def __init__(self, quant_config: GPTQMarlinConfig) -> None:
        self.quant_config = quant_config

        # Verify supported on platform.
        verify_marlin_supported(
            quant_type=self.quant_config.quant_type,
            group_size=self.quant_config.group_size,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        is_row_parallel = input_size != input_size_per_partition
        weight_loader = extra_weight_attrs.get("weight_loader")

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_config.quant_type,
            act_type=params_dtype,
            group_size=self.quant_config.group_size,
            zero_points=False,
            has_g_idx=self.quant_config.desc_act,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for GPTQMarlinLinearMethod", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        # Normalize group_size
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(
            self.quant_config.desc_act, self.quant_config.group_size, is_row_parallel
        ):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = input_size // group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = input_size_per_partition // group_size

        # Quantized weights
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader,
        )

        # Activation order
        g_idx = RowvLLMParameter(
            data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )

        qzeros_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader": weight_loader,
        }
        weight_scale_args = {
            "data": torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader": weight_loader,
        }

        if scales_and_zp_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1, **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        else:
            scales = GroupQuantScaleParameter(
                output_dim=1, input_dim=0, **weight_scale_args
            )
            qzeros = PackedvLLMParameter(
                input_dim=0,
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args,
            )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class GPTQMarlinMoEMethod(FusedMoEMethodBase):
    """MoE Marlin method with quantization."""

    def __init__(
        self,
        quant_config: GPTQMarlinConfig,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        if self.quant_config.quant_type.size_bits == 4:
            self.quant_type = scalar_types.uint4b8
        # elif self.quant_config.quant_type.size_bits == 8:
        #     self.quant_type = scalar_types.uint8b128
        else:
            raise ValueError("GPTQMarlinMoEMethod only supports int4 and int8 now.")
        self.use_marlin = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        self.is_k_full = (not self.quant_config.desc_act) or (
            intermediate_size_per_partition == intermediate_size_full
        )

        if self.quant_config.group_size != -1:
            scales_size13 = hidden_size // self.quant_config.group_size
            w2_scales_size = (
                intermediate_size_full
                if self.quant_config.desc_act
                else intermediate_size_per_partition
            )
            scales_size2 = w2_scales_size // self.quant_config.group_size
            strategy = FusedMoeWeightScaleSupported.GROUP.value
        else:
            scales_size13 = 1
            scales_size2 = 1
            strategy = FusedMoeWeightScaleSupported.CHANNEL.value

        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})
        # Fused gate_up_proj (column parallel)
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.quant_config.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        # down_proj (row parallel)
        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        # up_proj scales
        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)
        # down_proj scales
        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_scales, {"load_full_w2": self.quant_config.desc_act})
        # up_proj scales
        w13_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qzeros", w13_qzeros)
        set_weight_attrs(w13_qzeros, extra_weight_attrs)
        # down_proj scales
        w2_qzeros = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size2,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qzeros", w2_qzeros)
        set_weight_attrs(w2_qzeros, extra_weight_attrs)
        # don't shard the w2 scales when running act order
        set_weight_attrs(w2_qzeros, {"load_full_w2": self.quant_config.desc_act})
        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)
        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)
        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        device = layer.w13_qweight.device
        # layer.workspace = marlin_make_workspace_new(device, 4)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process act_order
        # if self.quant_config.desc_act:
            # Get sorting based on g_idx
        #     num_experts = layer.w13_g_idx.shape[0]
        #     w13_g_idx_sort_indices = torch.empty_like(layer.w13_g_idx)
        #     w2_g_idx_sort_indices = torch.empty_like(layer.w2_g_idx)
        #     w13_sorted_g_idx = torch.empty_like(layer.w13_g_idx)
        #     w2_sorted_g_idx = torch.empty_like(layer.w2_g_idx)
        #     for e in range(num_experts):
        #         w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_g_idx[e]).to(
        #             torch.int32
        #         )
        #         w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_g_idx[e]).to(
        #             torch.int32
        #         )
        #         w13_sorted_g_idx[e] = layer.w13_g_idx[e][w13_g_idx_sort_indices[e]]
        #         w2_sorted_g_idx[e] = layer.w2_g_idx[e][w2_g_idx_sort_indices[e]]
        #     replace_parameter(layer, "w13_g_idx", w13_sorted_g_idx)
        #     replace_parameter(layer, "w2_g_idx", w2_sorted_g_idx)
        #     replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        #     replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        # else:
        #     # Reset g_idx related tensors
        #     num_experts = layer.w13_g_idx.shape[0]
        #     device = layer.w13_g_idx.device
        #     layer.w13_g_idx = torch.nn.Parameter(
        #         torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        #         requires_grad=False,
        #     )
        #     layer.w2_g_idx = torch.nn.Parameter(
        #         torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        #         requires_grad=False,
        #     )
        #     layer.w13_g_idx_sort_indices = torch.nn.Parameter(
        #         torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        #         requires_grad=False,
        #     )
        #     layer.w2_g_idx_sort_indices = torch.nn.Parameter(
        #         torch.empty((num_experts, 0), dtype=torch.int32, device=device),
        #         requires_grad=False,
        #     )
        # # Repack weights
        # marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        #     layer.w13_qweight,
        #     layer.w13_g_idx_sort_indices,
        #     layer.w13_qweight.shape[1] * self.quant_config.pack_factor,
        #     layer.w13_qweight.shape[2],
        #     self.quant_config.quant_type.size_bits,
        # )
        # replace_parameter(layer, "w13_qweight", marlin_w13_qweight)
        # marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        #     layer.w2_qweight,
        #     layer.w2_g_idx_sort_indices,
        #     layer.w2_qweight.shape[1] * self.quant_config.pack_factor,
        #     layer.w2_qweight.shape[2],
        #     self.quant_config.quant_type.size_bits,
        # )
        # replace_parameter(layer, "w2_qweight", marlin_w2_qweight)
        # # Repack scales
        # marlin_w13_scales = marlin_moe_permute_scales(
        #     s=layer.w13_scales,
        #     size_k=layer.intermediate_size_per_partition,
        #     size_n=layer.w13_scales.shape[2],
        #     group_size=self.quant_config.group_size,
        # )
        # replace_parameter(layer, "w13_scales", marlin_w13_scales)
        # marlin_w2_scales = marlin_moe_permute_scales(
        #     s=layer.w2_scales,
        #     size_k=layer.w2_scales.shape[1]
        #     * (
        #         self.quant_config.group_size
        #         if self.quant_config.group_size != -1
        #         else self.quant_config.pack_factor
        #     ),
        #     size_n=layer.w2_scales.shape[2],
        #     group_size=self.quant_config.group_size,
        # )
        # replace_parameter(layer, "w2_scales", marlin_w2_scales)

        # if hasattr(layer, "w13_bias") and layer.w13_bias is not None:
        #     layer.w13_bias.data = marlin_permute_bias(layer.w13_bias)

        # if hasattr(layer, "w2_bias") and layer.w2_bias is not None:
        #     layer.w2_bias.data = marlin_permute_bias(layer.w2_bias)
        if self.quant_config.desc_act:
            raise NotImplementedError(
                "GPTQMarlinMoEMethod now not support  desc_act. please fix it")   
        w13_qweight_unpacked = unpack_k_batch(layer.w13_qweight)
        w13_qweight_repacked = pack_n_batch(w13_qweight_unpacked,self.quant_config.pack_factor,order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        replace_parameter(layer, "w13_qweight", w13_qweight_repacked)
        
        # quant vllm/model_executor/layers/quantization/utils/quant_utils.py#quantize_weights
        # if quant_type.has_bias():
        #     w_q += quant_type.bias 
        # use  quant_type.bias as zp,(ixformer support)
        w13_zp = torch.full_like(layer.w13_scales, self.quant_type.bias, dtype=torch.int32)
        w13_zp_pack = pack_n_batch(w13_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]).contiguous()
        replace_parameter(layer, "w13_qzeros", w13_zp_pack)
        
        w2_qweight_unpacked = unpack_k_batch(layer.w2_qweight)
        w2_qweight_repacked = pack_n_batch(w2_qweight_unpacked,self.quant_config.pack_factor,order_map=[0, 2, 4, 6, 1, 3, 5, 7])
        replace_parameter(layer, "w2_qweight", w2_qweight_repacked)
        
        w2_zp = torch.full_like(layer.w2_scales, self.quant_type.bias, dtype=torch.int32)
        w2_zp_pack = pack_n_batch(w2_zp, self.quant_config.pack_factor, order_map=[0, 2, 4, 6, 1, 3, 5, 7]).contiguous()
        replace_parameter(layer, "w2_qzeros", w2_zp_pack)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `GPTQMarlinMoEMethod` yet."
            )

        assert activation == "silu", "Only SiLU activation is supported."
        use_ep = expert_map is not None
        
        if use_ep:
            start_eid = layer.ep_rank * layer.local_num_experts
            end_eid = min((layer.ep_rank + 1) * layer.local_num_experts, global_num_experts)
            
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "GPTQMarlinMoEMethod Apply router weight on input is not supported for"
                "fused Marlin MoE method.") 
            
        if (hasattr(layer, "w13_bias") and layer.w13_bias is not None) or (hasattr(layer, "w2_bias") and layer.w2_bias is not None):
            raise NotImplementedError(
                "GPTQMarlinMoEMethod moe_w4a16_group_gemm not supported bias, please fix this") 
                  
        topk_weights, topk_ids, _ = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)
        
        
        num_tokens, num_experts = router_logits.shape

        if use_ep:
            hidden_size = x.shape[1]
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
                expand_tokens,
            ) = ixfops.moe_compute_token_index_ep(
                topk_ids=topk_ids,
                num_experts=num_experts,
                start_expert_id=start_eid,
                end_expert_id=end_eid,
            )
            if expert_sizes_cpu.sum() == 0:
                return torch.zeros(
                    (num_tokens, hidden_size),
                    device=x.device,
                    dtype=x.dtype,
                )
        else:
            expand_tokens = num_tokens * top_k
            (
                src_to_dst,
                sorted_token_ids,
                expert_sizes_gpu,
                expert_sizes_cpu,
            ) = ixfops.moe_compute_token_index(
                topk_ids=topk_ids,
                num_experts=num_experts,
            )
            expert_sizes_cpu = expert_sizes_gpu.cpu()

        # expand + reorder
        # TODO use kernel
        expand_hidden_states = ixfops.moe_expand_input(
            hidden_states=x,
            dst_to_src=sorted_token_ids,
            dst_tokens=expand_tokens,
            topk=top_k,
            src_to_dst=src_to_dst,
        )

        # w4a16 group gemm 1
        # pt_output_1: (expand_tokens, 2n) dtype
        pt_output_1 = ixfops.moe_w4a16_group_gemm(
            input=expand_hidden_states,
            weight=layer.w13_qweight,
            w_scales=layer.w13_scales,
            quant_type="awq",
            tokens_per_experts=expert_sizes_cpu,
            w_zeros=layer.w13_qzeros,
            group_size=self.quant_config.group_size,
            dst_to_src=None,
            format="NN",
            tokens_per_experts_gpu=expert_sizes_gpu,
        )

        # act
        pt_output_2 = ixfops.silu_and_mul(pt_output_1)

        # w4a16 group gemm 2 + reorder
        # pt_output_3: (expand_tokens, k) dtype
        if use_ep:
            pt_output_3 = torch.empty(
                (num_tokens * top_k, hidden_size),
                device=x.device,
                dtype=x.dtype,
            )

            ixfops.moe_w4a16_group_gemm(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                tokens_per_experts=expert_sizes_cpu,
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                output=pt_output_3,
                tokens_per_experts_gpu=expert_sizes_gpu,
            )

            reduce_mask = src_to_dst == -1
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=routed_scaling_factor,
                mask=reduce_mask,
            )
        else:
            pt_output_3 = ixfops.moe_w4a16_group_gemm(
                input=pt_output_2,
                weight=layer.w2_qweight,
                w_scales=layer.w2_scales,
                quant_type="awq",
                tokens_per_experts=expert_sizes_cpu,
                w_zeros=layer.w2_qzeros,
                group_size=self.quant_config.group_size,
                dst_to_src=sorted_token_ids,
                format="NN",
                tokens_per_experts_gpu=expert_sizes_gpu,
            )

            # mul + reduce_sum
            # final_hidden_states: (num_tokens, k)
            final_hidden_states = ixfops.moe_output_reduce_sum(
                input=pt_output_3.view(num_tokens, top_k, -1),
                topk_weight=topk_weights,
                scaling_factor=routed_scaling_factor
            )
        return final_hidden_states
        
        
        
        

        # return torch.ops.vllm.fused_marlin_moe(
        #     x,
        #     layer.w13_qweight,
        #     layer.w2_qweight,
        #     getattr(layer, "w13_bias", None),
        #     getattr(layer, "w2_bias", None),
        #     layer.w13_scales,
        #     layer.w2_scales,
        #     router_logits,
        #     topk_weights,
        #     topk_ids,
        #     quant_type_id=self.quant_type.id,
        #     apply_router_weight_on_input=apply_router_weight_on_input,
        #     global_num_experts=global_num_experts,
        #     expert_map=expert_map,
        #     g_idx1=layer.w13_g_idx,
        #     g_idx2=layer.w2_g_idx,
        #     sort_indices1=layer.w13_g_idx_sort_indices,
        #     sort_indices2=layer.w2_g_idx_sort_indices,
        #     workspace=layer.workspace,
        #     is_k_full=self.is_k_full)
