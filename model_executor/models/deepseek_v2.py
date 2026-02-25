# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm, LayerNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.gguf import GGUFConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
import ixformer.inference.functions as ops

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.logger import init_logger

logger = init_logger(__name__)
QUANT_CONFIG_FOR_GGUF = None
from torch.cuda import profiler


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias)

        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        self.add_shared_on_experts = quant_config.__class__.__name__ != "GGUFConfig" and config.n_shared_experts is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if self.add_shared_on_experts:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                extra_residual=shared_output,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits) * self.routed_scaling_factor
            if self.n_shared_experts is not None:
                final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)
            


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        vllm_config: Optional[VllmConfig] = None,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")
        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'

        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling and "factor" in rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(self.num_local_heads,
                              self.qk_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

        # DeepSeek V3.2 Indexer support (same as MLAAttention)
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            if int(prefix.split(".")[-2]) == 0:
                logger.warning(
                    "[DEBUG DeepseekV2Attention] is_v32=%s, "
                    "vllm_config_is_none=%s, q_lora_rank=%s, "
                    "topk_indices_buffer_is_none=%s",
                    self.is_v32, vllm_config is None,
                    q_lora_rank, topk_indices_buffer is None,
                )
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=not getattr(config, "indexer_rope_interleave",
                                          False),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

    def forward_opt(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q_latent_kpe = self.q_a_proj(hidden_states)[0]
            q, kv_a, k_pe = q_latent_kpe.split([self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q_latent_kpe = self.q_proj(hidden_states)[0]
            q, kv_a, k_pe = q_latent_kpe.split([self.num_heads * self.qk_head_dim, self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v_nope = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Use standard PyTorch ops instead of ixformer ops for debugging
        # Apply RoPE using standard rotary_emb
        # k_pe is 2D [num_tokens, rope_dim], need unsqueeze for rotary_emb
        q_pe, k_pe_rope = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Write back RoPE results
        q[..., self.qk_nope_head_dim:] = q_pe

        # Assemble k: [k_nope | k_pe_rope]
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe_rope  # broadcasts from [n,1,rope] to [n,heads,rope]

        # Assemble v: pad to qk_head_dim if needed
        v = torch.nn.functional.pad(
            v_nope, [0, self.qk_head_dim - self.v_head_dim], value=0
        )

        if not hasattr(DeepseekV2Attention, '_debug_fwd_logged'):
            DeepseekV2Attention._debug_fwd_logged = True
            logger.warning(
                "[DEBUG forward_opt] q.shape=%s, k.shape=%s, v.shape=%s, "
                "num_local_heads=%d, qk_head_dim=%d, v_head_dim=%d, "
                "qk_nope_head_dim=%d, qk_rope_head_dim=%d",
                q.shape, k.shape, v.shape,
                self.num_local_heads, self.qk_head_dim,
                self.v_head_dim, self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            )

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads, self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            kv_a, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split([self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0]
            kv_a, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split([self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        
        _, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v_nope = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.empty_like(q)
        v = torch.empty_like(q)
        ops.mla_rope(positions, q_pe, k_pe, k[...,self.qk_nope_head_dim:], self.rotary_emb.cos_sin_cache)
        ops.mla_copy_kv(k_nope, v_nope, k, v)

        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads, self.qk_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

    For more info see MLACommonImpl in: vllm/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        vllm_config: Optional[VllmConfig] = None,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        global QUANT_CONFIG_FOR_GGUF
        if QUANT_CONFIG_FOR_GGUF is None:
            import copy
            quant_config_for_gguf = copy.copy(quant_config) if quant_config is not None and quant_config.__class__.__name__ == "GGUFConfig" else quant_config
            if hasattr(quant_config_for_gguf, "keep_half_weight"):
                setattr(quant_config_for_gguf, "keep_half_weight", True)
            QUANT_CONFIG_FOR_GGUF = quant_config_for_gguf
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=QUANT_CONFIG_FOR_GGUF,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=QUANT_CONFIG_FOR_GGUF,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=QUANT_CONFIG_FOR_GGUF,
            prefix=f"{prefix}.kv_b_proj")
        from vllm import envs
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=QUANT_CONFIG_FOR_GGUF,
                                        prefix=f"{prefix}.o_proj")

        if rope_scaling:
            rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)
        if rope_scaling and "factor" in rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        # In the MLA backend, kv_cache includes both k_c and
        # pe (i.e. decoupled position embeddings). In particular,
        # the concat_and_cache_mla op requires
        #     k_c.size(1) + k_pe.size(1) == kv_cache.size(2)
        # i.e.
        #     kv_lora_rank + qk_rope_head_dim == head_size
        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.kv_lora_rank + self.qk_rope_head_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        # DeepSeek V3.2 Indexer support
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32 and self.debug_layer_idx == 0:
            logger.warning(
                "[DEBUG MLAAttention] layer=%d, is_v32=%s, "
                "vllm_config_is_none=%s, q_lora_rank=%s, "
                "topk_indices_buffer_is_none=%s",
                self.debug_layer_idx, self.is_v32,
                vllm_config is None, q_lora_rank,
                topk_indices_buffer is None,
            )
        if self.is_v32:
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=not getattr(config, "indexer_rope_interleave", False),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            kv_a, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split([self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            qr = self.q_a_layernorm(q)  # Save for indexer before q_b_proj
            kv_a = self.kv_a_layernorm(kv_a)

            # Call indexer if V3.2 (before q_b_proj)
            if self.is_v32 and self.indexer is not None:
                self.indexer(hidden_states, qr, positions, self.indexer_rope_emb)

            q = qr  # Use normalized q for mla_attn
        else:
            q = hidden_states
            latent_kpe = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_a, k_pe = latent_kpe.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            kv_a = self.kv_a_layernorm(kv_a)

        return self.mla_attn(q, kv_a, k_pe)

    def forward_opt(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q_latent_kpe = self.q_a_proj(hidden_states)[0]
            q, kv_a, k_pe, _ = q_latent_kpe.split([self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim, self.q_a_proj.output_padding_size], dim=1)
            qr = self.q_a_layernorm(q)  # Save for indexer
            kv_a = self.kv_a_layernorm(kv_a)

            # Call indexer if V3.2
            if self.is_v32 and self.indexer is not None:
                self.indexer(hidden_states, qr, positions, self.indexer_rope_emb)

            q = qr
        else:
            q = hidden_states
            latent_kpe = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_a, k_pe = latent_kpe.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=1)
            kv_a = self.kv_a_layernorm(kv_a)

        return self.mla_attn(q, kv_a, k_pe)


class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    """KV cache for DeepSeek V3.2 Indexer.

    For BI150: Uses bf16 directly instead of FP8 packing.
    """

    _debug_log_count = 0

    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        if DeepseekV32IndexerCache._debug_log_count == 0:
            logger.info(
                "[IndexerCache] Registered: prefix=%s, head_dim=%d, dtype=%s",
                prefix, head_dim, dtype,
            )
            DeepseekV32IndexerCache._debug_log_count += 1

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(  # Only has one vector instead of K + V
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> AttentionBackend:
        return DeepseekV32IndexerBackend


class Indexer(nn.Module):
    """DeepSeek V3.2 Sparse Attention Indexer.

    For BI150: Uses bf16 instead of FP8 quantization.
    """

    _debug_log_count = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        topk_indices_buffer: Optional[torch.Tensor],
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536

        # No tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        # BI150: No FP8 quantization, use bf16 directly
        self.topk_indices_buffer = topk_indices_buffer

        # BI150: Use bf16 cache instead of FP8 packed cache
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim,  # No FP8 scale packing
            dtype=torch.bfloat16,  # Use bf16 instead of uint8
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

        if Indexer._debug_log_count == 0:
            logger.info(
                "[Indexer] Initialized: index_topk=%d, head_dim=%d, "
                "q_lora_rank=%d, use_fp8=False (bi150 uses bf16)",
                self.topk_tokens, self.head_dim, q_lora_rank,
            )
            Indexer._debug_log_count += 1

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        """Forward pass for indexer.

        Args:
            hidden_states: [num_tokens, hidden_size]
            qr: [num_tokens, q_lora_rank] - q after q_a_layernorm
            positions: [num_tokens] position indices
            rotary_emb: RoPE embedding module
        Returns:
            topk_indices_buffer with top-k indices filled
        """
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
        # so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # BI150: No FP8 quantization, use bf16 directly
        # q stays as bf16: [num_tokens, n_head, head_dim]

        weights, _ = self.weights_proj(hidden_states)
        # Scale weights for attention
        weights = weights * self.softmax_scale * (self.n_head**-0.5)

        return self.indexer_op(hidden_states, q, k, weights)


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        vllm_config: Optional[VllmConfig] = None,
        topk_indices_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        if model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            vllm_config=vllm_config,
            topk_indices_buffer=topk_indices_buffer,
        )

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.routed_scaling_factor = config.routed_scaling_factor

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        if isinstance(self.mlp, DeepseekV2MoE) and \
            hidden_states.dtype == torch.float16:
            # This is a special case to avoid FP16 overflow
            hidden_states *= 1. / self.routed_scaling_factor
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if isinstance(self.mlp, DeepseekV2MLP) and \
            hidden_states.dtype == torch.float16:
            # This is a special case to avoid FP16 overflow
            hidden_states *= 1. / self.routed_scaling_factor
            residual *= 1. / self.routed_scaling_factor
        return hidden_states, residual


@support_torch_compile
class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.vocab_size = config.vocab_size

        # Detect DeepSeek V3.2 and create shared topk_indices_buffer
        self.is_v32 = hasattr(config, "index_topk")
        if self.is_v32:
            topk_tokens = config.index_topk
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=current_platform.device_type,
            )
            logger.info(
                "[DeepseekV2Model] V3.2 detected: index_topk=%d, "
                "topk_indices_buffer.shape=%s",
                topk_tokens, self.topk_indices_buffer.shape,
            )
        else:
            self.topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens")
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                vllm_config=vllm_config,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module, SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(vllm_config=vllm_config,
                                     prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.hidden_size,
                                          quant_config=quant_config)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
            "residual":
            torch.zeros((batch_size, self.config.hidden_size),
                        dtype=dtype,
                        device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        # Debug: print model structure info once
        indexer_params = [k for k in params_dict if '.indexer.' in k]
        attn_layers_with_indexer = set(
            k.rsplit('.indexer.', 1)[0] for k in indexer_params
        )
        sample_layer = 'model.layers.56.self_attn'
        sample_params = [k for k in params_dict if k.startswith(sample_layer)]
        logger.warning(
            "[DEBUG load_weights] use_mla=%s, is_v32=%s, "
            "total_params=%d, indexer_params=%d, "
            "layers_with_indexer=%s, "
            "params_under_%s=%s",
            getattr(self.config, 'use_mla', 'N/A'),
            hasattr(self.config, 'index_topk'),
            len(params_dict),
            len(indexer_params),
            sorted(attn_layers_with_indexer) if attn_layers_with_indexer else '[]',
            sample_layer,
            sample_params,
        )

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        opt_support_quant_method = ["GGUFLinearMethod", "UnquantizedLinearMethod", "CompressedTensorsW8A8Int8", "AWQMarlinLinearMethod"]
        # add your opt here..
        def inject_layer(layer, quant_method, is_mla):
            q_lora_rank = getattr(layer, "q_lora_rank", None)
            if quant_method in ["UnquantizedLinearMethod", "CompressedTensorsW8A8Int8"]:
                if q_lora_rank is not None:
                    layer.q_a_proj.weight.data = torch.cat([layer.q_a_proj.weight, layer.kv_a_proj_with_mqa.weight], dim=0)
                    if hasattr(layer.q_a_proj, "weight_scale"):
                        layer.q_a_proj.weight_scale.data = torch.cat([layer.q_a_proj.weight_scale, layer.kv_a_proj_with_mqa.weight_scale], dim=0)
                        del layer.kv_a_proj_with_mqa.weight_scale
                elif not is_mla:
                    layer.q_proj.weight.data = torch.cat([layer.q_proj.weight, layer.kv_a_proj_with_mqa.weight], dim=0)
                    if hasattr(layer.q_proj, "weight_scale"):
                        layer.q_proj.weight_scale.data = torch.cat([layer.q_proj.weight_scale, layer.kv_a_proj_with_mqa.weight_scale], dim=0)
                        del layer.kv_a_proj_with_mqa.weight_scale
                else:
                    return
                del layer.kv_a_proj_with_mqa.weight
                del layer.kv_a_proj_with_mqa
                layer.forward = layer.forward_opt
            elif quant_method == "GGUFLinearMethod":
                pass
            elif quant_method == "AWQMarlinLinearMethod":
                dtype = layer.kv_a_proj_with_mqa.qweight.dtype
                assert dtype == torch.int32
                if layer.q_lora_rank is not None:
                    layer.q_a_proj.qweight.data = torch.cat([layer.q_a_proj.qweight, layer.kv_a_proj_with_mqa.qweight], dim=1)
                    layer.q_a_proj.scales.data = torch.cat([layer.q_a_proj.scales, layer.kv_a_proj_with_mqa.scales], dim=1)
                    del layer.kv_a_proj_with_mqa.scales
                    layer.q_a_proj.qzeros.data = torch.cat([layer.q_a_proj.qzeros, layer.kv_a_proj_with_mqa.qzeros], dim=1)
                    del layer.kv_a_proj_with_mqa.qzeros
                elif not is_mla:
                    layer.q_proj.weight.data = torch.cat([layer.q_proj.weight, layer.kv_a_proj_with_mqa.weight], dim=1)
                    layer.q_proj.scales.data = torch.cat([layer.q_proj.scales, layer.kv_a_proj_with_mqa.scales], dim=1)
                    del layer.kv_a_proj_with_mqa.scales
                    layer.q_proj.qzeros.data = torch.cat([layer.q_proj.qzeros, layer.kv_a_proj_with_mqa.qzeros], dim=1)
                    del layer.kv_a_proj_with_mqa.qzeros
                else:
                    return

                del layer.kv_a_proj_with_mqa.qweight
                del layer.kv_a_proj_with_mqa
                layer.forward = layer.forward_opt
            else:
                pass
        
        for _, layer in self.model.named_modules():
            if layer.__class__.__name__ in ["DeepseekV2Attention","DeepseekV2MLAAttention"]:
                if hasattr(layer.kv_a_proj_with_mqa, "scheme"):
                    quant_method = layer.kv_a_proj_with_mqa.scheme.__class__.__name__
                else:
                    quant_method = layer.kv_a_proj_with_mqa.quant_method.__class__.__name__
                if quant_method not in opt_support_quant_method:
                    break

                inject_layer(layer, quant_method, is_mla = layer.__class__.__name__ == "DeepseekV2MLAAttention")
        
        return loaded_params


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


def get_spec_layer_idx_from_weight_name(config: PretrainedConfig,
                                        weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass
