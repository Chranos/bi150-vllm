# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import ClassVar

import torch

from vllm.attention.backends.abstract import (
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.distributed.parallel_state import get_dcp_group
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
import ixformer.inference.functions as ixf_ops
import vllm.envs as envs
from vllm import _custom_ops as ops

logger = init_logger(__name__)


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported"
            )

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        return super()._flash_attn_varlen_diff_headdims(
            q,
            k,
            v,
            return_softmax_lse=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_c_normed: torch.Tensor | None,
        k_pe: torch.Tensor | None,
        kv_c_and_k_pe_cache_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        decode_meta = attn_metadata.decode
        q_nope = self._k_up_proj(q_nope)
        q_nope = q_nope.view(-1, self.num_heads, self.kv_lora_rank)

        B = q_nope.shape[0]
        
        if self.dcp_world_size > 1:
            q = torch.cat([q_nope, q_pe], dim=-1)
            q = get_dcp_group().all_gather(q, dim=1)
            o = torch.empty(B,
                        q.shape[1],
                        self.kv_lora_rank,
                        dtype=q_nope.dtype,
                        device=q_nope.device)
            if envs.VLLM_USE_INT8_MLA:
                q_int8, q_scale = ops.quant_kv(q)
                attn_out, softmax_lse = ixf_ops.vllm_paged_attention_mla_int8(
                o,
                q_int8,
                q_scale,
                kv_c_and_k_pe_cache,
                kv_c_and_k_pe_cache_scale, 
                self.scale,
                attn_metadata.decode.block_table,
                attn_metadata.decode.seq_lens,
                attn_metadata.decode.max_decode_seq_len,
                return_softmax_lse=True   
        )
            else:
                attn_out, softmax_lse = ixf_ops.vllm_paged_attention_mla(
                    output=o,
                    query=q, 
                    kv_cache=kv_c_and_k_pe_cache, 
                    scale=self.scale, 
                    block_tables=attn_metadata.decode.block_table,
                    context_lens=attn_metadata.decode.seq_lens,
                    max_context_len=decode_meta.max_decode_seq_len,
                    return_softmax_lse=True)
            return attn_out, softmax_lse
            
        o = torch.empty(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q_nope.dtype,
                        device=q_nope.device)   

        if envs.VLLM_USE_INT8_MLA:
            q = torch.cat([q_nope, q_pe], dim=-1)
            q_int8, q_scale = ops.quant_kv(q)
            ixf_ops.vllm_paged_attention_mla_int8(
                o,
                q_int8,
                q_scale,
                kv_c_and_k_pe_cache,
                kv_c_and_k_pe_cache_scale, 
                self.scale,
                attn_metadata.decode.block_table,
                attn_metadata.decode.seq_lens,
                attn_metadata.decode.max_decode_seq_len,
                attn_metadata.decode.use_cuda_graph
        )
        else:
        # fused q concat & cache write
            ixf_ops.vllm_paged_attention_mla_fused(
                output=o,
                q_nope=q_nope,
                q_pe=q_pe.contiguous(),
                kv_cache=kv_c_and_k_pe_cache,
                scale=self.scale,
                block_tables=attn_metadata.decode.block_table,
                context_lens=attn_metadata.decode.seq_lens,
                max_context_len=decode_meta.max_decode_seq_len,
                k_c_normed=k_c_normed,
                k_pe=k_pe,
                use_cuda_graph=decode_meta.use_cuda_graph
            )
        return self._v_up_proj(o), None
