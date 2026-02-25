# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V3.2 Indexer Backend for BI150.

This is a simplified implementation that:
1. Uses bf16 instead of FP8 (BI150 ixformer doesn't support FP8)
2. Provides basic prefill/decode support
3. Uses PyTorch fallback for operations not available in ixformer
"""
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)

logger = init_logger(__name__)


def get_max_prefill_buffer_size(vllm_config: VllmConfig) -> int:
    """Calculate max prefill buffer size for indexer.

    For BI150, we use a simpler calculation since we're using bf16 instead of FP8.
    """
    max_model_len = vllm_config.model_config.max_model_len
    # Use a conservative multiplier for bf16
    return max_model_len * 20


class DeepseekV32IndexerBackend(AttentionBackend):
    """Attention backend for DeepSeek V3.2 Indexer on BI150."""

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER_BI150"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        # For BI150: use bf16 directly, no FP8 packing
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    """Metadata for a single prefill chunk."""
    block_table: torch.Tensor
    cu_seq_lens: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    """Metadata for prefill phase."""
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    """Metadata for decode phase."""
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor


@dataclass
class DeepseekV32IndexerMetadata:
    """Complete metadata for indexer attention."""
    seq_lens: torch.Tensor
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    head_dim: int

    # Prefill/decode split info
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    """Metadata builder for DeepSeek V3.2 Indexer on BI150.

    This is a simplified builder that handles basic prefill/decode scenarios.
    """
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    reorder_batch_threshold: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)

        self.decode_lens_buffer = torch.empty(
            (scheduler_config.max_num_seqs,), dtype=torch.int32, device=self.device
        )

    def _split_decodes_and_prefills(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> tuple[int, int, int, int]:
        """Split requests into decode and prefill based on query length."""
        query_lens = common_attn_metadata.query_start_loc_cpu[1:] - \
                     common_attn_metadata.query_start_loc_cpu[:-1]

        num_decodes = 0
        num_decode_tokens = 0

        for i, qlen in enumerate(query_lens):
            if qlen <= self.reorder_batch_threshold:
                num_decodes += 1
                num_decode_tokens += qlen.item()
            else:
                break

        num_prefills = common_attn_metadata.num_reqs - num_decodes
        num_prefill_tokens = common_attn_metadata.num_actual_tokens - num_decode_tokens

        return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens

    def build_one_prefill_chunk(
        self, reqs_start: int, reqs_end: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        block_table: torch.Tensor
    ) -> DeepseekV32IndexerPrefillChunkMetadata:
        """Build metadata for a single prefill chunk."""
        token_start = query_start_loc_cpu[reqs_start].item()
        token_end = query_start_loc_cpu[reqs_end].item()
        total_seq_lens = seq_lens_cpu[reqs_start:reqs_end].sum().item()

        cu_seq_lens = torch.cat([
            torch.zeros(1, dtype=torch.int32),
            seq_lens_cpu[reqs_start:reqs_end].cumsum(dim=0),
        ]).to(torch.int32).to(self.device)

        return DeepseekV32IndexerPrefillChunkMetadata(
            block_table=block_table[reqs_start:reqs_end],
            cu_seq_lens=cu_seq_lens,
            total_seq_lens=total_seq_lens,
            token_start=token_start,
            token_end=token_end,
            num_reqs=reqs_end - reqs_start,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        """Build indexer metadata from common attention metadata."""
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = \
            self._split_decodes_and_prefills(common_attn_metadata)

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        # Build prefill metadata
        prefill_metadata = None
        if num_prefills > 0:
            # Simple single-chunk approach for now
            chunks = [
                self.build_one_prefill_chunk(
                    num_decodes,
                    num_reqs,
                    query_start_loc_cpu,
                    common_attn_metadata.seq_lens_cpu,
                    common_attn_metadata.block_table_tensor,
                )
            ]
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks=chunks)

        # Build decode metadata
        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[:num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=common_attn_metadata.block_table_tensor[:num_decodes],
                seq_lens=common_attn_metadata.seq_lens[:num_decodes],
                decode_lens=decode_lens,
            )

        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            num_reqs=num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            head_dim=128,  # DeepSeek V3.2 indexer head_dim
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )
