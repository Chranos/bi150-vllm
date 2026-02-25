# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse Attention Indexer for BI150.

This is a simplified implementation that:
1. Uses bf16 instead of FP8 (BI150 ixformer doesn't support FP8)
2. Uses PyTorch fallback for operations not available in ixformer
3. Reuses ixformer ops where possible (mla_rope, mla_copy_kv, paged_attention)
"""
import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)

logger = init_logger(__name__)


class SparseAttnIndexer(nn.Module):
    """Sparse Attention Indexer for BI150.

    Simplified version that uses bf16 and PyTorch ops instead of
    FP8/DeepGEMM/custom CUDA kernels.
    """

    _debug_log_count = 0

    def __init__(
        self,
        k_cache,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for sparse attention indexer.

        Args:
            hidden_states: [num_tokens, hidden_size]
            q: [num_tokens, n_head, head_dim] in bf16
            k: [num_tokens, head_dim] in bf16
            weights: [num_tokens, n_head] pre-scaled weights
        Returns:
            topk_indices_buffer with top-k indices filled
        """
        attn_metadata = get_forward_context().attn_metadata

        # During profiling/dummy run, return buffer as-is
        if not isinstance(attn_metadata, dict):
            return self.topk_indices_buffer

        attn_metadata = attn_metadata[self.k_cache.prefix]
        assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)

        if SparseAttnIndexer._debug_log_count == 0:
            logger.info(
                "[SparseAttnIndexer] Forward: "
                "q.shape=%s, k.shape=%s, "
                "num_prefill_tokens=%d, num_decode_tokens=%d",
                q.shape, k.shape,
                attn_metadata.num_prefill_tokens,
                attn_metadata.num_decode_tokens,
            )
            SparseAttnIndexer._debug_log_count += 1

        slot_mapping = attn_metadata.slot_mapping
        kv_cache = self.k_cache.kv_cache[0]
        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0

        # Insert k into cache (bf16 scatter, replacing FP8 quantized cache)
        self._cache_k(k, kv_cache, slot_mapping)

        # Initialize topk_indices to -1
        self.topk_indices_buffer[:hidden_states.shape[0]] = -1

        if has_prefill:
            self._prefill_forward(q, kv_cache, weights, attn_metadata)

        if has_decode:
            self._decode_forward(
                q[:num_decode_tokens], kv_cache,
                weights[:num_decode_tokens], attn_metadata,
            )

        return self.topk_indices_buffer

    def _cache_k(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Insert k vectors into paged KV cache using bf16 (no FP8).

        Args:
            k: [num_tokens, head_dim]
            kv_cache: [num_blocks, block_size, head_dim]
            slot_mapping: [num_tokens] mapping tokens to cache slots
        """
        block_size = kv_cache.shape[1]
        block_idx = slot_mapping // block_size
        block_offset = slot_mapping % block_size
        valid = slot_mapping >= 0
        valid_bi = block_idx[valid]
        valid_bo = block_offset[valid]
        kv_cache[valid_bi, valid_bo, :] = k[valid].to(kv_cache.dtype)

    def _gather_k_from_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        total_seq_lens: int,
    ) -> torch.Tensor:
        """Gather k vectors from paged cache for prefill chunks.

        Replaces ops.cp_gather_indexer_k_quant_cache (which gathers FP8+scale).
        We gather bf16 directly.

        Args:
            kv_cache: [num_blocks, block_size, head_dim]
            block_table: [num_reqs, max_blocks_per_seq]
            cu_seq_lens: [num_reqs + 1] cumulative sequence lengths
            total_seq_lens: total number of tokens across all seqs
        Returns:
            k_gathered: [total_seq_lens, head_dim] in bf16
        """
        block_size = kv_cache.shape[1]
        head_dim = kv_cache.shape[2]
        num_reqs = block_table.shape[0]
        k_gathered = torch.empty(
            total_seq_lens, head_dim,
            dtype=kv_cache.dtype, device=kv_cache.device,
        )
        offset = 0
        for req_idx in range(num_reqs):
            seq_start = cu_seq_lens[req_idx].item()
            seq_end = cu_seq_lens[req_idx + 1].item()
            seq_len = seq_end - seq_start
            for pos in range(seq_len):
                block_idx = pos // block_size
                block_offset = pos % block_size
                physical_block = block_table[req_idx, block_idx]
                k_gathered[offset] = kv_cache[physical_block, block_offset]
                offset += 1
        return k_gathered

    def _prefill_forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: DeepseekV32IndexerMetadata,
    ):
        """Prefill forward pass using bf16 attention.

        Replaces fp8_mqa_logits with torch.bmm in bf16.
        Replaces torch.ops._C.top_k_per_row_prefill with torch.topk.

        Args:
            q: [num_tokens, n_head, head_dim]
            kv_cache: [num_blocks, block_size, head_dim]
            weights: [num_tokens, n_head]
            attn_metadata: metadata with prefill chunks
        """
        prefill_metadata = attn_metadata.prefill
        for chunk in prefill_metadata.chunks:
            # Gather k from cache for this chunk
            k_gathered = self._gather_k_from_cache(
                kv_cache,
                chunk.block_table,
                chunk.cu_seq_lens,
                chunk.total_seq_lens,
            )
            # q for this chunk: [chunk_tokens, n_head, head_dim]
            q_chunk = q[chunk.token_start:chunk.token_end]
            weights_chunk = weights[chunk.token_start:chunk.token_end]
            num_chunk_tokens = q_chunk.shape[0]
            n_head = q_chunk.shape[1]

            # Compute attention logits per request in chunk
            num_reqs = chunk.num_reqs
            for req_idx in range(num_reqs):
                # Get token range for this request
                if req_idx == 0:
                    tok_start = 0
                else:
                    tok_start = chunk.cu_seq_lens[req_idx].item() - \
                                chunk.cu_seq_lens[0].item()
                tok_end = chunk.cu_seq_lens[req_idx + 1].item() - \
                          chunk.cu_seq_lens[0].item()
                seq_len = tok_end - tok_start

                # Get k for this sequence
                k_seq = k_gathered[tok_start:tok_end]  # [seq_len, head_dim]

                # Get q tokens for this request (query tokens in prefill)
                # In prefill, each token attends to all previous tokens
                for tok_idx in range(seq_len):
                    global_tok_idx = chunk.token_start + tok_start + tok_idx
                    q_tok = q_chunk[tok_start + tok_idx]  # [n_head, head_dim]
                    w_tok = weights_chunk[tok_start + tok_idx]  # [n_head]

                    # Causal: only attend to positions 0..tok_idx
                    k_causal = k_seq[:tok_idx + 1]  # [causal_len, head_dim]
                    causal_len = k_causal.shape[0]

                    # Compute logits: q @ k^T -> [n_head, causal_len]
                    logits = torch.matmul(
                        q_tok.float(),  # [n_head, head_dim]
                        k_causal.float().T,  # [head_dim, causal_len]
                    )  # [n_head, causal_len]

                    # Apply weights and sum across heads
                    # logits: [n_head, causal_len], w_tok: [n_head]
                    weighted_logits = (logits * w_tok.float().unsqueeze(-1)).sum(dim=0)
                    # weighted_logits: [causal_len]

                    # Top-k selection
                    k_val = min(self.topk_tokens, causal_len)
                    _, topk_idx = torch.topk(weighted_logits, k_val)
                    self.topk_indices_buffer[global_tok_idx, :k_val] = topk_idx.to(
                        self.topk_indices_buffer.dtype
                    )

    def _decode_forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights: torch.Tensor,
        attn_metadata: DeepseekV32IndexerMetadata,
    ):
        """Decode forward pass using paged attention.

        Replaces fp8_paged_mqa_logits with gather + torch.bmm.
        Replaces torch.ops._C.top_k_per_row_decode with torch.topk.

        Args:
            q: [num_decode_tokens, n_head, head_dim]
            kv_cache: [num_blocks, block_size, head_dim]
            weights: [num_decode_tokens, n_head]
            attn_metadata: metadata with decode info
        """
        decode_metadata = attn_metadata.decode
        block_table = decode_metadata.block_table
        seq_lens = decode_metadata.seq_lens
        block_size = kv_cache.shape[1]
        num_decode_tokens = q.shape[0]

        for tok_idx in range(num_decode_tokens):
            seq_len = seq_lens[tok_idx].item()
            q_tok = q[tok_idx]  # [n_head, head_dim]
            w_tok = weights[tok_idx]  # [n_head]

            # Gather k from paged cache for this sequence
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            k_seq = []
            for blk_idx in range(num_blocks_needed):
                physical_block = block_table[tok_idx, blk_idx]
                start_pos = blk_idx * block_size
                end_pos = min(start_pos + block_size, seq_len)
                tokens_in_block = end_pos - start_pos
                k_seq.append(kv_cache[physical_block, :tokens_in_block])
            k_seq = torch.cat(k_seq, dim=0)  # [seq_len, head_dim]

            # Compute logits: q @ k^T -> [n_head, seq_len]
            logits = torch.matmul(
                q_tok.float(),  # [n_head, head_dim]
                k_seq.float().T,  # [head_dim, seq_len]
            )  # [n_head, seq_len]

            # Apply weights and sum across heads
            weighted_logits = (logits * w_tok.float().unsqueeze(-1)).sum(dim=0)
            # weighted_logits: [seq_len]

            # Top-k selection
            k_val = min(self.topk_tokens, seq_len)
            _, topk_idx = torch.topk(weighted_logits, k_val)
            self.topk_indices_buffer[tok_idx, :k_val] = topk_idx.to(
                self.topk_indices_buffer.dtype
            )
