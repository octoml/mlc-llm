from typing import Optional, Tuple, Union
from enum import Enum, auto

from dataclasses import dataclass

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl, reshape, expand_dims, concat, zeros, take, concat
from tvm.relax.op.nn import attention_var_len
from tvm.relax.testing import nn
from tvm.ir import VDevice
from tvm.script import relax as R
from tvm.script.ir_builder import tir as T

from ..quantization import QuantizationScheme
from .modules import ModuleList
from .param_manager import ParamManager
from .llama import (
    LlamaConfig,
    MixtralConfig,
    Linear,
    Embedding,
    LlamaRMSNorm,
    LlamaAttentionBase,
    LlamaDecoderLayer,
    get_param_quant_kind,
    setup_params,
    rotary_modulate_by_freq,
)


def apply_rotary_pos_emb(q, k, positions, position_embedding_base):
    def f_rotary_embedding(tensor, pos_tensor):
        def rotary_compute(*idx):
            pos = pos_tensor[idx[0]].astype("float32")
            return rotary_modulate_by_freq(
                tensor,
                idx,
                pos,
                position_embedding_base,
            )

        return tvm.te.compute(tensor.shape, rotary_compute, name="rotary")

    q_embed = nn.emit_te(f_rotary_embedding, q, positions, primfunc_name_hint="rotary_embedding")
    k_embed = nn.emit_te(f_rotary_embedding, k, positions, primfunc_name_hint="rotary_embedding")
    return q_embed, k_embed


class KVCacheType(Enum):
    VLLM = auto()
    FlashDecoding = auto()


@dataclass
class PrefillAttentionInput:
    seq_start: Optional[relax.Expr]  # (num_seq + 1,)
    indices_within_window: Optional[relax.Expr]  # (num_cached_total,)


@dataclass
class DecodeAttentionInput:
    seq_lens: relax.Expr  # (num_seq,)
    block_tables: Optional[relax.Expr]  # (num_seq, max_num_blocks_per_seq)


@dataclass
class EvaluateMultiQueryInput:
    seq_start: Optional[relax.Expr]  # (num_seq + 1,)
    query_start: relax.Expr  # (num_query_token + 1,)
    max_query_len: relax.Expr  # (), must be on CPU
    # The followings are only needed for our naive implementation of multi-query eval
    # with paged KV cache. They can be replaced with block_tables when a proper attention
    # kernel becomes available.
    past_slot_mapping: relax.Expr  # (num_past_token,)
    permute_indices_after_concat: relax.Expr  # (num_past_token + num_query_token,)


@dataclass
class AttentionInput:
    # KV cache and slot_mapping are not needed during memory profiling, hench they are optional.
    kv_cache: Optional[Tuple[relax.Expr, relax.Expr]]
    slot_mapping: Optional[relax.Expr]  # (num_query_token,)
    max_seqlen: relax.Expr  # (), must be on CPU
    aux_info: Union[PrefillAttentionInput, DecodeAttentionInput, EvaluateMultiQueryInput]


class AttentionBackend:
    def __init__(self, num_query_heads, num_key_value_heads, head_dim):
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

    def decode_attention(
        self,
        queries,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        num_seq,
        seqlen_q,
    ):
        pass

    def update_cache(self, keys, values, k_cache, v_cache, slot_mapping):
        pass

    def reconstruct_from_cache(self, k_cache, v_cache, past_slot_mapping):
        pass


class VllmAttention(AttentionBackend):
    block_size: int = 16

    def __init__(self, num_query_heads, num_key_value_heads, head_dim, max_context_length):
        super().__init__(num_query_heads, num_key_value_heads, head_dim)

        partition_size = 512  # partition_size in vLLM attention
        self.max_num_partitions = (max_context_length + partition_size - 1) // partition_size

    def decode_attention(
        self,
        queries,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        num_seq,
        seqlen_q,
    ):
        num_query_tokens = queries.struct_info.shape[0]
        exp_sums = nn.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr((num_query_tokens, self.num_query_heads, self.max_num_partitions)),
                dtype="float32",
                runtime_device_index=0,
            )
        )
        max_logits = nn.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr((num_query_tokens, self.num_query_heads, self.max_num_partitions)),
                dtype="float32",
                runtime_device_index=0,
            )
        )
        tmp_out = nn.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr(
                    (
                        num_query_tokens,
                        self.num_query_heads,
                        self.max_num_partitions,
                        self.head_dim,
                    )
                ),
                dtype=queries.struct_info.dtype,
                runtime_device_index=0,
            )
        )
        return nn.emit(
            relax.op.call_dps_packed(
                "tvm.contrib.vllm.single_query_cached_kv_attention",
                [
                    queries,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    16,  # block_size
                    max_context_len,
                    exp_sums,
                    max_logits,
                    tmp_out,
                ],
                out_sinfo=queries.struct_info,
            )
        )

    def update_cache(self, keys, values, k_cache, v_cache, slot_mapping):
        return nn.emit(
            relax.op.call_pure_packed(
                "tvm.contrib.vllm.reshape_and_cache",
                keys,
                values,
                k_cache,
                v_cache,
                slot_mapping,
                sinfo_args=[k_cache.struct_info, v_cache.struct_info],
            )
        )

    def reconstruct_from_cache(self, k_cache, v_cache, past_slot_mapping):
        num_kv_head = v_cache.struct_info.shape[1]
        head_size = v_cache.struct_info.shape[2]

        num_past_token = past_slot_mapping.struct_info.shape[0]
        kv_shape = (num_past_token, num_kv_head, head_size)
        kv_sinfo = relax.TensorStructInfo(kv_shape, k_cache.struct_info.dtype)

        return nn.emit(
            relax.op.call_pure_packed(
                "tvm.contrib.vllm.reconstruct_from_cache",
                k_cache,
                v_cache,
                past_slot_mapping,
                sinfo_args=[kv_sinfo, kv_sinfo],
            )
        )


class FlashDecodingAttention(AttentionBackend):
    block_size: int = 256

    def __init__(self, num_query_heads, num_key_value_heads, head_dim):
        super().__init__(num_query_heads, num_key_value_heads, head_dim)
        self.max_num_partitions = 128

    def decode_attention(
        self,
        queries,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        max_context_len,
        num_seq,
        seqlen_q,
    ):
        queries = nn.emit(
            reshape(queries, (num_seq, seqlen_q, self.num_query_heads, self.head_dim))
        )

        softmax_lse_accum = nn.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr((self.max_num_partitions, num_seq, self.num_query_heads, seqlen_q)),
                dtype="float32",
                runtime_device_index=0,
            )
        )
        output_accum = nn.emit(
            relax.op.builtin.alloc_tensor(
                relax.ShapeExpr(
                    (
                        self.max_num_partitions,
                        num_seq,
                        self.num_query_heads,
                        seqlen_q,
                        self.head_dim,
                    )
                ),
                dtype="float32",
                runtime_device_index=0,
            )
        )

        return R.call_dps_packed(
            "tvm.contrib.flash_attn.flash_decoding_with_paged_kvcache",
            [
                queries,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                softmax_lse_accum,
                output_accum,
            ],
            out_sinfo=queries.struct_info,
        )

    def update_cache(self, keys, values, k_cache, v_cache, slot_mapping):
        return nn.emit(
            relax.op.call_pure_packed(
                "tvm.contrib.flash_attn.update_cache",
                keys,
                values,
                k_cache,
                v_cache,
                slot_mapping,
                sinfo_args=[k_cache.struct_info, v_cache.struct_info],
            )
        )

    def reconstruct_from_cache(self, k_cache, v_cache, past_slot_mapping):
        num_kv_head = v_cache.struct_info.shape[2]
        head_size = v_cache.struct_info.shape[-1]

        num_past_token = past_slot_mapping.struct_info.shape[0]
        kv_shape = (num_past_token, num_kv_head, head_size)
        kv_sinfo = relax.TensorStructInfo(kv_shape, k_cache.struct_info.dtype)

        return nn.emit(
            relax.op.call_pure_packed(
                "tvm.contrib.flash_attn.reconstruct_from_cache",
                k_cache,
                v_cache,
                past_slot_mapping,
                sinfo_args=[kv_sinfo, kv_sinfo],
            )
        )


class LlamaAttentionBatched(LlamaAttentionBase):
    def __init__(self, config: LlamaConfig, kv_type: KVCacheType):
        super().__init__(config)
        if kv_type == KVCacheType.VLLM:
            max_context_length = config.sliding_window or config.max_sequence_length
            self.attn_backend = VllmAttention(
                self.num_query_heads, self.num_key_value_heads, self.head_dim, max_context_length
            )
        else:
            self.attn_backend = FlashDecodingAttention(
                self.num_query_heads, self.num_key_value_heads, self.head_dim
            )

        self.sliding_window = None

        if config.sliding_window:
            self.sliding_window = T.IntImm("int32", config.sliding_window)

    def forward(
        self,
        hidden_states: relax.Expr,  # (num_query_token, hidden_size) or (num_seq, seqlen_q, hidden_size)
        positions: relax.Expr,  # (num_query_token,), for batched RoPE
        attn_input: AttentionInput,
    ):
        num_query_tokens = positions.struct_info.shape[0]

        queries, keys, values = self.project_qkv(
            hidden_states,
            (num_query_tokens, self.num_query_heads, self.head_dim),
            (num_query_tokens, self.num_key_value_heads, self.head_dim),
        )

        queries, keys = apply_rotary_pos_emb(queries, keys, positions, self.position_embedding_base)

        if attn_input.kv_cache:
            # Paged KV cache update
            k_cache, v_cache = attn_input.kv_cache

            if isinstance(attn_input.aux_info, PrefillAttentionInput):
                indices_within_window = attn_input.aux_info.indices_within_window
            else:
                indices_within_window = None

            if indices_within_window:
                # Cache only the most recent keys and values within the window.
                keys_to_cache = nn.emit(take(keys, indices_within_window, axis=0))
                values_to_cache = nn.emit(take(values, indices_within_window, axis=0))
                slot_mapping = nn.emit(take(attn_input.slot_mapping, indices_within_window, axis=0))
            else:
                # For decode or prefill without sliding window, cache all keys / values.
                keys_to_cache = keys
                values_to_cache = values
                slot_mapping = attn_input.slot_mapping

            # kv caches are updated inplace, but make it look like a pure operation
            kv = self.attn_backend.update_cache(
                keys_to_cache, values_to_cache, k_cache, v_cache, slot_mapping
            )
            k_cache, v_cache = kv[0], kv[1]
        else:
            k_cache = v_cache = None

        if isinstance(attn_input.aux_info, EvaluateMultiQueryInput):
            assert k_cache and v_cache

            kv_tensors = self.attn_backend.reconstruct_from_cache(
                k_cache, v_cache, attn_input.aux_info.past_slot_mapping
            )
            keys_past, values_past = kv_tensors[0], kv_tensors[1]
            # Say we have past tokens [P1, P2, P3] and the current ones [C1, C2, C3].
            # Each of P1, C1 etc is a sequence of tokens.
            # After concat, we have [P1, P2, P3, C1, C2, C3], but batched sequences need to
            # be in the format [P1, C1, P2, C2, P3, C3]. This permutation is done by the take
            # op and the provided permutation indices.
            keys = nn.emit(
                take(
                    concat([keys_past, keys]),
                    attn_input.aux_info.permute_indices_after_concat,
                    axis=0,
                )
            )
            values = nn.emit(
                take(
                    concat([values_past, values]),
                    attn_input.aux_info.permute_indices_after_concat,
                    axis=0,
                )
            )
            seq_start_q = attn_input.aux_info.query_start
            max_seqlen_q = attn_input.aux_info.max_query_len
            seq_start_k = attn_input.aux_info.seq_start
            max_seqlen_k = attn_input.max_seqlen
        elif isinstance(attn_input.aux_info, PrefillAttentionInput):
            # prefill
            seq_start_q = seq_start_k = attn_input.aux_info.seq_start
            max_seqlen_q = max_seqlen_k = attn_input.max_seqlen
        else:
            # decode
            seq_start_q = seq_start_k = None
            max_seqlen_q = max_seqlen_k = None

        if seq_start_q:
            # Prefill or multi-query evaluation, batched attention over variable sequence lengths
            attn_output = nn.emit(
                attention_var_len(
                    nn.emit(expand_dims(queries, axis=0)),
                    nn.emit(expand_dims(keys, axis=0)),
                    nn.emit(expand_dims(values, axis=0)),
                    seq_start_q,
                    max_seqlen_q,
                    seq_start_k,
                    max_seqlen_k,
                    causal_mask="BottomRight",
                    window_size=self.sliding_window,
                )
            )
        else:
            # Decode, using vLLM or Flash-Decoding kernel
            assert isinstance(attn_input.aux_info, DecodeAttentionInput)

            if len(hidden_states.struct_info.shape) == 3:
                num_seq, seqlen_q, _ = hidden_states.struct_info.shape
            else:
                num_seq = hidden_states.struct_info.shape[0]
                seqlen_q = 1

            attn_output = self.attn_backend.decode_attention(
                queries,
                k_cache,
                v_cache,
                attn_input.aux_info.block_tables,
                attn_input.aux_info.seq_lens,
                attn_input.max_seqlen,
                num_seq,
                seqlen_q,
            )

        attn_output = nn.emit(reshape(attn_output, hidden_states.struct_info.shape))
        attn_output = self.o_proj(attn_output)

        return attn_output, (k_cache, v_cache)


class LlamaDecoderLayerBatched(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, kv_type: KVCacheType):
        super().__init__(config, False)
        self.self_attn = LlamaAttentionBatched(config, kv_type)

    def forward(
        self,
        hidden_states: relax.Expr,
        positions: relax.Expr,
        attn_input: AttentionInput,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            positions,
            attn_input,
        )

        hidden_states = self.post_self_attn(hidden_states, residual)

        return hidden_states, new_kv


def create_seq_start(seq_lens):
    # https://github.com/apache/tvm/issues/15851 for why we need to use Thrust
    cumsum = nn.emit(
        relax.op.call_dps_packed(
            "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
        )
    )
    return nn.emit(concat([zeros((1,), "int32"), cumsum]))


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.Var,
        kv_type: KVCacheType,
        sep_embed: bool = False,
    ):
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [LlamaDecoderLayerBatched(config, kv_type) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: relax.Expr,
        positions: relax.Expr,
        kv_caches: Optional[relax.Expr],
        slot_mapping: Optional[relax.Expr],
        max_seqlen: relax.Expr,
        attn_aux_info: Union[PrefillAttentionInput, DecodeAttentionInput, EvaluateMultiQueryInput],
    ):
        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        new_kvs = ()

        for idx, decoder_layer in enumerate(self.layers):
            if kv_caches:
                cache = (kv_caches[2 * idx], kv_caches[2 * idx + 1])
            else:
                cache = None

            attn_input = AttentionInput(cache, slot_mapping, max_seqlen, attn_aux_info)

            hidden_states, new_kv = decoder_layer(
                hidden_states,
                positions,
                attn_input,
            )
            new_kvs += new_kv

        return self.norm(hidden_states), new_kvs


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        cpu_device: VDevice,
        vocab_size_var: tvm.tir.SizeVar,
        kv_type: KVCacheType,
        sep_embed: bool = False,
    ):
        self.num_shards = config.num_shards
        self.cpu_device = cpu_device
        self.model = LlamaModel(config, vocab_size_var, kv_type, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads

        # Set the cached sin/cos to the maximum of 2048 and max seq len.
        # This will be eliminated further with online rotary embedding calculation.
        cache_len = te.var("cached_rotary_embedding_len", "int64")
        self.cos_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="cos_cached")
        self.sin_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="sin_cached")
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,  # (num_query_token,) or (num_seq, seqlen_q)
        positions: relax.Expr,  # (num_query_token,), for batched RoPE
        seq_lens: relax.Expr,  # (num_seq,)
        kv_caches: Optional[relax.Expr],  # For prefill and decode, not needed for evaluate
        slot_mapping: Optional[relax.Expr],  # (num_query_token,), Not needed for evaluate
        block_tables: Optional[relax.Expr],  # (num_seq, max_num_blocks_per_seq), for decode
        indices_within_window: Optional[
            relax.Expr
        ],  # (num_cached_total,), for prefill with sliding-window attention
        query_lens: Optional[relax.Expr],
        past_slot_mapping: Optional[relax.Expr],
        permute_indices_after_concat: Optional[relax.Expr],
    ):
        """
        In vLLM, the paged KV cache is simply a pair of tensors, one for keys and the other
        for values. The tensor has shape (num_blocks, num_kv_heads, head_size, block_size).
        (In practice, the key cache has a slightly different shape for an efficiency reason,
        but that's not important.)

        The mapping between sequences / tokens to blocks is specified by two inputs.
        - block_tables: A list of block IDs allocated for the sequence.
        - slot_mapping: A linear index into the 2D grid (num_blocks, block_size), for each token.

        Support for sliding-window attention is realized by making a block table a circular buffer.
        So the length of a block table for each sequence is at most ceil(window_size / block_size).

        With sliding window, not all past K / V values need to be cached during prefill.
        The last input, indices_within_window, tells which tokens among (num_query_token,) need to have
        their K / V values cached.
        """
        if self.num_shards > 1:
            input_ids = nn.emit(ccl.broadcast_from_worker0(input_ids))
            positions = nn.emit(ccl.broadcast_from_worker0(positions))
            seq_lens = nn.emit(ccl.broadcast_from_worker0(seq_lens))

            if slot_mapping:
                slot_mapping = nn.emit(ccl.broadcast_from_worker0(slot_mapping))

            if block_tables:
                block_tables = nn.emit(ccl.broadcast_from_worker0(block_tables))

            if indices_within_window:
                indices_within_window = nn.emit(ccl.broadcast_from_worker0(indices_within_window))

            if query_lens:
                query_lens = nn.emit(ccl.broadcast_from_worker0(query_lens))
                past_slot_mapping = nn.emit(ccl.broadcast_from_worker0(past_slot_mapping))
                permute_indices_after_concat = nn.emit(
                    ccl.broadcast_from_worker0(permute_indices_after_concat)
                )

        is_prompt = block_tables is None and query_lens is None

        if query_lens is not None:
            seq_start = create_seq_start(seq_lens)
            max_query_len = R.to_vdevice(R.max(query_lens), self.cpu_device)
            query_start = create_seq_start(query_lens)
            attn_aux_info = EvaluateMultiQueryInput(
                seq_start,
                query_start,
                max_query_len,
                past_slot_mapping,
                permute_indices_after_concat,
            )
        elif is_prompt:
            seq_start = create_seq_start(seq_lens)
            attn_aux_info = PrefillAttentionInput(seq_start, indices_within_window)
        else:
            seq_start = None
            attn_aux_info = DecodeAttentionInput(seq_lens, block_tables)

        # max_seqlen needs to be on CPU, so that vLLM and Flash Attention can directly get the
        # integer length by max_seqlen->data[0]. Otherwise, we need to repeatedly do cudaMemcpy
        # of a single int32.
        max_seqlen = R.to_vdevice(R.max(seq_lens), self.cpu_device)

        hidden_states, new_kvs = self.model(
            input_ids,
            positions,
            kv_caches,
            slot_mapping,
            max_seqlen,
            attn_aux_info,
        )

        if is_prompt:
            # Extract logits for the last token in each sequence

            def get_logits_last_tokens(x, seq_len_tensor, seq_start):
                return te.compute(
                    shape=(seq_len_tensor.shape[0], x.shape[-1]),
                    fcompute=lambda i, j: x[seq_start[i] + seq_len_tensor[i] - 1, j],
                    name="get_logits_last_tokens",
                )

            logits = self.lm_head(
                nn.emit_te(
                    get_logits_last_tokens,
                    hidden_states,
                    seq_lens,
                    seq_start,
                    primfunc_name_hint="get_logits_last_tokens",
                )
            )
        else:
            logits = self.lm_head(hidden_states)

        if logits.struct_info.dtype != "float32":
            logits = nn.emit(relax.op.astype(logits, "float32"))

        return logits, new_kvs


def get_inputs(
    num_query_token,
    num_seq,
    input_shape,
    config,
    kv_type=None,
    max_num_blocks_per_seq=None,
    sep_embed=False,
):
    hidden_size = config.hidden_size

    inputs = (
        nn.Placeholder(input_shape + (hidden_size,), dtype=config.dtype, name="inputs_embeds")
        if sep_embed
        else nn.Placeholder(input_shape, dtype="int32", name="input_ids")
    )

    seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
    positions = nn.Placeholder((num_query_token,), dtype="int32", name="positions")

    if kv_type:
        num_blocks = tvm.tir.Var("num_blocks", "int64")

        if kv_type == KVCacheType.VLLM:
            block_size = VllmAttention.block_size

            vec_size = 8  # 128 bit, fp16 x 8
            num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
            head_size = hidden_size // config.num_attention_heads

            k_cache_shape = (
                num_blocks,
                num_key_value_heads,
                head_size // vec_size,
                block_size,
                vec_size,
            )
            v_cache_shape = (num_blocks, num_key_value_heads, head_size, block_size)
        else:
            block_size = FlashDecodingAttention.block_size

            num_key_value_heads = config.get_num_key_value_heads() // config.num_shards
            head_size = hidden_size // config.num_attention_heads

            k_cache_shape = (num_blocks, block_size, num_key_value_heads, head_size)
            v_cache_shape = k_cache_shape

        get_cache_sinfo = lambda i: relax.TensorStructInfo(
            k_cache_shape if i % 2 == 0 else v_cache_shape, dtype="float16"
        )

        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [get_cache_sinfo(i) for i in range(config.num_hidden_layers * 2)]
            ),
        )
        slot_mapping = nn.Placeholder((num_query_token,), dtype="int32", name="slot_mapping")
    else:
        past_key_values = None
        slot_mapping = None
        block_tables = None

    if max_num_blocks_per_seq is None:
        block_tables = None
    else:
        block_tables = nn.Placeholder(
            (num_seq, max_num_blocks_per_seq), dtype="int32", name="block_tables"
        )

    return inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables


def create_evaluate_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    """Evaluate logits for the last token in each sequence. Same as prefill but without KV cache."""
    func_name = "evaluate"

    num_query_token = tvm.tir.SizeVar("num_tokens_excluding_cache", "int64")
    num_seq = tvm.tir.SizeVar("batch_size", "int64")

    with bb.function(func_name):
        model = LlamaForCausalLM(config, cpu_dev, tvm.tir.Var("vocab_size", "int64"), sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs, positions, seq_lens, _, _, _ = get_inputs(
            num_query_token, num_seq, (num_query_token,), config, sep_embed=sep_embed
        )

        with bb.dataflow():
            logits, _ = model(
                inputs,
                positions,
                seq_lens,
                kv_caches=None,
                slot_mapping=None,
                block_tables=None,
                indices_within_window=None,
                query_lens=None,
                past_slot_mapping=None,
                permute_indices_after_concat=None,
            )
            params = [
                inputs,
                positions,
                seq_lens,
            ] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    kv_type: KVCacheType,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    """Batched prefill with vLLM paged KV cache.

    The batched attention op is intended to be offloaded to CUTLASS or Flash Attention
    via BYOC.
    """
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    num_query_token = tvm.tir.SizeVar("num_tokens_excluding_cache", "int64")
    num_seq = tvm.tir.SizeVar("batch_size", "int64")

    num_inputs = 5

    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"), kv_type, sep_embed
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_query_token, num_seq, (num_query_token,), config, kv_type, sep_embed=sep_embed
        )

        with bb.dataflow():
            params = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
            ]

            inputs = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                None,  # block_tables
            ]

            if config.sliding_window:
                num_inputs += 1
                # The value of num_cached_total is between
                # num_query_token (if seq_len < sliding_window for all seq) and
                # num_seq * config.sliding_window (if seq_len > sliding_window for all seq)
                num_cached_total = tvm.tir.Var("num_cached_total", "int64")
                indices_within_window = nn.Placeholder(
                    (num_cached_total,), dtype="int32", name="indices_within_window"
                )
                inputs.append(indices_within_window)
                params.append(indices_within_window)
            else:
                inputs.append(None)

            inputs += [None, None, None]

            logits, new_kvs = model(*inputs)
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))

        bb.emit_func_output(gv, params + model.parameters())

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", num_inputs))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    kv_type: KVCacheType,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
) -> None:
    """Batched decoding with vLLM paged KV cache."""
    func_name = "decode"

    num_seq = tvm.tir.SizeVar("batch_size", "int64")

    func_names = ["decode"]

    if kv_type == KVCacheType.FlashDecoding:
        func_names.append("decode_multi_query")

    for func_name in func_names:
        max_num_blocks_per_seq = tvm.tir.SizeVar("max_num_blocks_per_seq", "int64")

        if func_name == "decode":
            num_query_token = num_seq
            input_shape = (num_query_token,)
        else:
            seqlen_q = tvm.tir.SizeVar("seqlen_q", "int64")
            num_query_token = num_seq * seqlen_q
            input_shape = (num_seq, seqlen_q)

        with bb.function(func_name):
            inputs, positions, seq_lens, past_key_values, slot_mapping, block_tables = get_inputs(
                num_query_token, num_seq, input_shape, config, kv_type, max_num_blocks_per_seq
            )

            with bb.dataflow():
                model = LlamaForCausalLM(
                    config, cpu_dev, tvm.tir.SizeVar("vocab_size", "int64"), kv_type
                )
                param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

                logits, new_kvs = model(
                    inputs,
                    positions,
                    seq_lens,
                    past_key_values,
                    slot_mapping,
                    block_tables,
                    None,
                    None,
                    None,
                    None,
                )
                params = [
                    inputs,
                    positions,
                    seq_lens,
                    past_key_values,
                    slot_mapping,
                    block_tables,
                ] + model.parameters()
                gv = bb.emit_output((logits, relax.Tuple(new_kvs)))
            bb.emit_func_output(gv, params)

        mod = bb.get()
        gv = mod.get_global_var(func_name)
        bb.update_func(gv, mod[gv].with_attr("num_input", 6))


def create_evaluate_multi_query_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    kv_type: KVCacheType,
    cpu_dev: VDevice,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "evaluate_multi_query"

    num_query_token = tvm.tir.SizeVar("num_tokens_excluding_cache", "int64")
    num_past_token = tvm.tir.SizeVar("num_tokens_in_cache", "int64")
    num_seq = tvm.tir.SizeVar("batch_size", "int64")
    seq_lens_sum = tvm.tir.SizeVar("seq_lens_sum", "int64")

    num_inputs = 8

    with bb.function(func_name):
        model = LlamaForCausalLM(
            config, cpu_dev, tvm.tir.Var("vocab_size", "int64"), kv_type, False
        )
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        input_ids, positions, seq_lens, past_key_values, slot_mapping, _ = get_inputs(
            num_query_token, num_seq, (num_query_token,), config, kv_type, sep_embed=False
        )

        query_lens = nn.Placeholder((num_seq,), dtype="int32", name="query_lens")

        # Replace them with block_tables when a proper attention kernel becomes available.
        past_slot_mapping = nn.Placeholder(
            (num_past_token,), dtype="int32", name="past_slot_mapping"
        )
        permute_indices_after_concat = nn.Placeholder(
            (seq_lens_sum,), dtype="int32", name="permute_indices_after_concat"
        )

        with bb.dataflow():
            params = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
            ]

            inputs = [
                input_ids,
                positions,
                seq_lens,
                past_key_values,
                slot_mapping,
                None,  # block_tables
                None,  # indices_within_window
            ]

            inputs += [query_lens, past_slot_mapping, permute_indices_after_concat]
            params += [query_lens, past_slot_mapping, permute_indices_after_concat]

            logits, new_kvs = model(*inputs)
            gv = bb.emit_output((logits, relax.Tuple(new_kvs)))

        bb.emit_func_output(gv, params + model.parameters())

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", num_inputs))


def get_model(args, hf_config):
    dtype = args.quantization.model_dtype
    sep_embed = False

    position_embedding_base = 10000

    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]

    # Llama-2 variants use `max_position_embeddings` to encode maximum sequence length in their hf model cards,
    # while Llama-1 variants use `max_sequence_length`.
    # Thus, use `max_sequence_length` if defined. Otherwise, use `max_position_embeddings`.
    # If none of them is defined, throw an error.
    if "mixtral" in args.model.lower():
        # FIXME
        config = MixtralConfig(
            **hf_config,
            dtype=dtype,
            max_sequence_length=hf_config["max_position_embeddings"],
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
            quantization_scheme=args.quantization,
        )
    elif "max_sequence_length" in hf_config:
        config = LlamaConfig(
            **hf_config,
            dtype=dtype,
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
        )
    elif "max_position_embeddings" in hf_config:
        config = LlamaConfig(
            **hf_config,
            dtype=dtype,
            max_sequence_length=hf_config["max_position_embeddings"],
            position_embedding_base=position_embedding_base,
            combine_matmul=True,
            num_shards=args.num_shards,
            build_model_only=args.build_model_only,
        )
    else:
        raise Exception(
            "The model config should contain information about maximum sequence length."
        )

    # If there is a user-provided maximum sequence length, override hf config.
    if args.max_seq_len != -1:
        config.max_sequence_length = args.max_seq_len

    keep_params_after_load = (
        isinstance(config, MixtralConfig) and args.quantization.name == "q4f16_ft"
    )
    param_manager = ParamManager(keep_params_after_load)
    bb = relax.BlockBuilder()

    # The CPU device to copy the result of relax.op.max(seq_lens) to CPU.
    cpu_dev = VDevice("llvm", 0, "global")

    if args.paged_kv_cache_type == "flash-decoding":
        kv_type = KVCacheType.FlashDecoding
    else:
        kv_type = KVCacheType.VLLM

    create_evaluate_func(bb, param_manager, config, cpu_dev, args.quantization, sep_embed)
    create_encoding_func(bb, param_manager, config, kv_type, cpu_dev, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, kv_type, cpu_dev, args.quantization)
    create_evaluate_multi_query_func(bb, param_manager, config, kv_type, cpu_dev, args.quantization)

    mod = bb.get()

    mod.update_global_info("vdevice", [cpu_dev])

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
