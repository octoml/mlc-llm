from typing import Optional, Tuple, List

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl, reshape, split, expand_dims
from tvm.relax.op.nn import attention_var_len
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import QuantizationScheme
from .modules import ModuleList
from .param_manager import ParamManager
from .llama import (
    LlamaConfig,
    Linear,
    Embedding,
    LlamaRMSNorm,
    LlamaMLP,
    get_param_quant_kind,
    emit_shard3d,
    setup_params,
)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, head_mapping, prefill):
        dtype = config.dtype
        self.num_shards = config.num_shards
        self.hidden_size = config.hidden_size
        self.num_key_value_heads = (
            config.num_key_value_heads is None
            and config.num_attention_heads
            or config.num_key_value_heads
        ) // config.num_shards
        self.num_query_heads = config.num_attention_heads // self.num_shards
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.position_embedding_base = config.position_embedding_base
        self.head_mapping = head_mapping
        self.prefill = prefill

        self.combine_matmul = config.combine_matmul
        if self.combine_matmul:
            self.query_key_value_proj = Linear(
                self.hidden_size,
                (self.num_query_heads + 2 * self.num_key_value_heads) * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.query_key_value_proj.weight.shard_dim = 0
        else:
            self.q_proj = Linear(
                self.hidden_size,
                self.num_query_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.k_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.v_proj = Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                dtype=dtype,
                bias=False,
            )
            self.q_proj.weight.shard_dim = 0
            self.k_proj.weight.shard_dim = 0
            self.v_proj.weight.shard_dim = 0

        self.o_proj = Linear(
            self.head_dim * self.num_query_heads, self.hidden_size, dtype=dtype, bias=False
        )
        self.o_proj.weight.shard_dim = 1

    def forward(
        self,
        hidden_states: relax.Expr,
        seq_lens: relax.Expr,
        seqstart_q: relax.Expr, # only for prefill
        max_seqlen_q: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr, # only for decode
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        num_tokens, hidden_size = hidden_states.struct_info.shape

        if self.combine_matmul:
            qkv_states = nn.emit(
                split(
                    self.query_key_value_proj(hidden_states),
                    indices_or_sections=[
                        self.num_query_heads * self.head_dim,
                        (self.num_query_heads + self.num_key_value_heads) * self.head_dim,
                    ],
                    axis=-1,
                )
            )
            query_states = relax.TupleGetItem(qkv_states, 0)
            key_states = relax.TupleGetItem(qkv_states, 1)
            value_states = relax.TupleGetItem(qkv_states, 2)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        queries = nn.emit(
            reshape(
                query_states,
                (num_tokens, self.num_query_heads, self.head_dim),
            ),
        )
        keys = nn.emit(
            reshape(
                key_states,
                (num_tokens, self.num_key_value_heads, self.head_dim),
            ),
        )
        values = nn.emit(
            reshape(
                value_states,
                (num_tokens, self.num_key_value_heads, self.head_dim),
            ),
        )

        # Paged KV cache update
        k_cache, v_cache = kv_cache

        nn.emit(
            tvm.script.ir_builder.relax.call_packed(
                "tvm.contrib.vllm.reshape_and_cache",
                keys,
                values,
                k_cache,
                v_cache,
                slot_mapping,
                sinfo_args=[R.Tuple()],
            )
        )

        if self.prefill:
            attn_output = nn.emit(
                attention_var_len(
                    nn.emit(expand_dims(queries, axis=0)),
                    nn.emit(expand_dims(keys, axis=0)),
                    nn.emit(expand_dims(values, axis=0)),
                    seqstart_q=seqstart_q,
                    max_seqlen_q=max_seqlen_q,
                    causal_mask="BottomRight",
                )
            )
        else:
            attn_output = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.vllm.single_query_cached_kv_attention",
                    [
                        queries,
                        k_cache,
                        v_cache,
                        self.head_mapping,
                        block_tables,
                        seq_lens,
                        16,  # block_size
                        max_seqlen_q,
                    ],
                    out_sinfo=queries.struct_info,
                )
            )

        attn_output = nn.emit(reshape(attn_output, (num_tokens, hidden_size)))
        attn_output = self.o_proj(attn_output)

        if self.num_shards > 1:
            attn_output = nn.emit(ccl.allreduce(attn_output, "sum"))

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, head_mapping, prefill: bool):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, head_mapping, prefill)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: relax.Expr,
        seq_lens: relax.Expr,
        seqstart_q: relax.Expr,
        max_seqlen_q: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            seq_lens=seq_lens,
            seqstart_q=seqstart_q,
            max_seqlen_q=max_seqlen_q,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        hidden_states = nn.emit(residual + hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.emit(residual + hidden_states)

        return hidden_states


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.Var,
        prefill: bool,
        sep_embed: bool = False,
    ):
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None
        self.prefill = prefill

        num_key_value_heads = (
            config.num_key_value_heads is None
            and config.num_attention_heads
            or config.num_key_value_heads
        )
        num_queries_per_kv = config.num_attention_heads // num_key_value_heads
        head_mapping = relax.const(
            tvm.nd.array(
                np.repeat(np.arange(num_key_value_heads, dtype="int32"), num_queries_per_kv)
            )
        )

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [
                LlamaDecoderLayer(config, head_mapping, prefill)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, dtype=config.dtype, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs: relax.Expr,
        seq_lens: relax.Expr,
        kv_caches: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr,
    ):
        if self.num_shards > 1:
            inputs = nn.emit(ccl.broadcast_from_worker0(inputs))

        if self.embed_tokens:
            inputs_embeds = self.embed_tokens(inputs)
        else:
            inputs_embeds = inputs

        hidden_states = inputs_embeds

        if self.prefill:
            seqstart_q = nn.emit(
                relax.op.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
                )
            )
        else:
            seqstart_q = None

        max_seqlen_q = R.max(seq_lens)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                seq_lens,
                seqstart_q,
                max_seqlen_q,
                (kv_caches[2 * idx], kv_caches[2 * idx + 1]),
                slot_mapping,
                block_tables,
            )

        return self.norm(hidden_states)


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        vocab_size_var: tvm.tir.Var,
        prefill: bool,
        sep_embed: bool = False,
    ):
        self.model = LlamaModel(config, vocab_size_var, prefill, sep_embed)
        self.lm_head = Linear(config.hidden_size, vocab_size_var, dtype=config.dtype, bias=False)

        ############ Rotary embedding constants ############
        assert config.hidden_size % config.num_attention_heads == 0
        head_dim = config.hidden_size // config.num_attention_heads

        # Set the cached sin/cos to the maximum of 2048 and max seq len.
        # This will be eliminated further with online rotary embedding calculation.
        cache_len = te.var("cache_len", "int64")
        self.cos_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="cos_cached")
        self.sin_cached = nn.Parameter((cache_len, head_dim), dtype=config.dtype, name="sin_cached")
        ############ End ############

    def forward(
        self,
        input_ids: relax.Expr,
        seq_lens: relax.Expr,
        kv_caches: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr, # only for decode
    ):
        hidden_states = self.model(input_ids, seq_lens, kv_caches, slot_mapping, block_tables)

        return hidden_states


def create_encoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
    sep_embed: bool = False,
) -> None:
    func_name = "prefill_with_embed" if sep_embed else "prefill"

    num_token = tvm.tir.Var("num_token", "int64")
    num_seq = tvm.tir.Var("num_seq", "int64")
    hidden_size = config.hidden_size

    with bb.function(func_name):
        model = LlamaForCausalLM(config, tvm.tir.Var("v", "int64"), True, sep_embed)
        param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

        inputs = (
            nn.Placeholder((num_token, hidden_size), dtype=config.dtype, name="inputs_embeds")
            if sep_embed
            else nn.Placeholder((num_token,), dtype="int32", name="input_ids")
        )
        seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        slot_mapping = nn.Placeholder((num_seq,), dtype="int32", name="slot_mapping")

        with bb.dataflow():
            logits = model(inputs, seq_lens, past_key_values, slot_mapping, None)
            params = [
                inputs,
                seq_lens,
                past_key_values,
                slot_mapping,
            ] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 4))


def create_decoding_func(
    bb: relax.BlockBuilder,
    param_manager: ParamManager,
    config: LlamaConfig,
    quant_scheme: QuantizationScheme,
) -> None:
    func_name = "decode"

    num_seq = tvm.tir.Var("num_seq", "int64")
    max_num_blocks_per_seq = tvm.tir.Var("max_num_blocks_per_seq", "int64")

    with bb.function(func_name):
        inputs = nn.Placeholder((num_seq,), dtype="int32", name="input_ids")
        seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        slot_mapping = nn.Placeholder((num_seq,), dtype="int32", name="slot_mapping")
        block_tables = nn.Placeholder(
            (num_seq, max_num_blocks_per_seq), dtype="int32", name="block_tables"
        )

        with bb.dataflow():
            model = LlamaForCausalLM(config, tvm.tir.Var("v", "int64"), False)
            param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

            logits = model(inputs, seq_lens, past_key_values, slot_mapping, block_tables)
            params = [
                inputs,
                seq_lens,
                past_key_values,
                slot_mapping,
                block_tables,
            ] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var(func_name)
    bb.update_func(gv, mod[gv].with_attr("num_input", 5))


def test():
    bb = relax.BlockBuilder()
    model_path = "/Users/masa/projects/dev/mlc-llm/mlc_llm/relax_model"
    import os
    import json

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as i_f:
        config = LlamaConfig(**json.load(i_f))

    func_name = "prefill"
    num_token = tvm.tir.Var("num_token", "int64")
    num_seq = tvm.tir.Var("num_seq", "int64")
    hidden_size = config.hidden_size

    with bb.function(func_name):
        model = LlamaForCausalLM(config, tvm.tir.Var("v", "int64"), True, False)

        inputs = nn.Placeholder((num_token,), dtype="int32", name="input_ids")
        seq_lens = nn.Placeholder((num_seq,), dtype="int32", name="seq_lens")
        past_key_values = relax.Var(
            "kv_cache",
            relax.TupleStructInfo(
                [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
            ),
        )
        slot_mapping = nn.Placeholder((num_seq,), dtype="int32", name="slot_mapping")

        with bb.dataflow():
            logits = model(inputs, seq_lens, past_key_values, slot_mapping, None)
            params = [
                inputs,
                seq_lens,
                past_key_values,
                slot_mapping,
            ] + model.parameters()
            gv = bb.emit_output(logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("prefill")
    bb.update_func(gv, mod[gv].with_attr("num_input", 4))

    print(mod)


def get_model(args, hf_config):
    model_name = args.model
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len
    sep_embed = False

    position_embedding_base = 10000
    max_position_embeddings = 2048
    if "rope_theta" in hf_config:
        position_embedding_base = hf_config["rope_theta"]
    if "max_position_embeddings" in hf_config:
        max_position_embeddings = hf_config["max_position_embeddings"]

    config = LlamaConfig(
        **hf_config,
        dtype=dtype,
        position_embedding_base=position_embedding_base,
        combine_matmul=True,
        num_shards=args.num_shards,
        build_model_only=args.build_model_only,
        convert_weight_only=args.convert_weight_only,
    )
    if max_seq_len != -1:
        config.max_sequence_length = max_seq_len

    param_manager = ParamManager()
    bb = relax.BlockBuilder()
    emit_shard3d(bb)

    create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
    create_decoding_func(bb, param_manager, config, args.quantization)

    mod = bb.get()

    if args.build_model_only:
        return mod, param_manager, None, config

    return setup_params(mod, param_manager, dtype, config, args)
