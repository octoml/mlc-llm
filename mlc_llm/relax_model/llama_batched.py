import math
from typing import Any, List, Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.op import ccl
from tvm.relax.testing import nn
from tvm.script import relax as R

from ..quantization import ParamQuantKind, QuantizationScheme
from .commons import create_metadata_func
from .modules import ModuleList
from .param_manager import ParamManager
from .llama import LlamaConfig, Linear, Embedding, LlamaRMSNorm, LlamaMLP


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
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
        seqstart_q: relax.Expr,
        max_seqlen_q: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr,
    ) -> Tuple[relax.Expr, Optional[relax.Expr], Optional[Tuple[relax.Expr]]]:
        from tvm.relax.op import (
            reshape,
            split,
            expand_dims
        )
        from tvm.relax.op.nn import attention_var_len

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

        nn.emit(relax.op.call_packed(
                "tvm.contrib.vllm.reshape_and_cache",
                keys,
                values,
                k_cache,
                v_cache,
                slot_mapping,
                sinfo_args=[R.Tuple()]))

        if True:
            # Prefill attention
            attn_output = nn.emit(attention_var_len(nn.emit(queries, axis=0),
                                                    nn.emit(keys, axis=0),
                                                    nn.emit(values, axis=0),
                                                    seqstart_q=seqstart_q,
                                                    max_seqlen_q=max_seqlen_q,
                                                    causal_mask="BottomRight"))
            attn_output = nn.emit(reshape(attn_output, (num_tokens, hidden_size)))
        else:
            # Decode attention
            pass

        attn_output = self.o_proj(attn_output)

        if self.num_shards > 1:
            attn_output = nn.emit(ccl.allreduce(attn_output, "sum"))

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
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
        seqstart_q: relax.Expr,
        max_seqlen_q: relax.Expr,
        kv_cache: relax.Expr,
        slot_mapping: relax.Expr,
        block_tables: relax.Expr,
    ) -> Tuple[relax.Expr, Optional[Tuple[relax.Expr, relax.Expr]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            seqstart_q=seqstart_q,
            max_seqlen_q=max_seqlen_q,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables
        )
        hidden_states = nn.emit(residual + hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = nn.emit(residual + hidden_states)

        return hidden_states, present_key_value


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.Var, sep_embed: bool = False):
        self.num_shards = config.num_shards
        self.padding_idx = config.pad_token_id
        self.embed_tokens = None

        if not sep_embed:
            self.embed_tokens = Embedding(vocab_size_var, config.hidden_size, dtype=config.dtype)

        self.layers = ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
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

        seqstart_q = nn.emit(relax.op.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan", seq_lens, out_sinfo=seq_lens.struct_info
                ))
        max_seqlen_q = R.max(seq_lens)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                seqstart_q,
                max_seqlen_q,
                kv_caches[idx],
                slot_mapping,
                block_tables,
            )

        return self.norm(hidden_states)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig, vocab_size_var: tvm.tir.Var, sep_embed: bool = False):
        self.model = LlamaModel(config, vocab_size_var, sep_embed)
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
        block_tables: relax.Expr,
    ):
        hidden_states = self.model(
            input_ids,
            seq_lens,
            kv_caches,
            slot_mapping,
            block_tables
        )

        return hidden_states


# def create_encoding_func(
#     bb: relax.BlockBuilder,
#     param_manager: ParamManager,
#     config: LlamaConfig,
#     quant_scheme: QuantizationScheme,
#     sep_embed: bool = False,
# ) -> None:
#     func_name = "prefill_with_embed" if sep_embed else "prefill"

#     bsz = 1
#     seq_len = tvm.tir.Var("n", "int64")
#     all_seq_len = tvm.tir.Var("m", "int64")
#     hidden_size = config.hidden_size
#     with bb.function(func_name):
#         model = LlamaForCausalLM(config, tvm.tir.Var("v", "int64"), sep_embed)
#         param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

#         inputs = (
#             nn.Placeholder((bsz, seq_len, hidden_size), dtype=config.dtype, name="inputs_embeds")
#             if sep_embed
#             else nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
#         )
#         all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
#         past_key_values = relax.Var(
#             "kv_cache",
#             relax.TupleStructInfo(
#                 [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
#             ),
#         )
#         with bb.dataflow():
#             logits, key_value_cache = model(
#                 inputs, all_seq_len_shape, past_key_values=past_key_values
#             )
#             params = [
#                 inputs,
#                 all_seq_len_shape,
#                 past_key_values,
#             ] + model.parameters()
#             gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
#         bb.emit_func_output(gv, params)

#     mod = bb.get()
#     gv = mod.get_global_var(func_name)
#     bb.update_func(gv, mod[gv].with_attr("num_input", 3))


# def create_decoding_func(
#     bb: relax.BlockBuilder,
#     param_manager: ParamManager,
#     config: LlamaConfig,
#     quant_scheme: QuantizationScheme,
# ) -> None:
#     func_name = "decode"

#     bsz = 1
#     all_seq_len = tvm.tir.Var("n", "int64")

#     with bb.function(func_name):
#         model = LlamaForCausalLM(config, tvm.tir.Var("v", "int64"))
#         param_manager.register_params(model, func_name, quant_scheme, get_param_quant_kind)

#         input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="input_ids")
#         all_seq_len_shape = relax.Var("all_seq_len", relax.ShapeStructInfo((all_seq_len,)))
#         past_key_values = relax.Var(
#             "kv_cache",
#             relax.TupleStructInfo(
#                 [relax.ObjectStructInfo() for _ in range(config.num_hidden_layers * 2)]
#             ),
#         )
#         with bb.dataflow():
#             logits, key_value_cache = model(
#                 input_ids, all_seq_len_shape, past_key_values=past_key_values
#             )
#             params = [
#                 input_ids,
#                 all_seq_len_shape,
#                 past_key_values,
#             ] + model.parameters()
#             gv = bb.emit_output((logits, relax.Tuple(key_value_cache)))
#         bb.emit_func_output(gv, params)

#     mod = bb.get()
#     gv = mod.get_global_var(func_name)
#     bb.update_func(gv, mod[gv].with_attr("num_input", 3))


# def create_kv_cache_func(bb: relax.BlockBuilder, config: LlamaConfig) -> None:
#     num_key_value_heads = (
#         config.num_attention_heads
#         if config.num_key_value_heads is None
#         else config.num_key_value_heads
#     ) // config.num_shards
#     init_shape = relax.ShapeExpr(
#         (
#             config.max_sequence_length,
#             num_key_value_heads,
#             config.hidden_size // config.num_attention_heads,  # head_dim
#         )
#     )
#     with bb.function("create_kv_cache", []):
#         with bb.dataflow():
#             zeros = bb.emit(relax.op.zeros(init_shape, config.dtype))
#             caches = []
#             f_kv_cache_create = relax.extern("vm.builtin.attention_kv_cache_create")
#             for _ in range(config.num_hidden_layers * 2):
#                 caches.append(
#                     bb.emit(
#                         relax.Call(
#                             f_kv_cache_create,
#                             args=[zeros, init_shape, relax.PrimValue(0)],
#                             sinfo_args=[relax.ObjectStructInfo()],
#                         )
#                     )
#                 )
#             gv = bb.emit_output(caches)
#         bb.emit_func_output(gv)


# def get_model(args, hf_config):
#     model_name = args.model
#     dtype = args.quantization.model_dtype
#     max_seq_len = args.max_seq_len
#     sep_embed = args.sep_embed

#     position_embedding_base = 10000
#     max_position_embeddings = 2048
#     if "rope_theta" in hf_config:
#         position_embedding_base = hf_config["rope_theta"]
#     if "max_position_embeddings" in hf_config:
#         max_position_embeddings = hf_config["max_position_embeddings"]

#     config = LlamaConfig(
#         **hf_config,
#         dtype=dtype,
#         position_embedding_base=position_embedding_base,
#         combine_matmul=True,
#         num_shards=args.num_shards,
#         build_model_only=args.build_model_only,
#         convert_weight_only=args.convert_weight_only,
#     )
#     if max_seq_len != -1:
#         config.max_sequence_length = max_seq_len

#     param_manager = ParamManager()
#     bb = relax.BlockBuilder()
#     emit_shard3d(bb)

#     if sep_embed:
#         create_embed_func(bb, param_manager, config, args.quantization)
#     create_encoding_func(bb, param_manager, config, args.quantization, sep_embed)
#     create_decoding_func(bb, param_manager, config, args.quantization)
#     create_kv_cache_func(bb, config)
#     create_softmax_func(bb, config)
#     create_metadata_func(
#         bb,
#         model_name=model_name,
#         max_window_size=config.max_sequence_length,
#         stop_tokens=[2],
#         add_prefix_space=False,
#     )

#     mod = bb.get()
#     for gv in mod.functions:
#         func = mod[gv]
#         if isinstance(func, relax.Function):
#             mod[gv] = func.with_attr(
#                 "tir_var_upper_bound",
#                 {
#                     "n": config.max_sequence_length,
#                     "m": config.max_sequence_length,
#                 },
#             )

#     if args.build_model_only:
#         return mod, param_manager, None, config

#     def f_convert_pname_fwd(pname: str) -> List[str]:
#         if not config.combine_matmul:
#             return [pname]

#         qkv_str = "query_key_value_proj"
#         gate_up_str = "gate_up_proj"
#         if qkv_str in pname:
#             return [
#                 pname.replace(qkv_str, "q_proj"),
#                 pname.replace(qkv_str, "k_proj"),
#                 pname.replace(qkv_str, "v_proj"),
#             ]
#         elif gate_up_str in pname:
#             return [
#                 pname.replace(gate_up_str, "gate_proj"),
#                 pname.replace(gate_up_str, "up_proj"),
#             ]
#         else:
#             return [pname]

#     def f_convert_param_bkwd(torch_pname: str, torch_param):
#         if not config.combine_matmul:
#             return [(torch_pname, torch_param.astype(dtype))]

#         combined_layers = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
#         if any([name in torch_pname for name in combined_layers]):
#             return None
#         return [(torch_pname, torch_param.astype(dtype))]

#     def f_compute_relax_param(relax_pname: str, torch_params: List[Any]):
#         # Expected to enter this function only for the combined linear matmul weights.
#         # Other weights are supposed to be loaded in `f_convert_param_bkwd` since
#         # each other relax param has a unique corresponding torch param.
#         if not config.combine_matmul:
#             # When matmul combination is not turned on, each relax param has a unique
#             # corresponding torch param, and this function is not expected to be entered.
#             raise NotImplementedError(
#                 "Matmul combination is not turned on, and the function "
#                 "is not expected to be entered"
#             )
#         num_shards = args.num_shards
#         hidden_size = config.hidden_size
#         head_dim = config.hidden_size // config.num_attention_heads

#         if "query_key_value_proj" in relax_pname:
#             q_heads = config.num_attention_heads
#             kv_heads = config.num_key_value_heads
#             if kv_heads is None:
#                 kv_heads = q_heads
#             q, k, v = torch_params
#             assert q.shape == (q_heads * head_dim, hidden_size)
#             assert k.shape == (kv_heads * head_dim, hidden_size)
#             assert v.shape == (kv_heads * head_dim, hidden_size)
#             q = q.reshape((num_shards, q_heads // num_shards, head_dim, hidden_size))
#             k = k.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size))
#             v = v.reshape((num_shards, kv_heads // num_shards, head_dim, hidden_size))
#             qkv = np.concatenate([q, k, v], axis=1)
#             qkv = qkv.reshape((-1, hidden_size)).astype(dtype)
#             return qkv
#         if "gate_up_proj" in relax_pname:
#             intermediate_size = config.intermediate_size
#             gate, up = torch_params
#             gate = gate.reshape((num_shards, intermediate_size // num_shards, hidden_size))
#             up = up.reshape((num_shards, intermediate_size // num_shards, hidden_size))
#             gate_up = np.concatenate([gate, up], axis=1)
#             gate_up = gate_up.reshape((-1, hidden_size)).astype(dtype)
#             return gate_up
#         raise ValueError("Unexpected param loading")

#     param_manager.set_param_loading_func(
#         args.model_path,
#         args.use_safetensors,
#         f_convert_pname_fwd,
#         f_convert_param_bkwd,
#         f_compute_relax_param,
#     )

#     device = tvm.cpu()
#     param_list = [None] * param_manager.nparam_to_load

#     head_dim = config.hidden_size / config.num_attention_heads
#     inv_freq = 1.0 / (
#         config.position_embedding_base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim)
#     )

#     # The following cos/sin values can be removed but **are kept for compatibility issues**.
#     t = np.arange(2048, dtype=inv_freq.dtype)
#     freqs = np.einsum("i,j->ij", t, inv_freq)
#     emb = np.concatenate((freqs, freqs), axis=-1)
#     param_list[-2] = tvm.nd.array(np.cos(emb).astype(config.dtype), device)
#     param_list[-1] = tvm.nd.array(np.sin(emb).astype(config.dtype), device)

#     return mod, param_manager, param_list, config
