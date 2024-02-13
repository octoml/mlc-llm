import tvm
from tvm import relax, te, tir
from tvm.relax.testing import nn
from tvm.script import relax as R, tir as T
from tvm import relax
from tvm.relax.frontend.nn import Tensor
from .llama import MixtralConfig, Linear
import functools


class MoELinear(nn.Module):
    def __init__(self, config: MixtralConfig, num_experts, in_features, out_features, bias=False):
        assert not bias, "bias not supported"
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_scheme = config.quantization_scheme

        if config.quantization_scheme.name == "q0f16":
            # weight is row major
            self.weight = nn.Parameter(
                (num_experts, out_features, in_features),
                dtype="float16",
            )
        elif config.quantization_scheme.name == "q4f16_ft":
            assert out_features % 8 == 0
            self.weight = nn.Parameter(
                (num_experts, in_features, out_features // 2),
                dtype="int8",
            )
            self.scales = nn.Parameter(
                (num_experts, out_features),
                dtype="float16",
            )
        else:
            assert False, "unsupported quantization scheme"

    def forward(self, x, rows_before):
        assert len(x.struct_info.shape) == 2
        total_rows = x.struct_info.shape[0]
        if self.quantization_scheme.name == "q0f16":
            return nn.emit(
                relax.call_dps_packed(
                    "cutlass.moe_gemm_f16f16",
                    [
                        x,
                        self.weight,
                        rows_before,
                        total_rows,
                        self.out_features,  # gemm_n
                        self.in_features,  # gemm_k
                        self.num_experts,
                    ],
                    out_sinfo=relax.TensorStructInfo(
                        (total_rows, self.out_features),
                        x.struct_info.dtype,
                    ),
                )
            )
        else:
            return nn.emit(
                relax.call_dps_packed(
                    "cutlass.moe_gemm_s4f16",
                    [
                        x,
                        self.weight,
                        self.scales,
                        rows_before,
                        total_rows,
                        self.out_features,  # gemm_n
                        self.in_features,  # gemm_k
                        self.num_experts,
                    ],
                    out_sinfo=relax.TensorStructInfo(
                        (total_rows, self.out_features),
                        x.struct_info.dtype,
                    ),
                )
            )


class MoEMLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.combine_matmul = config.combine_matmul

        self.num_shards = config.num_shards
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // self.num_shards

        self.down_proj = MoELinear(
            config, self.num_experts, intermediate_size, hidden_size, bias=False
        )
        if config.combine_matmul:
            self.gate_up_combined_proj = MoELinear(
                config,
                self.num_experts,
                hidden_size,
                2 * intermediate_size,
                bias=False,
            )
            # FIXME: rename to 'gate_up_proj' that's consistent with llama. using this name for now to avoid conflicting pname str replacing rules
            # TODO: check sharding is correct, note that the weight is row major
            self.gate_up_combined_proj.weight.shard_dim = 2
            self.gate_up_combined_proj.weight.shard_strategy = "moe_shard_gate_up"
            self.down_proj.weight.shard_dim = 1
            self.down_proj.weight.shard_strategy = "moe_shard_mlp_k"
        else:
            self.gate_proj = MoELinear(
                config, self.num_experts, config.hidden_size, config.intermediate_size, bias=False
            )
            self.up_proj = MoELinear(
                config, self.num_experts, config.hidden_size, config.intermediate_size, bias=False
            )

    def forward(self, hidden_states: relax.Expr, rows_before: relax.Expr):
        if self.combine_matmul:
            gate_up_results = nn.emit(
                relax.op.split(
                    self.gate_up_combined_proj(hidden_states, rows_before),
                    indices_or_sections=2,
                    axis=-1,
                )
            )
            gate_result = relax.TupleGetItem(gate_up_results, 0)
            up_result = relax.TupleGetItem(gate_up_results, 1)
        else:
            gate_result = self.gate_proj(hidden_states, rows_before)
            up_result = self.up_proj(hidden_states, rows_before)
        result = self.down_proj(nn.emit(relax.op.nn.silu(gate_result) * up_result), rows_before)
        return result


class MoE(nn.Module):
    def __init__(self, config: MixtralConfig):
        self.experts = MoEMLP(config)
        self.num_shards = config.num_shards
        self.gate = Linear(
            in_features=config.hidden_size,
            out_features=config.num_local_experts,
            bias=False,
            dtype=config.dtype,
        )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size

    def scatter(self, linear_out, indices):
        @T.prim_func
        def scatter_func(
            x_handle: T.handle,
            indices_handle: T.handle,
            out_handle: T.handle,
        ) -> None:
            total_rows = T.int64()
            x = T.match_buffer(x_handle, (total_rows, self.hidden_size), self.dtype)
            indices = T.match_buffer(indices_handle, (total_rows,), "int32")
            out = T.match_buffer(out_handle, (total_rows, self.hidden_size), self.dtype)
            T.func_attr({"global_symbol": "scatter", "tir.noalias": True})
            for i in range(total_rows):
                for j in range(self.hidden_size):
                    with T.block("scatter"):
                        vi, vj = T.axis.remap("SS", [i, j])
                        out[indices[vi], vj] = x[vi, vj]

        scatter = relax.BlockBuilder.current().add_func(scatter_func, "scatter")
        return nn.emit(
            relax.call_dps_packed(
                scatter,
                [linear_out, indices],
                out_sinfo=linear_out.struct_info,
            )
        )

    def get_token_indices(self, indices):
        def te_compute(x):
            return tvm.te.compute(
                x.shape,
                lambda *idx: tvm.tir.indexdiv(
                    x(*idx), tvm.runtime.const(self.num_experts_per_tok, dtype="int32")
                ).astype("int32"),
            )

        return nn.emit_te(te_compute, indices)

    def topk_mask(self, indices):
        from functools import reduce

        def te_topk_mask_op(topk_indices):
            ntokens = topk_indices.shape[0]
            assert topk_indices.shape[1] == self.num_experts_per_tok
            return te.compute(
                (ntokens, self.num_experts),
                lambda i, j: tir.expr.Select(
                    reduce(
                        lambda a, b: tir.Or(a, b),
                        [topk_indices[i, k] == j for k in range(self.num_experts_per_tok)],
                    ),
                    true_value=tir.const(1, "int32"),
                    false_value=tir.const(0, "int32"),
                ),
            )

        return nn.emit_te(te_topk_mask_op, indices)

    def get_indices(
        self, cumsum_colwise_flattened: relax.Expr, expert_indices: relax.Expr
    ) -> relax.Tuple:
        from tvm import relax
        from tvm.script import tir as T

        TX = 1024
        experts_per_tok = T.int32(self.num_experts_per_tok)

        @T.prim_func(private=True)
        def _func(
            var_cumsum: T.handle,
            var_expert_indices: T.handle,
            var_flattened_indices: T.handle,
            var_token_indices: T.handle,
        ):
            T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
            batch_size = T.SizeVar("batch_size", "int32")
            cumsum_len = T.SizeVar("cumsum_len", "int32")  # [experts_per_tok * batch_size]
            cumsum = T.match_buffer(var_cumsum, [cumsum_len], "int32")
            expert_indices = T.match_buffer(
                var_expert_indices, [batch_size, experts_per_tok], "int32"
            )
            flattened_indices = T.match_buffer(
                var_flattened_indices, [batch_size * experts_per_tok], "int32"
            )
            token_indices = T.match_buffer(
                var_token_indices, [batch_size * experts_per_tok], "int32"
            )
            for bj_o in T.thread_binding(
                0, T.ceildiv(batch_size * experts_per_tok, TX), "blockIdx.x"
            ):
                for bj_i in T.thread_binding(0, TX, "threadIdx.x"):
                    with T.block("indices"):
                        T.reads(expert_indices[:, :], cumsum[:])
                        T.writes(flattened_indices[:], token_indices[:])
                        if bj_o * TX + bj_i < batch_size * experts_per_tok:
                            b: T.int32 = T.floordiv(bj_o * TX + bj_i, experts_per_tok)
                            j: T.int32 = T.floormod(bj_o * TX + bj_i, experts_per_tok)
                            e: T.int32 = expert_indices[b, j]
                            flattened_indices[cumsum[e * batch_size + b] - 1] = (
                                b * experts_per_tok + j
                            )
                            token_indices[cumsum[e * batch_size + b] - 1] = b

        bb = relax.BlockBuilder.current()
        gvar = bb.add_func(_func, "get_indices")
        return bb.emit(
            relax.call_tir(
                gvar,
                [cumsum_colwise_flattened, expert_indices],
                out_sinfo=[
                    relax.TensorStructInfo(
                        [expert_indices.struct_info.shape[0] * self.num_experts_per_tok],
                        "int32",
                    )
                    for _ in range(2)
                ],
            )
        )

    def cumsum(self, data: relax.Expr) -> relax.Expr:
        return nn.emit(
            relax.call_dps_packed(
                "tvm.contrib.thrust.sum_scan",
                [data],
                out_sinfo=data.struct_info,
            )
        )

    def get_indptr(self, cumsum_colwise_flattened: relax.Expr) -> relax.Expr:
        from tvm import relax
        from tvm.script import tir as T

        @T.prim_func
        def get_expert_instance_indptr(
            var_cumsum_colwise_flattened: T.handle,
            var_expert_instance_indptr: T.handle,
            batch_size: T.int32,
        ):
            cumsum_colwise_flattened = T.match_buffer(
                var_cumsum_colwise_flattened, shape=[batch_size * self.num_experts], dtype="int32"
            )
            expert_instance_indptr = T.match_buffer(
                var_expert_instance_indptr, shape=[self.num_experts], dtype="int64"
            )

            for expert_id in range(self.num_experts):
                with T.block("indptr"):
                    vexpert_id = T.axis.spatial(self.num_experts, expert_id)
                    expert_instance_indptr[vexpert_id] = T.cast(
                        cumsum_colwise_flattened[(vexpert_id + 1) * batch_size - 1], "int64"
                    )

        bb = relax.BlockBuilder.current()
        gvar = bb.add_func(get_expert_instance_indptr, "get_expert_instance_indptr")
        return bb.emit(
            relax.call_tir(
                gvar,
                [cumsum_colwise_flattened],
                out_sinfo=relax.TensorStructInfo([self.num_experts], "int64"),
                tir_vars=[cumsum_colwise_flattened.struct_info.shape[0] // self.num_experts],
            )
        )

    def gating_softmax_topk(self, x, k):
        index_dtype = "int32"

        @T.prim_func
        def top2_softmax_func(
            x_handle: T.handle,
            out_handle: T.handle,
            out_index_handle: T.handle,
        ) -> None:
            total_rows = T.int64()
            x = T.match_buffer(x_handle, (total_rows, self.num_experts), self.dtype)
            out = T.match_buffer(out_handle, (total_rows, 2), self.dtype)
            out_index = T.match_buffer(out_index_handle, (total_rows, 2), index_dtype)
            local_top_k = T.alloc_buffer((2,), dtype=self.dtype, scope="local")
            local_top_k_index = T.alloc_buffer((2,), dtype=index_dtype, scope="local")
            local_top_k_f32 = T.alloc_buffer((2,), dtype="float32", scope="local")
            local_top_k_max = T.alloc_buffer((1,), dtype="float32", scope="local")
            T.func_attr({"tir.noalias": True, "tir.is_scheduled": True})
            for io in T.thread_binding(0, T.ceildiv(total_rows, T.int64(1024)), "blockIdx.x"):
                for ii in T.thread_binding(0, T.min(total_rows, T.int64(1024)), "threadIdx.x"):
                    with T.block("top2"):
                        vi = T.axis.spatial(total_rows, io * T.int64(1024) + ii)
                        T.where(io * T.int64(1024) + ii < total_rows)
                        with T.block("init"):
                            local_top_k[0] = T.min_value(self.dtype)
                            local_top_k_index[0] = 0
                        for k in range(self.num_experts):
                            with T.block("update"):
                                vk = T.axis.remap("S", [k])
                                if x[vi, vk] > local_top_k[0]:
                                    local_top_k[1] = local_top_k[0]
                                    local_top_k_index[1] = local_top_k_index[0]
                                    local_top_k[0] = x[vi, vk]
                                    local_top_k_index[0] = vk
                                elif x[vi, vk] > local_top_k[1]:
                                    local_top_k[1] = x[vi, vk]
                                    local_top_k_index[1] = vk
                        for j in T.unroll(2):
                            with T.block("cast"):
                                vj = T.axis.remap("S", [j])
                                local_top_k_f32[vj] = T.cast(local_top_k[j], "float32")
                        with T.block("max"):
                            local_top_k_max[0] = T.max(local_top_k_f32[0], local_top_k_f32[1])
                        for j in T.unroll(2):
                            with T.block("output"):
                                vj = T.axis.remap("S", [j])
                                out[vi, vj] = T.cast(
                                    T.exp(local_top_k_f32[j] - local_top_k_max[0])
                                    / (
                                        T.exp(local_top_k_f32[0] - local_top_k_max[0])
                                        + T.exp(local_top_k_f32[1] - local_top_k_max[0])
                                    ),
                                    self.dtype,
                                )
                                out_index[vi, vj] = local_top_k_index[vj]

        if k != 2:
            raise NotImplementedError("only support num_experts_per_token=2 for now")
        bb = relax.BlockBuilder.current()
        gvar = bb.add_func(top2_softmax_func, "top2_softmax")
        return bb.emit(
            relax.call_tir(
                gvar,
                [x],
                out_sinfo=[
                    relax.TensorStructInfo([x.struct_info.shape[0], k], x.struct_info.dtype),
                    relax.TensorStructInfo([x.struct_info.shape[0], k], index_dtype),
                ],
            )
        )

    def moe_sum(self, tokens):
        assert len(tokens.struct_info.shape) == 3

        def unrolled_sum(x):
            return tvm.te.compute(
                (x.shape[0], x.shape[2]),
                lambda i, j: functools.reduce(
                    tir.Add, [x[i, k, j] for k in range(self.num_experts_per_tok)]
                ),
                name="unrolled_sum",
            )

        return nn.emit_te(unrolled_sum, tokens)

    def forward(self, hidden_states):
        hidden_states_shape = hidden_states.struct_info.shape
        hidden_size = hidden_states_shape[-1]
        # reshape to 2D
        hidden_states = nn.emit(relax.op.reshape(hidden_states, (-1, hidden_size)))

        router_logits = self.gate(hidden_states)

        expert_weights, expert_indices = self.gating_softmax_topk(
            router_logits, k=self.num_experts_per_tok
        )

        expert_mask = self.topk_mask(expert_indices)
        mask_T_flattened = nn.emit(relax.op.flatten(relax.op.permute_dims(expert_mask)))

        cumsum_colwise_flattened = self.cumsum(mask_T_flattened)
        flattened_indices, token_indices = self.get_indices(
            cumsum_colwise_flattened, expert_indices
        )
        indptr = self.get_indptr(cumsum_colwise_flattened)

        gathered_x = nn.emit(relax.op.take(hidden_states, token_indices, axis=0))
        linear_out = self.experts(gathered_x, indptr)
        unpermuted = self.scatter(linear_out, flattened_indices)

        unflattened = nn.emit(
            relax.op.reshape(unpermuted, (-1, self.num_experts_per_tok, hidden_size))
        )
        expert_weights = nn.emit(
            relax.op.reshape(expert_weights, (-1, self.num_experts_per_tok, 1))
        )
        weighted_sum = self.moe_sum(nn.emit(unflattened * expert_weights))

        # reshape back to 3D
        weighted_sum = nn.emit(relax.op.reshape(weighted_sum, hidden_states_shape))
        return weighted_sum
