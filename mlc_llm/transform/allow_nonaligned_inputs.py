import tvm
from tvm import relax
from tvm import IRModule
from tvm.script import tir as T

from debug_utils import ExceptDebug


@relax.expr_functor.visitor
class CallTIRArgCollector(relax.expr_functor.PyExprVisitor):
    call_tir_op = tvm.ir.Op.get("relax.call_tir")

    def __init__(self):
        self.to_allow_nonaligned = set()
        self._token_params = set()

    @ExceptDebug
    def visit_function_(self, func):
        self._token_params = set(
            param for param in func.params if "input_ids" in param.name_hint
        )
        self.visit_expr(func.body)
        self._token_params = set()

    @ExceptDebug
    def visit_var_binding_(self, binding):
        if not isinstance(binding.value, relax.Call):
            return

        if binding.value.op != self.call_tir_op:
            return

        primfunc_gv, args, *_tir_vars = binding.value.args
        if "reshape" in primfunc_gv.name_hint:
            # Reshape is handled with a primitive, and passes on any
            # non-aligned input pointer.  Therefore, the output of
            # reshape() should be tracked as potentially unaligned.
            if args[0] in self._token_params:
                self._token_params.add(binding.var)
        else:
            self.to_allow_nonaligned.update(
                (primfunc_gv, i)
                for i, arg in enumerate(args)
                if arg in self._token_params
            )


def update_primfunc(func: tvm.tir.PrimFunc, arg_i: int) -> tvm.tir.PrimFunc:
    handle = func.params[arg_i]
    old_buf = func.buffer_map[handle]
    new_buf = tvm.tir.decl_buffer(
        # Override the previous data alignment
        data_alignment=1,
        # Copy all other contents
        shape=old_buf.shape,
        dtype=old_buf.dtype,
        name=old_buf.name,
        data=old_buf.data,
        strides=old_buf.strides,
        elem_offset=old_buf.elem_offset,
        scope=old_buf.scope,
        offset_factor=old_buf.offset_factor,
        buffer_type="auto_broadcast" if old_buf.buffer_type == 2 else "",
        axis_separators=old_buf.axis_separators,
    )

    def visit(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.same_as(old_buf):
            return tvm.tir.BufferLoad(new_buf, node.indices, node.span)
        elif isinstance(node, tvm.tir.BufferStore) and node.buffer.same_as(old_buf):
            return tvm.tir.BufferStore(new_buf, node.indices, node.value, node.span)

    body = tvm.tir.stmt_functor.ir_transform(
        func.body, preorder=lambda node: None, postorder=visit
    )
    buffer_map = {**func.buffer_map, handle: new_buf}

    return tvm.tir.PrimFunc(
        # Override the previous buffer_map and body
        buffer_map=buffer_map,
        body=body,
        # Copy all other contents
        params=func.params,
        ret_type=func.ret_type,
        attrs=func.attrs,
        span=func.span,
    )


@tvm.transform.module_pass(opt_level=0, name="AllowNonAlignedInputs")
class AllowNonAlignedInputs:
    @ExceptDebug
    def transform_module(
        self, mod: IRModule, ctx: tvm.transform.PassContext
    ) -> IRModule:
        collector = CallTIRArgCollector()
        for func in mod.functions.values():
            if isinstance(func, relax.Function):
                collector.visit_expr(func)

        for gv, arg_i in collector.to_allow_nonaligned:
            mod[gv] = update_primfunc(mod[gv], arg_i)
        return mod
