"""The SmoothQuant config"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Literal

import tvm

from tvm import IRModule, relax

from tvm.script import relax as R
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target
from tvm import dlight as dl
from mlc_llm.loader import QuantizeMapping
import os
from tvm.contrib import tvmjs


def _compile_quantize_func(mod: IRModule, target, device) -> Callable:
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
    )(mod)
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)  # pylint: disable=invalid-name
    return vm

def load_file(path):
    import json
    with open(path, 'r') as f:
        loaded_dict = json.load(f)
    return loaded_dict


def _gen_max_stat_func(bb, shape, dtype, func_name):
    inpt  =  relax.Var("w", relax.TensorStructInfo(shape, dtype))
    params = [inpt]

    with bb.function(func_name, params=params):
        with bb.dataflow():
            v = bb.emit(R.abs(inpt))
            v = bb.emit(R.max(v, axis=-2, keepdims=False))
            gv = bb.emit_output(v)
        bb.emit_func_output(gv)
    return func_name

def _gen_max_smoothed_func(bb, shape, dtype, func_name):
    inpt  =  relax.Var("w", relax.TensorStructInfo(shape, dtype))
    scale_smooth_const = relax.Var("sc", relax.TensorStructInfo((shape[1],), dtype))
    params = [inpt, scale_smooth_const]
    with bb.function(func_name, params=params):
        with bb.dataflow():
            inpt = bb.emit(R.multiply(inpt, scale_smooth_const))
            max_expr = bb.emit(R.max(inpt, axis=-1))
            min_expr = bb.emit(R.min(inpt, axis=-1))
            if inpt.struct_info.ndim > 2:
                max_expr = bb.emit.emit(R.squeeze(max_expr))
                min_expr = bb.emit.emit(R.squeeze(min_expr))
            gv = bb.emit_output((min_expr, max_expr))
        bb.emit_func_output(gv)
    return func_name


def shape_to_name(shp):
    out = ""
    for i in range(len(shp) -1):
        out += f"{shp[i]}x"
    out += f"{shp[-1]}"
    return out


@dataclass
class PreprocessSmoothQuantize:  # pylint: disable=too-many-instance-attributes
    name: str
    kind: str
    model_dtype: Literal["float16"]
    dump_calibration: Literal[""]

    def __post_init__(self):
        self.funcs_cache = {}
        self.conversion_funcs_cache = {}
        self.abs_max = {}
        self.min_max_smoothed = {}
        self.smooth_params = None
        self.param_to_smooth_factor = None
        return

    def preprocess_weight(
        self, weight: NDArray,
        name: str
    ) -> List[NDArray]:
        """
        Quantize weight with group quantization

        Parameters
        ----------
        weight : NDArray
            The original weight.

        axis : int
            The group axis.

        output_transpose : bool
            Whether to transpose the output quantized weight. Only 2D weight is supported.

        Returns
        ------
        ret: List[NDArray]
            The list of group quantized weights.
        """
        nm_shape = shape_to_name(weight.shape)
        if self.kind == "smooth-preprocess":
            func_name = "abs_max_calc"
            if not nm_shape in self.funcs_cache.keys():
                bb = relax.BlockBuilder()
                _gen_max_stat_func(bb, weight.shape, self.model_dtype, func_name)
                mod = bb.finalize()
                self.funcs_cache[nm_shape] = _compile_quantize_func(
                    mod, target=Target.from_device(weight.device), device=weight.device
                )[func_name]
            data = self.funcs_cache[nm_shape](weight)
            self.abs_max[name] = data.numpy()
            del data
        elif self.kind == "quantize-preprocess":
            func_name = "max_smoothed_func"
            if nm_shape not in self.conversion_funcs_cache.keys():
                bb = relax.BlockBuilder()
                _gen_max_smoothed_func(bb, weight.shape, self.model_dtype, func_name)
                mod = bb.finalize()
                self.conversion_funcs_cache[nm_shape] = _compile_quantize_func(
                    mod, target=Target.from_device(weight.device), device=weight.device
                )[func_name]
            scale_names = self.param_to_smooth_factor[name]
            scale = self.smooth_params[scale_names[1]].numpy()
            scale_dev = tvm.nd.array(scale, device=weight.device)
            data = self.conversion_funcs_cache[nm_shape](weight, scale_dev)
            self.min_max_smoothed[name] = [data[0].numpy(), data[1].numpy()]
            del data
            del scale_dev
        return [weight]

    def clear_data(self):
        self.funcs_cache = {}
        self.conversion_funcs_cache = {}
        self.abs_max = {}
        self.min_max_smoothed = {}

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with using smooth quantization
        currently all conversions are performed using compilation passes.
        ToDo: apply final model conversion using this pass.

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """
        class _Mutator(nn.Mutator):
            def __init__(self, config: PreprocessSmoothQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                """
                The visiting method for preprocessing for smooth quantization of nn.Module nodes.

                Parameters
                ----------
                name : str
                    The name of the current node

                node : nn.Module
                    The current node of nn.Module to mutate.

                Returns
                -------
                ret_node : Any
                    The new node to replace current node.
                """

                if isinstance(node, nn.Linear):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [weight_name]
                    self.quant_map.map_func[weight_name] = partial(
                        self.config.preprocess_weight,
                        name=weight_name,
                    )

                return self.visit(name, node)
        if self.kind == "quantize-preprocess":
            assert self.dump_calibration
            self.smooth_params = tvmjs.load_ndarray_cache(
                os.path.join(f"{self.dump_calibration}", "smooth"), tvm.cpu(0)
            )[0]
            self.param_to_smooth_factor = load_file(
                path=f"{self.dump_calibration}/smooth_scale2param.json"
            )["prefill"]
        self.clear_data()
        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)

        return model
