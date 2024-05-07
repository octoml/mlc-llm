"""A centralized registry of all existing quantization methods and their configurations."""
from typing import Any, Dict
from enum import Enum

from .awq_quantization import AWQQuantize
from .ft_quantization import FTQuantize
from .group_quantization import GroupQuantize
from .no_quantization import NoQuantize
from .per_tensor_quantization import PerTensorQuantize
from .smooth_quantization import SmoothQuantize

Quantization = Any
# TODO(jmcmahan): make this an abstract class + add `req_calibration` (default False) to better detect calibration requirement
"""Quantization is an object that represents an quantization algorithm. It is required to
have the following fields:

    name : str
        The name of the quantization algorithm, for example, "q4f16_1".

    kind : str
        The kind of quantization algorithm, for example, "group-quant", "faster-transformer".

It is also required to have the following method:

    def quantize_model(self, module: nn.Module) -> nn.Module:
        ...

    def quantize_weight(self, weight: tvm.runtime.NDArray) -> List[tvm.runtime.NDArray]:
        ...
"""

class MLCQuantization(str, Enum):
    q0f16 = "q0f16"
    q0f32 = "q0f32"
    q3f16_0 = "q3f16_0"
    q3f16_1 = "q3f16_1"
    q4f16_0 = "q4f16_0"
    q4f16_1 = "q4f16_1"
    q4f32_1 = "q4f32_1"
    q4f16_2 = "q4f16_2"
    q4f16_autoawq = "q4f16_autoawq"
    q4f16_ft = "q4f16_ft"
    fp8_e4m3_e4m3_max_calibration = "fp8_e4m3_e4m3_max_calibration"
    fp8_e4m3_e4m3_max = "fp8_e4m3_e4m3_max"
    ptq_e4m3_e4m3_max_calibration = "ptq_e4m3_e4m3_max_calibration"
    ptq_e4m3_e4m3_max = "ptq_e4m3_e4m3_max"
    smq_q8i8f16_0 = "smq_q8i8f16_0"
    smq_q8i8f16_1 = "smq_q8i8f16_1"
    smq_q8i8f16_2 = "smq_q8i8f16_2"
    smq_e4m3_float8_0 = "smq_e4m3_float8_0"
    smq_e4m3_float8_1 = "smq_e4m3_float8_1"
    smq_e4m3_float8_2 = "smq_e4m3_float8_2"
    smq_e5m2_float8_0 = "smq_e5m2_float8_0"
    smq_e5m2_float8_1 = "smq_e5m2_float8_1"
    smq_e5m2_float8_2 = "smq_e5m2_float8_2"
    fp16_max_calibration = "fp16_max_calibration"    

QUANTIZATION: Dict[str, Quantization] = {
    MLCQuantization.q0f16: NoQuantize(
        name=MLCQuantization.q0f16,
        kind="no-quant",
        model_dtype="float16",
    ),
    MLCQuantization.q0f32: NoQuantize(
        name=MLCQuantization.q0f32,
        kind="no-quant",
        model_dtype="float32",
    ),
    MLCQuantization.q3f16_0: GroupQuantize(
        name=MLCQuantization.q3f16_0,
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    MLCQuantization.q3f16_1: GroupQuantize(
        name=MLCQuantization.q3f16_1,
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    MLCQuantization.q4f16_0: GroupQuantize(
        name=MLCQuantization.q4f16_0,
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    MLCQuantization.q4f16_1: GroupQuantize(
        name=MLCQuantization.q4f16_1,
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    MLCQuantization.q4f32_1: GroupQuantize(
        name=MLCQuantization.q4f32_1,
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float32",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    MLCQuantization.q4f16_2: GroupQuantize(
        name=MLCQuantization.q4f16_2,
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=False,
        quantize_final_fc=False,
    ),
    MLCQuantization.q4f16_autoawq: AWQQuantize(
        name=MLCQuantization.q4f16_autoawq,
        kind="awq",
        group_size=128,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
    ),
    MLCQuantization.q4f16_ft: FTQuantize(
        name=MLCQuantization.q4f16_ft,
        kind="ft-quant",
        quantize_dtype="int4",
        storage_dtype="int8",
        model_dtype="float16",
    ),
    MLCQuantization.fp8_e4m3_e4m3_max_calibration: PerTensorQuantize(
        name=MLCQuantization.fp8_e4m3_e4m3_max_calibration,
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="uint8",
        model_dtype="float16",
        quantize_embedding=False,
        quantize_linear=False,
    ),
    MLCQuantization.fp8_e4m3_e4m3_max: PerTensorQuantize(
        name=MLCQuantization.fp8_e4m3_e4m3_max,
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="uint8",
        model_dtype="float16",
        quantize_embedding=False,
        quantize_linear=False,
    ),
    MLCQuantization.ptq_e4m3_e4m3_max_calibration: PerTensorQuantize(
        name=MLCQuantization.ptq_e4m3_e4m3_max_calibration,
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="e4m3_float8",
        model_dtype="float16",
        quantize_embedding=False,
        quantize_linear=True,
        # TODO(csullivan): Refactor sharding of calibration scale
        # to enable lm_head quantization for TP > 1
        quantize_final_fc=False,
    ),
    MLCQuantization.ptq_e4m3_e4m3_max: PerTensorQuantize(
        name=MLCQuantization.ptq_e4m3_e4m3_max,
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="e4m3_float8",
        model_dtype="float16",
        quantize_embedding=False,
        quantize_linear=True,
        quantize_final_fc=False,
    ),
    MLCQuantization.smq_q8i8f16_0: SmoothQuantize(
        name=MLCQuantization.smq_q8i8f16_0,
        kind="smoothquant",
        activation_dtype="int8",
        weight_dtype="int8",
        zero_point_dtype="int8",
        accumulator_dtype="int32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_q8i8f16_1: SmoothQuantize(
        name=MLCQuantization.smq_q8i8f16_1,
        kind="smoothquant",
        activation_dtype="int8",
        weight_dtype="int8",
        zero_point_dtype="int8",
        accumulator_dtype="int32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_q8i8f16_2: SmoothQuantize(
        name=MLCQuantization.smq_q8i8f16_2,
        kind="smoothquant",
        activation_dtype="int8",
        weight_dtype="int8",
        zero_point_dtype="int8",
        accumulator_dtype="int32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e4m3_float8_0: SmoothQuantize(
        name=MLCQuantization.smq_e4m3_float8_0,
        kind="smoothquant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e4m3_float8_1: SmoothQuantize(
        name=MLCQuantization.smq_e4m3_float8_1,
        kind="smoothquant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e4m3_float8_2: SmoothQuantize(
        name=MLCQuantization.smq_e4m3_float8_2,
        kind="smoothquant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e5m2_float8_0: SmoothQuantize(
        name=MLCQuantization.smq_e5m2_float8_0,
        kind="smoothquant",
        activation_dtype="e5m2_float8",
        weight_dtype="e5m2_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e5m2_float8_1: SmoothQuantize(
        name=MLCQuantization.smq_e5m2_float8_1,
        kind="smoothquant",
        activation_dtype="e5m2_float8",
        weight_dtype="e5m2_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.smq_e5m2_float8_2: SmoothQuantize(
        name=MLCQuantization.smq_e5m2_float8_2,
        kind="smoothquant",
        activation_dtype="e5m2_float8",
        weight_dtype="e5m2_float8",
        zero_point_dtype="float16",
        accumulator_dtype="float32",
        model_dtype="float16",
    ),
    MLCQuantization.fp16_max_calibration: PerTensorQuantize(
        name=MLCQuantization.fp16_max_calibration,
        kind="per-tensor-quant",
        activation_dtype="float16",
        weight_dtype="float16",
        storage_dtype="float16",
        model_dtype="float16",
        quantize_embedding=False,
        quantize_linear=False,
        no_scale=True,
    ),
}
