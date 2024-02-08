import ctypes
import json
import os
import pathlib
import struct

import typing
from typing import Union, Dict, List

import numpy as np
import structlog

import tvm

LOG = structlog.stdlib.get_logger(__name__)

# Safetensor names are defined by their enum name in the Rust
# implementation at
# https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs#L635
SAFETENSOR_TO_TVM_DTYPE = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    # FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    "F8_E5M2": "float8",
    # FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    "F8_E4M3": "float8",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "U64": "uint64",
}


class LazySafetensorDir:
    def __init__(self, dirpath: Union[str, pathlib.Path]):
        self.dirpath = pathlib.Path(dirpath)
        self._safetensor_files = [
            LazySafetensorFile(filepath)
            # safetensors.safe_open(filepath, "pt")
            for filepath in self.dirpath.glob("*.safetensors")
        ]

    @property
    def _file_lookup(self):
        if not hasattr(self, "_file_lookup_cache"):
            self._file_lookup_cache = {
                name: file for file in self._safetensor_files for name in file.tensors
            }

        return self._file_lookup_cache

    def __getitem__(self, name) -> tvm.runtime.NDArray:
        return self._file_lookup[name][name]


class LazySafetensorFile:
    def __init__(self, filepath: Union[str, pathlib.Path]):
        filepath = pathlib.Path(filepath)
        self.filepath = filepath
        self._handle = self.filepath.open("rb")
        self._tensors = None

    @property
    def tensors(self) -> Dict[str, "LazySafetensor"]:
        if self._tensors is not None:
            return self._tensors

        self._handle.seek(0, os.SEEK_END)
        file_size_bytes = self._handle.tell()
        self._handle.seek(0, os.SEEK_SET)

        # A uint64 header
        json_header_nbytes = struct.unpack("<Q", self._handle.read(8))[0]
        # Followed by that many bytes as a JSON packet
        json_header = self._handle.read(json_header_nbytes)
        header = json.loads(json_header)

        tensors = {}
        for name, entry in header.items():
            if name != "__metadata__":
                dtype = SAFETENSOR_TO_TVM_DTYPE[entry["dtype"]]
                shape = entry["shape"]

                # The data_offset is relative to the end of the JSON header,
                # *NOT* to the file itself.
                data_offsets = [
                    offset + json_header_nbytes + 8 for offset in entry["data_offsets"]
                ]

                assert len(data_offsets) == 2
                assert data_offsets[0] <= data_offsets[1] <= file_size_bytes

                nbytes = data_offsets[1] - data_offsets[0]
                expected_nbytes = np.prod(shape) * tvm.runtime.DataType(dtype).bits // 8

                assert nbytes == expected_nbytes

                tensors[name] = LazySafetensor(
                    self._handle,
                    name,
                    dtype=dtype,
                    shape=shape,
                    data_offset_in_file=data_offsets[0],
                )

        self._tensors = tensors
        return self._tensors

    def __getitem__(self, name: str) -> "LazySafetensor":
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


class LazySafetensor:
    def __init__(
        self,
        file_handle: typing.IO[bytes],
        name: str,
        dtype: str,
        shape: List[int],
        data_offset_in_file: int,
    ):
        self.file_handle = file_handle
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data_offset_in_file = data_offset_in_file

    @property
    def num_bytes(self) -> int:
        return np.prod(self.shape) * tvm.runtime.DataType(self.dtype).bits // 8

    def as_tvm_array(self) -> tvm.runtime.NDArray:
        # The `safetensors.safe_open` function is almost what we need,
        # but would either (a) require an extra data copy if going
        # through a Numpy array, or (b) fail TVM's alignment checks if
        # directly producing a pytorch array.
        #
        # Allocating the initial space using `tvm.nd.empty` avoids numpy-to-TVM conversions.
        arr = tvm.nd.empty(self.shape, self.dtype)

        byte_ptr_type = ctypes.POINTER(ctypes.c_byte)

        data_len = self.num_bytes

        data_ptr = ctypes.cast(arr.handle.contents.data, byte_ptr_type)

        # At some point, this will be exposed as
        # `ctypes.memoryview_at`.  However, as of 2024-02-01, the PR
        # hasn't merged to be available in preview versions, let alone
        # the current production Python version.
        #
        # https://github.com/python/cpython/pull/112018
        ctypes.pythonapi.PyMemoryView_FromMemory.argtypes = (
            byte_ptr_type,
            ctypes.c_ssize_t,
            ctypes.c_int,
        )
        ctypes.pythonapi.PyMemoryView_FromMemory.restype = ctypes.py_object
        read_write_flag = 0x200

        view = ctypes.pythonapi.PyMemoryView_FromMemory(
            data_ptr, ctypes.c_ssize_t(data_len), read_write_flag
        )

        self.file_handle.seek(self.data_offset_in_file)
        bytes_read = self.file_handle.readinto(view)

        assert bytes_read == data_len, "Could not read full array"

        return arr


get_free_memory_func = tvm.get_global_func("runtime.GetCudaFreeMemory")


def get_free_memory():
    text = get_free_memory_func()
    words = text.split()
    free = int(words[4])
    total = int(words[9])
    return free, total


get_relax_vm_memory_func = tvm.get_global_func("vm.memory_manager.get_used_memory")


def get_relax_vm_memory_usage(dev: tvm.runtime.Device) -> int:
    return get_relax_vm_memory_func(dev)


@tvm.register_func("mlc_serve.model.define_safetensors_get_item")
def define_safetensors_get_item(
    safetensors_dir: str,
    param_names: str,
    dev: tvm.runtime.Device,
) -> None:
    param_names = param_names.split("\n")

    safetensors = LazySafetensorDir(safetensors_dir)

    pid = os.getpid()

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        safetensor = safetensors[param_names[i]]

        bytes_free, bytes_total = get_free_memory()
        vm_memory_usage = get_relax_vm_memory_usage(dev)
        LOG.info(
            "Starting to load tensor to GPU",
            pid=pid,
            param=i,
            param_name=param_names[i],
            param_shape=safetensor.shape,
            param_dtype=safetensor.dtype,
            param_bytes=safetensor.num_bytes,
            bytes_free=bytes_free,
            bytes_total=bytes_total,
            vm_memory_usage=vm_memory_usage,
        )

        tvm_on_cpu = safetensor.as_tvm_array()
        tvm_on_gpu = tvm.nd.array(tvm_on_cpu, dev)

        bytes_free, bytes_total = get_free_memory()
        vm_memory_usage = get_relax_vm_memory_usage(dev)
        LOG.info(
            "Finished loading tensor to GPU",
            pid=pid,
            param=i,
            param_name=param_names[i],
            param_shape=safetensor.shape,
            param_dtype=safetensor.dtype,
            param_bytes=safetensor.num_bytes,
            bytes_free=bytes_free,
            bytes_total=bytes_total,
            vm_memory_usage=vm_memory_usage,
        )

        return tvm_on_gpu
