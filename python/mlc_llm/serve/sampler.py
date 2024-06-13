"""Classes denoting multi-modality data used in MLC LLM serving"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tvm
import tvm._ffi
from tvm.runtime import Object
from tvm.runtime.ndarray import NDArray
from tvm._ffi.runtime_ctypes import Device

from . import _ffi_api

# @tvm._ffi.register_object("mlc.serve.sampler")  # pylint: disable=protected-access
# class Sampler(Object):
#     """The base class of multi-modality data (text, tokens, embedding, etc)."""

#     def __init__(self):
#         pass

@tvm._ffi.register_object("mlc.serve.Sampler")  # pylint: disable=protected-access
class Sampler(Object):
#     """The class of sampler.

#     Parameters
#     ----------
#     text : str
#         The text string.
#     """

 def __init__(self):
        pass

@tvm._ffi.register_object("mlc.serve.GPUSampler")  # pylint: disable=protected-access
class GPUSampler(Sampler):
    """The class of text data, containing a text string.

    Parameters
    ----------
    text : str
        The text string.
    """

    def __init__(self, vocab_size: int, device: Device):
        self.__init_handle_by_constructor__(_ffi_api.GPUSampler, vocab_size, device)  # type: ignore  # pylint: disable=no-member


    @property
    def vocab_size(self) -> int:
        """The text data in `str`."""
        return 0 # type: ignore  # pylint: disable=no-member

    def sample(self, samples: NDArray) -> NDArray:
        """Return the image data."""
        return _ffi_api.GPUSamplerBatchDecode(self, samples)