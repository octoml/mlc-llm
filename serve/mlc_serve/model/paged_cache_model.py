from pathlib import Path
import structlog
from typing import List, Union

from .base import get_model_artifact_config
from .paged_cache_manager import CacheManager
from .tokenizer import HfTokenizerModule, ConversationTemplate, Tokenizer
from .torch_model import init_torch_model
from .tvm_model import init_tvm_model

from ..engine import MLCServeEngineConfig
from ..engine.model_module import (
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    TextGenerationResult,
    TextGenerator,
)

LOG = structlog.stdlib.get_logger(__name__)


class PagedCacheModelTextGenerator:
    def __init__(self, model: TextGenerator):
        self.model = model

    def generate(
        self, requests: List[Union[PrefillRequest, DecodeRequest]], kv_cache
    ) -> List[TextGenerationResult]:
        prefill_requests = [r for r in requests if isinstance(r, PrefillRequest)]
        decode_requests = [r for r in requests if isinstance(r, DecodeRequest)]

        out = []
        if prefill_requests:
            out.extend(self.model.generate(prefill_requests, kv_cache))
        if decode_requests:
            out.extend(self.model.generate(decode_requests, kv_cache))

        return out


class PagedCacheModelModule:
    engine_config: MLCServeEngineConfig
    text_generator: PagedCacheModelTextGenerator
    cache_manager: CacheManager
    tokenizer: Tokenizer
    conversation_template: ConversationTemplate

    def __init__(
        self,
        model_artifact_path: Path,
        engine_config: MLCServeEngineConfig,
    ):
        if engine_config.model_type == "tvm":
            model_artifact_config = get_model_artifact_config(model_artifact_path)
            model, cache_manager = init_tvm_model(model_artifact_config, engine_config)
            tokenizer_module = HfTokenizerModule(model_artifact_path.joinpath("model"))
        elif engine_config.model_type == "torch":
            model, cache_manager, model_artifact_config = init_torch_model(
                model_artifact_path, engine_config
            )
            tokenizer_module = HfTokenizerModule(model_artifact_path)
        else:
            raise RuntimeError(f"Unknown model type {engine_config.model_type}")

        self.engine_config = engine_config
        self.model_artifact_config = model_artifact_config
        self.text_generator = PagedCacheModelTextGenerator(model)
        self.cache_manager = cache_manager

        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

    def _check_implements_model_module(self) -> ModelModule:
        return self  # type: ignore
