from pathlib import Path
import structlog
from typing import List

from .base import get_model_artifact_config
from .paged_cache_manager import CacheManager
from .tokenizer import HfTokenizerModule, ConversationTemplate, Tokenizer
from .tvm_model import init_tvm_model

from ..engine import MLCServeEngineConfig
from ..engine.model_module import (
    DecodeRequest,
    ModelModule,
    PrefillRequest,
    EvalMultiQueryRequest,
    RequestType,
    TextGenerationResult,
    TextGenerator,
)

LOG = structlog.stdlib.get_logger(__name__)


class PagedCacheModelTextGenerator:
    def __init__(self, model: TextGenerator):
        self.model = model

    def generate(
        self,
        requests: List[RequestType],
        kv_cache,
    ) -> List[TextGenerationResult]:
        prefill_requests = []
        decode_requests = []
        multi_query_decode_requests = []
        multi_query_decode_request_ids = set()

        for r in requests:
            if isinstance(r, PrefillRequest):
                prefill_requests.append(r)
            elif isinstance(r, DecodeRequest):
                decode_requests.append(r)
            elif isinstance(r, EvalMultiQueryRequest):
                multi_query_decode_requests.append(r)
                multi_query_decode_request_ids.add(r.sequence_id.request_id)

        out = []

        if prefill_requests:
            prefill_res = self.model.generate(prefill_requests, kv_cache)

            if not multi_query_decode_requests:
                out.extend(prefill_res)
            else:
                # Prefill requests from restoration of evicted parallel-sampling requests
                # must not return outputs.
                for res in prefill_res:
                    if res.sequence_id.request_id not in multi_query_decode_request_ids:
                        out.append(res)

        if decode_requests:
            out.extend(self.model.generate(decode_requests, kv_cache))

        if multi_query_decode_requests:
            out.extend(self.model.generate(multi_query_decode_requests, kv_cache))

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
        model_artifact_config = get_model_artifact_config(model_artifact_path)

        # TODO(masahi): Make the model type configurable.
        model, cache_manager = init_tvm_model(model_artifact_config, engine_config)

        self.engine_config = engine_config
        self.model_artifact_config = model_artifact_config
        self.text_generator = PagedCacheModelTextGenerator(model)
        self.cache_manager = cache_manager

        tokenizer_module = HfTokenizerModule(model_artifact_path)
        self.tokenizer = tokenizer_module.tokenizer
        self.conversation_template = tokenizer_module.conversation_template

    def _check_implements_model_module(self) -> ModelModule:
        return self  # type: ignore
