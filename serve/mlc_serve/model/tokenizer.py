import structlog
from typing import List
from transformers import AutoTokenizer
from ..engine import ChatMessage
from pathlib import Path

LOG = structlog.stdlib.get_logger(__name__)


class Tokenizer:
    def __init__(self, hf_tokenizer, skip_special_tokens=True):
        self._tokenizer = hf_tokenizer
        self.eos_token_id = self._tokenizer.eos_token_id
        self.skip_special_tokens = skip_special_tokens
        self.all_special_ids = self._tokenizer.all_special_ids
        self.is_fast = self._tokenizer.is_fast

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=self.skip_special_tokens
        )

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return self._tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=self.skip_special_tokens
        )

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        assert tokens, f"tokens must be a valid List of tokens: {tokens}"
        return self._tokenizer.convert_tokens_to_string(tokens)


class ConversationTemplate:
    def __init__(self, hf_tokenizer):
        self._tokenizer = hf_tokenizer

    def apply(self, messages: list[ChatMessage]) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


class HfTokenizerModule:
    def __init__(self, tokenizer_path: Path):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            revision=None,
            tokenizer_revision=None,
        )
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.conversation_template = ConversationTemplate(hf_tokenizer)

        if not self.tokenizer.is_fast:
            LOG.warn("tokenizer.is_fast is false. Some models using an external tokenizer package, "
                     "such as QWen, might hit this condition but that does not imply that their "
                     "tokenizers are slow.")
