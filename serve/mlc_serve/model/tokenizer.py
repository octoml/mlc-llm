from typing import List, Optional, Union
from transformers import AutoTokenizer
from ..engine import ChatMessage
from pathlib import Path
from ..api.protocol import B_FUNC, E_FUNC
import json

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

    def apply(self, messages: list[ChatMessage], 
            tools: Optional[List] = None, 
            tool_choice: Optional[Union[str, object]] = None) -> str:
        
        first_user_prompt = True
        processed_messages = []

        for message in messages:
            if message.role == "user" and first_user_prompt and tools and tool_choice != "none":
                function_list = json.dumps(tools, indent=4)
                message.content = f"{B_FUNC}{function_list.strip()}{E_FUNC}{message.content.strip()}\n\n"
                first_user_prompt = False
            processed_messages.append({"role": message.role, "content": message.content})
        return self._tokenizer.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

class HfTokenizerModule:
    def __init__(self, model_artifact_path: Path):
        hf_tokenizer = AutoTokenizer.from_pretrained(
            model_artifact_path.joinpath("model"),
            revision=None, tokenizer_revision=None,
            trust_remote_code=False,
        )
        self.tokenizer = Tokenizer(hf_tokenizer)
        self.conversation_template = ConversationTemplate(hf_tokenizer)
