from . import RegexFSM, TransformerTokenizer
from .base_cache import BaseCache


class FSMCache(BaseCache):
    def __init__(self, tokenizer_path, tokenizer_args_dict):
        super().__init__()
        self.outlines_tokenizer = TransformerTokenizer(
            tokenizer_path, **tokenizer_args_dict
        )

    def init_value(self, regex):
        return RegexFSM(regex, self.outlines_tokenizer)
