from torchtitan.datasets.tokenizer.sentencepiece import SentencePieceTokenizer
from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer
from torchtitan.datasets.tokenizer.tokenizer import Tokenizer
from torchtitan.logging_utils import logger

from torchtitan_plugin.utils import JobConfig, logging_info

from .bbpe import ByteLevelBPETokenizer
from .hf_tokenizer import HFTokenizer


def create_tokenizer(tokenizer_type: str, tokenizer_path: str) -> Tokenizer:
    logging_info(f"Building {tokenizer_type} tokenizer locally from {tokenizer_path}")
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(tokenizer_path)
    elif tokenizer_type == "tiktoken":
        return TikTokenizer(tokenizer_path)
    elif tokenizer_type == "bbpe":
        return ByteLevelBPETokenizer(tokenizer_path)
    elif tokenizer_type == "hf":
        return HFTokenizer(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer type: {args.type}")
