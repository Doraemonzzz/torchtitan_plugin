from typing import List

from transformers import AutoTokenizer

from torchtitan_plugin.utils import logging_info

from .base_tokenizer import BaseTokenizer


class HFTokenizer(BaseTokenizer):
    """
    Tokenizing and encoding/decoding text based on a Huggingface Tokenizer.

    Args:
        tokenizer_path (str): The path to the Huggingface Tokenizer.
    """

    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)
        # reload tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        # BOS / EOS token IDs
        self._n_words: int = self.tokenizer.vocab_size
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            logging_info(f"Pad ID is not set, using the Eos ID instead.")
            self.pad_id = self.eos_id
        logging_info(
            f"HF Tokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        assert type(s) is str
        t = self.tokenizer.encode(s)

        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(t)
