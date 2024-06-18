import os
from typing import List

from tokenizers import ByteLevelBPETokenizer as ByteLevelBPETokenizer_
from torchtitan.logging_utils import logger

from .base_tokenizer import BaseTokenizer


class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Tokenizing and encoding/decoding text based on a ByteLevelBPE model.

    Args:
        tokenizer_path (str): The path to the ByteLevelBPE model file.
    """

    def __init__(self, tokenizer_path: str):
        super().__init__(tokenizer_path)
        # reload tokenizer
        self.tokenizer = ByteLevelBPETokenizer_.from_file(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt"),
        )

        # BOS / EOS token IDs
        self._n_words: int = self.tokenizer.get_vocab_size()
        self.bos_id: int = self.tokenizer.token_to_id("<s>")
        self.eos_id: int = self.tokenizer.token_to_id("</s>")
        self.pad_id: int = self.tokenizer.token_to_id("<pad>")
        logger.info(
            f"ByteLevelBPETokenizer built: #words {self.n_words}, BOS ID {self.bos_id}, EOS ID {self.eos_id}"
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
        t = self.tokenizer.encode(s).ids

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
