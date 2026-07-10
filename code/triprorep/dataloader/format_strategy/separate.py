"""
Separate format strategy: encodes type-wise tokens separately.
"""
import torch
from typing import Optional
from dataloader.format_strategy.base import DataFormatStrategy
from torch.nn import functional as F


class SeparateFormatStrategy(DataFormatStrategy):
    """
    Encodes seq_id, struct_id, fa_id, vox_id separately for each position.
    """
    
    def __init__(
        self,
        num_special_tokens: int = 4,
        seq_vocab_size: int = 21,
        bb_vocab_size: int = 21,
        fa_vocab_size: int = 21,
        mask_token_id: int = 3,
    ):
        """
        Args:
            num_special_tokens: Number of special tokens
            seq_vocab_size: Vocabulary size for sequence tokens
            struct_vocab_size: Vocabulary size for structure tokens
            mask_token_id: Token ID for mask token
        """
        self.num_special_tokens = num_special_tokens
        self.seq_vocab_size = seq_vocab_size
        self.bb_vocab_size = bb_vocab_size
        self.fa_vocab_size = fa_vocab_size
        self.mask_token_id = mask_token_id
        
        print(f"[SEPARATEFormatStrategy] Initialized with seq_vocab_size={seq_vocab_size}, struct_vocab_size={bb_vocab_size}, struct_fa_vocab_size={fa_vocab_size}")
        
        # Compute offsets for token ID encoding
        self.seq_offset = num_special_tokens
        self.struct_offset = num_special_tokens
        self.struct_fa_offset = num_special_tokens
    
    def get_output_length(self, original_seq_len: int) -> int:
        """Output length is original_seq_len (+ 2 if special tokens)."""
        return original_seq_len
    
    def get_seq_vocab_size(self) -> int:
        return self.seq_vocab_size + self.num_special_tokens

    def get_bb_vocab_size(self) -> int:
        return self.bb_vocab_size + self.num_special_tokens

    def get_fa_vocab_size(self) -> int:
        return self.fa_vocab_size + self.num_special_tokens
    
