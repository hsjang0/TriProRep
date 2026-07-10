"""
Base class for data format strategies.
"""
import torch
from abc import ABC, abstractmethod
from typing import Optional


class DataFormatStrategy(ABC):
    """Base class for data format strategies."""
    
    @abstractmethod
    def get_output_length(self, original_seq_len: int) -> int:
        """Get the output sequence length after formatting."""
        pass
    
    

