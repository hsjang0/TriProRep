"""
Base class for masking strategies.
"""
from dataclasses import dataclass
import torch
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class MaskOutput:
    """Container for masked tokens and their prediction targets."""
    masked_seq_ids: torch.Tensor
    masked_bb_ids: Optional[torch.Tensor]
    masked_fa_ids: Optional[torch.Tensor]
    seq_mask: torch.Tensor
    bb_mask: Optional[torch.Tensor]
    fa_mask: Optional[torch.Tensor]

class MaskStrategy(ABC):
    """Base class for masking strategies."""

    @abstractmethod
    def apply_mask(
        self,
        seq_ids: torch.Tensor,
        struct_ids: torch.Tensor,
        struct_ids_fa: torch.Tensor,
    ) -> MaskOutput:
        """
        Apply masking to sequence and structure tokens.

        Args:
            seq_ids: Sequence token IDs [L]
            struct_ids: Structure token IDs [L]
            struct_ids_fa: Full-atom structure token IDs [L]

        Returns:
            MaskOutput with masked tokens and boolean target masks.
        """
        pass
