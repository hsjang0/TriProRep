"""
Masking strategy implementations.
"""
from dataloader.mask_strategy.base import MaskStrategy, MaskOutput
from dataloader.mask_strategy.separate import SeparateMaskStrategy

__all__ = [
    "MaskOutput",
    "MaskStrategy",
    "SeparateMaskStrategy",
]
