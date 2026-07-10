"""
Shared typing helpers for model factories.
"""
from typing import Any, Protocol, runtime_checkable
from pytorch_lightning import LightningModule


@runtime_checkable
class ModelBuilder(Protocol):
    """Callable that builds a LightningModule."""

    def __call__(self, *args: Any, **kwargs: Any) -> LightningModule:
        ...
