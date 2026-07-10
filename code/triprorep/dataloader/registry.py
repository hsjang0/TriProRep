"""
Lightweight registry for datamodule builders.
"""
from typing import Callable, Dict, Iterable, TypeVar
import pytorch_lightning as pl


T = TypeVar("T", bound=pl.LightningDataModule)


class DataModuleRegistry:
    """Registry mapping datamodule names to builder callables."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        """Register a datamodule constructor."""
        self._builders[name] = builder

    def build(self, name: str, **kwargs) -> T:
        """Instantiate a registered datamodule."""
        if name not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise KeyError(f"Unknown datamodule '{name}'. Available: {available}")
        return self._builders[name](**kwargs)

    def names(self) -> Iterable[str]:
        """List available datamodule names."""
        return self._builders.keys()
