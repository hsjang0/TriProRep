"""
Lightweight registry for model variants.
"""
from typing import Callable, Dict, Iterable, TypeVar
from pytorch_lightning import LightningModule


T = TypeVar("T", bound=LightningModule)


class ModelRegistry:
    """Lightweight registry mapping variant names to builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        """Register a builder under a string key."""
        self._builders[name] = builder

    def build(self, name: str, **kwargs) -> T:
        """Instantiate a registered builder."""
        if name not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise KeyError(f"Unknown model variant '{name}'. Available: {available}")
        return self._builders[name](**kwargs)

    def get(self, name: str) -> Callable[..., T]:
        """Return the registered builder without constructing it."""
        if name not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise KeyError(f"Unknown model variant '{name}'. Available: {available}")
        return self._builders[name]

    def names(self) -> Iterable[str]:
        """List available variant names."""
        return self._builders.keys()
