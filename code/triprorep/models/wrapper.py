"""
Unified builder for backbone model variants (release subset — ELECTRA
pre-training only).
"""
from typing import Any
from pytorch_lightning import LightningModule

from models.registry import ModelRegistry
from models.backbones.backbone_models import (
    ProteinModel,
    ELECTRAProteinModel,
)


backbone_registry = ModelRegistry()
backbone_registry.register("ProteinModel", ProteinModel)
backbone_registry.register("ELECTRAProteinModel", ELECTRAProteinModel)


def build_backbone(name: str = "ProteinModel", **kwargs: Any) -> LightningModule:
    return backbone_registry.build(name, **kwargs)


__all__ = [
    "backbone_registry",
    "build_backbone",
]
