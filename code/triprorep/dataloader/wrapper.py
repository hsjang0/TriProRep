"""
Unified builder for datamodules (release subset — ELECTRA pre-training only).
"""
from dataloader.registry import DataModuleRegistry
from dataloader.datamodule import (
    ProteinDataModule,
    ComplexPretrainDataModule,
    ELECTRADataModule,
)


datamodule_registry = DataModuleRegistry()
datamodule_registry.register("ProteinDataModule", ProteinDataModule)
datamodule_registry.register("ComplexPretrainDataModule", ComplexPretrainDataModule)
datamodule_registry.register("ELECTRADataModule", ELECTRADataModule)


def build_datamodule(name: str, **kwargs):
    """Construct a datamodule by registry name."""
    return datamodule_registry.build(name, **kwargs)


__all__ = ["build_datamodule", "datamodule_registry"]
