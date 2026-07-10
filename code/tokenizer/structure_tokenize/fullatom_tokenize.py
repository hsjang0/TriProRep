"""
Thin wrappers around fullatom_tokenizer integration helpers for backward
compatibility with preprocessing scripts.
"""
from structure_tokenize.fullatom_tokenizer.integration import (
    load_fullatom_tokenizer,
    tokenize_fullatom_structure,
)

# Preserve legacy names used in preprocessing scripts
load_tokenizer = load_fullatom_tokenizer
get_struct_seq_fa = tokenize_fullatom_structure

__all__ = [
    "load_fullatom_tokenizer",
    "tokenize_fullatom_structure",
    "load_tokenizer",
    "get_struct_seq_fa",
]
