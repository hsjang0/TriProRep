from __future__ import annotations

import multiprocessing
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
from multiprocessing import Pool
from tqdm import tqdm


def _to_numpy_safe(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy_safe(v) for v in obj)
    return obj


@dataclass
class StructInitConfig:
    struct_format: str
    bb_ckpt_path: Optional[str] = None
    fa_ckpt_path: Optional[str] = None
    fa_cfg: Optional[str] = None
    label_map: Optional[dict] = None
    gpu_ids: Optional[List[int]] = None


class StructureWorkerContext:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.fa_model = None
        self.label_map: dict = {}
        self.device = "cpu"

    def reset(self) -> None:
        self.tokenizer = None
        self.model = None
        self.fa_model = None
        self.label_map = {}
        self.device = "cpu"

    def load_from_config(self, config: StructInitConfig) -> None:
        self.reset()
        self.label_map = config.label_map or {}
        self.device = select_device(config.gpu_ids)

        struct_format = config.struct_format
        if struct_format == "aminoaseed":
            from structure_tokenize.seq_backbone_tokenize import load_tokenizer

            self.model = load_tokenizer(config.bb_ckpt_path, device=self.device)
            print(f"Worker {os.getpid()} loaded aminoaseed model on {self.device}")
        elif struct_format == "fullatom_image":
            from structure_tokenize.seq_backbone_tokenize import load_tokenizer
            from structure_tokenize.fullatom_tokenize import load_tokenizer as load_fa

            self.model = load_tokenizer(config.bb_ckpt_path, device=self.device)
            self.fa_model = load_fa(config.fa_ckpt_path, config.fa_cfg, device=self.device)
            print(f"Worker {os.getpid()} loaded aminoaseed and fullatom models on {self.device}")
        else:
            raise ValueError(f"Unsupported struct_format: {struct_format}")


_CONTEXT = StructureWorkerContext()


def get_worker_context() -> StructureWorkerContext:
    return _CONTEXT


def configure_spawn_start_method() -> None:
    if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(gpu_ids: Optional[Sequence[int]]) -> str:
    device = "cpu"
    if gpu_ids:
        try:
            worker_rank = multiprocessing.current_process()._identity[0] - 1
        except Exception:
            worker_rank = 0
        device_id = gpu_ids[worker_rank % len(gpu_ids)]
        torch.cuda.set_device(device_id)
        device = f"cuda:{device_id}"
    elif torch.cuda.is_available():
        device = "cuda"
    return device


def select_num_workers(num_workers: Optional[int], gpu_ids: Optional[Sequence[int]], workers_per_gpu: int) -> int:
    if num_workers is None or num_workers <= 0:
        if gpu_ids:
            return max(1, workers_per_gpu * len(gpu_ids))
        return max(1, multiprocessing.cpu_count() // 8)
    return num_workers


def init_structure_worker(config: StructInitConfig) -> None:
    get_worker_context().load_from_config(config)


def encode_structure(
    pdb_path: str,
    struct_format: str,
    chain: Optional[str] = None,
    filter_low_plddt: bool = False,
) -> Optional[dict]:
    context = get_worker_context()
    chains = [chain] if chain else None

    if struct_format == "aminoaseed":
        from structure_tokenize.seq_backbone_tokenize import get_struct_seq

        kwargs = {"filter_low_plddt": filter_low_plddt}
        if chains:
            kwargs["chains"] = chains
        result = get_struct_seq(context.model, pdb_path, **kwargs)
        if result is not None and "pdb_id" not in result:
            result["pdb_id"] = Path(pdb_path).stem
        return _to_numpy_safe(result)

    if struct_format == "fullatom_image":
        from structure_tokenize.seq_backbone_tokenize import get_struct_seq
        from structure_tokenize.fullatom_tokenize import get_struct_seq_fa

        kwargs = {"filter_low_plddt": filter_low_plddt}
        if chains:
            kwargs["chains"] = chains
        result_bb = get_struct_seq(context.model, pdb_path, **kwargs)
        result_fa = get_struct_seq_fa(context.fa_model, pdb_path, chains=chains)

        if result_bb is not None and result_fa is not None:
            return _to_numpy_safe({
                "pdb_id": Path(pdb_path).stem,
                "seq_id": result_bb["seq_id"],
                "struct_id_aminoaseed": result_bb["struct_id"],
                "struct_id_fullatom": result_fa["fullatom_id"],
            })
        return None

    raise ValueError(f"Unsupported struct_format: {struct_format}")


def run_worker_pool(
    process_args: Iterable[tuple],
    process_fn: Callable,
    init_config: StructInitConfig,
    num_workers: int,
    desc: str,
) -> list:
    args_list = list(process_args)
    results = []
    with Pool(processes=num_workers, initializer=init_structure_worker, initargs=(init_config,)) as pool:
        for result in tqdm(pool.imap(process_fn, args_list), total=len(args_list), desc=desc):
            if result is not None:
                results.append(result)
    return results


__all__ = [
    "StructInitConfig",
    "StructureWorkerContext",
    "configure_spawn_start_method",
    "encode_structure",
    "get_worker_context",
    "init_structure_worker",
    "run_worker_pool",
    "seed_all",
    "select_device",
    "select_num_workers",
]
