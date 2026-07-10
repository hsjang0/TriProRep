"""
Shared utilities for homomer per-residue probes.

Core data formats
-----------------
**Per-encoder flat .pt** (from `extract_probing_features.py`):
  {emb_dir}/{train,valid,test}.pt = {
      'X':          fp16 [N_res, D],
      'pid_slices': list[(pid, start, end)],     # X[start:end] = residues of `pid`
  }
(no Y inside — labels live in a separate target pkl.
 D / n_proteins / n_res are derivable: X.shape[1] / len(pid_slices) / X.shape[0].)

**Homomer target pkl** (from `data/build_homomer_targets.py`):
  {pdb_id: {
      'L': int,
      'seq': str,
      'binding_site':           uint8 [L],
      'delta_sasa_mean':        float32 [L],
      'delta_sasa_max':         float32 [L],
      'levy_tier':              uint8 [L]   # 0=surface,1=interior,2=support,3=rim,4=core
      'levy_tier_mean_rank':    float32 [L] # ordinal regression target
      'bond_type':              uint8 [L, 5]  # multi-hot: hbond, salt, hydro, π-π, cation-π
      'rsasa_apo':              float32 [L]   # auxiliary regression target
  }}

**Split** (probing-side, NOT folding's):
  data/splits/{train,valid,test}.txt = list of pdb_ids per split
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Project root for `utils.compute_fmax`
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Targets / split / flat loading
# ---------------------------------------------------------------------------
def load_homomer_targets(pkl_path: str | Path) -> dict:
    """Returns {pdb_id: per-task arrays} from build_homomer_targets.py output."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_split_pids(splits_dir: str | Path) -> dict[str, list[str]]:
    """Returns {'train': [...], 'valid': [...], 'test': [...]}.

    Accepts the released upload naming (`splits_<split>.txt`) and the
    canonical fallback (`<split>.txt`) — first existing wins per split.
    """
    base = Path(splits_dir)
    out: dict[str, list[str]] = {}
    for s in ("train", "valid", "test"):
        for candidate in (base / f"{s}.txt", base / f"splits_{s}.txt"):
            if candidate.exists():
                out[s] = [l.strip() for l in candidate.read_text().splitlines() if l.strip()]
                break
        else:
            out[s] = []
    return out


def load_flat_split(emb_dir: Path, split: str):
    """Loads {emb_dir}/{split}.pt produced by `extract_probing_features.py`.

    Returns (X, pid_slices). Y is intentionally NOT loaded — task-specific
    labels are pulled from the target pkl at probe time.
    """
    blob = torch.load(emb_dir / f"{split}.pt", map_location="cpu")
    if "X" not in blob or "pid_slices" not in blob:
        raise RuntimeError(
            f"{emb_dir / f'{split}.pt'} is not in flat format "
            "(expected X + pid_slices). Re-run extract_probing_features.py.")
    return blob["X"], blob["pid_slices"]


# ---------------------------------------------------------------------------
# Per-task target gathering
# ---------------------------------------------------------------------------
TASK_KIND = {
    "binding_site":          "binary",
    "delta_sasa_mean":       "regression",
    "delta_sasa_max":        "regression",
    "levy_tier":             "multiclass",   # 5-class
    "levy_tier_mean_rank":   "regression",   # ordinal
    # Per-residue inter-chain interaction types, 5-class multilabel, computed
    # with PLIP (Salentin 2015; chain B treated as the peptide ligand). This
    # is the benchmark bond task.
    "bond_type_plip":        "multilabel",   # 5 classes — PLIP (benchmark)
    "bond_type":             "multilabel",   # 5 classes — legacy geometric (superseded by bond_type_plip)
    "rsasa_apo":             "regression",
}
TASK_NUM_CLASSES = {
    "levy_tier": 5,
    "bond_type_plip": 5,
    "bond_type": 5,
}
BOND_TYPE_NAMES = ["hbond", "salt_bridge", "hydrophobic", "pi_stack", "cation_pi"]
LEVY_TIER_NAMES = ["surface", "interior", "support", "rim", "core"]


def build_split_xy(X, pid_slices, targets, task, split_pids=None):
    """Return concatenated (X_split, Y_split) for the given task.

    pid_slices:    list[(pid, start, end)] — X[start:end] = residues of `pid`
    targets:       {pid: {task: array, ...}} (full pkl, will be filtered)
    task:          one of TASK_KIND keys
    split_pids:    optional set of pids; otherwise use all in pid_slices

    Y_split shape:
      binary / regression :      [N_res]
      multiclass          :      [N_res] (int64)
      multilabel          :      [N_res, K]
    """
    if task not in TASK_KIND:
        raise ValueError(f"unknown task: {task}")
    kind = TASK_KIND[task]
    keep = set(split_pids) if split_pids is not None else None
    X_list, Y_list = [], []
    miss_pid = miss_target = miss_len = 0
    for pid, s, e in pid_slices:
        if keep is not None and pid not in keep:
            continue
        rec = targets.get(pid)
        if rec is None:
            miss_pid += 1
            continue
        y = rec.get(task)
        if y is None:
            miss_target += 1
            continue
        L = e - s
        y_arr = np.asarray(y)
        if y_arr.shape[0] != L:
            miss_len += 1
            continue
        X_list.append(X[s:e])
        Y_list.append(torch.from_numpy(y_arr))
    if not X_list:
        raise RuntimeError(f"no aligned residues for task={task}")

    Xc = torch.cat(X_list, dim=0)
    Yc = torch.cat(Y_list, dim=0)

    # cast Y to torch types matching loss
    if kind == "binary":
        Yc = Yc.float()
    elif kind == "regression":
        Yc = Yc.float()
    elif kind == "multiclass":
        Yc = Yc.long()
    elif kind == "multilabel":
        Yc = Yc.float()

    print(f"  [build] task={task} kind={kind}  "
          f"residues={Xc.shape[0]} D={Xc.shape[1]} "
          f"miss_pid={miss_pid} miss_target={miss_target} miss_len={miss_len}",
          flush=True)
    return Xc, Yc, kind


# ---------------------------------------------------------------------------
# Metrics — PPI / regression / multiclass / multilabel field standards
# ---------------------------------------------------------------------------
def auprc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Trapezoidal AUPRC (sklearn-free)."""
    s = scores.detach().cpu().numpy().ravel()
    y = targets.detach().cpu().numpy().ravel().astype(np.int32)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    pre = tp / np.maximum(tp + fp, 1)
    rec = tp / max(y.sum(), 1)
    rec = np.concatenate(([0.0], rec))
    pre = np.concatenate(([1.0], pre))
    return float(np.sum((rec[1:] - rec[:-1]) * pre[1:]))


def auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Trapezoidal ROC-AUC (sklearn-free)."""
    s = scores.detach().cpu().numpy().ravel()
    y = targets.detach().cpu().numpy().ravel().astype(np.int32)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y_s = y[order]
    tpr = np.cumsum(y_s) / n_pos
    fpr = np.cumsum(1 - y_s) / n_neg
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))


def _spearman(p: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (sklearn-free)."""
    if len(p) < 2:
        return 0.0
    rp = np.argsort(np.argsort(p))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rp, ry)[0, 1])


def metrics_binary(logits: torch.Tensor, y: torch.Tensor) -> dict:
    """NaN-aware binary metrics (positions with non-finite targets are dropped)."""
    yf = y.float()
    valid = torch.isfinite(yf)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "acc@0.5": float("nan"), "pos_rate": float("nan"),
                "n_valid": 0}
    logits_v = logits[valid]
    yv = yf[valid]
    return {
        "auroc": auroc(logits_v, yv),
        "auprc": auprc(logits_v, yv),
        "acc@0.5": ((torch.sigmoid(logits_v) >= 0.5).float() == yv).float().mean().item(),
        "pos_rate": yv.mean().item(),
        "n_valid": n_valid,
    }


def metrics_multiclass(logits: torch.Tensor, y: torch.Tensor, K: int) -> dict:
    """Top-1 accuracy + per-class F1 + macro F1.
    Skips positions where y == -100 (CrossEntropyLoss ignore_index convention).
    """
    valid = (y != -100)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return {"acc": float("nan"), "macro_f1": float("nan"), "n_valid": 0}
    logits_v = logits[valid]
    y_v = y[valid]
    pred = logits_v.argmax(dim=-1)
    out = {"acc": (pred == y_v).float().mean().item(), "n_valid": n_valid}
    f1s = []
    for k in range(K):
        tp = ((pred == k) & (y_v == k)).sum().item()
        fp = ((pred == k) & (y_v != k)).sum().item()
        fn = ((pred != k) & (y_v == k)).sum().item()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        out[f"f1_class{k}"] = f1
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def metrics_regression(pred: torch.Tensor, y: torch.Tensor) -> dict:
    p = pred.detach().float().cpu().numpy().ravel()
    yn = y.detach().float().cpu().numpy().ravel()
    # NaN-aware: drop positions where target is NaN. For the dense regression
    # tasks (delta_sasa_*, rsasa_apo) this mask is a no-op.
    valid = np.isfinite(yn)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return {"mse": float("nan"), "mae": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"),
                "n_valid": n_valid}
    p, yn = p[valid], yn[valid]
    mse = float(((p - yn) ** 2).mean())
    mae = float(np.abs(p - yn).mean())
    var = float(yn.var())
    r2 = 1 - mse / max(var, 1e-9)
    pearson = float(np.corrcoef(p, yn)[0, 1]) if len(p) > 1 else 0.0
    spearman = _spearman(p, yn)
    return {"mse": mse, "mae": mae, "r2": r2,
            "pearson": pearson, "spearman": spearman,
            "n_valid": n_valid}


def metrics_multilabel(logits: torch.Tensor, y: torch.Tensor, K: int,
                       class_names=None) -> dict:
    """Per-class AUPRC + AUROC + per-class pos_rate. Mean over present classes."""
    out = {}
    aupr_list, auroc_list = [], []
    for k in range(K):
        nm = class_names[k] if class_names else f"class{k}"
        ap = auprc(logits[:, k], y[:, k])
        ar = auroc(logits[:, k], y[:, k])
        out[f"auprc_{nm}"] = ap
        out[f"auroc_{nm}"] = ar
        out[f"pos_rate_{nm}"] = y[:, k].float().mean().item()
        if not np.isnan(ap):
            aupr_list.append(ap)
        if not np.isnan(ar):
            auroc_list.append(ar)
    out["mean_auprc"] = float(np.mean(aupr_list)) if aupr_list else float("nan")
    out["mean_auroc"] = float(np.mean(auroc_list)) if auroc_list else float("nan")
    return out


def compute_metrics(logits, y, kind: str, K: int = 1, class_names=None):
    if kind == "binary":
        return metrics_binary(logits, y)
    if kind == "multiclass":
        return metrics_multiclass(logits, y, K)
    if kind == "regression":
        return metrics_regression(logits, y)
    if kind == "multilabel":
        return metrics_multilabel(logits, y, K, class_names)
    raise ValueError(kind)


def primary_score(metrics: dict, kind: str) -> float:
    """Single scalar for 'best epoch' selection + headline reporting.

      binary       → AUPRC      (robust to class imbalance, common in PPI)
      regression   → Pearson    (interpretable, paired with Spearman/R² in JSON)
      multiclass   → macro_F1   (handles imbalance across 5 Levy tiers)
      multilabel   → mean_AUPRC (sparse positive multilabel)
    """
    if kind == "binary":
        return metrics["auprc"]
    if kind == "multiclass":
        return metrics["macro_f1"]
    if kind == "regression":
        return metrics["pearson"]
    if kind == "multilabel":
        return metrics["mean_auprc"]
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# wandb / persistence / argparse — kept as-is for new probe scripts to use
# ---------------------------------------------------------------------------
def setup_wandb(args):
    if not getattr(args, "wandb_project", None):
        return None
    import wandb
    # --wandb_tags accepts a comma-separated string and splits to a list.
    raw_tags = getattr(args, "wandb_tags", None)
    tags = [t.strip() for t in raw_tags.split(",")] if raw_tags else None
    # Optional run-name override (encoder_task style).
    name = getattr(args, "wandb_run_name", None) or args.run_name
    return wandb.init(project=args.wandb_project,
                      entity=args.wandb_entity,
                      name=name,
                      tags=tags,
                      config=vars(args))


def finalize_wandb(wandb_run, results: dict, primary_key: str | None = None):
    if wandb_run is None:
        return
    summary = {f"final/{k}": v for k, v in results.items()
               if isinstance(v, (int, float))}
    if primary_key is not None and primary_key in results:
        summary["primary"] = results[primary_key]
    wandb_run.summary.update(summary)
    wandb_run.finish()


def save_results_json(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _round(v):
        return round(v, 4) if isinstance(v, float) else v
    walked = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            walked[k] = {k2: _round(v2) for k2, v2 in v.items()}
        else:
            walked[k] = _round(v)
    with open(out_path, "w") as f:
        json.dump(walked, f, indent=2)


def add_common_args(p):
    p.add_argument("--target_pkl", required=True,
                   help="Probing labels pkl (per-residue targets for the 4 tasks).")
    p.add_argument("--features_dir", required=True,
                   help="Dir with {train,valid,test}.pt produced by "
                        "extract_probing_features.py")
    p.add_argument("--task", required=True,
                   choices=list(TASK_KIND.keys()))
    p.add_argument("--probe_epochs", type=int, default=10)
    p.add_argument("--probe_lr", type=float, default=5e-4)
    p.add_argument("--probe_batch_size", type=int, default=16824,
                   help="Residue mini-batch size (benchmark default).")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--autocast", default="bf16",
                   choices=["bf16", "fp16", "off"])
    p.add_argument("--run_name", required=True)
    p.add_argument("--results_dir",
                   default="./results")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None,
                   help="Your W&B entity / team.")
    p.add_argument("--wandb_tags", default=None,
                   help="Comma-separated tags, e.g. 'ep10,task_binding_site'")
    p.add_argument("--wandb_run_name", default=None,
                   help="Override wandb run display name (default: --run_name)")
    return p


def autocast_dtype(name: str):
    return {"bf16": torch.bfloat16,
            "fp16": torch.float16,
            "off": torch.float32}[name]
