"""
Utility functions for evaluation and metrics.
"""
import torch
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
import torchmetrics as tm


def cknna_score(
    latent1: torch.Tensor,
    latent2: torch.Tensor,
    k: float = 0.1
):
    """
    Compute CKNNA (Cross-K-Nearest Neighbors Alignment) score.
    
    Args:
        latent1: First set of embeddings [N, D]
        latent2: Second set of embeddings [N, D]
        k: Fraction of neighbors to consider (default: 0.1)
    
    Returns:
        CKNNA score (float)
    """
    latent1 = F.normalize(latent1, dim=-1)
    latent2 = F.normalize(latent2, dim=-1)
    bs, d = latent1.shape
    match_k = int(bs * k)
    
    sim1 = latent1 @ latent1.T
    sim2 = latent2 @ latent2.T
    diag = torch.arange(bs, device=latent1.device)
    sim1[diag, diag] = -torch.inf
    sim2[diag, diag] = -torch.inf
    _, topk_1 = sim1.topk(match_k, dim=-1)
    _, topk_2 = sim2.topk(match_k, dim=-1)
    
    neigh_1 = torch.zeros(bs, bs, dtype=torch.bool, device=latent1.device)
    neigh_2 = torch.zeros_like(neigh_1)
    row_idx = torch.arange(bs, device=latent1.device).unsqueeze(-1).expand(-1, match_k)
    neigh_1[row_idx, topk_1] = True
    neigh_2[row_idx, topk_2] = True
    
    inter_counts = (neigh_1 & neigh_2).sum(dim=-1)
    overlaps = inter_counts.float() / float(match_k)
    cknna_score = overlaps.mean().item()
    
    return cknna_score


def compute_fmax(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    F1 score with the optimal threshold, Copied from TorchDrug.
    
    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.
    
    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    
    Returns:
        Fmax score (float)
    """
    pred = torch.sigmoid(pred)
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)
    
    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return {"fmax": all_f1.max().item()}


def compute_micro_fmax(pred: torch.Tensor, target: torch.Tensor, num_thresholds: int = 100) -> float:
    """
    Compute micro-averaged Fmax score via threshold sweeping.
    
    Micro-averaging pools all (protein, term) pairs together:
    - TP(t) = sum over all (i,c) where pred[i,c] >= t AND target[i,c] == 1
    - FP(t) = sum over all (i,c) where pred[i,c] >= t AND target[i,c] == 0
    - FN(t) = sum over all (i,c) where pred[i,c] < t AND target[i,c] == 1
    
    Then: P(t) = TP(t) / (TP(t) + FP(t)), R(t) = TP(t) / (TP(t) + FN(t))
    Fmax = max_t [2 * P(t) * R(t) / (P(t) + R(t))]
    
    Args:
        pred: Predictions [N, C] in [0, 1] (probabilities)
        target: Binary targets [N, C] in {0, 1}
        num_thresholds: Number of thresholds to sweep (default: 100)
    
    Returns:
        Fmax score (float)
    """
    pred = torch.sigmoid(pred)
    pred = pred.flatten()  # [N*C]
    target = target.flatten().float()  # [N*C]
    
    # Generate thresholds
    thresholds = torch.linspace(0.0, 1.0, num_thresholds, device=pred.device)
    
    best_f1 = 0.0
    
    for t in thresholds:
        # Binary predictions at threshold t
        pred_binary = (pred >= t).float()
        
        # Compute TP, FP, FN (micro-averaged over all pairs)
        tp = ((pred_binary == 1) & (target == 1)).sum().float()
        fp = ((pred_binary == 1) & (target == 0)).sum().float()
        fn = ((pred_binary == 0) & (target == 1)).sum().float()
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        # Compute F1
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        if f1 > best_f1:
            best_f1 = f1
    
    return {"micro_fmax": best_f1.item()}


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    device = logits.device
    acc_fn = tm.Accuracy(task="binary").to(device)

    preds = torch.sigmoid(logits)
    pred_labels = (preds > 0.5).long()
    acc = acc_fn(pred_labels, labels)
    return {"acc": acc.item()}


def compute_regression_metrics(logits: torch.Tensor, labels: torch.Tensor):
    device = logits.device
    mse_fn = tm.MeanSquaredError().to(device)
    spearman_fn = tm.SpearmanCorrCoef().to(device)
    r2_fn = tm.R2Score().to(device)
    pearson_fn = tm.PearsonCorrCoef().to(device)
    
    logits = logits.float()
    labels = labels.float()

    mse = mse_fn(logits, labels)
    spearman = spearman_fn(logits, labels)
    r2 = r2_fn(logits, labels)
    pearson = pearson_fn(logits, labels)
    return {
        "mse": mse.item(),
        "spearman": spearman.item(),
        "r2": r2.item(),
        "pearson": pearson.item()
    }


"""
Code adopted from ProteinGym:
https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_DMS_benchmarks.py#L213
"""
def minmax_np(x):
    return ( (x - np.min(x)) / (np.max(x) - np.min(x)) ) 


def calc_ndcg(y_true, y_score): # benign-focused comparison
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    top = 10
    k = int(np.floor(len(y_true) * (top / 100.0)))

    gains = minmax_np(y_true)
    ranks = np.argsort(np.argsort(-y_score)) + 1

    mask_k = ranks <= k
    ranks_k = ranks[mask_k]
    gains_k = gains[mask_k]

    mask_nonzero = gains_k != 0
    ranks_fil = ranks_k[mask_nonzero]
    gains_fil = gains_k[mask_nonzero]

    # if none of the ranks made it return 0
    if len(ranks_fil) == 0:
        return 0.0

    dcg = np.sum([g / np.log2(r + 1) for r, g in zip(ranks_fil, gains_fil)])

    ideal_ranks = np.argsort(np.argsort(-gains)) + 1
    mask_k_ideal = ideal_ranks <= k
    ideal_ranks_k = ideal_ranks[mask_k_ideal]
    ideal_gains_k = gains[mask_k_ideal]

    mask_nonzero_ideal = ideal_gains_k != 0
    ideal_ranks_fil = ideal_ranks_k[mask_nonzero_ideal]
    ideal_gains_fil = ideal_gains_k[mask_nonzero_ideal]

    if len(ideal_ranks_fil) == 0:
        return 0.0

    idcg = np.sum([g / np.log2(r + 1) for r, g in zip(ideal_ranks_fil, ideal_gains_fil)])

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def calc_toprecall(y_true, y_score, top_true=10, top_model=10):  
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    
    top_true = (y_true >= np.percentile(y_true, 100-top_true))
    top_model = (y_score >= np.percentile(y_score, 100-top_model))
    
    TP = (top_true) & (top_model)
    recall = TP.sum() / (top_true.sum()) if top_true.sum() > 0 else 0
    
    return (recall)


def _format_atom_name(atom_name: str, element: str | None) -> str:
    """
    PDB atom-name alignment rules are a little quirky.
    This heuristic works well for typical protein atoms (N, CA, C, O, CB, ...).
    """
    atom_name = (atom_name or "").strip()
    element = (element or "").strip()

    # If element is 1 char (C, N, O, S, H), atom name is usually right-justified in 4 cols
    # e.g., " N  ", " CA ", " CB "
    if len(element) == 1:
        return f"{atom_name:>4}"[:4]
    # If element is 2 chars (FE, ZN, CL...), left-justify often works better
    return f"{atom_name:<4}"[:4]


def dataframe_to_pdb(
    df: pd.DataFrame,
    output_pdb: str,
    *,
    record_name: str = "ATOM",  # could be "HETATM" if you want
    write_models: bool = True,
    occupancy: float = 1.00,
    bfactor: float = 0.00,
):
    required = ["aid", "atom_name", "resname", "chain", "residue", "x", "y", "z", "element"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure stable ordering (PDBs are typically ordered by model, chain, residue, aid)
    sort_cols = [c for c in ["model", "chain", "residue", "aid"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    has_model = "model" in df.columns
    col_idx = {name: df.columns.get_loc(name) for name in df.columns}
    values = df.to_numpy()
    lines: list[str] = []

    with open(output_pdb, "w") as f:
        current_model = None

        for row in values:
            model = int(row[col_idx["model"]]) if has_model and pd.notna(row[col_idx["model"]]) else 0

            if write_models and (current_model is None or model != current_model):
                if current_model is not None:
                    lines.append("ENDMDL\n")
                current_model = model
                # PDB MODEL numbers are typically 1-based
                lines.append(f"MODEL     {current_model + 1:4d}\n")

            serial = int(row[col_idx["aid"]])
            residue_raw = str(row[col_idx["residue"]]).strip()
            residue_match = re.match(r"(-?\d+)([A-Za-z]?)", residue_raw)
            if not residue_match:
                raise ValueError(f"Invalid residue value: {residue_raw!r}")
            resseq = int(residue_match.group(1))
            icode = residue_match.group(2) or " "
            chain_raw = row[col_idx["chain"]]
            chain = str(chain_raw)[0] if pd.notna(chain_raw) and str(chain_raw) else "A"
            resname = str(row[col_idx["resname"]]).strip()[:3].upper()
            element = str(row[col_idx["element"]]).strip().upper()[:2]
            atom_name = _format_atom_name(str(row[col_idx["atom_name"]]), element)

            x = float(row[col_idx["x"]]); y = float(row[col_idx["y"]]); z = float(row[col_idx["z"]])

            # PDB v3 ATOM/HETATM format (classic fixed-width)
            # Columns:  1-6  record, 7-11 serial, 13-16 atom, 18-20 resname,
            #          22 chain, 23-26 resseq, 31-38 x, 39-46 y, 47-54 z,
            #          55-60 occ, 61-66 bfac, 77-78 element
            line = (
                f"{record_name:<6}"
                f"{serial:5d} "
                f"{atom_name}"
                f" "
                f"{resname:>3} "
                f"{chain:1}"
                f"{resseq:4d}"
                f"{icode:1}   "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"{occupancy:6.2f}{bfactor:6.2f}"
                f"          "
                f"{element:>2}\n"
            )
            lines.append(line)

        if write_models and current_model is not None:
            lines.append("ENDMDL\n")
        lines.append("END\n")
        f.writelines(lines)
