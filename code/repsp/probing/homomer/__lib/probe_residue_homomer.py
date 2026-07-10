#!/usr/bin/env python
"""
Task-aware per-residue probe for homomer-defined targets.

Architecture: 2-hidden-layer MLP head, no input LayerNorm, hidden=1280.
Output dim auto-set by task:
  binary       : 1   (sigmoid)
  multiclass K : K   (softmax)
  regression   : 1
  multilabel K : K   (per-class sigmoid)

Loss / metric also dispatch by task (see _homomer_common.compute_metrics).

Usage:
  python probe_residue_homomer.py \
      --target_pkl    .../probing_labels.pkl \
      --features_dir  .../<encoder>/   # contains train.pt / valid.pt / test.pt
      --task          binding_site | delta_sasa_mean | levy_tier | ...
      --run_name      <encoder>_<task>
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from _homomer_common import (
    add_common_args, autocast_dtype,
    BOND_TYPE_NAMES, LEVY_TIER_NAMES,
    build_split_xy, compute_metrics, finalize_wandb,
    load_flat_split, load_homomer_targets,
    primary_score, save_results_json, setup_wandb,
    TASK_KIND, TASK_NUM_CLASSES,
)


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """No input LayerNorm. n_hidden hidden Linear+GELU+Dropout blocks → output.

    n_hidden=2 (default): D_in → H → H → D_out
    n_hidden=1 (ablation):          D_in → H → D_out
    n_hidden=0 (linear probe):      D_in → D_out
    """
    def __init__(self, D_in, D_hidden=1280, D_out=1, dropout=0.1, n_hidden=2):
        super().__init__()
        layers = []
        prev = D_in
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, D_hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = D_hidden
        layers.append(nn.Linear(prev, D_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------
def _make_loss(kind: str, pos_weight: torch.Tensor | None = None):
    if kind == "binary":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if kind == "multiclass":
        # ignore_index=-100 lets a task flag positions to skip in the loss
        # (e.g. undefined / endpoint residues) without affecting training.
        return nn.CrossEntropyLoss(ignore_index=-100)
    if kind == "regression":
        return nn.MSELoss()
    if kind == "multilabel":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    raise ValueError(kind)


def _maybe_pin(t):
    if t is None:
        return None
    try:
        return t.contiguous().pin_memory()
    except Exception:
        return t.contiguous()


@torch.no_grad()
def _stream_predict(head, x_cpu, bs, device, ac_dtype, kind):
    out = []
    for s in range(0, x_cpu.shape[0], bs):
        e = min(s + bs, x_cpu.shape[0])
        x_mb = x_cpu[s:e].to(device, non_blocking=True).float()
        with torch.autocast(device_type="cuda", dtype=ac_dtype):
            o = head(x_mb)
        if kind == "binary" or kind == "regression":
            o = o.squeeze(-1)
        out.append(o.float().detach().cpu())
    return torch.cat(out, dim=0)


def run_probe(train, val, test, device, kind, K, num_epochs, lr, batch_size,
              eval_every, weight_decay, ac_dtype, hidden_dim, dropout,
              n_hidden=2, wandb_run=None, pos_weight_mode="none"):
    tx, ty = train
    sx, sy = test
    vx, vy = (val if val is not None else (None, None))
    D = tx.shape[1]

    if kind == "binary":
        D_out = 1
    elif kind == "regression":
        D_out = 1
    else:
        D_out = K

    head = MLPHead(D, D_hidden=hidden_dim, D_out=D_out, dropout=dropout,
                   n_hidden=n_hidden).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  [probe] D={D} hidden={hidden_dim} n_hidden={n_hidden} D_out={D_out} "
          f"kind={kind} head_params={n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=weight_decay)

    # Optional positive class upweighting for imbalanced binary / multilabel.
    pos_weight_t = None
    if kind in ("binary", "multilabel") and pos_weight_mode != "none":
        if kind == "binary":
            ty_finite = ty[torch.isfinite(ty)]
            n_pos = float((ty_finite == 1).sum().item())
            n_neg = float((ty_finite == 0).sum().item())
        else:  # multilabel: per-class
            n_pos = (ty == 1).float().sum(dim=0)
            n_neg = (ty == 0).float().sum(dim=0)
        if pos_weight_mode == "auto":
            if kind == "binary":
                w = max(1.0, n_neg / max(1.0, n_pos))
                pos_weight_t = torch.tensor(w, device=device, dtype=torch.float32)
                print(f"  [probe] pos_weight=auto → {w:.2f}  "
                      f"(n_pos={int(n_pos)}, n_neg={int(n_neg)})", flush=True)
            else:
                w = torch.maximum(torch.ones_like(n_pos), n_neg / torch.clamp(n_pos, min=1.0))
                pos_weight_t = w.to(device, dtype=torch.float32)
                print(f"  [probe] pos_weight=auto (per-class): {w.tolist()}", flush=True)
        else:
            try:
                w = float(pos_weight_mode)
            except ValueError:
                raise SystemExit(f"--pos_weight must be 'none', 'auto', or a float; got {pos_weight_mode}")
            pos_weight_t = torch.tensor(w, device=device, dtype=torch.float32)
            print(f"  [probe] pos_weight=fixed → {w}", flush=True)
    crit = _make_loss(kind, pos_weight=pos_weight_t)

    tx, ty = _maybe_pin(tx), ty.contiguous()  # ty stays on CPU until upload
    if vx is not None:
        vx, vy = _maybe_pin(vx), vy.contiguous()
    sx, sy = _maybe_pin(sx), sy.contiguous()
    sy_dev = sy.to(device)
    vy_dev = vy.to(device) if vy is not None else None

    N = tx.shape[0]
    if batch_size <= 0:
        batch_size = 16824

    best_val, best_epoch, best_test = -float("inf"), 0, None
    global_step = 0
    for epoch in range(num_epochs):
        head.train()
        perm = torch.randperm(N)
        for s in range(0, N, batch_size):
            idx = perm[s : s + batch_size]
            x_mb = tx[idx].to(device, non_blocking=True).float()
            y_mb = ty[idx].to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=ac_dtype):
                logits = head(x_mb)
                if kind == "binary" or kind == "regression":
                    logits = logits.squeeze(-1)
                if kind == "regression":
                    logits = logits.float()  # MSE in fp32 stable
                    # Mask non-finite targets. For the dense regression tasks
                    # (delta_sasa_*, rsasa_apo) this mask is a no-op.
                    mask = torch.isfinite(y_mb)
                    if not bool(mask.any()):
                        continue
                    loss = crit(logits[mask], y_mb[mask])
                elif kind == "binary":
                    # Mask non-finite binary targets. For dense binary tasks
                    # (binding_site) this is a no-op.
                    mask = torch.isfinite(y_mb)
                    if not bool(mask.any()):
                        continue
                    loss = crit(logits[mask], y_mb[mask])
                else:
                    loss = crit(logits, y_mb)
            loss.backward()
            opt.step()
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(),
                               "epoch": epoch + 1}, step=global_step)

        if vx is not None and ((epoch + 1) % eval_every == 0
                                or epoch == num_epochs - 1):
            head.eval()
            v_logits = _stream_predict(head, vx, batch_size, device, ac_dtype, kind)
            t_logits = _stream_predict(head, sx, batch_size, device, ac_dtype, kind)
            class_names = (BOND_TYPE_NAMES if kind == "multilabel" else None)
            v_m = compute_metrics(v_logits, vy_dev.cpu(), kind, K, class_names)
            t_m = compute_metrics(t_logits, sy_dev.cpu(), kind, K, class_names)
            v_score = primary_score(v_m, kind)
            t_score = primary_score(t_m, kind)
            print(f"  epoch={epoch+1:3d}  val={v_score:.4f}  test={t_score:.4f}",
                  flush=True)
            if wandb_run is not None:
                wandb_run.log({**{f"val/{k}": v for k, v in v_m.items()},
                               **{f"test/{k}": v for k, v in t_m.items()},
                               "epoch": epoch + 1}, step=global_step)
            if v_score > best_val:
                best_val = v_score
                best_epoch = epoch + 1
                best_test = t_m

    head.eval()
    final_logits = _stream_predict(head, sx, batch_size, device, ac_dtype, kind)
    class_names = (BOND_TYPE_NAMES if kind == "multilabel" else None)
    final = compute_metrics(final_logits, sy_dev.cpu(), kind, K, class_names)

    out = {f"final/{k}": v for k, v in final.items()}
    if best_test is not None:
        out["best_val"] = best_val
        out["best_epoch"] = best_epoch
        for k, v in best_test.items():
            out[f"best_val_test/{k}"] = v
    return out


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--hidden_dim", type=int, default=1280)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_hidden_layers", type=int, default=2,
                   help="Hidden Linear+GELU+Dropout blocks. 2=default (benchmark), "
                        "1 or 0=lighter probe ablations.")
    p.add_argument("--pos_weight", type=str, default="none",
                   help="BCE pos_weight for binary/multilabel. "
                        "'none' (default), 'auto' (=n_neg/n_pos from train), "
                        "or a fixed float.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for torch / numpy / python RNG (controls head "
                        "init + train shuffle ordering). default 0 is the "
                        "benchmark setting.")
    args = p.parse_args()

    # Set deterministic-ish seed across torch/numpy/python.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ac = autocast_dtype(args.autocast)
    wandb_run = setup_wandb(args)

    print(f"[probe] task={args.task}", flush=True)
    targets = load_homomer_targets(args.target_pkl)
    print(f"  {len(targets)} target entries", flush=True)

    features_dir = Path(args.features_dir)
    K = TASK_NUM_CLASSES.get(args.task, 1)

    # Build (X, Y) per split — the {split}.pt's `pid_slices` field already
    # carries the pid list for that split (set at extraction time), so no
    # external split file is needed here.
    def _split(s):
        X, pid_slices = load_flat_split(features_dir, s)
        return build_split_xy(X, pid_slices, targets, args.task, split_pids=None)

    print("[load] train", flush=True);   tX, tY, kind = _split("train")
    print("[load] valid", flush=True);   vX, vY, _    = _split("valid")
    print("[load] test",  flush=True);   sX, sY, _    = _split("test")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    res = run_probe((tX, tY), (vX, vY), (sX, sY), device, kind, K,
                    num_epochs=args.probe_epochs, lr=args.probe_lr,
                    batch_size=args.probe_batch_size,
                    eval_every=args.eval_every,
                    weight_decay=args.weight_decay,
                    ac_dtype=ac, hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    n_hidden=args.n_hidden_layers,
                    wandb_run=wandb_run,
                    pos_weight_mode=args.pos_weight)

    print(f"\n=====  {args.run_name}  task={args.task}  =====", flush=True)
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)

    payload = {
        "run_name": args.run_name,
        "task": args.task,
        "kind": kind,
        "features_dir": str(features_dir),
        "target_pkl": args.target_pkl,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_batch_size": args.probe_batch_size,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "autocast": args.autocast,
        **res,
    }
    save_results_json(
        Path(args.results_dir) / args.run_name / f"{args.task}.json", payload)
    print(f"\nSaved → {Path(args.results_dir) / args.run_name / f'{args.task}.json'}",
          flush=True)
    finalize_wandb(wandb_run, res, primary_key="best_val")


if __name__ == "__main__":
    main()
