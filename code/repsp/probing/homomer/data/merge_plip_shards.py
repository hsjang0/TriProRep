"""Merge sharded PLIP partial pkls into the target pkl with bond_type_plip.

Each shard pkl is {pid: bond_type_plip ndarray}. We load all shards, merge
into the target pkl as a new field 'bond_type_plip', and save the augmented pkl.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_pkl', type=Path, required=True)
    ap.add_argument('--shards_dir', type=Path, required=True)
    ap.add_argument('--out_pkl', type=Path, required=True)
    args = ap.parse_args()

    print(f'[load] target_pkl = {args.target_pkl}', flush=True)
    with open(args.target_pkl, 'rb') as f:
        targets = pickle.load(f)
    print(f'  N pids = {len(targets)}', flush=True)

    shard_pkls = sorted(args.shards_dir.glob('shard_*.pkl'))
    print(f'[load] {len(shard_pkls)} shard pkls from {args.shards_dir}',
          flush=True)
    merged = {}
    for sp in shard_pkls:
        with open(sp, 'rb') as f:
            d = pickle.load(f)
        before = len(merged)
        merged.update(d)
        added = len(merged) - before
        print(f'  {sp.name}: {len(d)} pids ({added} new, {len(d)-added} dup)',
              flush=True)
    print(f'[merged] total {len(merged)} unique pids')

    n_added = 0
    for pid, arr in merged.items():
        if pid not in targets:
            continue
        targets[pid]['bond_type_plip'] = arr
        n_added += 1
    print(f'[merge] {n_added} pids augmented with bond_type_plip')

    # stats
    names = ['hbond', 'salt_bridge', 'hydrophobic', 'pi_stack', 'cation_pi']
    n_total = 0
    n_pos = np.zeros(5, dtype=np.int64)
    for pid, d in targets.items():
        arr = d.get('bond_type_plip')
        if arr is None: continue
        arr = np.asarray(arr).astype(np.int64)
        n_total += arr.shape[0]
        n_pos += arr.sum(axis=0)
    print(f'\n[stats] total residues = {n_total:,}')
    for k, name in enumerate(names):
        print(f'  {name:>14s}: {int(n_pos[k]):>10,}  ({100*n_pos[k]/max(1,n_total):.3f} %)')

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_pkl.with_suffix(args.out_pkl.suffix + '.tmp')
    print(f'\n[save] {args.out_pkl}', flush=True)
    with open(tmp, 'wb') as f:
        pickle.dump(targets, f)
    tmp.replace(args.out_pkl)
    sz = args.out_pkl.stat().st_size / 1e6
    print(f'  saved ({sz:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
