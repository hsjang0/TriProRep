"""Re-compute homomer bond_type 5-class multilabel using PLIP directly
(Salentin et al. 2015, Nucleic Acids Research) instead of hand-coded
geometric criteria.

Treatment for homodimer:
    - Set config.PEPTIDES = ['B']: PLIP treats chain B as a peptide ligand
      and detects inter-chain interactions between chain A (protein) and
      chain B (ligand) residues.
    - For each interaction, PLIP reports both resnr (chain A side) and
      resnr_l (chain B side). Both residue positions get the corresponding
      bond_type bit set (OR aggregation across A_1 and A_2 chain copies),
      since the homodimer has identical sequence and label is per
      sequence position.

5-class layout (matches existing bond_type field):
    [hbond, salt_bridge, hydrophobic, pi_stack, cation_pi]

Usage:
    python \\
        enrich_bond_type_plip.py \\
        --target_pkl  data/homomer_target_set.pkl \\
        --holo_idx    data/holo_idx.pkl \\
        --out_pkl     data/homomer_target_set.pkl \\
        --workers     32
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np


N_BOND = 5
IDX_HBOND, IDX_SALT, IDX_HYDRO, IDX_PI, IDX_CATPI = 0, 1, 2, 3, 4


def _plip_one(pdb_path: str, L: int) -> np.ndarray:
    """Returns (L, 5) uint8 multilabel: per-residue inter-chain interactions
    detected by PLIP. OR-aggregated over chain A and chain B copies."""
    from plip.structure.preparation import PDBComplex
    from plip.basic import config
    config.PEPTIDES = ['B']
    config.NOPDBCANMAP = True

    c = PDBComplex()
    c.load_pdb(pdb_path)
    c.analyze()

    out = np.zeros((L, N_BOND), dtype=np.uint8)

    def _set(idx_class: int, resnr: int):
        if 1 <= resnr <= L:
            out[resnr - 1, idx_class] = 1

    for _, iset in c.interaction_sets.items():
        # Hydrogen bonds
        for hb in list(iset.hbonds_pdon) + list(iset.hbonds_ldon):
            _set(IDX_HBOND, hb.resnr)
            _set(IDX_HBOND, hb.resnr_l)
        # Salt bridges
        for sb in list(iset.saltbridge_lneg) + list(iset.saltbridge_pneg):
            _set(IDX_SALT, sb.resnr)
            _set(IDX_SALT, sb.resnr_l)
        # Hydrophobic contacts
        for hc in iset.hydrophobic_contacts:
            _set(IDX_HYDRO, hc.resnr)
            _set(IDX_HYDRO, hc.resnr_l)
        # π-stacking (pi-pi)
        for ps in iset.pistacking:
            _set(IDX_PI, ps.resnr)
            _set(IDX_PI, ps.resnr_l)
        # cation-π (both protein-cation and ligand-cation)
        for pc in list(iset.pication_paro) + list(iset.pication_laro):
            _set(IDX_CATPI, pc.resnr)
            _set(IDX_CATPI, pc.resnr_l)

    return out


def _compute_one(args):
    pid, holo_path, L = args
    if holo_path is None or not Path(holo_path).exists():
        return pid, None, 'no_path'
    try:
        # Silence PLIP logging
        logging.getLogger().setLevel(logging.ERROR)
        for name in ('plip', 'plip.structure.preparation',
                     'plip.basic.config', 'plip.exchange.report'):
            logging.getLogger(name).setLevel(logging.ERROR)
        out = _plip_one(str(holo_path), L)
        return pid, out, 'ok'
    except Exception as exc:
        return pid, None, f'fail:{type(exc).__name__}:{exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_pkl', type=Path, required=True)
    ap.add_argument('--holo_idx', type=Path, required=True)
    ap.add_argument('--out_pkl', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard_idx', type=int, default=0,
                    help='this shard\'s index in [0, n_shards)')
    ap.add_argument('--n_shards', type=int, default=1,
                    help='split pids into N shards. Each shard processes '
                         'pids[shard_idx::n_shards]. Output is partial dict '
                         '{pid: array} only — merge step required after all '
                         'shards finish.')
    ap.add_argument('--partial_only', action='store_true',
                    help='output only {pid: bond_type_plip} dict instead of '
                         'full augmented target pkl. Used by sharded runs.')
    args = ap.parse_args()

    print(f'[load] target_pkl = {args.target_pkl}', flush=True)
    with open(args.target_pkl, 'rb') as f:
        targets = pickle.load(f)
    print(f'  N pids = {len(targets)}', flush=True)

    print(f'[load] holo_idx = {args.holo_idx}', flush=True)
    with open(args.holo_idx, 'rb') as f:
        holo_idx = pickle.load(f)

    pids = list(targets.keys())
    if args.limit:
        pids = pids[:args.limit]
    if args.n_shards > 1:
        pids = pids[args.shard_idx::args.n_shards]
        print(f'[shard] {args.shard_idx+1}/{args.n_shards} → '
              f'{len(pids)} pids', flush=True)
    work = [(p, holo_idx.get(p), targets[p]['L']) for p in pids]
    print(f'[run] PLIP on {len(work)} pids with {args.workers} workers',
          flush=True)

    n_ok = n_fail = 0
    fail_kinds: dict[str, int] = {}
    progress_every = max(1, len(work) // 100)

    with mp.Pool(args.workers) as pool:
        for i, (pid, arr, status) in enumerate(
            pool.imap_unordered(_compute_one, work, chunksize=4)
        ):
            if status == 'ok':
                # Replace existing bond_type (hand-coded geometric) with PLIP output
                targets[pid]['bond_type_plip'] = arr
                n_ok += 1
            else:
                n_fail += 1
                kind = status.split(':', 1)[0]
                fail_kinds[kind] = fail_kinds.get(kind, 0) + 1
            if (i + 1) % progress_every == 0:
                pct = 100 * (i + 1) / len(work)
                print(f'  {i+1}/{len(work)} ({pct:.1f}%)  ok={n_ok}  fail={n_fail}',
                      flush=True)

    print(f'\n[done] ok={n_ok}  fail={n_fail}  fail_kinds={fail_kinds}', flush=True)

    if n_ok > 0:
        names = ['hbond', 'salt_bridge', 'hydrophobic', 'pi_stack', 'cation_pi']
        n_total = 0
        n_pos = np.zeros(N_BOND, dtype=np.int64)
        for pid, d in targets.items():
            arr = d.get('bond_type_plip')
            if arr is None: continue
            arr = np.asarray(arr).astype(np.int64)
            n_total += arr.shape[0]
            n_pos += arr.sum(axis=0)
        print(f'[stats] total residues = {n_total:,}')
        for k, name in enumerate(names):
            print(f'  {name:>14s}: {int(n_pos[k]):>10,}  '
                  f'({100*n_pos[k]/max(1,n_total):.3f} %)')

    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out_pkl.with_suffix(args.out_pkl.suffix + '.tmp')
    if args.partial_only:
        # write only {pid: bond_type_plip} dict for this shard
        partial = {pid: d['bond_type_plip']
                   for pid, d in targets.items()
                   if 'bond_type_plip' in d}
        print(f'\n[save partial] {args.out_pkl}  ({len(partial)} pids)',
              flush=True)
        with open(tmp, 'wb') as f:
            pickle.dump(partial, f)
    else:
        print(f'\n[save full] {args.out_pkl}', flush=True)
        with open(tmp, 'wb') as f:
            pickle.dump(targets, f)
    tmp.replace(args.out_pkl)
    sz = args.out_pkl.stat().st_size / 1e6
    print(f'  saved ({sz:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
