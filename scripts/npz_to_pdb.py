#!/usr/bin/env python
"""Convert Boltz-style .npz structures to PDB files.

Reads:
    <apo_dir>/af-<afid>.npz    (monomer: one chain A)
    <holo_dir>/af-<afid>.npz   (homodimer: chains A + B)

Writes:
    <out_dir>/monomer/<AF-id>_monomer.pdb
    <out_dir>/homodimer/<AF-id>.pdb

Runs one process per split file. Resumable: existing output PDBs are skipped.

Usage:
    python scripts/npz_to_pdb.py \\
        --split /path/to/train_homodimer.txt \\
        --apo_dir /mnt/.../co-folding/full/apo_targets/structures \\
        --holo_dir /mnt/.../co-folding/full/holo_targets/structures \\
        --out_dir /mnt/.../REPSP_PDB \\
        --workers 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

# atomic-number -> element symbol (covers all standard biomolecule atoms)
_ELEMENT = {
    1:  "H",  5:  "B",  6:  "C",  7:  "N",  8:  "O",  9:  "F",
    11: "NA", 12: "MG", 15: "P",  16: "S",  17: "CL", 19: "K",
    20: "CA", 25: "MN", 26: "FE", 27: "CO", 28: "NI", 29: "CU",
    30: "ZN", 34: "SE", 35: "BR", 53: "I",
}


def _atom_name(name_bytes) -> str:
    """4-byte int8 array to ASCII atom name, e.g. [78,0,0,0] -> 'N'."""
    return bytes(int(b) for b in name_bytes if b > 0).decode("ascii", "replace")


def _fmt_atom_name(name: str) -> str:
    """PDB columns 13-16 name field with the goofy indentation rule."""
    # Element-symbol atoms start at column 14 unless the name is 4 chars.
    if len(name) >= 4:
        return name[:4]
    return f" {name:<3s}"


def _load(path: Path):
    with np.load(path, allow_pickle=True) as z:
        return {"atoms": z["atoms"], "residues": z["residues"], "chains": z["chains"]}


def _iter_pdb_lines(record):
    """Yield ATOM/TER/END lines for a Boltz .npz record."""
    atoms = record["atoms"]
    residues = record["residues"]
    chains = record["chains"]

    serial = 1
    for c in chains:
        chain_id = str(c["name"])[:1] or "A"
        c_start_res = int(c["res_idx"])
        c_num_res   = int(c["res_num"])
        for res in residues[c_start_res : c_start_res + c_num_res]:
            resname = str(res["name"])[:3]
            resseq  = int(res["res_idx"]) + 1
            a_start = int(res["atom_idx"])
            a_num   = int(res["atom_num"])
            for atom in atoms[a_start : a_start + a_num]:
                if not bool(atom["is_present"]):
                    continue
                aname = _atom_name(atom["name"])
                element = _ELEMENT.get(int(atom["element"]), "")
                x, y, z_ = (float(v) for v in atom["coords"])
                yield (
                    f"ATOM  {serial:5d} {_fmt_atom_name(aname)} {resname:>3s} "
                    f"{chain_id}{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z_:8.3f}"
                    f"{1.00:6.2f}{0.00:6.2f}          {element:>2s}\n"
                )
                serial += 1
        yield f"TER   {serial:5d}      {resname:>3s} {chain_id}{resseq:4d}\n"
        serial += 1
    yield "END\n"


def _one(args) -> tuple[str, str | None]:
    """Worker: (afid, apo_dir, holo_dir, out_dir) -> (afid, error_or_None)."""
    afid, apo_dir, holo_dir, out_dir = args
    apo_dir  = Path(apo_dir)
    holo_dir = Path(holo_dir)
    out_dir  = Path(out_dir)
    lower = afid.lower()

    homo_pdb = out_dir / "homodimer" / f"{afid}.pdb"
    mono_pdb = out_dir / "monomer"   / f"{afid}_monomer.pdb"

    try:
        if not mono_pdb.exists():
            src = apo_dir / f"{lower}.npz"
            if not src.exists():
                return afid, f"missing_apo:{lower}.npz"
            rec = _load(src)
            with open(mono_pdb.with_suffix(mono_pdb.suffix + ".tmp"), "w") as f:
                for line in _iter_pdb_lines(rec):
                    f.write(line)
            mono_pdb.with_suffix(mono_pdb.suffix + ".tmp").replace(mono_pdb)

        if not homo_pdb.exists():
            src = holo_dir / f"{lower}.npz"
            if not src.exists():
                return afid, f"missing_holo:{lower}.npz"
            rec = _load(src)
            with open(homo_pdb.with_suffix(homo_pdb.suffix + ".tmp"), "w") as f:
                for line in _iter_pdb_lines(rec):
                    f.write(line)
            homo_pdb.with_suffix(homo_pdb.suffix + ".tmp").replace(homo_pdb)
    except Exception as e:
        return afid, f"{type(e).__name__}: {e}"
    return afid, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path,
                    help="Text file with one AFid per line.")
    ap.add_argument("--apo_dir", required=True, type=Path)
    ap.add_argument("--holo_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    (args.out_dir / "monomer").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "homodimer").mkdir(parents=True, exist_ok=True)

    afids = [l.strip() for l in args.split.read_text().splitlines() if l.strip()]
    print(f"[npz2pdb] {args.split.name}: {len(afids)} AFids  workers={args.workers}",
          flush=True)

    todo = [(a, str(args.apo_dir), str(args.holo_dir), str(args.out_dir))
            for a in afids]

    n_ok = n_err = 0
    err_by_kind: dict[str, int] = {}
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        for i, (afid, err) in enumerate(pool.imap_unordered(_one, todo, chunksize=64), 1):
            if err is None:
                n_ok += 1
            else:
                n_err += 1
                kind = err.split(":", 1)[0]
                err_by_kind[kind] = err_by_kind.get(kind, 0) + 1
            if i % 2000 == 0 or i == len(todo):
                rate = i / (time.time() - t0 + 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"  [{i:>7}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"{rate:.0f} af/s  eta {eta/60:.1f} min  "
                      f"errs={dict(sorted(err_by_kind.items(), key=lambda x: -x[1])[:3])}",
                      flush=True)

    print(f"[npz2pdb] DONE  ok={n_ok}  err={n_err}  errs={err_by_kind}  "
          f"wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
