#!/usr/bin/env python
"""
Build per-residue homomer-defined target labels for binding-site probing.

Per pdb_id (apo monomer + holo dimer), compute and aggregate (A1+A2):
  - binding_site (binary, OR)               
  - delta_sasa_{mean,max} (regression)         
  - levy_tier (5-class) / mean_rank (ord.)  
  - bond_type (multi-hot 5 classes)
  - rsasa_apo (regression aux)

Output: pickle {pdb_id: {L, seq, ...np.array per task}}
"""

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from tqdm import tqdm


# -------- Constants --------
CONTACT_CA_THRESHOLD = 8.0
LEVY_RSASA_CUT = 0.25  # Levy 2010, Miller 1987

THREE_TO_ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLU':'E','GLN':'Q',
    'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
    'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
}

# Tien 2013 (PLoS ONE) max ASA per residue (Ų)
MAX_ASA = {
    'A':121.,'R':265.,'N':187.,'D':187.,'C':148.,'E':214.,'Q':214.,'G':97.,
    'H':216.,'I':195.,'L':191.,'K':230.,'M':203.,'F':228.,'P':154.,'S':143.,
    'T':163.,'W':264.,'Y':255.,'V':165.,
}

# H-bond donor/acceptor (Arpeggio-aligned). Note PRO N is excluded as donor.
HBOND_DONOR = {  # heavy-atom side
    'N','ND1','NE2','NZ','NE','NH1','NH2','ND2','NE1','OG','OG1','OH','SG'
}
HBOND_ACCEPTOR = {
    'O','OXT','OD1','OD2','OE1','OE2','OG','OG1','OH','ND1','NE2',
}
# Per-residue donor/acceptor restrictions (asymmetric ASN/GLN handling)
ASN_DONOR_ONLY = {'ND2'}
ASN_ACCEPTOR_ONLY = {'OD1'}
GLN_DONOR_ONLY = {'NE2'}
GLN_ACCEPTOR_ONLY = {'OE1'}

# Salt-bridge centroid atoms (Barlow-Thornton 1983)
POS_CENTROID = {'LYS': ['NZ'],
                'ARG': ['NH1', 'NH2', 'CZ', 'NE']}
NEG_CENTROID = {'ASP': ['OD1', 'OD2'],
                'GLU': ['OE1', 'OE2']}

# Aromatic ring atoms
AROMATIC_RING = {
    'PHE': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TYR': ['CG','CD1','CD2','CE1','CE2','CZ'],
    'TRP': ['CD2','CE2','CE3','CZ2','CZ3','CH2'],
    'HIS': ['CG','ND1','CD2','CE1','NE2'],
}
# Cation centers (Gallivan-Dougherty 1999)
CATION = {'LYS': ['NZ'], 'ARG': ['NH1', 'NH2', 'CZ']}

# Hydrophobic residues (Arpeggio convention; CYS dropped per agent review).
HYDROPHOBIC_RES = {'ALA','VAL','LEU','ILE','MET','PHE','PRO','TRP','TYR'}
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}

BOND_TYPES = ['hbond', 'salt_bridge', 'hydrophobic', 'pi_stack', 'cation_pi']
N_BOND = len(BOND_TYPES)

# Levy 5-class (Levy 2010 JMB)
SURFACE, INTERIOR, SUPPORT, RIM, CORE = 0, 1, 2, 3, 4
N_TIERS = 5


# -------- PDB index --------
def index_pdb_dir(root: Path):
    """Build {pdb_id: path} by scanning root/<subdir>/<id>.pdb."""
    out = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.suffix == '.pdb':
                out[f.stem] = f
    return out


# -------- Per-residue helpers --------
def get_aa_residues(chain):
    out = []
    for res in chain:
        if res.id[0] != ' ':
            continue
        if res.resname in THREE_TO_ONE:
            out.append((res.id[1], res.resname, res))
    return out


def compute_sasa(struct):
    sr = ShrakeRupley(probe_radius=1.4, n_points=960)
    sr.compute(struct, level='R')
    out = {}
    for ch in struct[0]:
        for res in ch:
            if res.id[0] != ' ':
                continue
            try:
                out[(ch.id, res.id[1])] = res.sasa
            except AttributeError:
                pass
    return out


def get_atoms(residue):
    names, coords = [], []
    for at in residue.get_atoms():
        if at.element == 'H':
            continue
        names.append(at.name)
        coords.append(at.coord)
    return np.asarray(names), (np.asarray(coords) if coords else
                                np.empty((0, 3), dtype=np.float32))


def hbond_donor_ok(residue, atom_name):
    """Per-residue donor restrictions."""
    if residue.resname == 'PRO' and atom_name == 'N':
        return False
    if residue.resname == 'ASN' and atom_name in ASN_ACCEPTOR_ONLY:
        return False
    if residue.resname == 'GLN' and atom_name in GLN_ACCEPTOR_ONLY:
        return False
    return atom_name in HBOND_DONOR


def hbond_acceptor_ok(residue, atom_name):
    if residue.resname == 'ASN' and atom_name in ASN_DONOR_ONLY:
        return False
    if residue.resname == 'GLN' and atom_name in GLN_DONOR_ONLY:
        return False
    return atom_name in HBOND_ACCEPTOR


def n_o_angle_ok(donor_res, donor_atom_idx, acceptor_atom_coord,
                 atom_names, atom_coords):
    """Cheap H-less angle check: donor's heavy-atom predecessor → donor → acceptor.
    Reject if predecessor → donor → acceptor angle < 90° (sterically impossible)."""
    # For backbone N: predecessor is the previous Cα or C; we use the residue's CA.
    # For sidechain N/O: parent atom is approximated by the Cα.
    if 'CA' not in atom_names:
        return True  # no check available
    ca_idx = list(atom_names).index('CA')
    v1 = atom_coords[ca_idx] - atom_coords[donor_atom_idx]
    v2 = acceptor_atom_coord - atom_coords[donor_atom_idx]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return True
    cos_ang = np.dot(v1, v2) / (n1 * n2)
    cos_ang = np.clip(cos_ang, -1.0, 1.0)
    return np.degrees(np.arccos(cos_ang)) > 90.0


def classify_bond_pair(res1, res2, threshold_atom=5.0,
                       hbond_dist=3.5, salt_dist=4.0,
                       pi_dist=5.5, cation_pi_dist=6.0):
    n1, n2 = res1.resname, res2.resname
    a1_names, a1_coords = get_atoms(res1)
    a2_names, a2_coords = get_atoms(res2)
    if len(a1_coords) == 0 or len(a2_coords) == 0:
        return set()
    dmat = np.linalg.norm(a1_coords[:, None] - a2_coords[None, :], axis=-1)
    if dmat.min() > threshold_atom:
        return set()

    bonds = set()

    # ---- salt bridge (centroid distance, Barlow-Thornton 1983)
    def _centroid_pair(pos_res, neg_res, pos_anames, pos_acoords,
                       neg_anames, neg_acoords):
        if pos_res.resname not in POS_CENTROID or neg_res.resname not in NEG_CENTROID:
            return False
        pa = POS_CENTROID[pos_res.resname]
        na = NEG_CENTROID[neg_res.resname]
        pi = [i for i, x in enumerate(pos_anames) if x in pa]
        ni = [i for i, x in enumerate(neg_anames) if x in na]
        if not pi or not ni:
            return False
        pc = pos_acoords[pi].mean(axis=0)
        nc = neg_acoords[ni].mean(axis=0)
        return np.linalg.norm(pc - nc) < salt_dist

    if _centroid_pair(res1, res2, a1_names, a1_coords, a2_names, a2_coords):
        bonds.add('salt_bridge')
    elif _centroid_pair(res2, res1, a2_names, a2_coords, a1_names, a1_coords):
        bonds.add('salt_bridge')

    # ---- H-bond (distance + cheap angle)
    found = False
    for i, ni_at in enumerate(a1_names):
        if found:
            break
        for j, nj_at in enumerate(a2_names):
            if dmat[i, j] >= hbond_dist:
                continue
            d_a = (hbond_donor_ok(res1, ni_at) and
                   hbond_acceptor_ok(res2, nj_at) and
                   n_o_angle_ok(res1, i, a2_coords[j], a1_names, a1_coords))
            a_d = (hbond_donor_ok(res2, nj_at) and
                   hbond_acceptor_ok(res1, ni_at) and
                   n_o_angle_ok(res2, j, a1_coords[i], a2_names, a2_coords))
            if d_a or a_d:
                bonds.add('hbond')
                found = True
                break

    # ---- π-π stacking (Burley-Petsko 1985)
    if n1 in AROMATIC_RING and n2 in AROMATIC_RING:
        r1 = AROMATIC_RING[n1]; r2 = AROMATIC_RING[n2]
        idx1 = [i for i, x in enumerate(a1_names) if x in r1]
        idx2 = [i for i, x in enumerate(a2_names) if x in r2]
        if len(idx1) >= 3 and len(idx2) >= 3:
            c1 = a1_coords[idx1].mean(axis=0)
            c2 = a2_coords[idx2].mean(axis=0)
            if np.linalg.norm(c1 - c2) < pi_dist:
                v1 = a1_coords[idx1[1]] - a1_coords[idx1[0]]
                v2 = a1_coords[idx1[2]] - a1_coords[idx1[0]]
                norm1 = np.cross(v1, v2)
                norm1 = norm1 / (np.linalg.norm(norm1) + 1e-9)
                w1 = a2_coords[idx2[1]] - a2_coords[idx2[0]]
                w2 = a2_coords[idx2[2]] - a2_coords[idx2[0]]
                norm2 = np.cross(w1, w2)
                norm2 = norm2 / (np.linalg.norm(norm2) + 1e-9)
                ang = np.degrees(np.arccos(
                    np.clip(abs(np.dot(norm1, norm2)), 0, 1)))
                if ang < 30 or (60 <= ang <= 90):
                    bonds.add('pi_stack')

    # ---- cation-π (Gallivan-Dougherty 1999)
    def _cation_pi(cat_res, ar_res, cat_anames, cat_acoords, ar_anames, ar_acoords):
        if cat_res.resname not in CATION or ar_res.resname not in AROMATIC_RING:
            return False
        ca = CATION[cat_res.resname]
        ra = AROMATIC_RING[ar_res.resname]
        ci = [i for i, x in enumerate(cat_anames) if x in ca]
        ri = [i for i, x in enumerate(ar_anames) if x in ra]
        if not ci or len(ri) < 3:
            return False
        cat_pos = cat_acoords[ci].mean(axis=0)
        ring_c = ar_acoords[ri].mean(axis=0)
        if np.linalg.norm(cat_pos - ring_c) >= cation_pi_dist:
            return False
        v1 = ar_acoords[ri[1]] - ar_acoords[ri[0]]
        v2 = ar_acoords[ri[2]] - ar_acoords[ri[0]]
        nv = np.cross(v1, v2); nv = nv / (np.linalg.norm(nv) + 1e-9)
        d = cat_pos - ring_c; d = d / (np.linalg.norm(d) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(nv, d)), 0, 1)))
        return ang < 60.0  # cation roughly above ring plane

    if _cation_pi(res1, res2, a1_names, a1_coords, a2_names, a2_coords):
        bonds.add('cation_pi')
    if _cation_pi(res2, res1, a2_names, a2_coords, a1_names, a1_coords):
        bonds.add('cation_pi')

    # ---- hydrophobic (both residues hydrophobic, side-chain heavy atoms)
    if (n1 in HYDROPHOBIC_RES and n2 in HYDROPHOBIC_RES
            and 'salt_bridge' not in bonds and 'hbond' not in bonds
            and 'pi_stack' not in bonds and 'cation_pi' not in bonds):
        sc1 = ~np.isin(a1_names, list(BACKBONE_ATOMS))
        sc2 = ~np.isin(a2_names, list(BACKBONE_ATOMS))
        if sc1.any() and sc2.any():
            d_sc = dmat[sc1][:, sc2]
            if d_sc.size > 0 and d_sc.min() < threshold_atom:
                bonds.add('hydrophobic')

    return bonds


def classify_bonds_chain_pair(c1_residues, c2_residues, prefilter_ca=10.0):
    out = {}
    ca1 = [(rid, rn, res, res['CA'].coord) for rid, rn, res in c1_residues
           if 'CA' in res]
    ca2 = [(rid, rn, res, res['CA'].coord) for rid, rn, res in c2_residues
           if 'CA' in res]
    if not ca1 or not ca2:
        return out
    ca1_arr = np.asarray([x[3] for x in ca1])
    ca2_arr = np.asarray([x[3] for x in ca2])
    d_ca = np.linalg.norm(ca1_arr[:, None] - ca2_arr[None, :], axis=-1)
    pairs = np.argwhere(d_ca < prefilter_ca)
    for ai, bi in pairs:
        rid1, _, res1, _ = ca1[ai]
        rid2, _, res2, _ = ca2[bi]
        b = classify_bond_pair(res1, res2)
        if b:
            ch1 = res1.parent.id; ch2 = res2.parent.id
            for x in b:
                out.setdefault((ch1, rid1), set()).add(x)
                out.setdefault((ch2, rid2), set()).add(x)
    return out


def levy_class(rsasa_apo_i, rsasa_holo_i, has_contact_i,
               cut=LEVY_RSASA_CUT):
    """Levy 2010 5-class:
      surface  : exposed in apo, no interface contact
      interior : buried in apo, no interface contact
      support  : buried in apo, has interface contact
      rim      : exposed in apo, has contact, still exposed in holo
      core     : exposed in apo, has contact, buried in holo
    """
    apo_buried = rsasa_apo_i <= cut
    holo_buried = rsasa_holo_i <= cut
    if has_contact_i:
        if apo_buried:
            return SUPPORT
        return CORE if holo_buried else RIM
    return INTERIOR if apo_buried else SURFACE


def process_one(args):
    pdb_id, apo_path, holo_path = args
    try:
        parser = PDBParser(QUIET=True)
        apo = parser.get_structure(pdb_id, str(apo_path))
        holo = parser.get_structure(pdb_id, str(holo_path))

        apo_chain = next(iter(apo[0]))
        apo_res = get_aa_residues(apo_chain)
        if not apo_res:
            return pdb_id, 'empty_apo'
        L = len(apo_res)
        seq = ''.join(THREE_TO_ONE[r[1]] for r in apo_res)
        apo_resnums = [r[0] for r in apo_res]

        holo_chains = list(holo[0])
        per_chain = {ch.id: get_aa_residues(ch) for ch in holo_chains}
        per_chain = {k: v for k, v in per_chain.items() if len(v) == L}
        if len(per_chain) < 2:
            return pdb_id, 'holo_chain_len_mismatch'

        apo_sasa = compute_sasa(apo)
        holo_sasa = compute_sasa(holo)

        # rsasa_apo (per-residue, from apo monomer)
        rsasa_apo = np.zeros(L, dtype=np.float32)
        for k, (rnum, rname, _) in enumerate(apo_res):
            sa = apo_sasa.get((apo_chain.id, rnum), 0.0)
            ms = MAX_ASA.get(THREE_TO_ONE[rname], 200.0)
            rsasa_apo[k] = min(sa / ms, 1.0)

        # ΔSASA + rsasa_holo per chain
        dsasa_per = {}
        rsasa_holo_per = {}
        for cid, residues in per_chain.items():
            dsa = np.zeros(L, dtype=np.float32)
            rsh = np.zeros(L, dtype=np.float32)
            for k, (rnum, rname, _) in enumerate(residues):
                s_apo = apo_sasa.get((apo_chain.id, apo_resnums[k]), 0.0)
                s_holo = holo_sasa.get((cid, rnum), 0.0)
                dsa[k] = max(s_apo - s_holo, 0.0)
                ms = MAX_ASA.get(THREE_TO_ONE[rname], 200.0)
                rsh[k] = min(s_holo / ms, 1.0)
            dsasa_per[cid] = dsa
            rsasa_holo_per[cid] = rsh

        # binding-site (Cα 8Å) per chain
        chain_ids = list(per_chain.keys())
        ca_per = {}
        for cid, residues in per_chain.items():
            cas = np.full((L, 3), np.inf, dtype=np.float32)
            for k, (_, _, res) in enumerate(residues):
                if 'CA' in res:
                    cas[k] = res['CA'].coord
            ca_per[cid] = cas

        bs_per = {}
        for cid in chain_ids:
            own = ca_per[cid]
            mask = np.zeros(L, dtype=np.uint8)
            for cid2 in chain_ids:
                if cid2 == cid:
                    continue
                d = np.linalg.norm(own[:, None] - ca_per[cid2][None, :], axis=-1)
                mask |= (d < CONTACT_CA_THRESHOLD).any(axis=1).astype(np.uint8)
            bs_per[cid] = mask

        # Aggregate
        bs_agg = np.stack(list(bs_per.values())).any(axis=0).astype(np.uint8)
        dsasa_arr = np.stack(list(dsasa_per.values()))
        dsasa_mean = dsasa_arr.mean(axis=0).astype(np.float32)
        dsasa_max = dsasa_arr.max(axis=0).astype(np.float32)

        # Levy tier
        tier_per = {}
        for cid in chain_ids:
            tiers = np.zeros(L, dtype=np.uint8)
            for k in range(L):
                tiers[k] = levy_class(
                    rsasa_apo[k], rsasa_holo_per[cid][k], bs_per[cid][k] == 1
                )
            tier_per[cid] = tiers
        tier_mat = np.stack(list(tier_per.values()))
        levy_tier = tier_mat.max(axis=0).astype(np.uint8)
        levy_mean_rank = tier_mat.mean(axis=0).astype(np.float32)

        # Bond type (chain pair → multi-hot per residue)
        c1_id, c2_id = chain_ids[0], chain_ids[1]
        bond_dict = classify_bonds_chain_pair(per_chain[c1_id], per_chain[c2_id])
        bond_mh = np.zeros((L, N_BOND), dtype=np.uint8)
        for (cid, rnum), bset in bond_dict.items():
            if cid not in per_chain:
                continue
            for k, (rid_k, _, _) in enumerate(per_chain[cid]):
                if rid_k == rnum:
                    for b in bset:
                        if b in BOND_TYPES:
                            bond_mh[k, BOND_TYPES.index(b)] = 1
                    break

        return pdb_id, {
            'L': L,
            'seq': seq,
            'binding_site': bs_agg,
            'delta_sasa_mean': dsasa_mean,
            'delta_sasa_max': dsasa_max,
            'levy_tier': levy_tier,
            'levy_tier_mean_rank': levy_mean_rank,
            'bond_type': bond_mh,
            'rsasa_apo': rsasa_apo,
        }
    except Exception as e:
        return pdb_id, f'err: {type(e).__name__}: {e}'


# -------- Worker init for fork-shared globals --------
_apo_idx_g = None
_holo_idx_g = None

def _init_workers(apo_idx, holo_idx):
    global _apo_idx_g, _holo_idx_g
    _apo_idx_g = apo_idx
    _holo_idx_g = holo_idx

def _worker(pid):
    ap = _apo_idx_g.get(pid)
    hp = _holo_idx_g.get(pid)
    if ap is None or hp is None:
        return pid, 'missing_pdb'
    return process_one((pid, ap, hp))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--splits_dir', required=True)
    p.add_argument('--apo_root', default='/path/to/afdb_apo_pdb')
    p.add_argument('--holo_root', default='/path/to/afdb_holo_pdb')
    p.add_argument('--out_pkl', required=True)
    p.add_argument('--n_workers', type=int, default=16)
    p.add_argument('--max_n', type=int, default=None)
    p.add_argument('--apo_idx_cache', default=None)
    p.add_argument('--holo_idx_cache', default=None)
    args = p.parse_args()

    print(f'[index] apo_root={args.apo_root}', flush=True)
    if args.apo_idx_cache and Path(args.apo_idx_cache).exists():
        with open(args.apo_idx_cache, 'rb') as f:
            apo_idx = pickle.load(f)
        print(f'  loaded apo_idx cache: {len(apo_idx)}', flush=True)
    else:
        apo_idx = index_pdb_dir(Path(args.apo_root))
        print(f'  indexed {len(apo_idx)} apo PDBs', flush=True)
        if args.apo_idx_cache:
            p = Path(args.apo_idx_cache)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + '.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(apo_idx, f)
            tmp.replace(p)

    print(f'[index] holo_root={args.holo_root}', flush=True)
    if args.holo_idx_cache and Path(args.holo_idx_cache).exists():
        with open(args.holo_idx_cache, 'rb') as f:
            holo_idx = pickle.load(f)
        print(f'  loaded holo_idx cache: {len(holo_idx)}', flush=True)
    else:
        holo_idx = index_pdb_dir(Path(args.holo_root))
        print(f'  indexed {len(holo_idx)} holo PDBs', flush=True)
        if args.holo_idx_cache:
            p = Path(args.holo_idx_cache)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + '.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(holo_idx, f)
            tmp.replace(p)

    pids = []
    for s in ('train', 'valid', 'test'):
        pp = Path(args.splits_dir) / f'{s}.txt'
        if pp.exists():
            for ln in pp.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    pids.append(ln)
    pids = list(dict.fromkeys(pids))
    if args.max_n:
        pids = pids[:args.max_n]
    print(f'[run] processing {len(pids)} pdb_ids with {args.n_workers} workers',
          flush=True)

    out, skipped = {}, {}
    if args.n_workers <= 1:
        for pid in tqdm(pids):
            ap = apo_idx.get(pid); hp = holo_idx.get(pid)
            if ap is None or hp is None:
                skipped['missing_pdb'] = skipped.get('missing_pdb', 0) + 1
                continue
            pid2, res = process_one((pid, ap, hp))
            if isinstance(res, dict):
                out[pid2] = res
            else:
                skipped[res] = skipped.get(res, 0) + 1
    else:
        with mp.Pool(args.n_workers, initializer=_init_workers,
                     initargs=(apo_idx, holo_idx)) as pool:
            for pid, res in tqdm(pool.imap_unordered(_worker, pids),
                                 total=len(pids)):
                if isinstance(res, dict):
                    out[pid] = res
                else:
                    skipped[res] = skipped.get(res, 0) + 1

    print(f'\n[done] kept {len(out)} / {len(pids)}', flush=True)
    print(f'[done] skipped: {skipped}', flush=True)

    # Atomic write: tmp file + rename. Long pkl write (~hundreds of MB) is
    # vulnerable to mid-write SIGKILL; without atomic rename the final file
    # at args.out_pkl could be partially written and corrupt any prior valid copy.
    out_path = Path(args.out_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp_path, 'wb') as f:
        pickle.dump(out, f)
    tmp_path.replace(out_path)
    sz = out_path.stat().st_size / 1e6
    print(f'[done] saved → {out_path} ({sz:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
