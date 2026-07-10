# K-Fold Structure Release: Upload Manifest

Code → **GitHub**, weights + data → **HuggingFace**.

- **Upload as** = the clean, human-readable name to publish under. The files
  on disk keep their original (often cryptic) names, **rename at upload
  time only**, don't rename the source files.
- Only the full-atom VQ-VAE tokenization is shipped; internal
  `fullatom_image` suffixes are dropped from upload names.
- `Source` = where the artifact currently lives (original name kept for
  reproducibility).

---

## 1. Model weights → HuggingFace Hub

One repo per size (e.g. `<org>/triprorep-650M`): the full
Lightning `.ckpt` (re-training) + the encoder-only `.pt` (inference,
`strict=True` into `ProteinNetEncoder`).

| Upload as | Source | Size | Notes |
|---|---|---:|---|
| `35M.ckpt`        | `/mnt/.../final_checkpoints/35M.ckpt`         | 244 MB | ELECTRA encoder, 35M |
| `150M.ckpt`       | `/mnt/.../final_checkpoints/150M.ckpt`        | 768 MB | ELECTRA encoder, 150M |
| `650M.ckpt`       | `/mnt/.../final_checkpoints/650M.ckpt`        | 3.2 GB | ELECTRA encoder, 650M |
| `3B.ckpt`         | `/mnt/.../final_checkpoints/3B.ckpt`          | 14 GB  | ELECTRA encoder, 3B |
| `3B_encoder.pt`   | `/mnt/.../final_checkpoints/3B_encoder.pt`    | 9.8 GB | encoder-only state_dict (571 tensors) |

> Dropped: `3B_finetune.{ckpt,pt}`, stage-2 IndepApo fine-tune. Not public
> (excluded from the code release + leaks into the probing/folding splits).

---

## 2. Tokenizers → HuggingFace Hub (one small repo)

Needed to go raw PDB → token IDs.

| Upload as | Source | Size | Role |
|---|---|---:|---|
| `backbone_tokenizer.pt`  | `/home/.../k-fold-V2/data/aminoaseed.pt`         | 536 MB | backbone (bb) token VQ-VAE, taken from [StructTokenBench](https://github.com/KatarinaYuan/StructTokenBench), all credit to its authors |
| `fullatom_tokenizer.pt`  | `/home/.../k-fold-V2/data/fa_tokenizer.ckpt` | 173 MB | full-atom (fa) token VQ-VAE |

---

## 3. Pre-training tokenized LMDB → HuggingFace Datasets

The ELECTRA pre-training corpus (seq + bb + fa tokens per structure).

| Upload as | Source | Size | Content |
|---|---|---:|---|
| `pretrain_structure_tokens` (LMDB) | `/cache/ssahn_lab/lmdb/atlas_pdb_all` | 609 GB | 83.6M ATLAS + PDB structures; `{seq, bb, fa}` per entry. Used by `ELECTRADataModule`. |

> Large, release as sharded HF dataset or external mirror. Per-entry schema:
> `code/tokenizer/tokenize_example.py`.

---

## 4. Benchmark raw PDBs → `REPSP_PDB/` (not redistributed)

We do **not** redistribute the PDB files (AFDB license terms). Users
build `REPSP_PDB/` locally with `scripts/npz_to_pdb.py` from the
Boltz-format structures under `/mnt/.../co-folding/full/{apo,holo}_targets/`
(bundled with the folding pipeline), or by fetching the NVIDIA/AFDB-Multimer
chunk tars at
`https://ftp.ebi.ac.uk/pub/databases/alphafold/collaborations/nvda/`
(`homodimer_metadata.csv` maps each AFid to a `chunk_NNNN.tar`; download
only the chunks referenced by our splits).

Resulting layout, referenced by every downstream tool in the release:

```
REPSP_PDB/
├── monomer/<AF-id>_monomer.pdb    # chain A (apo prediction)
└── homodimer/<AF-id>.pdb          # chain A + chain B (holo dimer)
```

The AFid in both filenames is the same homodimer identifier taken from
`splits/{folding,probing}/{train,valid,test}.txt`.

The per-entry EBI API (`/api/prediction/{AFid}`) serves only the subset
of AFDB-Multimer that survives the current metadata refresh (about 85%
of our split). The `collaborations/nvda/` chunk tars are the full
archive and cover 100%.

---

## 5. Benchmark tokenized LMDB → HuggingFace Datasets

Full-atom tokenization of the AFDB-Multimer apo+holo above (apo = AlphaFold
monomer, holo = dimer); keyed by `AF-{id}`.

| Upload as | Source | Size | Content |
|---|---|---:|---|
| `benchmark_apo_holo_tokens` (LMDB) | `/mnt/.../lmdb/apo_holo_afdb_fullatom_image` | (large) | `{apo_{seq,bb,fa}_{A,B}, holo_{seq,bb,fa}_{A,B}, contact_map}` per `AF-{id}`. Source for feature extraction. |

---

## 6. Train / valid / test split → HuggingFace Datasets

**Shared by folding and co-folding** (same `AF-{id}` partition). One
file per split, containing homodimer AFids. Each AFid resolves to two
on-disk PDBs under `REPSP_PDB/`: `monomer/<AF-id>_monomer.pdb` for the
apo side and `homodimer/<AF-id>.pdb` for the holo side.

| Upload as | Source | Size | Content |
|---|---|---:|---|
| `splits/folding/{train,valid,test}.txt` | `/mnt/.../co-folding/splits/afdb_multimer_reps/train_valid_test/{train,valid,test}.txt` (LMDB-cleaned) | small | 390,627 / 400 / 1,000 homodimer AFids |

---

## 7. Folding / co-folding training inputs → HuggingFace Datasets

Wired in `code/repsp/folding/configs/data/{afdb_monomer,afdb_multimer}.yaml`.

| Upload as | Source | Size | Content |
|---|---|---:|---|
| `boltz_apo_{tokens,targets}`  | `/mnt/.../co-folding/full/apo_{tokens,targets}`                       | ~16 GB | Boltz-tokenized apo monomer + structure targets (folding) |
| `boltz_holo_{tokens,targets}` | `/mnt/.../co-folding/afdb_multimer_boltz_{tokenized,targets}` (+ `symmetry.pkl`) | ~120 GB | Boltz-tokenized holo dimer + targets (co-folding) |
| `features/<encoder>.lmdb`     | `/mnt/.../co-folding/features/afdb_monomer/apo_<encoder>/features.lmdb` | ~325 GB ea (3B: ~650 GB) | per-residue `[L_A, D] fp16` (chain A only), REPA target (folding) + apo conditioning (co-folding; tiled A∥A inline at load) + probing input. One per encoder (`ours[_35M/150M/3B]`, `esm2`, `saprot`, `esm3`, `mifst`, `prost_t5`, `splm`). |

> Features are huge. Minimum useful release: `ours` (one size) + the
> baselines you want to compare. Probing only needs the ~40k split-subset
> of each (the staging step slices it).

---

## 8. Probing assets (homomer) → HuggingFace Datasets / GitHub

| Upload as | Source | Size | Content |
|---|---|---:|---|
| `probing_features/<encoder>/{train,valid,test}.pt` | produced by `code/repsp/probing/homomer/__lib/extract_probing_features.py` from the §7 `features/<encoder>.lmdb` + split files below | ~10–50 GB per encoder | per-split flat tensors (`X: fp16 [N_res, D]` + `pid_slices: list[(pid, start, end)]`), the probe reads these directly, no further extraction. One dir per encoder. |
| `probing/labels.pkl` | `/home/.../probing/homomer/data/homomer_target_set_plip.pkl` | 574 MB | per-residue labels for the 4 tasks (`binding_site`, `delta_sasa_mean`, `levy_tier`, `bond_type_plip`) |
| `splits/probing/{train,valid,test}.txt` | `/home/.../probing/homomer/data/keep_pids/{train,valid,test}.txt` (LMDB-cleaned) | small | 39,100 / 400 / 1,000 homodimer AFids (10 % of folding train + full valid/test) |

---

## 9. Source code → GitHub

`k-fold-structure-release/code/`, three components:

* **`triprorep/`**: ELECTRA pre-training (configs, train launcher, models, dataloader).
* **`repsp/`**: benchmark: `probing/homomer/` + `folding/` (REPA + co-folding).
* **`tokenizer/`**: PDB → (seq, bb, fa) tokens: example + `structure_tokenize/` + bundled tokenizer ckpts.

Excluded: IndepApo stage-2, EC/GO + conventional probes, cameo22, SaProt-vocab
fork, SelfFlow, voxel FT, paper/scratch.

---

## Release order (suggested)

1. Code → GitHub.
2. Tokenizers (§2) + model weights (§1) → HF Hub.
3. Probing assets (§8, small) + splits (§6, small) → HF Datasets.
4. Benchmark tokenized LMDB (§5) + folding/co-folding inputs (§7), large,
   release the encoders you actually compare.
5. Pre-training corpus (§3, 609 GB), largest single asset, sharded LMDB upload.
6. (Raw PDBs are NOT uploaded; §4 ships only the resolver/downloader script.)
7. Update the GitHub README with HF links + download instructions.
