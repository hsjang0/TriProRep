# Homomer probing: current setting

Per-residue probing of frozen pretrained protein encoders against
homomer-defined targets. Apo monomer sequence in → per-residue holo-dimer
property out.

## Encoder

- **Frozen** pretrained encoder (no gradient flow through encoder)
- Input: apo monomer sequence
- Output: per-residue features `x_i ∈ R^D`
- `D` varies by encoder:

| tier | encoders | `D` |
|---|---|---|
| Small (35M) | ESM2, SaProt, Ours | 480 |
| Medium (150M) | ESM2, Ours | 640 |
| Large (650M) | ESM2, SaProt, Ours, S-PLM (704M) | 1280 |
| Structure (650M) | MIF-ST | 256 |
| Huge (1.4–3B) | ESM3 (1536), ProstT5 (2048), ESM2 3B / Ours 2.85B | 1536 / 2048 / 2560 |

## Probe head (task-agnostic)

- **2-hidden-layer MLP**: `D → 1280 → 1280 → D_out`
- Activation: **GELU**
- **Dropout 0.1** between layers
- **No input LayerNorm** (raw frozen features fed directly)
- `D_out` task-dependent (see below)

## Task-specific output / loss

| task | kind | `D_out` | loss |
|---|---|---|---|
| `binding_site` | binary | 1 | `BCEWithLogitsLoss` |
| `delta_sasa_mean` | regression | 1 | `MSELoss` (fp32) |
| `levy_tier` | 5-class | 5 | `CrossEntropyLoss` |
| `bond_type_plip` | multi-label | 5 | `BCEWithLogitsLoss` |

`bond_type_plip` = 5 inter-chain interaction classes (hbond / salt-bridge /
hydrophobic / π-π / cation-π) detected by **PLIP** (Salentin 2015), with chain B
treated as the peptide ligand. It supersedes the legacy hand-coded geometric
`bond_type` produced by `build_homomer_targets.py`.

## Optimization

| | |
|---|---|
| optimizer | AdamW (β = (0.9, 0.999), eps = 1e-8, PyTorch defaults) |
| learning rate | **5e-4** constant (no schedule, no warmup) |
| weight decay | 0.01 |
| epochs | **10** |
| mini-batch | **16,824 residues** (per-residue, not per-protein) |
| precision | bf16 autocast forward+loss; MSE logits cast to fp32 |

## Data

- **Split**: deterministic ~10% subsample of folding train + full valid + full test
  - train: 39,100 sequences
  - valid: 400
  - test: 1,000
- folding split itself: cluster-level via MMseqs2 `easy-linclust`
  (`--min-seq-id 0.5 -c 0.8 --cov-mode 1`) → 404,961 reps → 390,861 / 400 / 1,000

### Label aggregation rules (homodimer A1∥A2 → 1 per position)

Each residue position appears in both chain instances; encoder is chain-blind,
so labels are folded into a single per-position target:

| task | rule |
|---|---|
| `binding_site` | **OR** (union) |
| `delta_sasa_mean` | **mean** |
| `levy_tier` | **max-rank** (most-buried wins; 0 surface → 4 core) |
| `bond_type_plip` | **union** (multi-hot OR per class) |

## Train loop

- Per-epoch eval on **val + test**
- **Model selection**: best epoch chosen by **val primary score**
- Primary score per task kind:

| kind | primary | secondary |
|---|---|---|
| binary | AUPRC | AUROC |
| regression | Pearson | Spearman |
| multiclass | macro F1 | accuracy |
| multilabel | mean AUPRC | mean AUROC |

- Reported test metric = `best_val_test/*` (test metric at val-best epoch)

## Compute

- **Node**: gpu23 (8 × B200, 192 cores, 1.9 TB RAM)
- **Concurrency**: 1 GPU per encoder, 4 tasks **sequential** within each GPU
- **CPU thread cap**: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4` (~32 threads peak)
- 14 encoders × 4 tasks = **56 probe runs** per ablation
- Wall-clock per ablation: ~80–90 min (8 + 6 = 14 over 8 GPUs)

## WandB

- project: `homomer probing` (set your own)
- run name: `<encoder>_<task>` (unique)
- tags: `[ep10, task_<task>, encoder_<run>]` (filterable in UI)
- entity: `<your-wandb-entity>`

## Code layout

- `__lib/probe_residue_homomer.py`: probe head + train loop
- `__lib/_homomer_common.py`: task dispatch, metrics, wandb plumbing
- `__lib/extract_probing_features.py`: encoder features LMDB → per-split flat `.pt`
- `__lib/verify_split_consistency.py`: split-membership sanity check
- `data/build_homomer_targets.py`: builds binding_site / delta_sasa / levy_tier (BioPython + Shrake-Rupley) + the legacy geometric `bond_type`
- `data/enrich_bond_type_plip.py`: computes `bond_type_plip` with PLIP (sharded; chain B as peptide ligand)
- `data/merge_plip_shards.py`: merges PLIP shards into the target pkl as `bond_type_plip`
- `data/homomer_target_set.pkl`: prebuilt per-residue labels incl. `bond_type_plip` (shipped target)
- `data/splits/`: train / valid / test pid lists

## File output

- per-task: `results/<run>/<task>.json` (each contains all metrics + best_epoch + config)
- 14 × 4 = 56 JSONs per ablation
