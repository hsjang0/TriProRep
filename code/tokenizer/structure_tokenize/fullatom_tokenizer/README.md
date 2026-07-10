# fullatom_tokenizer: FA-VQVAE Tokenizer (Inference-Only)

Self-contained inference package for the SE(3)-invariant full-atom VQVAE tokenizer. Encodes per-residue 3D geometry into discrete structure tokens (1 of 512). The model operates in canonical local frames for rotation/translation invariance.

This package contains *only* the runtime path. No dataset, training loop, or loss code. The bundled checkpoint (`fullatom_tokenizer.pt`) was trained with the full `favqvae/` package (a sibling directory) and frozen here for downstream use.

## Quick Start

```bash
# Tokenize a single PDB
python -m structure_tokenize.fullatom_tokenizer.inference \
    --pdb /path/to/protein.pdb \
    --output tokens.npy
# (--ckpt defaults to the bundled fullatom_tokenizer.pt)

# Verify the slim package still reproduces tokens from the original favqvae
python -m structure_tokenize.fullatom_tokenizer.verify
```

Or call from Python:

```python
from structure_tokenize.fullatom_tokenizer.inference import tokenize
tokens = tokenize("/path/to/protein.pdb", device="cuda")  # np.int64 array of shape [L]
```

## 1. Input Representation

### 1.1 Coordinate Canonicalization

Coordinates are transformed into SE(3)-invariant local residue frames built from backbone atoms (N, CA, C).

Per residue:

$$
\text{origin} = C_\alpha,\quad
\hat{x} = \frac{N - C_\alpha}{\|N - C_\alpha\|},\quad
\hat{z} = \frac{\hat{x} \times (C - C_\alpha)}{\|\hat{x} \times (C - C_\alpha)\|},\quad
\hat{y} = \hat{z} \times \hat{x}
$$

$$
R = [\hat{x},\; \hat{y},\; \hat{z}]^T \in \mathbb{R}^{3 \times 3}
$$

All 37 atom positions are expressed in the local frame:

$$
\mathbf{x}_\text{canonical} = R \cdot (\mathbf{x} - \text{origin}) \quad \in \mathbb{R}^{L \times 37 \times 3}
$$

The representation is invariant to global rotation and translation — the encoder sees only local geometry.

### 1.2 Sidechain Angle Features

Four chi torsion angles (chi1–chi4) per residue, each discretized into 20 bins over $[-\pi, \pi]$ + 1 overflow bin, one-hot encoded and concatenated with a 4-d validity mask.

| Step | Operation | Shape |
|------|-----------|-------|
| 1 | OpenFold `atom37_to_torsion_angles` | sin/cos pairs |
| 2 | `atan2(sin, cos)` | $[-\pi, \pi]$ |
| 3 | Bucketize into 21 bins | bin index |
| 4 | One-hot + flatten + append validity mask | [B, L, 88] |

## 2. Architecture

### 2.1 Encoder — `AtomisticImageEncoder`

`tokenizer_models/fullatom_image_encoder.py` — outputs latent `z ∈ [B, L, 256]`.

- Atom-level features: relative coords + bond lengths + backbone angles + sidechain angles
- Pairwise features: 37×37 distance image per residue, processed by a small 2D ResNet (`PairImageBackbone`)
- 6 atom transformer layers with pair-biased attention
- Pool 37 atoms → 1 residue vector → project to codebook dim (256)

Attention is intra-residue only (atom × atom within a residue); inter-residue context flows through fused sequence-level features.

### 2.2 Quantizer — `EMAQuantizer`

`tokenizer_models/quantizers.py` — codebook of 512 entries × 256 dims.

- Nearest-neighbor lookup → token index per residue
- EMA codebook updates (decay 0.99) and dead-code reset are training-only; in eval mode the codebook is fixed.

### 2.3 Decoder — `VanillaStructureTokenDecoder`

`tokenizer_models/decoder.py` — 4-layer transformer (d_model = 256). Built when the model is instantiated (to load checkpoint weights) but **not invoked** by the tokenizer path: `inference.py` calls the model with `use_as_tokenizer=True`, which returns immediately after the quantizer.

## 3. Inference Pipeline

```
PDB
  → strip HETATM/HOH
  → WrappedProteinChain.from_pdb()
  → atom37 coordinates [L, 37, 3]
  → centroid centering
  → Encoder:
       canonicalize coords (local frames)
       extract sidechain angles (chi1–4)
       atom transformer (6 layers)
       pool to residue level
       project to 256-dim
  → EMA Quantizer:
       nearest codebook lookup
       → structure token index per residue [L]
  → return {"fullatom_id": np.ndarray[int64, L]}
```

## 4. Package Layout

```
fullatom_tokenizer/
├── inference.py              CLI + tokenize() helper
├── verify.py                 Regression check vs frozen reference
├── integration.py            load_fullatom_tokenizer, tokenize_fullatom_structure
├── configs/
│   └── pretrain_fullatom_image.yaml
├── reference/
│   └── tokens.npz            frozen tokens for 10 CAMEO PDBs (gold for verify.py)
├── tokenizer_models/
│   ├── vqvae.py              VQVAEModel (encoder + quantizer + decoder)
│   ├── fullatom_image_encoder.py
│   ├── decoder.py
│   ├── quantizers.py
│   ├── modules/              attention + pair-update building blocks
│   └── layers/               transformer stack / blocks / rotary / geom attention
└── utils/
    ├── protein_chain.py      WrappedProteinChain (PDB/CIF loader)
    ├── coord_utils.py        residue-frame construction
    └── angle_utils.py        bond angle helper
```

## 5. Regenerating the Reference Snapshot

`reference/tokens.npz` is a frozen output from the original `favqvae` package. Re-create it from a sibling `favqvae/` checkout when either the upstream tokenizer logic or the checkpoint changes:

```python
# minimal recipe — uses the original favqvae alongside this package
import numpy as np
from pathlib import Path
from favqvae.integration import load_fullatom_tokenizer, tokenize_fullatom_structure

ckpt = "fullatom_tokenizer.pt"
pdbs = sorted(Path("/path/to/pdbs").glob("*.pdb"))[:10]
model = load_fullatom_tokenizer(ckpt, "pretrain_fullatom_image", device="cuda")
data = {p.name: np.asarray(tokenize_fullatom_structure(model, p)["struct_id"], np.int64)
        for p in pdbs}
np.savez("structure_tokenize/fullatom_tokenizer/reference/tokens.npz",
         _paths=np.array([str(p) for p in pdbs]), **data)
```

Then run `python -m structure_tokenize.fullatom_tokenizer.verify` to confirm the slim package still matches.
