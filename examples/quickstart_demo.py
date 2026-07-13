"""Zero-arg demo. Loads the 650M encoder, embeds the bundled example PDB.

Run from the repo root:

    python examples/quickstart_demo.py

Prints the embedding shape. First run downloads the encoder checkpoint +
tokenizers from HuggingFace (about 3 GB) and caches them under HF_HOME.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code" / "triprorep"))

from inference import load_encoder, embed_pdb

PDB = _HERE.parent / "_assets" / "example_monomer.pdb"
REPO = "k-fold-structure/triprorep-650M"

encoder = load_encoder("650M", hf_repo=REPO)
features = embed_pdb(encoder, str(PDB), hf_repo=REPO)

print(f"input:      {PDB}")
print(f"features:   shape={tuple(features.shape)} dtype={features.dtype}")
print(f"  (L, D) = ({features.shape[0]} residues, {features.shape[1]}-dim fp16)")
