"""Convenience helpers for our pretrained encoder.

Four entry points:

    load_encoder(size, ...)       build the model + load weights → eval-mode encoder
    encode(encoder, seq, bb, fa)  forward chain-A tokens → [L, D] fp16 numpy
    embed_pdb(encoder, pdb_path)  one-shot PDB → [L, D] fp16 numpy
    tokenize_pdb(pdb_path)        one-shot PDB → {seq, bb, fa} discrete token IDs
                                  (no encoder forward. Use these tokens with
                                   any downstream model, ours or yours.)

Weights / tokenizers can be loaded from a local path or pulled from
HuggingFace Hub:

    encoder  = load_encoder("650M", hf_repo="k-fold-structure/triprorep-650M")
    features = embed_pdb(encoder, "your_protein.pdb")
    tokens   = tokenize_pdb("your_protein.pdb",
                            hf_repo="k-fold-structure/triprorep-650M")

The +4 token shift, special-token reservation, and Lightning prefix
stripping are handled internally.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

# Make `from models import build_backbone` resolve no matter where this
# file is imported from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import build_backbone  # noqa: E402

_PAD_ID = 2
_NUM_SPECIAL = 4
_SIZES = ("35M", "150M", "650M", "3B")


def load_encoder(
    size: str,
    *,
    ckpt: Optional[str] = None,
    hf_repo: Optional[str] = None,
    hf_filename: Optional[str] = None,
    config: Optional[str] = None,
    device: str = "cuda",
):
    """Build + load the pretrained ELECTRA encoder.

    Args:
        size: one of "35M" / "150M" / "650M" / "3B" (selects the bundled config).
        ckpt: local path to weights (Lightning ``.ckpt`` or encoder-only ``.pt``).
        hf_repo: HuggingFace repo id; if given, downloads ``{size}.ckpt``
            (or ``hf_filename``) and uses it as ``ckpt``.
        hf_filename: override the filename to pull from ``hf_repo``.
        config: override the bundled config path.
        device: where to place the encoder ("cuda", "cpu", "cuda:1", …).

    Returns:
        The encoder (``nn.Module``) in ``eval()`` mode on ``device``.
    """
    if size not in _SIZES:
        raise ValueError(f"size must be one of {_SIZES}; got {size!r}")
    if ckpt is None and hf_repo is None:
        raise ValueError("Pass either ckpt= (local) or hf_repo= (HuggingFace).")
    if ckpt is None:
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(hf_repo, hf_filename or f"{size}.ckpt")
    if config is None:
        config = str(_HERE / "configs" / f"pretrain_{size}" / "pretrain_electra.yaml")

    cfg  = OmegaConf.load(config)
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    dcfg = OmegaConf.to_container(cfg.data,  resolve=True)
    fa   = dcfg.get("fa_vocab_size", 0)
    model = build_backbone(
        mcfg.pop("name"),
        pad_token_id   = _PAD_ID,
        seq_vocab_size = 33 + _NUM_SPECIAL,
        bb_vocab_size  = dcfg.get("bb_vocab_size", 512) + _NUM_SPECIAL,
        fa_vocab_size  = fa + _NUM_SPECIAL if fa else 0,
        **mcfg,
    )

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("_orig_mod.", "").replace("pretrained_model.", ""): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_params = sum(1 for _ in model.state_dict())
    if missing:
        frac = len(missing) / max(n_params, 1)
        msg = (f"[load_encoder] {len(missing)} / {n_params} model params were "
               f"NOT populated from the checkpoint ({100*frac:.1f}%). "
               f"First few: {list(missing)[:5]}")
        if frac > 0.10:
            raise RuntimeError(
                msg + "\nThis usually means a checkpoint / config mismatch "
                "(random weights would silently ship). Refusing to proceed. "
                "Verify `size` matches the checkpoint or pass an explicit "
                "`config=` override.")
        print(msg, file=sys.stderr)
    if unexpected:
        print(f"[load_encoder] {len(unexpected)} unexpected keys in checkpoint "
              f"(ignored). First few: {list(unexpected)[:5]}", file=sys.stderr)

    enc = model.discriminator if hasattr(model, "discriminator") else model
    return enc.eval().to(device)


@torch.no_grad()
def encode(encoder, seq_ids, bb_ids, fa_ids=None) -> np.ndarray:
    """Single-chain forward → ``[L, D]`` fp16 numpy features.

    Pass the raw token IDs from our benchmark LMDB (``apo_seq_A`` / ``apo_bb_A``
    / ``apo_fa_A``); the ``+_NUM_SPECIAL`` shift required by the encoder is
    applied inside.
    """
    device = next(encoder.parameters()).device
    seq = torch.from_numpy(np.asarray(seq_ids, dtype=np.int64) + _NUM_SPECIAL).unsqueeze(1).to(device)
    bb  = torch.from_numpy(np.asarray(bb_ids,  dtype=np.int64) + _NUM_SPECIAL).unsqueeze(1).to(device)
    fa  = (torch.from_numpy(np.asarray(fa_ids, dtype=np.int64) + _NUM_SPECIAL).unsqueeze(1).to(device)
           if fa_ids is not None else None)
    attn = torch.ones(1, seq.shape[0], dtype=torch.bool, device=device)
    emb = encoder(seq, bb, attn, fa, ret_embeddings=True)   # [1, L, D]
    return emb[0].cpu().numpy().astype(np.float16)


@torch.no_grad()
def encode_batch(encoder, records) -> list[np.ndarray]:
    """Batched chain-A forward, one padded batch → list of per-record ``[L_i, D]``
    fp16 numpy features (unpadded).

    ``records`` is a list of ``(seq_ids, bb_ids, fa_ids)`` triples. All three
    arrays per record must have the same length ``L_i``; ``fa_ids`` may be
    ``None`` for every record (must be uniformly None or uniformly non-None).

    Padding uses ``_PAD_ID``; the attention mask is set to ``True`` only on
    real positions so pad tokens do not leak into the representation.
    """
    if not records:
        return []
    have_fa = records[0][2] is not None
    device = next(encoder.parameters()).device
    B = len(records)
    lengths = [len(r[0]) for r in records]
    L_max = max(lengths)

    seq = torch.full((L_max, B), _PAD_ID, dtype=torch.int64, device=device)
    bb  = torch.full((L_max, B), _PAD_ID, dtype=torch.int64, device=device)
    fa  = (torch.full((L_max, B), _PAD_ID, dtype=torch.int64, device=device)
           if have_fa else None)
    attn = torch.zeros((B, L_max), dtype=torch.bool, device=device)
    for i, (s, b, f) in enumerate(records):
        L = len(s)
        seq[:L, i] = torch.from_numpy(np.asarray(s, dtype=np.int64) + _NUM_SPECIAL).to(device)
        bb[:L, i]  = torch.from_numpy(np.asarray(b, dtype=np.int64) + _NUM_SPECIAL).to(device)
        if have_fa:
            fa[:L, i] = torch.from_numpy(np.asarray(f, dtype=np.int64) + _NUM_SPECIAL).to(device)
        attn[i, :L] = True

    emb = encoder(seq, bb, attn, fa, ret_embeddings=True)   # [B, L_max, D]
    emb = emb.to(torch.float16).cpu().numpy()
    return [emb[i, :L] for i, L in enumerate(lengths)]


# ---------------------------------------------------------------------------
# PDB → features (one-shot helper, downloads + caches tokenizers on first call)
# ---------------------------------------------------------------------------
# Tracks the (bb_ckpt, fa_ckpt, fa_cfg, device) tuple the tokenizer worker
# was initialized with. `None` means uninitialized. A subsequent call with a
# different tuple emits a warning explaining the sticky state, then re-inits.
_TOKENIZER_STATE: Optional[tuple[str, str, str, str]] = None


def _parse_gpu_index(device: str) -> Optional[int]:
    """`cuda` -> 0, `cuda:3` -> 3, `cpu` -> None."""
    if not device or not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0


def reset_tokenizers() -> None:
    """Forget the current tokenizer worker so the next `embed_pdb` /
    `tokenize_pdb` call re-initializes with fresh ckpts + device."""
    global _TOKENIZER_STATE
    _TOKENIZER_STATE = None


def _init_tokenizers(bb_ckpt: str, fa_ckpt: str, fa_cfg: str = "pretrain_fullatom_image",
                     device: str = "cuda") -> None:
    """Lazy tokenizer worker setup, keyed on (bb_ckpt, fa_ckpt, fa_cfg, device).

    The PDB tokenizer code (``code/tokenizer/``) must be importable. Adds it
    to ``sys.path`` automatically if it lives next to this package.
    """
    global _TOKENIZER_STATE
    identity = (bb_ckpt, fa_ckpt, fa_cfg, device)
    if _TOKENIZER_STATE == identity:
        return
    if _TOKENIZER_STATE is not None:
        print(
            f"[triprorep] re-initializing tokenizer worker: prior state was "
            f"{_TOKENIZER_STATE}, requested {identity}. Call `reset_tokenizers()` "
            f"first to silence this warning.", file=sys.stderr,
        )
    tok_root = _HERE.parent / "tokenizer"   # code/tokenizer/ sibling to code/triprorep/
    if not tok_root.exists():
        raise RuntimeError(
            f"Tokenizer package not found at {tok_root}. Clone the GitHub "
            "release and run from its root, or point PYTHONPATH at "
            "code/tokenizer/.")
    if str(tok_root) not in sys.path:
        sys.path.insert(0, str(tok_root))
    from preprocess import StructInitConfig, init_structure_worker
    gpu_idx = _parse_gpu_index(device)
    init_structure_worker(StructInitConfig(
        struct_format="fullatom_image",
        bb_ckpt_path=bb_ckpt,
        fa_ckpt_path=fa_ckpt,
        fa_cfg=fa_cfg,
        gpu_ids=[gpu_idx] if gpu_idx is not None else None,
    ))
    _TOKENIZER_STATE = identity


def embed_pdb(
    encoder,
    pdb_path: str,
    *,
    hf_repo: Optional[str] = None,
    bb_ckpt: Optional[str] = None,
    fa_ckpt: Optional[str] = None,
    chain: Optional[str] = "A",
) -> np.ndarray:
    """One-shot: ``pdb_path`` → ``[L, D]`` fp16 features.

    Downloads + initializes the (backbone, full-atom) tokenizers on first call.
    Pass ``hf_repo`` to pull them from the same HF model repo, or pre-supply
    local paths via ``bb_ckpt`` / ``fa_ckpt``.

    Args:
        encoder: returned by ``load_encoder(...)``.
        pdb_path: path to the input PDB file.
        hf_repo: HF model repo (e.g. ``"k-fold-structure/triprorep-650M"``).
            Required on first call unless ``bb_ckpt`` + ``fa_ckpt`` are set.
        bb_ckpt: local path to ``backbone_tokenizer.pt`` (overrides HF download).
        fa_ckpt: local path to ``fullatom_tokenizer.pt`` (overrides HF download).
        chain: chain ID to extract (default: "A"; pass explicit None to let the
            tokenizer pick the first chain the source file iterates, which is
            non-deterministic on multi-chain PDBs).
    """
    if bb_ckpt is None or fa_ckpt is None:
        if hf_repo is None:
            raise ValueError("Provide hf_repo= or both bb_ckpt + fa_ckpt.")
        from huggingface_hub import hf_hub_download
        bb_ckpt = bb_ckpt or hf_hub_download(hf_repo, "backbone_tokenizer.pt")
        fa_ckpt = fa_ckpt or hf_hub_download(hf_repo, "fullatom_tokenizer.pt")
    device = str(next(encoder.parameters()).device)
    _init_tokenizers(bb_ckpt, fa_ckpt, device=device)

    from preprocess import encode_structure
    result = encode_structure(pdb_path, "fullatom_image", chain=chain)
    if result is None:
        raise RuntimeError(f"Tokenization failed for {pdb_path}")
    return encode(
        encoder,
        result["seq_id"],
        result["struct_id_aminoaseed"],
        result["struct_id_fullatom"],
    )


@torch.no_grad()
def tokenize_pdb(
    pdb_path: str,
    *,
    hf_repo: Optional[str] = None,
    bb_ckpt: Optional[str] = None,
    fa_ckpt: Optional[str] = None,
    chain: Optional[str] = "A",
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """One-shot: ``pdb_path`` → ``{seq, bb, fa}`` discrete token IDs (no encoder).

    Returns three per-residue int arrays of shape ``(L,)`` for the selected chain:

    * ``seq``: ESM2-style amino-acid token IDs (vocab = 33 + 4 special).
    * ``bb``:  backbone-geometry token IDs (codebook size 512, StructTokenBench).
    * ``fa``:  full-atom token IDs (codebook size 512, our full-atom VQ-VAE).

    Drop these into any downstream model (your classifier / generator / etc.).
    Pass the same array trio to ``encode(...)`` to get the encoder embedding.

    Args:
        pdb_path: path to the input PDB file.
        hf_repo: HF model repo to pull the tokenizers from on first call
            (e.g. ``"k-fold-structure/triprorep-650M"``).
        bb_ckpt / fa_ckpt: local tokenizer paths (override HF download).
        chain: chain ID to extract (default: ``"A"``; pass explicit ``None``
            to let the tokenizer pick the first chain the source iterates,
            which is non-deterministic on multi-chain PDBs).
        device: ``"cuda"`` (default), ``"cuda:N"``, or ``"cpu"``.
    """
    if bb_ckpt is None or fa_ckpt is None:
        if hf_repo is None:
            raise ValueError("Provide hf_repo= or both bb_ckpt + fa_ckpt.")
        from huggingface_hub import hf_hub_download
        bb_ckpt = bb_ckpt or hf_hub_download(hf_repo, "backbone_tokenizer.pt")
        fa_ckpt = fa_ckpt or hf_hub_download(hf_repo, "fullatom_tokenizer.pt")
    _init_tokenizers(bb_ckpt, fa_ckpt, device=device)

    from preprocess import encode_structure
    result = encode_structure(pdb_path, "fullatom_image", chain=chain)
    if result is None:
        raise RuntimeError(f"Tokenization failed for {pdb_path}")
    return {
        "seq": np.asarray(result["seq_id"], dtype=np.int64),
        "bb":  np.asarray(result["struct_id_aminoaseed"], dtype=np.int64),
        "fa":  np.asarray(result["struct_id_fullatom"], dtype=np.int64),
    }
