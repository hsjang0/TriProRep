"""
Helpers that bridge the PLM codebase with the favqvae tokenizer.
"""
from pathlib import Path
import tempfile
import os
import sys
from typing import Dict, Iterable, Optional

import biotite.structure.io.pdb as pdb
import torch
from omegaconf import OmegaConf

if __package__ is None or __package__ == "":
    # add the tokenizer component root (parent of structure_tokenize/)
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from structure_tokenize.fullatom_tokenizer.tokenizer_models.vqvae import VQVAEModel
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from structure_tokenize.fullatom_tokenizer.utils.protein_chain import WrappedProteinChain
from openfold.np.residue_constants import restypes as openfold_restypes

_seq_tokenizer: Optional[EsmSequenceTokenizer] = None


def load_fullatom_tokenizer(ckpt_path: str, cfg_name: str, device: str = "cuda") -> VQVAEModel:
    """
    Load the VQVAE tokenizer weights and freeze the model for inference-only use.
    """
    cfg_path = Path(__file__).resolve().parent / "configs" / f"{cfg_name}.yaml"
    model_cfg = OmegaConf.load(cfg_path).model
    model_cfg.quantizer._need_init = False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt.get("module", ckpt))
    new_state_dict = {}
    for key, value in state_dict.items():
        # Strip Lightning module prefix
        if key.startswith("model."):
            key = key[6:]
        # Strip torch.compile prefix (compiled checkpoints use _orig_mod.)
        key = key.replace("_orig_mod.", "")
        new_state_dict[key] = value

    model = VQVAEModel(model_cfg)
    incompat = model.load_state_dict(new_state_dict, strict=False)

    # Allow known non-critical keys
    allowed_missing = {"_restype_lut", "chi_angle_proj"}  # CPU-only LUT, rebuilt in __init__; chi head absent in older ckpts
    allowed_unexpected = {"quantizer._need_init", "quantizer._ema_initialized"}
    missing = [k for k in incompat.missing_keys if not any(a in k for a in allowed_missing)]
    unexpected = [k for k in incompat.unexpected_keys if k not in allowed_unexpected]
    if missing or unexpected:
        missing_msg = f"missing keys: {missing}" if missing else ""
        unexpected_msg = f"unexpected keys: {unexpected}" if unexpected else ""
        details = ", ".join([msg for msg in [missing_msg, unexpected_msg] if msg])
        raise RuntimeError(f"Error(s) in loading state_dict for VQVAEModel ({details})")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _get_seq_tokenizer() -> EsmSequenceTokenizer:
    global _seq_tokenizer
    if _seq_tokenizer is None:
        _seq_tokenizer = EsmSequenceTokenizer()
    return _seq_tokenizer


def tokenize_fullatom_structure(
    model: VQVAEModel,
    pdb_path: Path,
    chains: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """
    Tokenize a structure file into sequence and structure IDs using the favqvae tokenizer.
    """
    pdb_path = Path(pdb_path)

    def _filter_pdb(path: Path) -> tuple[Path, bool]:
        if path.suffix.lower() != ".pdb":
            return path, False
        fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        os.close(fd)
        with open(path, "r") as src, open(tmp_path, "w") as dst:
            for line in src:
                if line.startswith("HETATM"):
                    continue
                if line.startswith("ATOM") and line[17:20].strip() == "HOH":
                    continue
                dst.write(line)
        return Path(tmp_path), True

    filtered_path, is_temp = _filter_pdb(pdb_path)
    try:
        if chains is None:
            try:
                atom_array = pdb.PDBFile.read(str(filtered_path)).get_structure(model=1)
            except (ValueError, Exception) as e:
                print(f"Warning: failed to parse {pdb_path}: {e}")
                return None
            chains = list(dict.fromkeys(atom_array.chain_id)) if len(atom_array.chain_id) > 0 else ["A"]
        chain_id = next(iter(chains))
        try:
            pdb_chain_full = WrappedProteinChain.from_pdb(filtered_path, chain_id=chain_id)
        except (ValueError, Exception) as e:
            print(f"Warning: failed to load chain {chain_id} from {pdb_path}: {e}")
            return None
        device = next(model.parameters()).device
        if len(pdb_chain_full) == 0:
            print(f"Warning: empty chain {chain_id} in {pdb_path}")
            return None
        coords = torch.from_numpy(pdb_chain_full.atom37_positions).float().to(device)  # (L, 37, 3)
        atom_mask = torch.all(
            torch.isfinite(coords) & (coords < 1e6),
            dim=-1,
        )
        num_atoms = atom_mask.sum()
        if num_atoms == 0:
            print(f"Warning: no valid atoms for chain {chain_id} in {pdb_path}")
            return None
        # Match training pipeline: invalid atoms → inf (not nan/0).
        # The collate_fn in training pads with torch.inf; _prepare_inputs
        # then checks isfinite to build the mask.  We replicate that here.
        coords = coords.masked_fill((~atom_mask)[..., None], float('inf'))
        coords = coords.unsqueeze(0)  # [1, L, 37, 3]
        residue_index = torch.from_numpy(pdb_chain_full.residue_index).long().unsqueeze(0).to(device)  # [1, L]

        if coords.numel() == 0 or coords.shape[1] == 0:
            print(f"Warning: no coordinates for chain {chain_id} in {pdb_path}")
            return None
        seq = pdb_chain_full.sequence
        if not any(aa in openfold_restypes for aa in seq):
            print(f"Warning: no standard residues for chain {chain_id} in {pdb_path}")
            return None
    except Exception as e:
        print(f"Warning: unexpected error processing {pdb_path}: {e}")
        return None
    finally:
        if is_temp and filtered_path.exists():
            filtered_path.unlink()

    attention_mask = torch.isfinite(coords[:, :, 0, 0])  # [1, L]
    pdb_chain = [pdb_chain_full]

    seq_tokenizer = _get_seq_tokenizer()
    seq_id = seq_tokenizer.encode(seq, add_special_tokens=False)
    seq_id = torch.tensor(seq_id, dtype=torch.int64, device=device).unsqueeze(0)  # [1, L]
    input_list = (coords, attention_mask, residue_index, seq_id, pdb_chain)

    try:
        with torch.no_grad(), torch.autocast(device_type=device.type if isinstance(device, torch.device) else device, enabled=False):
            _, fullatom_id, _ = model(input_list, use_as_tokenizer=True)
        fullatom_id = fullatom_id.squeeze(0)  # [L]
        return {
            "fullatom_id": fullatom_id.cpu().numpy(),
        }
    except Exception as e:
        print(f"Warning: inference failed for {pdb_path}: {e}")
        return None


__all__ = ["load_fullatom_tokenizer", "tokenize_fullatom_structure"]

