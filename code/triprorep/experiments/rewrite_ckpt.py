import argparse
import os
from typing import Any, Dict

import torch


def rewrite_checkpoint(src: str, dst: str) -> None:
    checkpoint = torch.load(src, map_location="cpu")
    state_dict: Dict[str, Any] = checkpoint.get("state_dict", {})

    if any("_orig_mod." in key for key in state_dict):
        checkpoint["state_dict"] = {
            key.replace("_orig_mod.", ""): value for key, value in state_dict.items()
        }
        state_dict = checkpoint["state_dict"]

    prefix = "pretrained_model."
    if any(key.startswith(prefix) for key in state_dict):
        checkpoint["state_dict"] = {
            (key[len(prefix) :] if key.startswith(prefix) else key): value
            for key, value in state_dict.items()
        }
        state_dict = checkpoint["state_dict"]

    checkpoint["state_dict"] = {
        key: value for key, value in state_dict.items() if not key.startswith("head.")
    }

    target_dir = os.path.dirname(dst)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    torch.save(checkpoint, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip _orig_mod. prefix from checkpoint keys."
    )
    parser.add_argument("--ckpt", help="Path to the source checkpoint.")
    parser.add_argument("--target", help="Path to save the rewritten checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    rewrite_checkpoint(args.ckpt, args.target)


if __name__ == "__main__":
    main()
