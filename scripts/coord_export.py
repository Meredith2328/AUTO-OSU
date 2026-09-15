"""Turn an accelerate checkpoint of the coordinate model into the two files the inference pipeline loads.

    python scripts/coord_export.py <checkpoint-dir> <out-dir>

accelerate.save_state writes the registered objects as custom_checkpoint_N.pkl in registration order:
0 = EMA model state dict, 1 = tokenizer state dict (see coord/osu_diffusion/train.py). The pipeline's
load_diff_model expects <out-dir>/model_ema.pkl and <out-dir>/tokenizer.pkl.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    ema, tok = src / "custom_checkpoint_0.pkl", src / "custom_checkpoint_1.pkl"
    for p in (ema, tok):
        if not p.exists():
            raise SystemExit(f"missing {p}; contents: {[q.name for q in src.iterdir()]}")
    ema_state = torch.load(ema, map_location="cpu", weights_only=False)
    tok_state = torch.load(tok, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ema_state.values() if hasattr(v, "numel"))
    print(f"ema state: {len(ema_state)} tensors, {n_params / 1e6:.1f}M params; tokenizer keys: {list(tok_state)[:6]}")
    shutil.copyfile(ema, dst / "model_ema.pkl")
    shutil.copyfile(tok, dst / "tokenizer.pkl")
    print(f"-> {dst}/model_ema.pkl, {dst}/tokenizer.pkl")


if __name__ == "__main__":
    main()
