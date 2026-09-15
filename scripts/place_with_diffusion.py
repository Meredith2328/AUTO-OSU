"""Re-place an existing beatmap's objects with Mapperatorinator's osu-diffusion position model.

Keeps our rhythm (times, object types, slider lengths, hitsounds, combos, timing) and only regenerates
x/y positions and slider anchor shapes, i.e. "our rhythm layer + their coordinate layer".

Run with the Mapperatorinator environment:
    external/Mapperatorinator/.venv/Scripts/python scripts/place_with_diffusion.py --osz "out/ml_v0/masked/....osz" --out out/ml_v0/masked_diff
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPP = ROOT / "external" / "Mapperatorinator"


def main() -> None:
    for stream in (sys.stdout, sys.stderr):      # Japanese titles vs the Windows console code page
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--osz", required=True, help="beatmap package to re-place (first .osu inside is used)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--difficulty", type=float, default=None, help="star rating for the diffusion condition")
    ap.add_argument("--cs", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--version", default=None, help="difficulty name written to the new .osu")
    ap.add_argument("--cfg-scale", type=float, default=1.0, help="classifier-free guidance for the diffusion")
    ap.add_argument("--diff-ckpt", default=None, help="local directory with model_ema.pkl + tokenizer.pkl (our coordinate model); default = released osu-diffusion")
    ap.add_argument("--zero-distances", action="store_true", help="give the model no spacing hint (use with our from-scratch model)")
    ap.add_argument("--steps", type=int, default=None, help="sampling steps per tenth of the noise schedule, e.g. 10 -> 100 steps over the full process")
    ap.add_argument("--max-stretch", type=float, default=1.5,
                    help="scale slider anchors up to this factor to reach the required length before lowering SV (their default 1.5; use ~6 for our from-scratch model)")
    ap.add_argument("--random-init", action="store_true",
                    help="start from pure noise instead of refining the existing positions (the released osu-diffusion "
                         "checkpoint is a refiner trained on <=100/1000 noise steps, so leave this off)")
    a = ap.parse_args()

    # work dir without spaces (hydra override quoting) holding the extracted .osu and audio
    out_dir = (ROOT / a.out).resolve() if not Path(a.out).is_absolute() else Path(a.out)
    work = out_dir / ".work_diff"
    work.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.osz) as z:
        osu_name = next(n for n in z.namelist() if n.endswith(".osu"))
        audio_name = next(n for n in z.namelist() if n.lower().endswith((".mp3", ".ogg")))
        (work / "map.osu").write_bytes(z.read(osu_name))
        (work / ("audio" + Path(audio_name).suffix)).write_bytes(z.read(audio_name))
    osu_path = work / "map.osu"
    audio_path = work / ("audio" + Path(audio_name).suffix)
    diff_ckpt = Path(a.diff_ckpt).resolve() if a.diff_ckpt else None   # resolve before chdir

    os.chdir(MAPP)
    sys.path.insert(0, str(MAPP))
    import hydra
    import torch
    from omegaconf import OmegaConf
    import config  # noqa: F401  (registers the structured configs)
    from inference import compile_args, get_config, load_diff_model, setup_inference_environment
    from diffusion_pipeline import DiffisionPipeline
    from osuT5.osuT5.inference import Postprocessor
    from osuT5.osuT5.tokenizer import Tokenizer
    from osuT5.osuT5.dataset.osu_parser import OsuParser
    from slider import Beatmap

    overrides = [
        f"beatmap_path='{osu_path}'", f"audio_path='{audio_path}'", f"output_path='{out_dir}'",
        "gamemode=0", "generate_positions=true", f"random_init={str(a.random_init).lower()}", "export_osz=false",
        f"seed={a.seed}", f"diff_cfg_scale={a.cfg_scale}",
    ]
    if a.difficulty is not None:
        overrides.append(f"difficulty={a.difficulty}")
    if diff_ckpt:
        overrides.append(f"diff_ckpt='{diff_ckpt}'")
    if a.steps:
        overrides.append("timesteps=[" + ",".join([str(a.steps)] * 10) + "]")
    if a.cs is not None:
        overrides.append(f"circle_size={a.cs}")
    with hydra.initialize_config_dir(config_dir=str(MAPP / "configs" / "inference"), version_base="1.1"):
        cfg = hydra.compose(config_name="v32", overrides=overrides)
    args = OmegaConf.to_object(cfg)
    compile_args(args)
    if a.difficulty is not None:
        args.difficulty = a.difficulty
    if a.version:
        args.version = a.version
    setup_inference_environment(args.seed)
    print(f"device {args.device}  difficulty {args.difficulty}  cs {args.circle_size}  creator {args.creator}  version {args.version}")

    # our map -> Mapperatorinator events (exact positions, which the diffusion model will replace)
    tokenizer = Tokenizer.from_pretrained("OliBomby/Mapperatorinator-v32", subfolder="gamemode=0")
    parser = OsuParser(args.train, tokenizer)
    parser.position_precision = 1
    parser.position_split_axes = True
    bm = Beatmap.from_path(osu_path)
    events, _ = parser.parse(bm)
    timing = [tp for tp in bm.timing_points if tp.parent is None]
    print(f"{len(bm._hit_objects)} hit objects -> {len(events)} events, {len(timing)} red line(s)")

    diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)
    generation_config, beatmap_config = get_config(args)
    pipeline = DiffisionPipeline(args, diff_model, diff_tokenizer, None)
    pipeline.zero_distances = a.zero_distances
    with torch.no_grad():
        events = pipeline.generate(events=events, generation_config=generation_config, timing=timing, verbose=True)

    post = Postprocessor(args)
    post.max_slider_stretch = a.max_stretch
    result = post.generate(events=events, beatmap_config=beatmap_config, timing=timing)
    name = f"{beatmap_config.artist} - {beatmap_config.title} ({beatmap_config.creator}) [{beatmap_config.version}]"
    out_osz = out_dir / f"{name}.osz"
    post.export_osz(str(out_osz), result, f"{name}.osu", str(audio_path), None)
    (out_dir / f"{name}.osu").write_text(result, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"-> {out_osz}")


if __name__ == "__main__":
    main()
