"""Dedicated mania 4K inference; rules here only enforce legal note intervals."""
from __future__ import annotations

import numpy as np
import torch

from ..beatmap import Circle, Hold
from ..difficulty import DifficultyPreset
from ..rhythm import RhythmEvent, Sections
from ..timing import Timing
from .dataset import gather_patches
from .mania_data import FRAME_MS, grid_from_redlines
from .mania_model import ManiaNet


def choose_device(requested: str = "auto") -> str:
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this PyTorch runtime")
    if requested not in ("cpu", "mps"):
        raise ValueError("Mania model device must be auto, mps, or cpu")
    torch.set_num_threads(2)
    if requested == "mps":
        torch.mps.set_per_process_memory_fraction(
            min(1.0, (1024 ** 3) / torch.mps.recommended_max_memory()))
    return requested


@torch.inference_mode()
def predict_logits(model: ManiaNet, mel: np.ndarray, times: np.ndarray, phase: np.ndarray,
                   beat_ms: np.ndarray, target_nps: float, od: float, device: str,
                   chunk: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lane_output, chord_output, duration_output = [], [], []
    cond = torch.tensor([[target_nps / 12.0, od / 10.0]], dtype=torch.float32, device=device)
    model.eval()
    halo = 2 * sum(2 ** i for i in range(model.cfg.blocks))
    for start in range(0, len(times), chunk):
        lo, hi = max(0, start - halo), min(len(times), start + chunk + halo)
        sl = slice(lo, hi)
        frames = np.rint(times[sl] / FRAME_MS).astype(np.int64)
        patches = gather_patches(mel, frames).astype(np.float32) / 255.0
        lane_logits, chord_logits, duration_logits = model(torch.from_numpy(patches).unsqueeze(0).to(device),
                       torch.from_numpy(phase[sl].astype(np.int64)).unsqueeze(0).to(device),
                       torch.from_numpy(beat_ms[sl].astype(np.float32)).unsqueeze(0).to(device),
                       cond)
        kept = slice(start - lo, start - lo + min(chunk, len(times) - start))
        lane_output.append(lane_logits[0, kept].float().cpu().numpy())
        chord_output.append(chord_logits[0, kept].float().cpu().numpy())
        duration_output.append(duration_logits[0, kept].float().cpu().numpy())
    if not lane_output:
        return (np.zeros((0, 4, 5), np.float32), np.zeros((0, 16), np.float32),
                np.zeros((0, 4, 33), np.float32))
    return (np.concatenate(lane_output, axis=0), np.concatenate(chord_output, axis=0),
            np.concatenate(duration_output, axis=0))


def _probabilities(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def decode_predictions(logits: np.ndarray, chord_logits: np.ndarray,
                       duration_logits: np.ndarray, times: np.ndarray, *,
                       threshold: float, duration_ms: float, shift_ms: int = 0,
                       hold_threshold: float = 0.45, seed: int = 0,
                       mask_temperature: float = 1.0
                       ) -> list[tuple[int, int, int, float]]:
    """Return (start ms, lane, end ms, confidence); a tap has end==start."""
    if not len(times):
        return []
    chords = _probabilities(chord_logits)
    durations = _probabilities(duration_logits)
    any_onset = 1 - chords[:, 0]
    rng = np.random.default_rng(seed)
    conditional = _probabilities(chord_logits[:, 1:] / mask_temperature)
    masks = np.asarray([rng.choice(np.arange(1, 16), p=p) if any_onset[i] >= threshold else 0
                        for i, p in enumerate(conditional)], dtype=np.int64)
    free_after = [-1] * 4
    notes = []
    for i, t in enumerate(times):
        if any_onset[i] < threshold:
            continue
        for lane in range(4):
            if not masks[i] & (1 << lane) or i <= free_after[lane]:
                continue
            start = int(round(float(t))) - shift_ms
            if start < 0 or start >= duration_ms:
                continue
            end_index = i
            hold_prob = 1 - durations[i, lane, 0]
            if hold_prob >= hold_threshold:
                length = int(np.argmax(durations[i, lane, 1:])) + 1
                end_index = min(len(times) - 1, i + length)
            end = min(int(duration_ms), int(round(float(times[end_index]))) - shift_ms)
            if end - start < 60:
                end_index, end = i, start
            free_after[lane] = end_index
            notes.append((start, lane, end, float(any_onset[i])))
    return sorted(notes)


def generate_model_objects(mel: np.ndarray, timing: Timing, duration_ms: float,
                           preset: DifficultyPreset, sections: Sections, model: ManiaNet,
                           checkpoint: dict, device: str, shift_ms: int,
                           target_nps: float | None = None, threshold: float | None = None,
                           seed: int = 0
                           ) -> tuple[list[RhythmEvent], list[Circle | Hold]]:
    if target_nps is None:
        target_nps = float(checkpoint["difficulty_nps"][preset.name])
    if not np.isfinite(target_nps) or target_nps <= 0:
        raise ValueError("mania target NPS must be finite and positive")
    threshold = float(checkpoint.get("threshold", 0.35) if threshold is None else threshold)
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("mania onset threshold must be finite and in (0, 1]")
    hold_threshold = float(checkpoint.get("hold_threshold", 0.45))
    red = [(timing.offset_ms, timing.beat_length, 4)]
    times, phase, beats = grid_from_redlines(red, duration_ms)
    lane_logits, chord_logits, duration_logits = predict_logits(
        model, mel, times, phase, beats, target_nps, preset.od, device)
    decoded = decode_predictions(lane_logits, chord_logits, duration_logits, times, threshold=threshold,
                                 duration_ms=duration_ms, shift_ms=shift_ms,
                                 hold_threshold=hold_threshold, seed=seed,
                                 mask_temperature=float(checkpoint.get("mask_temperature", 1.0)))
    events, objects = [], []
    lane_x = (64, 192, 320, 448)
    for start, lane, end, confidence in decoded:
        raw_start, raw_end = start + shift_ms, end + shift_ms
        beat = timing.beat_at(raw_start)
        hold = end > start
        events.append(RhythmEvent(time=raw_start, beat=beat, kind="hold" if hold else "circle",
                                  strength=confidence, intensity=sections.at(beat), lane=lane,
                                  end_time=raw_end if hold else 0,
                                  end_beat=timing.beat_at(raw_end) if hold else 0.0))
        if hold:
            objects.append(Hold(lane_x[lane], 192, raw_start, end=raw_end))
        else:
            objects.append(Circle(lane_x[lane], 192, raw_start))
    return events, objects
