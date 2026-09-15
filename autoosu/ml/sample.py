"""Generate rhythm events for a song with a trained tick model (either paradigm), in the same
RhythmEvent format the rule-based rhythm layer produces, so placement / hitsounds / export are shared."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from ..audio import AudioAnalysis
from ..difficulty import DifficultyPreset
from ..rhythm import RhythmEvent, Sections, _combos_and_hitsounds, effective_grid, tick_features
from ..timing import Timing
from .dataset import BOS, FRAME_MS, GRID, cond_vector, extra_features, gather_patches, local_loudness
from .model import TickTransformer, sample_ar, sample_masked
from .prepare_data import HOP, MEL_HI, MEL_LO, N_FFT, N_MELS, SR


def song_mel_uint8(audio_path: str | Path) -> np.ndarray:
    """Same features as prepare_data, computed from any audio file."""
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    y = y / (float(np.abs(y).max()) or 1.0)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=8000)
    lm = np.log(mel + 1e-6).T
    return (np.clip((lm - MEL_LO) / (MEL_HI - MEL_LO), 0, 1) * 255).astype(np.uint8)


@torch.no_grad()
def generate_rhythm(model: TickTransformer, mel: np.ndarray, timing: Timing, preset: DifficultyPreset,
                    analysis: AudioAnalysis, sections: Sections, *, star_rating: Optional[float] = None,
                    year: int = 2023, density: Optional[float] = None, temperature: float = 0.9,
                    none_bias: float = 0.0, decode_steps: int = 12, context: int = 1024, seed: int = 0,
                    device: str = "cuda") -> List[RhythmEvent]:
    """density: target objects per measure (conditioning); None = let the model decide."""
    bl = timing.beat_length
    step = bl / GRID
    first_beat = math.ceil(timing.beat_at(0.0))
    n_ticks = int((len(mel) * FRAME_MS - timing.ms_at(first_beat)) / step)
    beats = first_beat + np.arange(n_ticks) / GRID
    times = timing.offset_ms + beats * bl
    metrical = (np.arange(n_ticks) % (4 * GRID)).astype(np.int64)
    frames = np.round(times / FRAME_MS).astype(np.int64)
    loud = local_loudness(mel, frames)
    dens = np.full(n_ticks, float(density), dtype=np.float32) if density is not None else None
    extra = extra_features(loud, np.full(n_ticks, bl), dens)
    audio = torch.from_numpy(gather_patches(mel, frames).astype(np.float32) / 255.0).to(device)
    extra_t = torch.from_numpy(extra).to(device)
    met = torch.from_numpy(metrical).to(device)
    cond = torch.from_numpy(cond_vector(dict(
        sr=star_rating if star_rating is not None else preset.star,
        cs=preset.cs, ar=preset.ar, od=preset.od, hp=preset.hp, year=year))).to(device)

    gen = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    if model.cfg.mode == "ar":
        labels = sample_ar(model, audio, met, extra_t, cond, temperature=temperature, none_bias=none_bias,
                           context=context, generator=gen, prev0=BOS)
    else:
        labels = sample_masked(model, audio, met, extra_t, cond, steps=decode_steps, temperature=temperature,
                               none_bias=none_bias, generator=gen)

    # tick labels -> events; drum / melody features for hitsounds and placement come from the analyser
    grid = effective_grid(analysis, timing, preset)
    feats = {round(t.beat, 6): t for t in tick_features(analysis, timing, grid)}
    events: List[RhythmEvent] = []
    i = 0
    while i < n_ticks:
        c = labels[i]
        if c in (1, 2, 5):
            b = float(beats[i])
            tk = feats.get(round(b, 6))
            ev = RhythmEvent(time=timing.ms_at(b), beat=b, intensity=sections.at(b),
                             strength=tk.score if tk else 0.5, percussive=tk.percussive if tk else 0.5,
                             kick=tk.kick if tk else 0.0, snare=tk.snare if tk else 0.0)
            if c == 2:
                j = i + 1
                while j < n_ticks and labels[j] == 3:
                    j += 1
                end = j if j < n_ticks and labels[j] == 4 else j - 1
                end = max(end, i + 1)
                if end < n_ticks:
                    ev.kind, ev.end_beat = "slider", float(beats[end])
                    ev.end_time = timing.ms_at(ev.end_beat)
                    i = end
            elif c == 5:
                j = i
                while j + 1 < n_ticks and labels[j + 1] == 5:
                    j += 1
                if (j - i) * step >= preset.min_spinner_ms:
                    ev.kind, ev.end_beat = "spinner", float(beats[j])
                    ev.end_time = timing.ms_at(ev.end_beat)
                    i = j
                else:
                    i += 1
                    continue
            events.append(ev)
        i += 1
    _combos_and_hitsounds(events, preset)
    return events
