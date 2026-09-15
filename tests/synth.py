"""Synthetic test song: drums + sustained melody at a known BPM/offset.

Run directly to create a wav for manual testing:
    python tests/synth.py samples/synth_128.wav --bpm 128 --bars 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def _env(n: int, decay: float, sr: int) -> np.ndarray:
    return np.exp(-np.arange(n) / sr * decay)


def make_song(path: Path, bpm: float = 128.0, bars: int = 24, sr: int = 22050,
              offset_s: float = 0.5, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    bl = 60.0 / bpm
    total = offset_s + bars * 4 * bl + 1.0
    y = np.zeros(int(total * sr))

    def add(t: float, sig: np.ndarray) -> None:
        i = int(round(t * sr))
        j = min(len(y), i + len(sig))
        if j > i:
            y[i:j] += sig[: j - i]

    n = int(0.18 * sr)
    tt = np.arange(n) / sr
    f = 150 * np.exp(-tt * 25) + 45
    kick = np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, 18, sr) * 0.9

    n = int(0.15 * sr)
    snare = (rng.normal(size=n) * 0.5 + np.sin(2 * np.pi * 190 * np.arange(n) / sr) * 0.4) * _env(n, 30, sr) * 0.6

    n = int(0.05 * sr)
    hat = np.diff(rng.normal(size=n + 1)) * _env(n, 110, sr) * 0.22

    scale = [220.0, 246.9, 261.6, 293.7, 329.6, 349.2, 392.0, 440.0]
    bass_notes = [55.0, 73.4, 65.4, 82.4]
    for bar in range(bars):
        chorus = 8 <= bar < 16 or bar >= 20
        # bass line: one note per bar, changes on the downbeat
        t = offset_s + bar * 4 * bl
        n = int(3.9 * bl * sr)
        tt = np.arange(n) / sr
        freq = bass_notes[bar % 4]
        bass = (np.sin(2 * np.pi * freq * tt) + 0.5 * np.sin(2 * np.pi * 2 * freq * tt))
        add(t, bass * np.minimum(1.0, tt / 0.01) * np.minimum(1.0, (3.9 * bl - tt) / 0.05) * 0.3)
        for beat in range(4):
            t = offset_s + (bar * 4 + beat) * bl
            if beat in (0, 2):
                add(t, kick)
            else:
                add(t, snare)
            add(t, hat)
            add(t + bl / 2, hat)
            if chorus and beat in (1, 3):
                add(t + bl / 2, kick * 0.7)
            if chorus and beat == 3:
                add(t + bl * 0.75, snare * 0.5)
        # melody: one sustained note every 2 beats, held for ~90 % of its slot
        for half in range(2):
            t = offset_s + (bar * 4 + half * 2) * bl
            freq = scale[int(rng.integers(len(scale)))]
            dur = 1.8 * bl
            n = int(dur * sr)
            tt = np.arange(n) / sr
            tone = sum(np.sin(2 * np.pi * freq * k * tt) / k for k in (1, 2, 3))
            env = np.minimum(1.0, tt / 0.01) * np.minimum(1.0, (dur - tt) / 0.05)
            add(t, tone * env * (0.28 if chorus else 0.18))

    y /= max(np.abs(y).max(), 1e-9) / 0.9
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y.astype(np.float32), sr)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, default=128.0)
    ap.add_argument("--bars", type=int, default=24)
    ap.add_argument("--offset", type=float, default=0.5)
    a = ap.parse_args()
    print(make_song(Path(a.out), bpm=a.bpm, bars=a.bars, offset_s=a.offset))
