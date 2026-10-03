"""Beat This! (Foscarin, Schlüter & Widmer, ISMIR 2024) beat/downbeat activations, without torchaudio.

The network comes from the ``beat_this`` package; its spectrogram front end (torchaudio
MelSpectrogram, slaney mel, frame-length-normalised magnitude STFT, log1p(1000 x)) is reproduced
here with torch.stft and a librosa filterbank so the frozen Windows app needs no torchaudio.
The 80 MB checkpoint is looked up in the model folders (``beat_this-final0.ckpt``) and otherwise
downloaded once into the torch hub cache.
"""
from __future__ import annotations

import inspect
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

SR = 22050
N_FFT = 1024
HOP = 441                     # 50 fps
CHECKPOINT = "final0"
CHECKPOINT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp"

_MODEL = None
_FBANK: Optional[torch.Tensor] = None


def log_mel(y: np.ndarray) -> torch.Tensor:
    """(frames, 128) log-mel spectrogram identical to beat_this.preprocessing.LogMelSpect."""
    global _FBANK
    import librosa

    if _FBANK is None:
        _FBANK = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=128, fmin=30, fmax=11000,
                                                      htk=False, norm=None)).float()
    x = torch.as_tensor(np.asarray(y, np.float32))
    spec = torch.stft(x, N_FFT, HOP, N_FFT, torch.hann_window(N_FFT), center=True, pad_mode="reflect",
                      normalized=True, onesided=True, return_complex=True).abs()
    return torch.log1p(1000.0 * (_FBANK @ spec)).T


def _checkpoint(device: str) -> dict:
    from ..models import candidate_dirs

    name = f"beat_this-{CHECKPOINT}.ckpt"
    for d in candidate_dirs():
        if (d / name).is_file():
            return torch.load(d / name, map_location=device, weights_only=True)
    return torch.hub.load_state_dict_from_url(f"{CHECKPOINT_URL}/{CHECKPOINT}.ckpt", file_name=name,
                                              map_location=device)


def load_model(device: str = "cpu"):
    global _MODEL
    if _MODEL is None:
        from beat_this.model.beat_tracker import BeatThis

        ck = _checkpoint(device)
        hp = {k: v for k, v in ck["hyper_parameters"].items() if k in inspect.signature(BeatThis).parameters}
        model = BeatThis(**hp)
        model.load_state_dict({k[len("model."):] if k.startswith("model.") else k: v
                               for k, v in ck["state_dict"].items()})
        _MODEL = model.to(device).eval()
    return _MODEL


def activations(y: np.ndarray, sr: int, device: str = "cpu", chunk: int = 1500, border: int = 6
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat logits at 50 fps (same chunking as beat_this' Audio2Frames)."""
    if sr != SR:
        import librosa

        y = librosa.resample(np.asarray(y, np.float32), orig_sr=sr, target_sr=SR)
    model = load_model(device)
    spect = log_mel(y).to(device)
    n = len(spect)
    starts = np.arange(-border, n - border, chunk - 2 * border)
    if n > chunk - 2 * border:
        starts[-1] = n - (chunk - border)
    beat = torch.full((n,), -1000.0)
    down = torch.full((n,), -1000.0)
    with torch.inference_mode():
        for s in reversed(starts.tolist()):                  # earlier chunks win on overlaps
            piece = spect[max(s, 0):min(s + chunk, n)]
            piece = F.pad(piece, (0, 0, max(0, -s), max(0, min(border, s + chunk - n))))
            out = model(piece[None])
            b, d = out["beat"][0][border:-border], out["downbeat"][0][border:-border]
            lo, hi = s + border, s + chunk - border
            beat[max(lo, 0):min(hi, n)] = b[max(0, -lo):max(0, -lo) + min(hi, n) - max(lo, 0)].float().cpu()
            down[max(lo, 0):min(hi, n)] = d[max(0, -lo):max(0, -lo) + min(hi, n) - max(lo, 0)].float().cpu()
    return beat.numpy(), down.numpy()
