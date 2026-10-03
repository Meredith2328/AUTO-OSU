"""Mechanical checks for the optional dedicated mania model (no quality claims)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from autoosu.ml.mania_data import _split_groups
from autoosu.ml.mania_infer import predict_logits
from autoosu.ml.mania_model import ManiaNet
from autoosu.ml.mania_train import objective


def test_song_group_split_unions_title_and_audio():
    rows = [dict(song_key="same|title", audio_hash="a"),
            dict(song_key="same|title", audio_hash="b"),
            dict(song_key="different|title", audio_hash="b"),
            dict(song_key="other|song", audio_hash="c"),
            dict(song_key="last|song", audio_hash="d")]
    _split_groups(rows, 42)
    assert rows[0]["split"] == rows[1]["split"] == rows[2]["split"]
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_empty_window_has_finite_loss_and_gradient():
    net = ManiaNet()
    audio = torch.zeros(1, 12, 16, 64)
    phase = torch.arange(12)[None]
    beats = torch.full((1, 12), 500.0)
    condition = torch.zeros(1, 2)
    outputs = net(audio, phase, beats, condition)
    weights = torch.tensor([0.5, 1.0, 4.0, 1.0, 1.5])
    for label_value in (0, -100):
        labels = torch.full((1, 12, 4), label_value)
        durations = torch.full_like(labels, -100)
        loss = objective(*outputs, labels, durations, weights)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)


def test_inference_chunks_agree_at_boundaries():
    torch.manual_seed(3)
    model = ManiaNet().eval()
    mel = np.random.default_rng(3).integers(0, 255, (450, 64), dtype=np.uint8)
    times = np.arange(80, dtype=np.float32) * 50
    phase = np.arange(80, dtype=np.int64) % 32
    beat = np.full(80, 400, dtype=np.float32)
    a = predict_logits(model, mel, times, phase, beat, 6.0, 7.0, "cpu", chunk=19)
    b = predict_logits(model, mel, times, phase, beat, 6.0, 7.0, "cpu", chunk=256)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-5)
