import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoosu import batch
from autoosu.generate import cached_model


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"input")
    return path


def test_scan_recursion_case_and_output_exclusion(tmp_path):
    touch(tmp_path/"b.WAV")
    touch(tmp_path/"a.mp3")
    touch(tmp_path/"readme.txt")
    touch(tmp_path/"sub"/"video.MP4")
    touch(tmp_path/".work"/"audio.mp3")
    touch(tmp_path/"output"/"generated.mp3")
    assert [p.name for p in batch.scan_inputs(tmp_path, exclude_dirs=[tmp_path/"output"])] == ["a.mp3", "b.WAV"]
    assert [p.name for p in batch.scan_inputs(tmp_path, True, [tmp_path/"output"])] == ["a.mp3", "b.WAV", "video.MP4"]


def test_outputs_do_not_collide_for_same_stem(tmp_path):
    root = tmp_path/"input"
    files = [root/"same.wav", root/"same.mp3", root/"disc2"/"same.wav"]
    paths = [batch.output_for(p, root, tmp_path/"out") for p in files]
    assert len(set(paths)) == 3
    assert paths == [batch.output_for(p, root, tmp_path/"out") for p in files]


def test_failure_continues_with_shared_models_and_persistent_report(tmp_path, monkeypatch):
    source, destination = tmp_path/"input", tmp_path/"out"
    for name in ("a.wav", "broken.wav", "c.wav"):
        touch(source/name)
    caches = []
    def generate(path, diffs, out, *, model_cache, **kwargs):
        caches.append(id(model_cache))
        if path.name == "broken.wav":
            raise ValueError("invalid audio")
        out.mkdir(parents=True)
        osz = out/"same artist - same title.osz"
        osz.write_bytes(path.name.encode())
        return SimpleNamespace(osz=osz, device="cpu")
    monkeypatch.setattr(batch, "generate", generate)
    result = batch.generate_batch(source, ["Hard"], destination, log=lambda _: None)
    assert result.succeeded == 2 and result.failed == 1
    assert len(set(caches)) == 1
    data = json.loads(result.report.read_text(encoding="utf-8"))
    assert [i['status'] for i in data['items']] == ["done", "error", "done"]
    assert "invalid audio" in data['items'][1]['error']
    assert Path(data['items'][0]['osz']).read_bytes() == b"a.wav"
    assert Path(data['items'][2]['osz']).read_bytes() == b"c.wav"


def test_cancel_finishes_current_and_marks_remaining(tmp_path, monkeypatch):
    source = tmp_path/"input"
    for name in ("a.wav", "b.wav", "c.wav"):
        touch(source/name)
    cancel = threading.Event()
    def generate(path, diffs, out, **kwargs):
        cancel.set()
        return SimpleNamespace(osz=out/"map.osz", device="cpu")
    monkeypatch.setattr(batch, "generate", generate)
    result = batch.generate_batch(source, ["Hard"], tmp_path/"out", cancel=cancel, log=lambda _: None)
    assert result.cancelled and result.succeeded == 1
    assert [i.status for i in result.items] == ["done", "cancelled", "cancelled"]
    assert json.loads(result.report.read_text())['cancelled']


def test_output_can_be_parent_of_input(tmp_path, monkeypatch):
    source = tmp_path/"input"
    touch(source/"song.wav")
    monkeypatch.setattr(batch, "generate", lambda p, d, o, **kw: SimpleNamespace(osz=o/"map.osz", device="cpu"))
    assert batch.generate_batch(source, ["Hard"], tmp_path, log=lambda _: None).succeeded == 1


def test_empty_folder_and_same_output_fail_before_inference(tmp_path):
    with pytest.raises(ValueError, match="different"):
        batch.generate_batch(tmp_path, ["Hard"], tmp_path)
    with pytest.raises(ValueError, match="No supported"):
        batch.generate_batch(tmp_path, ["Hard"], tmp_path/"out")


def test_cache_invalidates_when_model_file_changes(tmp_path):
    checkpoint = touch(tmp_path/"model.pt")
    calls = []
    def loader(path, device):
        result = object()
        calls.append(result)
        return result
    cache = {}
    first = cached_model(cache, "rhythm", str(checkpoint), "cpu", loader)
    assert cached_model(cache, "rhythm", str(checkpoint), "cpu", loader) is first
    checkpoint.write_bytes(b"different checkpoint")
    assert cached_model(cache, "rhythm", str(checkpoint), "cpu", loader) is not first
    assert len(calls) == 2
