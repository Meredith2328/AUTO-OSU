from autoosu import models


def test_find_model_uses_env_dir_and_checks_size(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOOSU_MODELS", str(tmp_path))
    assert models.candidate_dirs()[0] == tmp_path
    monkeypatch.setattr(models, "candidate_dirs", lambda: [tmp_path])   # ignore the real models folder
    spec = models.MODELS["rhythm"]
    fake = tmp_path / spec.filename
    fake.write_bytes(b"x" * 10)
    assert models.find_model("rhythm") is None          # wrong size = not a valid model file
    fake.write_bytes(b"x" * spec.size)
    assert models.find_model("rhythm") == fake


def test_release_urls():
    for spec in models.MODELS.values():
        assert spec.url.startswith("https://github.com/") and spec.url.endswith(spec.filename)
        assert len(spec.sha256) == 64
