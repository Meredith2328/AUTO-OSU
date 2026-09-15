"""Where the trained models live and how to fetch them.

Search order for a model file: $AUTOOSU_MODELS, the `models` folder next to the executable /
repository, then ~/.autoosu/models. Missing files are downloaded from the GitHub release.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

RELEASE_BASE = "https://github.com/kanze1/AUTO-OSU/releases/download/models-v0"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    filename: str
    size: int
    sha256: str
    description: str

    @property
    def url(self) -> str:
        return f"{RELEASE_BASE}/{self.filename}"


MODELS: Dict[str, ModelSpec] = {
    "rhythm": ModelSpec("rhythm", "rhythm_v0.pt", 57389131,
                        "e01267c647e103a36801109e3c34f508dd2d04d228b07b4a299cddbd55dd52f6",
                        "masked tick transformer v0: when to place circles / sliders / spinners"),
    "coord": ModelSpec("coord", "coord_v0.pt", 261135487,
                       "52163effa16b7cb8172013245deef5c87cc022023e42501f5a61007215252040",
                       "coordinate diffusion (DiT-B) v0: where to place them"),
}


def app_root() -> Path:
    """Folder of the executable when frozen, else the repository root."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def candidate_dirs() -> List[Path]:
    dirs: List[Path] = []
    env = os.environ.get("AUTOOSU_MODELS")
    if env:
        dirs.append(Path(env))
    dirs.append(app_root() / "models")
    dirs.append(Path.home() / ".autoosu" / "models")
    return dirs


def find_model(name: str) -> Optional[Path]:
    spec = MODELS[name]
    for d in candidate_dirs():
        p = d / spec.filename
        if p.is_file() and p.stat().st_size == spec.size:
            return p
    return None


def _writable_dir() -> Path:
    for d in candidate_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_test"
            probe.write_bytes(b"")
            probe.unlink()
            return d
        except OSError:
            continue
    raise RuntimeError("no writable models directory")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(spec: ModelSpec, dest: Path, progress: Optional[Callable[[float, str], None]] = None) -> Path:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(spec.url, headers={"User-Agent": "AUTO-OSU"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or spec.size)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if progress:
                progress(got / max(total, 1), f"{spec.filename} {got / 1e6:.0f}/{total / 1e6:.0f} MB")
    if sha256_of(tmp) != spec.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {spec.filename}; download again")
    tmp.replace(dest)
    return dest


def ensure_model(name: str, progress: Optional[Callable[[float, str], None]] = None) -> Path:
    """Return the local path of a model, downloading it first if needed."""
    found = find_model(name)
    if found:
        return found
    spec = MODELS[name]
    dest = _writable_dir() / spec.filename
    return download(spec, dest, progress)
