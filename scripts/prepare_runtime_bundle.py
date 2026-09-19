"""Bundle the matching inference source for the frozen app's uv installer."""
from pathlib import Path
import zipfile


def write_bundle(root: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [root/name for name in ("pyproject.toml", "README.md", "LICENSE")]
    files += sorted((root/"autoosu").rglob("*.py"))
    files += [root/"autoosu/ml/coord/LICENSE"]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print(write_bundle(root, root/'build/runtime-source.zip'))
