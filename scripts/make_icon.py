"""Turn the OC avatar into the app assets.

    python scripts/make_icon.py <avatar.png>

Writes autoosu/assets/avatar.png (square crop, 512 px, used in the window header and as the
window icon) and assets/icon.ico (multi-size, used by PyInstaller for the exe icon).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    src = Path(sys.argv[1])
    img = Image.open(src).convert("RGBA")
    side = min(img.size)
    left, top = (img.width - side) // 2, 0            # keep the top: faces sit high in portraits
    sq = img.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS)
    out_png = ROOT / "autoosu" / "assets" / "avatar.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sq.save(out_png)

    # round icon with a thin gold ring, transparent corners
    ring = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, 511, 511), fill=255)
    ring.paste(sq, (0, 0), mask)
    ImageDraw.Draw(ring).ellipse((4, 4, 507, 507), outline=(212, 165, 44, 255), width=10)
    ico_dir = ROOT / "assets"
    ico_dir.mkdir(exist_ok=True)
    ring.save(ico_dir / "icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    ring.save(ico_dir / "icon.png")
    print(f"-> {out_png}\n-> {ico_dir / 'icon.ico'}")


if __name__ == "__main__":
    main()
