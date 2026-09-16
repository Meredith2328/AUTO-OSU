"""Turn the OC avatar into the app assets.

    python scripts/make_icon.py <avatar.png>

The source may have a baked-in white / checkerboard "transparency" pattern: it is keyed out by
flood-filling the light background from the border, then the edge is feathered.  Writes
autoosu/assets/avatar.png (square, transparent, 512 px; window header + window icon) and
assets/icon.ico + assets/icon.png (round, thin gold ring; exe icon).
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
GOLD = (230, 169, 60, 255)


def key_out_background(img: Image.Image, tol: int = 28) -> Image.Image:
    """Make the light background transparent: pixels that are nearly white / light grey and
    connected to the image border become alpha 0, with a soft edge."""
    rgb = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w, _ = rgb.shape
    # candidate background: bright, low saturation (white and the grey checker squares)
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    light = (mn >= 200 - tol) & ((mx - mn) <= tol)
    # flood fill from the border through candidate pixels only
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if light[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if light[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and light[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    alpha = np.where(seen, 0, 255).astype(np.uint8)
    a = Image.fromarray(alpha, "L").filter(ImageFilter.GaussianBlur(1.2))
    # pull the matte in by a pixel so no light fringe survives
    a = a.point(lambda v: 0 if v < 140 else min(255, int((v - 140) * 255 / 115)))
    out = img.convert("RGBA")
    out.putalpha(a)
    return out


def main() -> None:
    src = Path(sys.argv[1])
    img = key_out_background(Image.open(src))
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
    disc = Image.new("RGBA", (512, 512), (24, 24, 27, 255))      # graphite backdrop behind the character
    disc.alpha_composite(sq)
    ring.paste(disc, (0, 0), mask)
    ImageDraw.Draw(ring).ellipse((5, 5, 506, 506), outline=GOLD, width=12)
    ico_dir = ROOT / "assets"
    ico_dir.mkdir(exist_ok=True)
    ring.save(ico_dir / "icon.ico", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    ring.save(ico_dir / "icon.png")
    print(f"-> {out_png}\n-> {ico_dir / 'icon.ico'}")


if __name__ == "__main__":
    main()
