#!/usr/bin/env python3
"""Regenerate the README visuals: sample image grid + coin-count distribution chart.

Usage:  python3 archive/tools/make_readme_assets.py
Writes: assets/samples.jpg, assets/distribution-light.svg, assets/distribution-dark.svg
"""
import json
import os
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "dataset")
OUT = os.path.join(HERE, "assets")

# dataviz reference palette: series-1 blue, text tokens per mode
THEME = {
    "light": {"bar": "#2a78d6", "text": "#0b0b0b", "muted": "#52514e", "axis": "#d8d7d2"},
    "dark": {"bar": "#3987e5", "text": "#ffffff", "muted": "#c3c2b7", "axis": "#3a3a37"},
}


def load_labels():
    with open(os.path.join(DATA, "labels.json")) as fh:
        return json.load(fh)


def make_samples(labels, cols=4, rows=2, tile=200, pad=3):
    """One tile per chosen image, each stamped with its coin count."""
    by_count = {}
    for name, count in sorted(labels.items()):
        by_count.setdefault(count, name)
    picks = [by_count[c] for c in sorted(by_count)][: cols * rows]

    W = cols * tile + (cols + 1) * pad
    H = rows * tile + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), "#2f2f2d")  # neutral seam, reads on light and dark READMEs
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()

    for i, name in enumerate(picks):
        img = Image.open(os.path.join(DATA, "images", name)).convert("RGB").resize((tile, tile))
        x = pad + (i % cols) * (tile + pad)
        y = pad + (i // cols) * (tile + pad)
        sheet.paste(img, (x, y))
        badge = f"{labels[name]}"
        bw = 34 if len(badge) < 2 else 46
        draw.rounded_rectangle([x + 8, y + 8, x + 8 + bw, y + 42], 8, fill="#0b0b0bcc")
        draw.text((x + 8 + bw / 2, y + 25), badge, font=font, fill="#ffffff", anchor="mm")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "samples.jpg")
    sheet.save(path, quality=88)
    print("wrote", path, sheet.size)


def make_distribution(labels, mode):
    """Single-series bar chart: how many images have each coin count."""
    c = Counter(labels.values())
    counts = sorted(c)
    vmax = max(c.values())
    t = THEME[mode]

    W, H = 720, 230
    left, right, top, bottom = 34, 14, 26, 34
    pw, ph = W - left - right, H - top - bottom
    slot = pw / len(counts)
    bar_w = slot - 10  # 2px+ surface gap between adjacent bars

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
        f'role="img" aria-label="Images per coin count, 1 to 12 coins">'
    ]
    # recessive gridlines + y ticks
    for gv in range(0, vmax + 1, 5):
        y = top + ph - (gv / vmax) * ph
        p.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W-right}" y2="{y:.1f}" stroke="{t["axis"]}" stroke-width="1"/>')
        p.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="{t["muted"]}">{gv}</text>')
    # bars with 4px rounded data-ends anchored to the baseline
    for i, k in enumerate(counts):
        v = c[k]
        bh = (v / vmax) * ph
        x = left + i * slot + (slot - bar_w) / 2
        y = top + ph - bh
        r = min(4, bh / 2)
        p.append(
            f'<path d="M{x:.1f},{top+ph} V{y+r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},-{r:.1f} '
            f'H{x+bar_w-r:.1f} a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f} V{top+ph} Z" fill="{t["bar"]}"/>'
        )
        p.append(f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="11" fill="{t["text"]}">{v}</text>')
        p.append(f'<text x="{x+bar_w/2:.1f}" y="{top+ph+17:.1f}" text-anchor="middle" font-size="11" fill="{t["muted"]}">{k}</text>')
    p.append(f'<line x1="{left}" y1="{top+ph}" x2="{W-right}" y2="{top+ph}" stroke="{t["axis"]}" stroke-width="1"/>')
    p.append(f'<text x="{left}" y="14" font-size="12" fill="{t["muted"]}">images per coin count</text>')
    p.append(f'<text x="{W-right}" y="{H-4}" text-anchor="end" font-size="11" fill="{t["muted"]}">coins in image</text>')
    p.append("</svg>")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"distribution-{mode}.svg")
    with open(path, "w") as fh:
        fh.write("\n".join(p))
    print("wrote", path)


if __name__ == "__main__":
    labels = load_labels()
    make_samples(labels)
    for mode in ("light", "dark"):
        make_distribution(labels, mode)
