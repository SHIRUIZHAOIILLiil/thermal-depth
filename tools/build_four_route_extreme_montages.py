"""Build 4x4 montages of each route's worst/best Val sample across all routes.

For each of the four routes, pick the sample where that route's own AbsRel is
worst (or best), then place the four routes' untouched official ``vis/*.png``
for that same sample side by side.  Pixels are never modified; labels are
drawn only on separate margin bands.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "lotus_line_v2"
ROUTES = [
    ("Thermal-VAE", "thermal_vae_mode_official_val_full"),
    ("UNet-only", "full_epoch1_unet_only_official_val_full"),
    ("Adapter-only", "full_epoch1_v2_4_official_val_full"),
    ("Joint", "full_epoch1_joint_official_val_full"),
]
OUTPUT = BASE / "route_comparison_montages"
LABEL_BAND = 34
ROW_HEADER = 44
FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def load_font(size):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_per_sample(route_dir: Path):
    scores = {}
    with (route_dir / "per_sample_metrics.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[list(row.values())[0]] = float(row["abs_relative_difference"])
    return scores


def vis_path(route_dir: Path, sample_id: str) -> Path:
    return route_dir / "vis" / (sample_id.replace("/", "_") + ".png")


def build(kind: str, pick) -> dict:
    per_route = {name: read_per_sample(BASE / d) for name, d in ROUTES}
    common = set.intersection(*(set(s) for s in per_route.values()))
    if not common:
        raise RuntimeError("Routes share no common sample ids.")
    rows = []
    for name, _ in ROUTES:
        scores = per_route[name]
        sample = pick(common, key=lambda k: scores[k])
        rows.append((name, sample))

    tile = Image.open(vis_path(BASE / ROUTES[0][1], rows[0][1]))
    width, height = tile.size
    cell_h = LABEL_BAND + height
    canvas = Image.new(
        "RGB", (width * 4, (ROW_HEADER + cell_h) * 4), (24, 24, 24)
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(20)
    header_font = load_font(24)

    manifest = {"kind": kind, "rows": []}
    for row_index, (owner, sample) in enumerate(rows):
        top = (ROW_HEADER + cell_h) * row_index
        short = sample.split("/")[-1]
        seq = sample.split("/")[1].strip("_")
        header = f"{kind} sample of {owner}: {seq} {short}"
        draw.text((8, top + 8), header, fill=(255, 220, 120), font=header_font)
        row_record = {"owner": owner, "sample": sample, "absrel": {}}
        for col, (name, d) in enumerate(ROUTES):
            image = Image.open(vis_path(BASE / d, sample)).convert("RGB")
            if image.size != (width, height):
                raise RuntimeError(f"Size mismatch for {name} {sample}")
            x = width * col
            y = top + ROW_HEADER
            score = _SCORES[name][sample]
            label = f"{name}  AbsRel {score:.3f}"
            color = (140, 235, 140) if name == owner and kind == "Best" else (
                (250, 140, 140) if name == owner else (230, 230, 230)
            )
            draw.text((x + 8, y + 7), label, fill=color, font=font)
            canvas.paste(image, (x, y + LABEL_BAND))
            row_record["absrel"][name] = score
        manifest["rows"].append(row_record)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / f"{kind.lower()}_of_each_route_4x4.png"
    canvas.save(out_path)
    manifest["montage"] = str(out_path)
    return manifest


_SCORES = {}


def main() -> None:
    for name, d in ROUTES:
        _SCORES[name] = read_per_sample(BASE / d)
    results = [build("Worst", max), build("Best", min)]
    summary_path = OUTPUT / "montage_manifest.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for result in results:
        print(result["montage"])
        for row in result["rows"]:
            values = "  ".join(f"{k}={v:.3f}" for k, v in row["absrel"].items())
            print(f"  [{row['owner']}] {row['sample'].split('/')[-1]}: {values}")


if __name__ == "__main__":
    main()
