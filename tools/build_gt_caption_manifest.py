"""Write captions from the ground truth itself, to measure the ceiling on language.

Every caption this project has trained on came from a vision-language model
looking at the image, and the completed 2x2 says the model uses only the fact
that text is present. Two explanations remain: the model cannot turn described
content into geometry, or the captions are not accurate enough about this frame.
Nothing settles that while the text keeps coming from a model that has to guess.

So take the guessing out. These captions are computed from the frame's own lidar
-- the same measurement the loss is scored against -- and state its geometry
directly. If text carrying correct depth still does not help, then no captioner,
prompt or thermal VLM will, and the whole line can be closed. If it helps, the
ceiling is real and caption quality is worth the cost of chasing.

⚠️ THIS IS AN ORACLE, NOT A METHOD. Monocular depth estimation has no ground
truth at inference, so a system captioned this way cannot be deployed and its
score cannot be quoted beside the others as if it were one. It measures an upper
bound and must be labelled as one wherever it appears.

Two choices worth stating. The captions are templated, so phrasing is constant
across frames and only the geometry varies -- which is exactly what isolates
content from wording in the correct-versus-shuffled comparison. And they lean on
ratios and ordering rather than absolute metres, because the evaluation fits a
per-image scale and shift: a global scale in the text is removed before the
metric sees it, and only relative structure survives.

    python tools/build_gt_caption_manifest.py --manifest <in.jsonl> \
        --ms2-root <root> --output <out.jsonl>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

COLUMNS = ("left", "centre", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-valid", type=int, default=200,
                        help="低于这个有效像素数就判这一帧无法生成，宁可报错也不编")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def round_metres(value: float) -> int:
    """Coarse on purpose. Reporting 37.4 m from a lidar median would claim a
    precision the measurement does not have, and the extra digits cost tokens
    that carry no ordering information."""
    if value < 10:
        return int(round(value))
    if value < 30:
        return int(round(value / 2) * 2)
    return int(round(value / 5) * 5)


def describe(depth: np.ndarray, valid: np.ndarray, min_valid: int) -> str | None:
    if int(valid.sum()) < min_valid:
        return None
    height, width = depth.shape
    edges = (0, width // 3, 2 * width // 3, width)

    per_column = {}
    for name, (start, stop) in zip(COLUMNS, zip(edges[:-1], edges[1:])):
        mask = valid[:, start:stop]
        if int(mask.sum()) >= min_valid // 4:
            per_column[name] = float(np.median(depth[:, start:stop][mask]))
    if not per_column:
        return None

    nearest = float(np.percentile(depth[valid], 2))
    farthest = float(np.percentile(depth[valid], 98))
    nearest_column = min(per_column, key=per_column.get)
    order = sorted(per_column, key=per_column.get)

    parts = [
        f"The nearest surface is about {round_metres(nearest)} m and the scene "
        f"extends to about {round_metres(farthest)} m."
    ]
    # "on the left" but "in the centre" -- the wrong preposition would appear in
    # twenty thousand captions and read as a template artefact rather than English.
    preposition = {"left": "on", "right": "on", "centre": "in"}
    parts.append(
        "Typical distance is "
        + ", ".join(f"{round_metres(per_column[name])} m {preposition[name]} the {name}"
                    for name in COLUMNS if name in per_column)
        + "."
    )
    if len(order) > 1:
        parts.append("From nearest to farthest: " + ", ".join(order) + ".")
        ratio = per_column[order[-1]] / max(per_column[order[0]], 1e-6)
        if ratio >= 1.5:
            parts.append(f"The {order[-1]} is roughly {ratio:.1f} times as far as the {nearest_column}.")
    return " ".join(parts)


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    written, failed = [], []
    for row in rows:
        relative = row.get("thermal_depth_path") or row.get("depth_path")
        if not relative:
            failed.append((row.get("id", "?"), "no thermal-view GT path"))
            continue
        depth = np.asarray(Image.open(args.ms2_root / relative), dtype=np.float32) / args.depth_scale
        valid = np.isfinite(depth) & (depth > args.min_depth) & (depth < args.max_depth)
        caption = describe(depth, valid, args.min_valid)
        if caption is None:
            failed.append((row.get("id", "?"), f"only {int(valid.sum())} valid lidar pixels"))
            continue
        new_row = dict(row)
        new_row["caption"] = caption
        new_row["caption_prompt_version"] = "gt_geometry_v1"
        new_row["caption_model"] = "none (computed from lidar)"
        new_row["caption_model_id"] = ""
        new_row["caption_status"] = "ok"
        new_row["caption_input_modality"] = "gt_depth"
        # Loud, because this number must never be quoted as a deployable result.
        new_row["caption_is_oracle"] = True
        written.append(new_row)

    print(f"[gt-caption] {len(rows)} 行 -> 写出 {len(written)}；无法生成 {len(failed)}")
    for identifier, reason in failed[:5]:
        print(f"    {identifier}: {reason}")
    if failed:
        raise SystemExit(
            f"有 {len(failed)} 帧生成不了 caption。空 caption 会被当成空 prompt 训练，"
            f"等于混进另一个实验臂 —— 先决定这些帧怎么处理，再重建。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = [len(row["caption"].split()) for row in written]
    print(f"[gt-caption] {args.output}")
    print(f"[gt-caption] 词数 中位 {sorted(lengths)[len(lengths)//2]}  最大 {max(lengths)}")
    print(f"[gt-caption] 样例: {written[0]['caption']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
