"""Copy manifests with every frame's caption replaced by one fixed sentence.

For the caption-control arms: a model trained, validated and tested with the same
sentence on every frame, so the text pathway is exercised but carries no
frame-specific content. Everything else in each row -- paths, ids, split -- is left
exactly as it was, so the arm differs from the captioned arm in the caption alone.
`lotus/utils/ms2_thermal_dataset.py` reads `row["caption"]` for training and the
evaluation pipeline reads the same field under `--val-caption-mode correct`, so
pointing TRAIN/VAL/TEST_MANIFEST at these copies is the whole change.

Row order and line count are preserved: evaluation seeds its noise by line number.

    python tools/make_fixed_caption_manifest.py --tag tplA \
        --text "A complex 3D scene with varying objects at different distances." \
        --out-dir <dir> <manifest> [<manifest> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="Short name that goes into the output filenames.")
    ap.add_argument("--text", required=True, help="The one sentence every frame receives.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("manifests", type=Path, nargs="+")
    args = ap.parse_args()
    text = args.text.strip()
    if not text:
        raise SystemExit("!! empty --text: that is the caption-free arm, which already exists")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.manifests:
        lines = src.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
                continue
            row = json.loads(line)
            row["caption_original_sha1"] = hashlib.sha1(str(row.get("caption", "")).encode()).hexdigest()[:12]
            row["caption"] = text
            row["caption_fixed"] = args.tag
            # These described the generated caption and would now be wrong.
            for key in ("caption_clip_tokens", "caption_trimmed", "caption_status"):
                row.pop(key, None)
            out.append(json.dumps(row, ensure_ascii=False))
        dst = args.out_dir / f"{src.stem}_fixed-{args.tag}.jsonl"
        dst.write_text("\n".join(out) + "\n", encoding="utf-8")
        assert len(dst.read_text(encoding="utf-8").splitlines()) == len(lines)
        print(f"[{args.tag}] {src.name} -> {dst}  ({len(lines)} lines)")
    print(f"[{args.tag}] every caption is now: {text!r}")


if __name__ == "__main__":
    main()
