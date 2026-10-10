"""Official-subset copies of the three test manifests, compact and line-preserving.

The published MS2 benchmark scores every tenth frame (`official_frames`). This writes,
per condition, the rows of our test manifest that fall in that subset, twice:

  <name>_official.jsonl         compact, for tools that only need the frame list;
  <name>_official_padded.jsonl  every kept row on the line it had in the full
                                manifest, blank lines in between. Our evaluator seeds
                                each frame's noise by its line number, so this copy
                                reproduces the full-manifest predictions exactly at a
                                tenth of the cost.

Frames the manifest lacks (the one day frame without a caption) are reported.

    python tools/make_official_subset_manifest.py --ms2-root <root> --manifest-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_qualitative_frames import CONDITIONS, official_frames  # noqa: E402

MANIFESTS = {
    "day": "ms2_test_day3_common_thermalcap_20260821.jsonl",
    "night": "ms2_test_night3_thermalcap_20260821.jsonl",
    "rainy": "ms2_test_rainy3_thermalcap_20260821.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for cond, list_file in CONDITIONS.items():
        want = {f["id"] for f in official_frames(args.ms2_root, list_file)}
        src = args.manifest_dir / MANIFESTS[cond]
        kept, padded = [], []
        for index, line in enumerate(src.read_text(encoding="utf-8").splitlines()):
            if line.strip() and json.loads(line)["id"] in want:
                kept.append(line)
                padded += [""] * (index - len(padded))
                padded.append(line)
        stem = Path(MANIFESTS[cond]).stem
        (args.out / f"{stem}_official.jsonl").write_text("\n".join(kept) + "\n", encoding="utf-8")
        (args.out / f"{stem}_official_padded.jsonl").write_text("\n".join(padded) + "\n", encoding="utf-8")
        missing = len(want) - len(kept)
        print(f"[{cond}] {len(kept)} of {len(want)} official frames"
              + (f"  ({missing} not in {src.name})" if missing else ""))


if __name__ == "__main__":
    main()
