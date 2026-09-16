"""Cut a manifest down to the official MS2 evaluation frames before inference.

`subset_to_official.py` does this after the fact, re-averaging per-frame CSVs,
which is right when the predictions already exist. For a model we are about to
run it is the wrong end: our day manifest holds 23,311 frames and the official
protocol scores 2,332 of them, so inference at full frame rate writes ten times
the predictions to compare a table that will be read on the tenth. At roughly
655 KB per float32 prediction that is 15 GB per condition per model, 45 GB for
one model over three conditions -- and the baseline table has five models in it.

The two frame sets differ by under 0.0003 AbsRel, measured, so this buys disk
rather than costing accuracy.

Official frames the manifest does not carry are reported, not dropped in
silence: a footnote reading "2,331 of the official 2,332" is honest, a bare
2,331 is not.

    python tools/subset_manifest_to_official.py \
        --manifest $SCRATCH/manifests/.../ms2_test_day3_common_thermalcap_20260821.jsonl \
        --ms2-root $SCRATCH/data/ms2 --test-env test_day \
        --out $SCRATCH/manifests/official_subset/ms2_test_day_official.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_ms2_supdepth_baselines import official_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ms2-root", required=True, type=Path)
    parser.add_argument("--test-env", required=True,
                        choices=("test_day", "test_night", "test_rain"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-step", type=int, default=10,
                        help="configs/Base/Base_Sup_Mono_Depth.yaml test.sample_step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted = [row["id"] for row in
              official_frames(args.ms2_root, args.test_env, args.sample_step)]
    wanted_set = set(wanted)

    rows = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("id") in wanted_set:
                rows[row["id"]] = line

    # Emit in the official order, so the macro average is over the same list the
    # published numbers average over.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sample_id in wanted:
            if sample_id in rows:
                handle.write(rows[sample_id] + "\n")

    missing = [s for s in wanted if s not in rows]
    print(f"\n官方帧 {len(wanted)}   清单里有 {len(rows)}   -> {args.out}")
    if missing:
        print(f"⚠️ 清单缺 {len(missing)} 帧，报表脚注要写「{len(rows)} / {len(wanted)}」：")
        for sample_id in missing[:10]:
            print(f"    {sample_id}")
        if len(missing) > 10:
            print(f"    …还有 {len(missing) - 10} 帧")


if __name__ == "__main__":
    main()
