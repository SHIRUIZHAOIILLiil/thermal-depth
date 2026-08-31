"""Compare backbones' val AbsRel on the frames they actually share.

Marigold's val ran at stride 200 (72 frames) and the Lotus arms at stride 4
(3,563). Reading 0.134 against 0.078 across those two sets confounds the
backbone with the subsample, and per-frame AbsRel varies enough that 72 frames
can move the mean on its own. The per-sample CSVs already hold every frame, so
the shared subset costs nothing to compute -- and it is the only comparison
that says anything.
"""
import csv
import os
from pathlib import Path

RUNS = Path(os.environ["IRIS_RUNS"]) / "eval"

WANTED = [
    ("Marigold cap", "marigold_full8_thermalcap_val_*_s20000",
     "eval_eval_affine_invariant_depth_space_per_sample.csv"),
    ("Marigold nocap", "marigold_full8_nocap_val_*_s20000",
     "eval_eval_affine_invariant_depth_space_per_sample.csv"),
    ("Lotus-D nocap", "lotusd_full8_nocap_val_c26a5859_s20000", "eval_eval_per_sample.csv"),
    ("Lotus-G nocap", "iris_ms2_full8_nocap_val_ca6868a0_s20000", "eval_eval_per_sample.csv"),
]


def load(pattern, filename):
    for directory in sorted(RUNS.glob(pattern)):
        path = directory / filename
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        if not rows:
            continue
        key = next((k for k in rows[0] if k in ("id", "frame_id", "sample_id")), None)
        if key is None:
            raise SystemExit(f"{path}: no id column among {list(rows[0])[:8]}")
        return path, {r[key]: float(r["abs_rel"]) for r in rows}
    return None, {}


def main():
    loaded = []
    for label, pattern, filename in WANTED:
        path, table = load(pattern, filename)
        print(f"{label:16s} {len(table):5d} 帧  {path}")
        if table:
            loaded.append((label, table))
    if len(loaded) < 2:
        raise SystemExit("至少要两条才有得比")

    print("\n各自完整 val（帧集不同，不可直接对比）")
    for label, table in loaded:
        print(f"  {label:16s} {sum(table.values()) / len(table):.5f}  ({len(table)} 帧)")

    shared = set.intersection(*(set(t) for _, t in loaded))
    print(f"\n共有的 {len(shared)} 帧上（这个才可比）")
    if not shared:
        raise SystemExit("没有共有帧 —— 检查 id 列是否同一种写法")
    for label, table in loaded:
        print(f"  {label:16s} {sum(table[k] for k in shared) / len(shared):.5f}")


if __name__ == "__main__":
    main()
