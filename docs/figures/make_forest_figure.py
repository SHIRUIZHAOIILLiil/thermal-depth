#!/usr/bin/env python3
"""Every prompt comparison as a point with its interval, on one axis.

The paper's results are differences of two ten-thousandths on a metric whose level
is a tenth.  Prose can state them, and a side-by-side of depth maps cannot show
them at all -- at this size the two predictions are the same picture.  What the
reader actually needs to see is which differences clear zero and by how much, and
that is a forest plot.

It carries the methodological point in the same frame.  Each row is drawn twice:
the thick bar is the moving-block interval the paper quotes, the thin one behind
it is the frame-level bootstrap.  Where a thin bar clears zero and the thick one
does not, the reader is looking at a result that an independence assumption would
have sold them.

    python docs/figures/make_forest_figure.py
    python docs/figures/make_forest_figure.py --output-dir "$PAPER/figures"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# Rows are grouped by the question they answer, not by the arm, because the
# contrast that matters is between the two arms inside one question.
COMPARISONS = ("content", "presence")
CONDITIONS = ("day", "rain", "night")
ARMS = (("iris_ms2_full8_nocap", "caption-free"),
        ("iris_ms2_full8_thermalcap", "thermal caption"),
        ("iris_ms2_full8_rgbcap", "RGB caption"))

# Short enough to sit over its own panel without running into the next one; the
# full statement of each question belongs in the caption.
QUESTION = {
    "content": "Content: own vs. another caption",
    "presence": "Presence: caption vs. none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stats", type=Path,
                        default=ROOT / "docs" / "data" / "paired_stats_threearm_20260824.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument("--metric", default="abs_rel")
    parser.add_argument("--comparisons", nargs="*", default=None,
                        help="Subset of content/presence/total. The teaser shows content "
                             "only, because that is the question the paper is about.")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--name", default="forest_effects",
                        help="Output stem, so a subset does not overwrite the full figure.")
    parser.add_argument("--width", type=float, default=7.1)
    parser.add_argument("--step", default=None,
                        help="Restrict to one checkpoint per arm; default keeps the "
                             "selected one and drops the sensitivity re-run.")
    return parser.parse_args()


def load(path: Path, metric: str) -> dict:
    """(arm, condition, comparison) -> row, keeping one checkpoint per arm.

    The caption arm appears twice for daytime, at its selected checkpoint and at
    the one the sensitivity analysis used.  Plotting both would put two bars on
    one row without saying why, so the later step wins and the analysis keeps its
    own place in the text.
    """
    best: dict = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        if row["metric"] != metric:
            continue
        key = (row["arm"], row["condition"], row["comparison"])
        if key not in best or int(row["step"]) > int(best[key]["step"]):
            best[key] = row
    return best


def main() -> int:
    args = parse_args()
    rows = load(args.stats, args.metric)
    comparisons = tuple(args.comparisons) if args.comparisons else COMPARISONS
    conditions = tuple(args.conditions) if args.conditions else CONDITIONS

    labels, blocks, iids, points, marks = [], [], [], [], []
    separators = []
    for comparison in comparisons:
        if labels:
            separators.append(len(labels) - 0.5)
        for condition in conditions:
            for arm_key, arm_name in ARMS:
                row = rows.get((arm_key, condition, comparison))
                if row is None:
                    continue
                labels.append(f"{condition}, {arm_name}")
                points.append(float(row["mean_difference"]))
                blocks.append((float(row["ci95_low"]), float(row["ci95_high"])))
                iids.append((float(row["iid_ci95_low"]), float(row["iid_ci95_high"])))
                marks.append(row["separable_from_zero"] == "True")

    # Side by side, not stacked.  Nine rows per question is tall in one column and
    # short across two, and the row labels are the same in both panels, so only the
    # left one carries them.  Independent x-axes: presence effects run five to ten
    # times larger than content effects and would flatten them on a shared scale.
    split = int(separators[0] + 0.5) if separators else len(labels)
    groups = [(0, split), (split, len(labels))] if len(comparisons) > 1 else [(0, len(labels))]

    fig, axes = plt.subplots(
        1, len(groups), figsize=(args.width, 0.26 * (groups[0][1] - groups[0][0]) + 1.25),
        gridspec_kw={"wspace": 0.08}, squeeze=False,
    )
    axes = axes[0]

    for ax, (lo, hi), comparison in zip(axes, groups, comparisons):
        ax.axvline(0, color="0.35", lw=0.9, zorder=1)
        for y, i in enumerate(range(lo, hi)):
            # The frame-level interval sits behind in a warm tone -- it must not be
            # confusable with a block interval that failed to clear zero, which is grey.
            ax.plot(iids[i], (y, y), color="#e8c39a", lw=4.6, solid_capstyle="butt", zorder=2)
            colour = "#1f5fa0" if marks[i] else "#8c8c8c"
            ax.plot(blocks[i], (y, y), color=colour, lw=1.7, solid_capstyle="butt", zorder=3)
            ax.plot([points[i]], [y], "o", ms=4.0, color=colour, zorder=4)

        ax.set_yticks(range(hi - lo))
        ax.invert_yaxis()
        ax.set_ylim(hi - lo - 0.45, -0.6)
        ax.set_title(QUESTION[comparison], fontsize=8.5, style="italic",
                     color="0.2", loc="left", pad=5)
        ax.tick_params(axis="x", labelsize=7.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[0].set_yticklabels([labels[i] for i in range(*groups[0])], fontsize=8)
    if len(axes) > 1:
        axes[1].set_yticklabels([])
        axes[1].tick_params(axis="y", length=0)
    XLABEL = {"content": "AbsRel, permuted $-$ matched",
              "presence": "AbsRel, permuted $-$ empty",
              "total": "AbsRel, matched $-$ empty"}
    for ax, comparison in zip(axes, comparisons):
        ax.set_xlabel(XLABEL[comparison], fontsize=8)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / (args.name + ".pdf")
    fig.savefig(destination, bbox_inches="tight")
    print(f"{len(labels)} rows  ->  {destination}")
    print(f"  separable at block 200: {sum(marks)}/{len(marks)}")
    flipped = sum(1 for b, i in zip(blocks, iids)
                  if (b[0] > 0 or b[1] < 0) != (i[0] > 0 or i[1] < 0))
    print(f"  block and IID disagree on: {flipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
