"""How much of each test frame the thermal-view LiDAR actually scores.

Every MS2 metric is computed only where the projected LiDAR returns a value inside
the official mask (1e-3 m < d < 80 m). This reports, per condition and over the
official evaluation subset, the fraction of the 256x640 frame that mask covers --
the number behind "our metrics say nothing about the rest of the frame".

No inference; reads the GT PNGs only.

    python tools/lidar_coverage.py --ms2-root <root>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_qualitative_frames import CONDITIONS, official_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    args = ap.parse_args()
    overall = []
    for cond, list_file in CONDITIONS.items():
        cov = []
        for f in official_frames(args.ms2_root, list_file):
            if not Path(f["depth_path"]).is_file():
                continue
            d = np.asarray(Image.open(f["depth_path"]), np.float32) / 256.0
            cov.append(float(((d > 1e-3) & (d < 80.0)).mean()))
        if not cov:
            print(f"{cond:5s} no GT under {args.ms2_root} -- skipped")
            continue
        overall += cov
        q = np.percentile(cov, (10, 50, 90))
        print(f"{cond:5s} frames {len(cov):5d}   mean {statistics.fmean(cov):.1%}   "
              f"median {q[1]:.1%}   p10 {q[0]:.1%}   p90 {q[2]:.1%}")
    print(f"all   frames {len(overall):5d}   mean {statistics.fmean(overall):.1%}")


if __name__ == "__main__":
    main()
