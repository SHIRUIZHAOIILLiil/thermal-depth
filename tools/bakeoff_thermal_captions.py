"""Decide whether an RGB-trained VLM can actually read MS2 thermal frames.

The MS2 captions this project trains on describe the RGB camera, while the depth
model is fed the thermal one. So the text names things the model cannot see --
colours above all -- from a viewpoint that is not its own, and the completed 2x2
says the model uses only the fact that text is present, not what it says.

Captioning the thermal frame instead is the obvious fix, but it puts an
RGB-trained captioner on out-of-distribution input, and a VLM handed something
it cannot read does not fail loudly: it writes a fluent sentence anyway. This
scores that risk on a few dozen frames before 22k are committed.

Four measures, chosen because each fails differently:

  chromatic colour rate  A thermal frame has no colour. "yellow bus" is
                         therefore invention, full stop. Achromatic words
                         (white/black/grey/dark) are counted separately because
                         they legitimately describe a greyscale image.
  unique fraction        A captioner that cannot see its input falls back on a
                         generic sentence, so near-duplicates across frames are
                         the signature of reading nothing.
  self-similarity        The same thing measured continuously: mean pairwise
                         Jaccard over content words. Low is good.
  RGB agreement          Jaccard against this frame's own RGB caption. The two
                         cameras see one scene, so a captioner that is reading
                         should overlap; near-zero overlap means it is not.

  python tools/bakeoff_thermal_captions.py prepare --manifest <train.jsonl> \
      --frames 40 --output <subset.jsonl>
  python tools/bakeoff_thermal_captions.py score --subset <subset.jsonl> \
      --captions grayscale=<a.jsonl> magma=<b.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from itertools import combinations
from pathlib import Path

CHROMATIC = {
    "red", "blue", "green", "yellow", "orange", "purple", "violet", "pink",
    "brown", "gold", "golden", "silver", "beige", "tan", "turquoise", "cyan",
    "maroon", "navy", "olive", "teal", "colorful", "colourful",
}
# Legitimate in a greyscale image, so tracked but never held against a caption.
ACHROMATIC = {"white", "black", "grey", "gray", "dark", "bright", "pale", "light"}

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "with", "is",
    "are", "this", "that", "these", "those", "it", "its", "as", "by", "for",
    "from", "into", "their", "there", "which", "while", "image", "scene",
    "shows", "depicts", "showing", "featuring", "appears", "appear", "visible",
    "several", "some", "other", "others", "including", "such", "very", "more",
    "most", "near", "far", "left", "right", "center", "centre",
}

WORD = re.compile(r"[a-z]+")


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cmd_prepare(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.manifest)
    if args.frames == 0:
        # Every row, same absolute-path rewrite. The full manifests need it just
        # as much as a subset does: their paths are relative to the MS2 root,
        # and the captioner resolves against the manifest's own directory.
        step, picked = 1, [dict(row) for row in rows]
    else:
        if len(rows) < args.frames:
            raise SystemExit(f"{args.manifest} has {len(rows)} rows, fewer than --frames {args.frames}")
        # Evenly spaced, never consecutive: neighbouring MS2 frames are almost the
        # same picture, and duplicate-rate would then measure the road, not the model.
        step = len(rows) // args.frames
        picked = [dict(rows[i * step]) for i in range(args.frames)]

    # The manifest's paths are relative to the MS2 root, but the captioner
    # resolves them against the manifest's own directory. A subset written
    # anywhere else would therefore point at nothing -- and the captioner would
    # report every frame as a read failure rather than as a wrong path. Write
    # them absolute, and prove the first one resolves before going further.
    root = args.ms2_root.resolve()
    for row in picked:
        for key in ("thermal_path", "rgb_path", "depth_path", "thermal_depth_path", "rgb_depth_path"):
            value = row.get(key)
            if value and not Path(value).is_absolute():
                # as_posix() so the file never carries backslashes: a Windows-style
                # path is not absolute under POSIX, so a consumer on the other
                # system silently joins it onto its own directory instead of
                # failing, and every frame comes back as a read error.
                row[key] = (root / value).as_posix()
    for key in ("thermal_path", "rgb_path"):
        probe = picked[0].get(key)
        if probe and not Path(probe).is_file():
            raise SystemExit(f"{picked[0]['id']}: {key} does not resolve to a file:\n  {probe}\n"
                             f"Check --ms2-root (given: {root})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[prepare] {len(picked)} frames every {step}th row -> {args.output}")
    print(f"[prepare] first {picked[0]['id']}  last {picked[-1]['id']}")
    print(f"[prepare] paths rewritten absolute against {root}; first thermal frame verified")
    if str(root)[1:2] == ":":
        # Last line, so it is the one still on screen. The existence check above
        # passes on Windows and says nothing about WSL, so this has to be stated
        # rather than discovered forty frames into a captioning run.
        # Plain ASCII: run from a GBK console, an emoji here aborts the program.
        print(f"[prepare] WARNING: these paths are Windows-style ({root}). Whatever reads "
              f"this manifest must run on Windows too -- under WSL a path like E:\\... is "
              f"not absolute, so it gets joined onto the manifest's own directory and every "
              f"frame fails to load. For WSL, re-run prepare there with --ms2-root /mnt/e/...")
    return 0


def score_arm(captions: dict[str, str], rgb: dict[str, str]) -> dict:
    texts = [c for c in captions.values() if c.strip()]
    if not texts:
        return {"n": 0}
    words = [content_words(t) for t in texts]
    chromatic = sum(1 for w in words if w & CHROMATIC)
    achromatic = sum(1 for w in words if w & ACHROMATIC)

    pairs = list(combinations(range(len(words)), 2))
    self_sim = statistics.mean(
        len(words[i] & words[j]) / max(len(words[i] | words[j]), 1) for i, j in pairs
    ) if pairs else 0.0

    agreements = []
    for image_id, text in captions.items():
        reference = rgb.get(image_id, "")
        if text.strip() and reference.strip():
            a, b = content_words(text), content_words(reference)
            agreements.append(len(a & b) / max(len(a | b), 1))

    return {
        "n": len(texts),
        "chromatic": chromatic / len(texts),
        "achromatic": achromatic / len(texts),
        "unique": len({t.strip().lower() for t in texts}) / len(texts),
        "self_sim": self_sim,
        "rgb_agree": statistics.mean(agreements) if agreements else float("nan"),
        "words": statistics.mean(len(t.split()) for t in texts),
    }


def join_key(value: str) -> str:
    """Key on the input image, not on `image_id`.

    The manifest's `image_id` is the bare frame number (`000000`), which repeats
    in every sequence -- 10-59-33 and 11-37-46 both have one. Joining on it would
    silently mix two sequences together, and at full scale roughly half the
    frames collide. The image path is what actually identifies a frame.
    """
    return Path(str(value).replace("\\", "/")).as_posix().lower()


def cmd_score(args: argparse.Namespace) -> int:
    subset = read_jsonl(args.subset)
    rgb = {join_key(row["thermal_path"]): row.get("caption", "") for row in subset}
    order = [join_key(row["thermal_path"]) for row in subset]
    labels_by_key = {join_key(row["thermal_path"]): row["id"] for row in subset}

    arms = {}
    for spec in args.captions:
        if "=" not in spec:
            raise SystemExit(f"--captions wants label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        rows = read_jsonl(Path(path))
        arms[label] = {
            join_key(r.get("input_path") or r.get("thermal_path") or r.get("image_id", "")):
                r.get("caption", "")
            for r in rows if r.get("status", "ok") == "ok"
        }
        matched = len(set(arms[label]) & set(order))
        if matched == 0:
            sample = next(iter(arms[label]), "<空>")
            raise SystemExit(
                f"{label}: 没有一条 caption 能和子集对上。\n"
                f"  caption 侧的键: {sample}\n  子集侧的键:    {order[0]}\n"
                f"两边的路径写法不一致（Windows vs WSL？）"
            )
        if matched < len(order):
            print(f"[warn] {label}: {matched}/{len(order)} 帧对上")

    print(f"\n{len(subset)} 帧；RGB caption 作为参照（同一场景、另一台相机）\n")
    header = f"{'':16s}" + "".join(f"{k:>14s}" for k in arms) + f"{'rgb(现用)':>14s}"
    print(header)
    print("-" * len(header))

    scored = {label: score_arm(caps, rgb) for label, caps in arms.items()}
    scored["rgb"] = score_arm(rgb, rgb)

    rows = [
        ("有效条数", "n", "{:.0f}"),
        ("彩色词率 ↓", "chromatic", "{:.1%}"),
        ("消色词率", "achromatic", "{:.1%}"),
        ("唯一句子占比 ↑", "unique", "{:.1%}"),
        ("句间自相似 ↓", "self_sim", "{:.3f}"),
        ("与RGB重合 ↑", "rgb_agree", "{:.3f}"),
        ("平均词数", "words", "{:.1f}"),
    ]
    for label, key, fmt in rows:
        cells = "".join(fmt.format(scored[a].get(key, float("nan"))).rjust(14) for a in scored)
        print(f"{label:16s}{cells}")

    print("\n判读：彩色词率是硬判据 —— 热像没有颜色，出现即为编造。"
          "\n      唯一句子占比低、自相似高 = 看不见输入、在套模板。"
          "\n      与 RGB 重合接近 0 = 描述的不是同一个场景。"
          "\n      （rgb 那一列是现用 caption，作为量表的参照，不是候选。）")

    shown = order[: args.examples]
    print(f"\n=== 前 {len(shown)} 帧逐条对照 ===")
    for key in shown:
        print(f"\n[{labels_by_key[key]}]")
        for label, caps in arms.items():
            print(f"  {label:10s} {caps.get(key, '<缺>')}")
        print(f"  {'rgb(现用)':10s} {rgb.get(key, '<缺>')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="抽一个均匀分布的小 manifest")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--ms2-root", type=Path, required=True,
                   help="manifest 里的相对路径按它展开成绝对路径。必填 —— 猜错会让每一帧"
                        "都读不到图，而 captioner 只会报读取失败，不会说路径错了。")
    p.add_argument("--frames", type=int, default=40,
                   help="0 = 全部行（只做绝对路径改写，不抽样）")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_prepare)

    s = sub.add_parser("score", help="给各候选打分")
    s.add_argument("--subset", type=Path, required=True)
    s.add_argument("--captions", nargs="+", required=True, metavar="标签=路径")
    s.add_argument("--examples", type=int, default=5)
    s.set_defaults(func=cmd_score)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
