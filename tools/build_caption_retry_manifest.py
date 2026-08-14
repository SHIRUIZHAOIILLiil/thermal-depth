"""Collect the frames whose caption has to be generated again.

Two kinds fail the acceptance check for MS2 thermal captions:

  over budget   The CLIP text encoder takes 77 tokens and silently truncates the
                rest, so an over-long caption is not a long caption -- it is a
                caption with an unknown ending. 7.9% of the first pass landed
                here, up to 139 words against a median of 47.
  quality error The generator's own gate rejected the output (runaway word
                repetition, mostly), and the row carries no caption at all.

Trimming the long ones back is the wrong repair for this format. The captions are
prose, and their trailing clause is the near-to-far ordering -- "progressively
distant ones receding into the background" -- so dropping the tail removes the
depth content the prompt exists to elicit. `trim_to_budget` assumes the P3
Near/Middle/Far layout, where the tail really is the least important part; that
assumption does not carry over.

So: regenerate these frames under a prompt that asks for the same content in
fewer words, and leave the 92% that already fit exactly as they are.

    python tools/build_caption_retry_manifest.py \
        --captions <captions.jsonl> --manifest <abs manifest.jsonl> \
        --output <retry.jsonl> --token-budget 75
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="第一遍生成的 caption jsonl")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="生成时用的那个绝对路径 manifest（帧的来源）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=75)
    parser.add_argument("--clip-tokenizer", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--clip-tokenizer-subfolder", default="tokenizer")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join_key(value: str) -> str:
    """Key on the image path. `image_id` is the bare frame number and every MS2
    sequence has a 000000, so joining on it silently merges sequences."""
    return Path(str(value).replace("\\", "/")).as_posix().lower()


def main() -> int:
    args = parse_args()
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        args.clip_tokenizer, subfolder=args.clip_tokenizer_subfolder or None
    )

    def count(text: str) -> int:
        # Same convention as the Iris acceptance validator: BOS/EOS in, no truncation.
        return len(tokenizer(text)["input_ids"])

    captions = read_jsonl(args.captions)
    rows = read_jsonl(args.manifest)
    by_key = {join_key(r["thermal_path"]): r for r in rows}

    over, failed, fine, unmatched = [], [], 0, 0
    # Built once. Inside the `missing` comprehension below it would be rebuilt per
    # row: twenty thousand rows times twenty thousand captions is four hundred
    # million key constructions, which does not fail -- it just never finishes.
    captioned = set()
    for entry in captions:
        key = join_key(entry.get("input_path") or entry.get("thermal_path") or "")
        captioned.add(key)
        row = by_key.get(key)
        if row is None:
            unmatched += 1
            continue
        if entry.get("status") != "ok" or not entry.get("caption", "").strip():
            failed.append(row)
        elif count(entry["caption"]) > args.token_budget:
            over.append(row)
        else:
            fine += 1

    if unmatched:
        print(f"[warn] {unmatched} 条 caption 在 manifest 里找不到对应帧")

    missing = [row for key, row in by_key.items() if key not in captioned]

    retry = over + failed + missing
    seen, unique = set(), []
    for row in retry:
        key = join_key(row["thermal_path"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in unique:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(by_key)
    print(f"[retry] 合格 {fine}/{total}"
          f"  超预算 {len(over)}  生成失败 {len(failed)}  完全缺失 {len(missing)}")
    print(f"[retry] 需要重跑 {len(unique)} 帧 ({len(unique)/max(total,1):.1%}) -> {args.output}")
    if not unique:
        print("[retry] 没有要重跑的，可以直接进 manifest 重建")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
