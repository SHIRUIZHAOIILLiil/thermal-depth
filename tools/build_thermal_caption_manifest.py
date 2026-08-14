"""Write a manifest whose captions describe the image the model is actually fed.

The captions this project trains on were generated from the RGB camera while the
depth model sees the thermal one, so 30% of them name a colour that cannot exist
in the model's input, from a viewpoint that is not its own. The completed 2x2
showed the model uses only the presence of that text, not its content. This
rebuilds the manifest against captions generated from the thermal frame itself.

Three things it has to get right:

  join      The caption files key on `image_id`, which is the bare frame number,
            and every MS2 sequence has a 000000. Joining on it would merge the
            two training sequences. The image path identifies a frame, so the
            key is its last few components -- independent of whether one side
            wrote /mnt/e/... and the other E:\\... or a root-relative path.
  precedence A frame can appear in more than one caption file: the second pass
            regenerated the 7.9% that overran the CLIP budget. The later prompt
            version wins, and file order must not decide it.
  budget    The CLIP text encoder takes 77 tokens and silently drops the rest,
            so an over-long caption is one with an unknown ending. Anything
            still over after the retry pass is trimmed at a clause boundary and
            counted; a handful is noise, a systematic count is a problem with
            the prompt and should stop the build.

The source manifest is never modified, and every non-caption field and the row
order are copied verbatim, so the result stays frame-for-frame comparable with
the RGB-caption runs.

    python tools/build_thermal_caption_manifest.py \
        --manifest <original.jsonl> --captions <a.jsonl> <b.jsonl> \
        --output <new.jsonl>
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Later version wins when a frame appears more than once.
PROMPT_PRECEDENCE = {"thermal_depth_v1": 1, "thermal_depth_v2": 2, "thermal_depth_v2_short": 3}
KEY_DEPTH = 5  # sync_data/_<sequence>/thr/img_left/<frame>.png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="原 manifest，不会被修改")
    parser.add_argument("--captions", type=Path, nargs="+", required=True,
                        help="一个或多个 caption jsonl（含重跑那一批）")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=75)
    parser.add_argument("--max-trimmed", type=int, default=20,
                        help="超过这个条数就判失败：个位数是噪声，成百上千说明 prompt 有问题")
    parser.add_argument("--clip-tokenizer", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--clip-tokenizer-subfolder", default="tokenizer")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_key(value: str) -> str:
    """Last few path components, lowercased. Root- and platform-independent."""
    parts = Path(str(value).replace("\\", "/")).as_posix().lower().split("/")
    return "/".join(parts[-KEY_DEPTH:])


def trim_to_budget(caption: str, count, budget: int) -> tuple[str, bool]:
    """Drop trailing clauses until it fits. Last resort only -- in this prose
    format the tail carries the near-to-far ordering, so trimming costs depth
    content, which is why the retry pass exists to make it rare."""
    caption = " ".join(caption.split())
    if count(caption) <= budget:
        return caption, False
    parts = re.split(r"(?<=[.;,])\s+", caption)
    while len(parts) > 1:
        parts.pop()
        candidate = " ".join(parts).rstrip(" ,;")
        if not candidate.endswith("."):
            candidate += "."
        if count(candidate) <= budget:
            return candidate, True
    words = caption.split()
    while words and count(" ".join(words) + ".") > budget:
        words.pop()
    return (" ".join(words) + ".") if words else "", True


def main() -> int:
    args = parse_args()
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        args.clip_tokenizer, subfolder=args.clip_tokenizer_subfolder or None
    )

    def count(text: str) -> int:
        # Same convention as the Iris acceptance validator: BOS/EOS in, no truncation.
        return len(tokenizer(text)["input_ids"])

    chosen: dict[str, dict] = {}
    per_version = Counter()
    for path in args.captions:
        for row in read_jsonl(path):
            if row.get("status") != "ok" or not (row.get("caption") or "").strip():
                continue
            source = row.get("input_path") or row.get("thermal_path") or ""
            if not source:
                continue
            version = row.get("prompt_version", "")
            per_version[version] += 1
            key = frame_key(source)
            rank = PROMPT_PRECEDENCE.get(version, 0)
            if key not in chosen or rank > PROMPT_PRECEDENCE.get(chosen[key].get("prompt_version", ""), 0):
                chosen[key] = row

    rows = read_jsonl(args.manifest)
    written, missing, trimmed, used = [], [], 0, Counter()
    for row in rows:
        key = frame_key(row.get("thermal_path") or row.get("depth_path") or "")
        entry = chosen.get(key)
        if entry is None:
            missing.append(row.get("id", "?"))
            continue
        caption, was_trimmed = trim_to_budget(entry["caption"], count, args.token_budget)
        trimmed += int(was_trimmed)
        used[entry.get("prompt_version", "")] += 1

        new_row = dict(row)  # every non-caption field and the row order kept as-is
        new_row["caption"] = caption
        new_row["caption_clip_tokens"] = count(caption)
        new_row["caption_prompt_version"] = entry.get("prompt_version", "")
        new_row["caption_model"] = entry.get("caption_model", "")
        new_row["caption_model_id"] = entry.get("caption_model_id", "")
        new_row["caption_status"] = "ok"
        # The two fields that say this caption describes the thermal camera. A
        # manifest without them is indistinguishable from the RGB-caption one.
        new_row["caption_input_modality"] = entry.get("input_modality", "thermal")
        new_row["caption_thermal_render"] = entry.get("thermal_render", "")
        new_row["caption_trimmed"] = bool(was_trimmed)
        written.append(new_row)

    print(f"[captions] 读入 {sum(per_version.values())} 条，按版本: {dict(per_version)}")
    print(f"[captions] 去重后 {len(chosen)} 帧，采用版本: {dict(used)}")
    print(f"[manifest] {len(rows)} 行 -> 写出 {len(written)}；缺 caption {len(missing)}；截断 {trimmed}")

    if missing:
        raise SystemExit(f"有 {len(missing)} 帧没有 caption，前几个: {missing[:5]}。"
                         f"\n补齐后再重建 —— 空 caption 会被当成'空 prompt'训练，等于混进另一个实验臂。")
    if trimmed > args.max_trimmed:
        raise SystemExit(f"截断了 {trimmed} 条，超过 --max-trimmed {args.max_trimmed}。"
                         f"\n这不是噪声，是 prompt 的长度约束不够 —— 重跑超标帧，别让工具替你删内容。")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = sorted(row["caption_clip_tokens"] for row in written)
    print(f"[done] {args.output}")
    print(f"[done] CLIP token 中位 {lengths[len(lengths)//2]}  p90 {lengths[int(len(lengths)*0.9)]}  "
          f"最大 {lengths[-1]}  超预算 {sum(1 for x in lengths if x > args.token_budget)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
