#!/bin/bash
# 把 caption_ms2.sbatch 的分片输出合回一个文件，并核对没有缺帧。
#
# 分片是按 rows[i::n] 跨步切的，彼此不相交，所以直接 cat 就是完整集合 ——
# 但必须核对总数：array 里任何一片被 Slurm 杀掉、或超时没写完，
# 都会安静地少掉一千多帧，而下游 build_thermal_caption_manifest.py
# 只会按图路径 join，缺的帧会变成"没有 caption"而不是报错。
#
#   bash $HOME/Iris/slurm/merge_captions.sh train
#   bash $HOME/Iris/slurm/merge_captions.sh val
#   bash $HOME/Iris/slurm/merge_captions.sh test

set -euo pipefail

SPLIT="${1:?用法: merge_captions.sh <train|val|test> [prompt_version]}"
PROMPT_VERSION="${2:-thermal_depth_v3_1}"

: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"
SRC_DIR="${OUT_ROOT:-$SCRATCH/captions/${PROMPT_VERSION}/${SPLIT}}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$SCRATCH/manifests/capinput/ms2_${SPLIT}_capinput_20260818.jsonl}"
MERGED="$SRC_DIR/captions_${SPLIT}_${PROMPT_VERSION}_merged.jsonl"

[[ -d "$SRC_DIR"        ]] || { echo "!! 找不到分片目录：$SRC_DIR"; exit 1; }
[[ -f "$INPUT_MANIFEST" ]] || { echo "!! 找不到输入清单：$INPUT_MANIFEST"; exit 1; }

# 排除上一次合并的产物，否则重跑会把自己算进去
mapfile -t SHARDS < <(find "$SRC_DIR" -name "captions_${SPLIT}_shard*of*_internvl3_8b_thermal_${PROMPT_VERSION}.jsonl" | sort)
(( ${#SHARDS[@]} > 0 )) || { echo "!! $SRC_DIR 下没有分片输出"; exit 1; }

echo "=== 找到 ${#SHARDS[@]} 个分片 ==="
for f in "${SHARDS[@]}"; do printf "  %6d  %s\n" "$(wc -l < "$f")" "$(basename "$f")"; done

cat "${SHARDS[@]}" > "$MERGED"

python - "$MERGED" "$INPUT_MANIFEST" <<'PY'
import json, sys, collections
merged, expected = sys.argv[1], sys.argv[2]

# ⚠️ 千万别用 image_id 做 join 键：它是**裸帧号**，而每条 MS2 序列都有 000000。
# train 横跨两条序列，按裸帧号比对会得出"9507 个重复、全部缺失"这种看着像
# 灾难、其实只是键空间对不上的结论。按图路径末 5 段比对 —— 这也是
# tools/build_thermal_caption_manifest.py 用的键，两边保持一致。
KEY_DEPTH = 5  # sync_data/_<sequence>/thr/img_left/<frame>.png

def frame_key(value):
    parts = str(value).replace("\\", "/").lower().split("/")
    return "/".join(parts[-KEY_DEPTH:])

want = {frame_key(json.loads(l)["thermal_path"]) for l in open(expected, encoding="utf-8")}
rows = [json.loads(l) for l in open(merged, encoding="utf-8")]
got = collections.Counter(
    frame_key(r.get("thermal_path") or r.get("input_path") or "") for r in rows
)
status = collections.Counter(r.get("status", "?") for r in rows)

dupes = {k: v for k, v in got.items() if v > 1}
missing = want - set(got)
extra = set(got) - want

print(f"\n=== 合并结果 {merged} ===")
print(f"期望 {len(want)} 帧，实得 {len(rows)} 行、{len(got)} 个唯一帧")
print(f"状态分布: {dict(status)}")
if dupes:
    print(f"!! 重复帧 {len(dupes)} 个（前 5）: {list(dupes)[:5]}")
if missing:
    print(f"!! 缺失 {len(missing)} 帧（前 5）: {sorted(missing)[:5]}")
    print("   补跑：重投同一条 sbatch 命令即可，--resume 会跳过已成功的行")
if extra:
    print(f"!! 多出 {len(extra)} 帧不在输入清单里（前 5）: {sorted(extra)[:5]}")

bad = [r for r in rows if r.get("status") != "ok"]
if bad:
    # 质量检查拦下来的（多为词语重复），不算缺帧：它们有行、只是没有可用 caption。
    # 数量少可以直接放着（下游按帧 join，这些帧就是没有 caption），
    # 多了就得回头看 prompt。
    reasons = collections.Counter(str(r.get("error"))[:60] for r in bad)
    print(f"!! 非 ok 行 {len(bad)} 条 = {len(bad)/len(rows):.3%}，原因分布:")
    for reason, n in reasons.most_common(5):
        print(f"   {n:5d}  {reason}")

sys.exit(1 if (missing or dupes or extra) else 0)
PY

echo
echo "=== OK：$MERGED ==="
