#!/bin/bash
#SBATCH --job-name=qualpred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 定性图（fig:qual）要的全部原料，一个作业出齐，打成一个包，下载一次。
#
# 三帧是 tools/pick_qualitative_frames.py 排出来、2026-10-01 选定的：
#   day   2021-08-06-11-23-45_005720
#   night 2021-08-13-22-03-03_006139
#   rainy 2021-08-06-16-45-28_005762
#
# 三个模型都**走它们出主表数字的那条路**，只是这次把逐帧预测留下来：
#   Ours    train_route_suite.py，s43 step18000，correct prompt，ssi_log
#   DA2 v1  run_external_depth_inference.py，step16000，ssi_disparity
#   PPD v2  run_external_depth_inference.py，e014-s030000，ssi_log
#
# ⭐ 自检：三帧的逐帧 AbsRel 要和当初 test 留下的逐帧 CSV 对上。对上了，图上画的
#    就是主表打分的那个预测；对不上，说明推理设置和主表不一致 —— 结果表里会标
#    出来，不会悄悄画出一张别的模型的图。
#    Ours 的热像拉伸在 pipeline 里默认 minmax、run 名里写的是 pct，两种都跑，
#    由自检决定哪个是当初那个。
#
# PPD 的环境和代码在 sc23sz 名下读不到。代码用 $PPD_REPO（学长 scratch 下的上游克隆，
# 427a86f）；环境是叠在 iris 上的 venv，只把 timm 换成他们要的 0.9.1（iris 的 timm
# 太旧，没有 timm.layers）。借来的环境数值上对不对，同样交给自检判。建法：
#   python -m venv --system-site-packages /mnt/scratch/dwdh0363/ppd_venv
#   /mnt/scratch/dwdh0363/ppd_venv/bin/pip install -q timm==0.9.1
#
# ⚠️ 2026-10-01：sc23sz 名下 DA2 v1 的 model.safetensors 和 PPD e014 是 -rw-------+，
#    ACL 掩码 ---，给学长的 r-x 实际不生效（getfacl 显示 #effective:---）。只有
#    文件主人或管理员能改。拿到可读副本后用 DA2_CKPT= / PPD_CKPT= 指过去重投；
#    已经出了预测的模型会跳过。
#
# 用法（学长账号，一行，见 docs 里的交接）：
#   sbatch ~/Iris/slurm/qual_figure_preds.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set +e   # 一个模型挂了不连累另外两个；每一步自己报

S=/mnt/scratch/sc23sz/runs
OURS_CKPT="${OURS_CKPT:-$S/iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt}"
DA2_CKPT="${DA2_CKPT:-$S/da2_thermal/v1/step16000}"
PPD_CKPT="${PPD_CKPT:-$S/ppd_ms2/v2_1024/checkpoints/e014-s030000.ckpt}"
export PPD_REPO="${PPD_REPO:-/mnt/scratch/dwdh0363/ppd_repo}"
PPD_PY="${PPD_PY:-/mnt/scratch/dwdh0363/ppd_venv/bin/python}"
PPD_SEM="${PPD_SEM:-$PPD_REPO/checkpoints/depth_anything_v2_vitl.pth}"
OUT="${OUT:-$IRIS_RUNS/analysis/qual_figure}"

# 可用 DAY_FRAME / NIGHT_FRAME / RAINY_FRAME 换帧（同时给 OUT 换目录，免得
# 「已有三帧预测就跳过」把旧帧当成新帧）。
declare -A FRAME=(
  [day]="${DAY_FRAME:-2021-08-06-11-23-45_005720}"
  [night]="${NIGHT_FRAME:-2021-08-13-22-03-03_006139}"
  [rainy]="${RAINY_FRAME:-2021-08-06-16-45-28_005762}"
)
declare -A MFST=(
  [day]=ms2_test_day3_common_thermalcap_20260821.jsonl
  [night]=ms2_test_night3_thermalcap_20260821.jsonl
  [rainy]=ms2_test_rainy3_thermalcap_20260821.jsonl
)

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 输入检查 ==="
ok=1
# -r 看的是权限位，被 ACL 掩码盖掉的读权限它不一定看得出；真读一个字节才算数。
for p in "$OURS_CKPT" "$DA2_CKPT/model.safetensors" "$PPD_CKPT" "$PPD_REPO/ppd/models/ppd.py" "$PPD_SEM" "$PPD_PY"; do
  if head -c 1 "$p" > /dev/null 2>&1; then echo "  ok   $p"; else echo "  !!   读不了 $p"; ok=0; fi
done
done3() { [[ "$(ls "$1" 2>/dev/null | grep -c '[.]npy$')" == 3 ]]; }

# --- 1. 三帧子清单 + GT/热像原件 ----------------------------------------------
python - "$OUT" "$IRIS_MANIFEST_DIR" "$IRIS_MS2_ROOT" \
  "day:${FRAME[day]}:${MFST[day]}" "night:${FRAME[night]}:${MFST[night]}" \
  "rainy:${FRAME[rainy]}:${MFST[rainy]}" <<'PY'
import json, shutil, sys
from pathlib import Path
out, mdir, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
(out / "inputs").mkdir(parents=True, exist_ok=True)
rows, at, ordinals = [], {}, []
for spec in sys.argv[4:]:
    cond, fid, mname = spec.split(":")
    with (mdir / mname).open(encoding="utf-8") as h:   # 与 train_route_suite 的计数方式一致
        hit = [(i, json.loads(l)) for i, l in enumerate(h)
               if l.strip() and json.loads(l).get("id") == fid]
    if len(hit) != 1:
        raise SystemExit(f"!! {mname} 里 id={fid} 找到 {len(hit)} 行")
    index, row = hit[0]
    if index in at:
        raise SystemExit(f"!! {fid} 与 {at[index]['id']} 在各自清单里同在第 {index} 行，补空行的办法不适用")
    at[index] = row
    rows.append(row)
    # run_ms2_lotus_thermal_vae_official.py 的种子是 seed + 该帧在**非空行**里的序号
    with (mdir / mname).open(encoding="utf-8") as h:
        ordinal = sum(1 for i, l in enumerate(h) if i < index and l.strip())
    ordinals.append(f"{cond} {ordinal}\n")
    print(f"[帧] {cond:5s} 在 {mname} 第 {index} 行（非空行序号 {ordinal}）")
    for kind, key in (("thermal", "thermal_path"), ("gt", "thermal_depth_path")):
        src = row.get(key) or (row.get("depth_path") if kind == "gt" else None)
        src = Path(src) if Path(src).is_absolute() else root / src
        shutil.copy2(src, out / "inputs" / f"{cond}_{fid}_{kind}.png")
    print(f"[帧] {cond:5s} {fid}  caption: {str(row.get('caption', ''))[:70]}…")
(out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
# Ours 的噪声种子是 seed + 该行在清单里的行号（train_route_suite.py 的 manifest_index，
# 空行也计数）。三帧挤在一份三行清单里，种子就和主表那次不同，逐帧差到 0.003
# （2026-10-01 作业 8024289）。所以给 Ours 另写一份：每帧放回它原来的行号，中间补空行。
padded = []
for index in sorted(at):
    padded += [""] * (index - len(padded))
    padded.append(json.dumps(at[index]))
(out / "frames_ours.jsonl").write_text("\n".join(padded) + "\n", encoding="utf-8")
(out / "ordinals.txt").write_text("".join(ordinals), encoding="utf-8")
PY
[[ -s "$OUT/frames.jsonl" ]] || { echo "!! 子清单没建成，停"; exit 1; }
FR="$OUT/frames.jsonl"

# --- 2. Ours -------------------------------------------------------------------
for st in percentile minmax; do
  echo; echo "=== Ours  (thermal stretch = $st) ==="
  done3 "$OUT/ours_$st/raw_predictions" && { echo "[skip] 已有三帧预测"; continue; }
  IRIS_THERMAL_STRETCH=$st python tools/train_route_suite.py \
    --route b_thermal_unet --output-dir "$OUT/ours_$st" --ms2-root "$IRIS_MS2_ROOT" \
    --train-manifest "$OUT/frames_ours.jsonl" --val-manifest "$OUT/frames_ours.jsonl" \
    --eval-checkpoint "$OURS_CKPT" \
    --val-caption-mode correct --align-mode ssi_log --num-inference-steps 1 \
    --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
    || echo "!! Ours ($st) 失败"
done

# --- 2b. Lotus-G 零样本（主表「Lotus-G 未适配」那一行；备案版图用它当「适配前」）------
# 当初 slurm/lotus_zeroshot.sbatch 存过逐帧 .npy；还在就直接拷，最准。不在就逐帧单独
# 推理：该脚本的种子是 seed + 该帧在清单非空行里的序号，单帧清单里序号是 0，所以把
# --seed 设成 20260701 + 原序号，噪声就和当初一样。
echo; echo "=== Lotus-G zero-shot ==="
L0="$OUT/lotus0"
if done3 "$L0/raw_predictions"; then echo "[skip] 已有三帧预测"; else
  mkdir -p "$L0/raw_predictions"
  while read -r cond ordinal; do
    fid="${FRAME[$cond]}"
    old="$S/analysis/lotus_zeroshot_$cond/raw_predictions/$fid.npy"
    if head -c 1 "$old" > /dev/null 2>&1; then
      cp "$old" "$L0/raw_predictions/" && echo "[copy] $cond 当初的逐帧预测还在，直接拷"
      continue
    fi
    grep -F "\"id\": \"$fid\"" "$FR" > "$L0/$cond.jsonl"
    rm -rf "$L0/run_$cond"
    IRIS_THERMAL_STRETCH=percentile IRIS_DECODE_CLAMP=0 python tools/run_ms2_lotus_thermal_vae_official.py \
      --manifest "$L0/$cond.jsonl" --ms2-root "$IRIS_MS2_ROOT" --output-dir "$L0/run_$cond" \
      --max-samples 0 --caption-mode empty --save-raw-pred --seed $((20260701 + ordinal)) \
    && cp "$L0/run_$cond/raw_predictions/$fid.npy" "$L0/raw_predictions/" \
    || echo "!! Lotus-G 零样本 $cond 失败"
  done < "$OUT/ordinals.txt"
fi
python tools/run_official_ms2_evaluation.py --manifest "$FR" --data-root "$IRIS_MS2_ROOT" \
  --prediction-dir "$L0/raw_predictions" --route iris-lotus --align ssi_disparity \
  --output-dir "$L0" || echo "!! Lotus-G 零样本打分失败"

# --- 3. DA2 v1 -----------------------------------------------------------------
echo; echo "=== DA2 v1 ==="
export IRIS_THERMAL_STRETCH=percentile
if done3 "$OUT/da2/raw_predictions"; then echo "[skip] 已有三帧预测"; else
python tools/run_external_depth_inference.py --model depth_anything_v2 --manifest "$FR" \
  --data-root "$IRIS_MS2_ROOT" --out-dir "$OUT/da2/raw_predictions" --stretch percentile \
  --weights "$DA2_CKPT" \
&& python tools/run_official_ms2_evaluation.py --manifest "$FR" --data-root "$IRIS_MS2_ROOT" \
  --prediction-dir "$OUT/da2/raw_predictions" --route iris-lotus --align ssi_disparity \
  --output-dir "$OUT/da2" \
|| echo "!! DA2 失败"
fi

# --- 4. PPD v2 -----------------------------------------------------------------
echo; echo "=== PPD v2 ==="
if done3 "$OUT/ppd/raw_predictions"; then echo "[skip] 已有三帧预测"
elif PYTHONPATH="$PPD_REPO" "$PPD_PY" -c "from ppd.models.ppd import PixelPerfectDepth" 2> "$OUT/ppd_import.err"; then
  "$PPD_PY" tools/run_external_depth_inference.py --model pixel_perfect_depth --manifest "$FR" \
    --data-root "$IRIS_MS2_ROOT" --out-dir "$OUT/ppd/raw_predictions" --stretch percentile \
    --ppd-checkpoint "$PPD_CKPT" --ppd-semantics "$PPD_SEM" \
  && python tools/run_official_ms2_evaluation.py --manifest "$FR" --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$OUT/ppd/raw_predictions" --route iris-lotus --align ssi_log \
    --output-dir "$OUT/ppd" \
  || echo "!! PPD 失败"
else
  echo "!! $PPD_PY import 不了 PPD："; tail -n 3 "$OUT/ppd_import.err"
fi

# --- 5. 自检：逐帧 AbsRel 对当初的 test CSV ----------------------------------
echo; echo "=== 自检：这三帧的 AbsRel，本次 vs 主表那次 ==="
python - "$OUT" "$S" "${FRAME[day]}" "${FRAME[night]}" "${FRAME[rainy]}" <<'PY' | tee "$OUT/selfcheck.txt"
import csv, glob, sys
from pathlib import Path
out, S, ids = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]

def table(paths):
    got = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                fid = r.get("sample_id") or r.get("id")
                key = "abs_rel" if "abs_rel" in r else next((k for k in r if "abs_rel" in k), None)
                if fid in ids and key:
                    got[fid] = float(r[key])
    return got

ref = {
    "ours": glob.glob(f"{S}/eval/sd2_cap_pct_logtn_s43_test_*_s18000_correct/eval_*log_space*per_sample.csv"),
    "lotus0": glob.glob(f"{S}/analysis/lotus_zeroshot_*/official/metrics/per_image.csv"),
    "da2":  glob.glob(f"{S}/analysis/da2_test_s16000_*/**/per_image.csv", recursive=True),
    "ppd":  glob.glob(f"{S}/analysis/ppdv2_test_*/**/per_image.csv", recursive=True),
}
new = {
    "ours_percentile": glob.glob(f"{out}/ours_percentile/eval_*per_sample.csv"),
    "ours_minmax":     glob.glob(f"{out}/ours_minmax/eval_*per_sample.csv"),
    "lotus0":          [f"{out}/lotus0/metrics/per_image.csv"],
    "da2":             [f"{out}/da2/metrics/per_image.csv"],
    "ppd":             [f"{out}/ppd/metrics/per_image.csv"],
}
print(f"{'模型':16s} {'帧':28s} {'主表那次':>9s} {'本次':>9s} {'差':>9s}")
for name, paths in new.items():
    r = table(ref[name.split("_")[0]])
    n = table([p for p in paths if Path(p).is_file()])
    for fid in ids:
        a, b = r.get(fid), n.get(fid)
        if a is None or b is None:
            print(f"{name:16s} {fid:28s} {a if a is not None else '缺':>9} {b if b is not None else '缺':>9}  ⚠️")
            continue
        mark = "✅" if abs(a - b) < 5e-4 else "⚠️"
        print(f"{name:16s} {fid:28s} {a:9.5f} {b:9.5f} {b - a:+9.5f}  {mark}")
PY

# --- 6. 打包 -------------------------------------------------------------------
cd "$OUT/.."
B="$(basename "$OUT")"
tar czf "$B.tar.gz" \
  "$B/inputs" "$B/frames.jsonl" "$B/selfcheck.txt" \
  $(ls -d "$B"/ours_*/raw_predictions "$B/lotus0/raw_predictions" "$B/da2/raw_predictions" \
          "$B/ppd/raw_predictions" 2>/dev/null)
echo
echo "[完成] 下载这一个文件：$(pwd)/$B.tar.gz  ($(du -h "$B.tar.gz" | cut -f1))"
