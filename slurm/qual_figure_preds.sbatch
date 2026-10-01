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
# 427a86f），环境借 iris —— 推理那条路只用到 torch/torchvision/timm/cv2/omegaconf/scipy。
# 借来的环境数值上对不对，同样交给自检判。
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
PPD_SEM="${PPD_SEM:-$PPD_REPO/checkpoints/depth_anything_v2_vitl.pth}"
OUT="${OUT:-$IRIS_RUNS/analysis/qual_figure}"

declare -A FRAME=(
  [day]=2021-08-06-11-23-45_005720
  [night]=2021-08-13-22-03-03_006139
  [rainy]=2021-08-06-16-45-28_005762
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
for p in "$OURS_CKPT" "$DA2_CKPT" "$PPD_CKPT" "$PPD_REPO/ppd/models/ppd.py" "$PPD_SEM"; do
  if [[ -r "$p" ]]; then echo "  ok   $p"; else echo "  !!   读不到 $p"; ok=0; fi
done
[[ -r "$PPD_CKPT" ]] && { head -c 1 "$PPD_CKPT" > /dev/null 2>&1 || { echo "  !!   $PPD_CKPT 能看见但读不出（ACL）"; ok=0; }; }

# --- 1. 三帧子清单 + GT/热像原件 ----------------------------------------------
python - "$OUT" "$IRIS_MANIFEST_DIR" "$IRIS_MS2_ROOT" \
  "day:${FRAME[day]}:${MFST[day]}" "night:${FRAME[night]}:${MFST[night]}" \
  "rainy:${FRAME[rainy]}:${MFST[rainy]}" <<'PY'
import json, shutil, sys
from pathlib import Path
out, mdir, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
(out / "inputs").mkdir(parents=True, exist_ok=True)
rows = []
for spec in sys.argv[4:]:
    cond, fid, mname = spec.split(":")
    hit = [json.loads(l) for l in (mdir / mname).read_text(encoding="utf-8").splitlines()
           if l.strip() and json.loads(l).get("id") == fid]
    if len(hit) != 1:
        raise SystemExit(f"!! {mname} 里 id={fid} 找到 {len(hit)} 行")
    row = hit[0]
    rows.append(row)
    for kind, key in (("thermal", "thermal_path"), ("gt", "thermal_depth_path")):
        src = row.get(key) or (row.get("depth_path") if kind == "gt" else None)
        src = Path(src) if Path(src).is_absolute() else root / src
        shutil.copy2(src, out / "inputs" / f"{cond}_{fid}_{kind}.png")
    print(f"[帧] {cond:5s} {fid}  caption: {str(row.get('caption', ''))[:70]}…")
(out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
PY
[[ -s "$OUT/frames.jsonl" ]] || { echo "!! 子清单没建成，停"; exit 1; }
FR="$OUT/frames.jsonl"

# --- 2. Ours -------------------------------------------------------------------
for st in percentile minmax; do
  echo; echo "=== Ours  (thermal stretch = $st) ==="
  IRIS_THERMAL_STRETCH=$st python tools/train_route_suite.py \
    --route b_thermal_unet --output-dir "$OUT/ours_$st" --ms2-root "$IRIS_MS2_ROOT" \
    --train-manifest "$FR" --val-manifest "$FR" --eval-checkpoint "$OURS_CKPT" \
    --val-caption-mode correct --align-mode ssi_log --num-inference-steps 1 \
    --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
    || echo "!! Ours ($st) 失败"
done

# --- 3. DA2 v1 -----------------------------------------------------------------
echo; echo "=== DA2 v1 ==="
export IRIS_THERMAL_STRETCH=percentile
python tools/run_external_depth_inference.py --model depth_anything_v2 --manifest "$FR" \
  --data-root "$IRIS_MS2_ROOT" --out-dir "$OUT/da2/raw_predictions" --stretch percentile \
  --weights "$DA2_CKPT" \
&& python tools/run_official_ms2_evaluation.py --manifest "$FR" --data-root "$IRIS_MS2_ROOT" \
  --prediction-dir "$OUT/da2/raw_predictions" --route iris-lotus --align ssi_disparity \
  --output-dir "$OUT/da2" \
|| echo "!! DA2 失败"

# --- 4. PPD v2 -----------------------------------------------------------------
echo; echo "=== PPD v2 ==="
if PYTHONPATH="$PPD_REPO" python -c "from ppd.models.ppd import PixelPerfectDepth" 2> "$OUT/ppd_import.err"; then
  python tools/run_external_depth_inference.py --model pixel_perfect_depth --manifest "$FR" \
    --data-root "$IRIS_MS2_ROOT" --out-dir "$OUT/ppd/raw_predictions" --stretch percentile \
    --ppd-checkpoint "$PPD_CKPT" --ppd-semantics "$PPD_SEM" \
  && python tools/run_official_ms2_evaluation.py --manifest "$FR" --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$OUT/ppd/raw_predictions" --route iris-lotus --align ssi_log \
    --output-dir "$OUT/ppd" \
  || echo "!! PPD 失败"
else
  echo "!! iris 环境 import 不了 PPD："; tail -n 3 "$OUT/ppd_import.err"
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
    "da2":  glob.glob(f"{S}/analysis/da2_test_s16000_*/**/per_image.csv", recursive=True),
    "ppd":  glob.glob(f"{S}/analysis/ppdv2_test_*/**/per_image.csv", recursive=True),
}
new = {
    "ours_percentile": glob.glob(f"{out}/ours_percentile/eval_*per_sample.csv"),
    "ours_minmax":     glob.glob(f"{out}/ours_minmax/eval_*per_sample.csv"),
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
tar czf qual_figure.tar.gz \
  qual_figure/inputs qual_figure/frames.jsonl qual_figure/selfcheck.txt \
  $(ls -d qual_figure/ours_*/raw_predictions qual_figure/da2/raw_predictions \
          qual_figure/ppd/raw_predictions 2>/dev/null)
echo
echo "[完成] 下载这一个文件：$(pwd)/qual_figure.tar.gz  ($(du -h qual_figure.tar.gz | cut -f1))"
