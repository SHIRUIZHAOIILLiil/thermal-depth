#!/bin/bash
#SBATCH --job-name=pgt_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断，不是基准结果。在测试集**全帧**上，用 slurm/build_test_pseudo_gt.sbatch 存下的
# 「激光 + 校准后 AnyThermal」参照给一个模型打分：dense（全部像素）和 filled（只补全像素）。
# 参照在补全区就是 AnyThermal，并用测试激光校准过 —— 量的是与补全目标的一致性，不是准确度。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MODEL  ours_img | ours_img_nocap | ours_cap | ours_nocap | newcrf | anythermal
#   COND   day3_common | night3 | rainy3
#   KEEP_RAW=1  打完分保留逐帧预测（每个模型每个条件约 15 GB）。默认删。
#
# 我们的模型用**原始**完整测试清单推理：噪声种子 = 行号，与主表那次逐帧相同，
# 所以顺带得到的激光分数必须复现原来的全帧结果（自检）。
#
#   for M in ...; do for C in ...; do MODEL=$M COND=$C sbatch --dependency=afterok:<build> \
#       --export=ALL,MODEL,COND ~/Iris/slurm/pseudo_gt_eval.sbatch; done; done

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set +e

: "${MODEL:?必须指定 MODEL}"; : "${COND:?必须指定 COND}"
S=/mnt/scratch/sc23sz/runs
BASE="${BASE:-/mnt/scratch/dwdh0363/baselines}"
PGT="${PGT:-$IRIS_RUNS/analysis/test_pseudo_gt_full}"
OUT="$IRIS_RUNS/analysis/pseudo_gt_eval_full/${MODEL}_$COND"
ORIG="$IRIS_MANIFEST_DIR/ms2_test_${COND}_thermalcap_20260821.jsonl"
mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
for v in dense filled; do
  [[ -f "$PGT/manifests/ms2_test_${COND}_thermalcap_20260821_pseudo_$v.jsonl" ]] \
    || { echo "!! 找不到参照清单 $v —— 先跑 build_test_pseudo_gt.sbatch"; exit 1; }
done

RAW="$OUT/raw_predictions"
case "$MODEL" in
  ours_*)
    case "$MODEL" in
      ours_img)       CKPT=$S/iris_ms2/sd2_cap_pct_logtn_img_s43/converted/step16000_weights.pt;   PROMPT=correct ;;
      ours_img_nocap) CKPT=$S/iris_ms2/sd2_nocap_pct_logtn_img_s43/converted/step16000_weights.pt; PROMPT=empty ;;
      ours_cap)       CKPT=$S/iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt;       PROMPT=correct ;;
      ours_nocap)     CKPT=$S/iris_ms2/sd2_nocap_pct_logtn_s43/converted/step18000_weights.pt;     PROMPT=empty ;;
      *) echo "!! 不认识的 MODEL $MODEL"; exit 1 ;;
    esac
    ALIGN=ssi_log
    if [[ "$(ls "$OUT/infer/raw_predictions" 2>/dev/null | wc -l)" -eq 0 ]]; then
      echo "=== 推理（原清单，种子与主表一致）==="
      IRIS_THERMAL_STRETCH=percentile python tools/train_route_suite.py \
        --route b_thermal_unet --output-dir "$OUT/infer" --ms2-root "$IRIS_MS2_ROOT" \
        --train-manifest "$ORIG" --val-manifest "$ORIG" --eval-checkpoint "$CKPT" \
        --val-caption-mode "$PROMPT" --align-mode ssi_log --num-inference-steps 1 \
        --condition-latent mode --val-stride 1 --save-raw-pred --local-files-only \
        || { echo "!! 推理失败"; exit 1; }
    fi
    RAW="$OUT/infer/raw_predictions"
    echo "=== 激光自检（应复现原全帧结果）==="
    python -c "import json,glob;d=json.load(open(glob.glob('$OUT/infer/eval_*log_space.json')[0]));print('lidar',d['abs_rel'],d['rmse'],d['a1'])"
    ;;
  newcrf)
    ALIGN=ssi
    env=$(case "$COND" in day3_common) echo test_day;; night3) echo test_night;; rainy3) echo test_rain;; esac)
    if [[ "$(ls "$OUT/bench/raw_predictions/newcrf" 2>/dev/null | wc -l)" -eq 0 ]]; then
      ids=$(python -c "import json,sys;print(' '.join(json.loads(l)['id'] for l in open(sys.argv[1],encoding='utf-8') if l.strip()))" "$ORIG")
      python tools/run_ms2_supdepth_baselines.py --models newcrf --bmsd-root "$BASE/SupDepth4Thermal" \
        --ckpt-dir "$BASE/supdepth4thermal" --ms2-root "$IRIS_MS2_ROOT" --test-env "$env" \
        --sample-step 1 --output-dir "$OUT/bench" --save-pred-ids $ids 2>&1 | grep -v "saved prediction" | tail -n 12 \
        || { echo "!! NeWCRF 推理失败"; exit 1; }
    fi
    RAW="$OUT/bench/raw_predictions/newcrf"
    ;;
  anythermal)
    ALIGN=ssi
    RAW=$S/anythermal/official_$COND/raw_predictions
    ;;
  *) echo "!! 不认识的 MODEL $MODEL"; exit 1 ;;
esac

echo "=== 打分：dense / filled（只保留有预测的帧）==="
for v in dense filled; do
  man="$OUT/manifest_$v.jsonl"
  python - "$PGT/manifests/ms2_test_${COND}_thermalcap_20260821_pseudo_$v.jsonl" "$RAW" "$man" <<'PY'
import json, sys
from pathlib import Path
src, raw, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
kept = [l for l in rows if (raw / f"{json.loads(l)['id']}.npy").is_file()]
dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"[{dst.name}] {len(kept)} of {len(rows)} frames have a prediction")
PY
  python tools/run_official_ms2_evaluation.py --manifest "$man" --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$RAW" --route iris-lotus --align "$ALIGN" --output-dir "$OUT/score_$v" 2>&1 | tail -n 3
  python -c "import json;s=json.load(open('$OUT/score_$v/metrics/summary.json'))['image_wise'];st=s['statistics'];print('$v  frames',s['image_count'],' AbsRel %.4f  RMSE %.3f  d1 %.4f'%(st['abs_rel']['mean'],st['rmse']['mean'],st['a1']['mean']))" \
    || echo "!! 读不了 $OUT/score_$v/metrics/summary.json"
done

if [[ "${KEEP_RAW:-0}" != "1" && "$MODEL" != "anythermal" ]]; then
  rm -rf "$OUT/infer/raw_predictions" "$OUT/bench/raw_predictions"
  echo "[清理] 逐帧预测已删（KEEP_RAW=1 可保留）"
fi
echo "[完成] $OUT"
