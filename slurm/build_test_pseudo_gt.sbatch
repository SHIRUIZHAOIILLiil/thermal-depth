#!/bin/bash
#SBATCH --job-name=pgt_build
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断用参照，不是真值。测试集**全帧**（不抽帧）的「激光 + 校准后 AnyThermal」稠密深度，
# 存成 MS2 GT 同格式的 16 位 PNG，外加指向它的清单。只读图、拟合两个参数、写文件，
# 不需要 GPU，所以不写 --partition（走默认 CPU 分区）。
# 生成方式见 tools/build_test_pseudo_gt.py；和训练目标、和 score_on_pseudo_gt.py 用同一个校准函数。
#
#   sbatch ~/Iris/slurm/build_test_pseudo_gt.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

OUT="${OUT:-$IRIS_RUNS/analysis/test_pseudo_gt_full}"
cd "$IRIS_REPO"
git log --oneline -1
M="$IRIS_MANIFEST_DIR"
python tools/build_test_pseudo_gt.py --ms2-root "$IRIS_MS2_ROOT" \
  --anythermal-root /mnt/scratch/sc23sz/runs/anythermal --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "$M/ms2_test_day3_common_thermalcap_20260821.jsonl" \
  "$M/ms2_test_night3_thermalcap_20260821.jsonl" \
  "$M/ms2_test_rainy3_thermalcap_20260821.jsonl"
echo
echo "=== 产物 ==="
for m in day3_common night3 rainy3; do
  echo "$m  dense $(ls "$OUT/$m/dense" | wc -l)  filled $(ls "$OUT/$m/filled" | wc -l)"
done
wc -l "$OUT"/manifests/*.jsonl
du -sh "$OUT"
