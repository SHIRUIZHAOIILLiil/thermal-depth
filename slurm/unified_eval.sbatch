#!/bin/bash
#SBATCH --job-name=unified_eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 把多条路线放到 V1 统一协议（docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md）下评估：
# 每条线先按注册的 adapter 把原始输出换算成正深度，再统一在深度空间做仿射对齐。
# 这是 b/c1/c2 与 AnyThermal 唯一能合法同表的方式 —— 旧的 official CLI 里
# ssi 与 ssi_disparity 互斥，强行统一必然打崩一方（见 WEEK3 文档 §8）。
#
# 不吃 GPU：pipeline.py 全程 numpy，不 import torch。
#
# ⚠️ 一把尺子是靠**结构**保证的，不是靠自觉：四条线在同一个作业里、读同一份
#    渲染好的 ruler.yaml，逐条只覆盖 route / prediction_dir / output_dir。
#    要改判据就改下面那段 heredoc，四条线一起变。
#
# ⚠️ 只放 thr 视角 GT 的线。a 线是 rgb 视角 GT，混进来就是拿两张考卷比分数
#    （AIRE 那轮踩过）。pipeline 会把 gt_view 写死成 thermal-view 记进元数据。
#
# ⚠️ 磁盘：每样本存 2 个 npy（native + affine_aligned），4 条线 × 2543 帧
#    ≈ 2 万文件 / 约 10GB，全部落 $SCRATCH。出完表就清。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   TAG_ROOT      输出根目录名，默认 v1_test_16-08-46
#   VAL_MANIFEST  评测集，默认 test 16-08-46
#   MIN_DEPTH     默认 1e-3，与官方 BMSD 协议一致（ms2_eval/official_protocol.py:32）
#                 —— 改了它，V1 的数就不再能和 0.0821/0.0884 对照读
#
# ── 用法 ────────────────────────────────────────────────────────────
#   sbatch ~/Iris/slurm/unified_eval.sbatch
#   # 先验一次（不读预测，只查 manifest/GT）：
#   AUDIT_ONLY=1 sbatch -J unified_audit --time=00:20:00 \
#     --export=ALL,AUDIT_ONLY ~/Iris/slurm/unified_eval.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

TAG_ROOT="${TAG_ROOT:-v1_test_16-08-46}"
OUT_ROOT="$IRIS_RUNS/unified/$TAG_ROOT"
VAL_MANIFEST="${VAL_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl}"
[[ -f "$VAL_MANIFEST" ]] || { echo "!! 找不到 manifest: $VAL_MANIFEST"; exit 1; }
mkdir -p "$OUT_ROOT"

# tag|route|raw 预测目录。route 必须是 ms2_eval/adapters.py 里注册过的名字。
# b/c1/c2 都是 Lotus 视差输出，共用 iris-lotus 声明（adapter-only / adapter+u-net
# 的换算式逐字相同，换标签只改记录不改数值）。
ENTRIES=(
  "b_e05|iris-lotus|$IRIS_RUNS/eval/b_e05_raw/raw_predictions"
  "c1|iris-lotus|$IRIS_RUNS/eval/c1_raw/raw_predictions"
  "c2|iris-lotus|$IRIS_RUNS/eval/c2_raw/raw_predictions"
  "anythermal|anythermal|$SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions"
)

# 排队几小时后才发现某条线的 raw 不在，是最不值的一种失败
for entry in "${ENTRIES[@]}"; do
  dir="${entry##*|}"
  [[ -d "$dir" ]] || { echo "!! 找不到 raw 预测: $dir"; exit 1; }
  echo "raw: ${entry%%|*} -> $(ls "$dir" | wc -l) 帧"
done

RULER="$OUT_ROOT/ruler.yaml"
cat > "$RULER" <<YAML
protocol_version: MS2-unified-depth-evaluation-v1
output_dir: $OUT_ROOT/_placeholder
audit_only: ${AUDIT_ONLY:-false}
gt_audit_samples: 8
visualization_samples: 8

dataset:
  manifest: $VAL_MANIFEST
  root: $IRIS_MS2_ROOT
  input_view: left-thermal
  gt_view: thermal-view-filtered-lidar
  depth_scale: 256.0
  allow_legacy_depth_path: false

evaluation:
  min_depth_m: ${MIN_DEPTH:-1.0e-3}
  max_depth_m: 80.0
  spearman: true
  image_wise_primary: true
  global_pixel_secondary: true
  bootstrap_iterations: 2000
  bootstrap_confidence: 0.95
  bootstrap_seed: 20260630

model:
  route: null
  checkpoint: null
  prediction_dir: null
  epsilon: 1.0e-6
  clip_min: null
  clip_max: null

visualization:
  colormap: magma_r
  depth_range_m: [0.1, 80.0]
  error_max_m: 10.0
  per_image_minmax_for_main_figures: false
YAML
echo "=== 尺子 ==="; cat "$RULER"

cd "$IRIS_REPO"
# common.sh 开了 set -euo pipefail，所以这里不能用 `cond && a || b`（条件为假时
# 整个复合命令返回 1，直接把作业带走），也不能靠 `cmd; rc=$?` 接住失败
# —— 一条线炸了要接着跑剩下的，才知道是四条全错还是只错一条。
AUDIT=()
if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then AUDIT=(--audit-only); fi

failed=0
for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r tag route pred <<< "$entry"
  echo; echo "=== [$tag] route=$route ==="
  # set -u 下空数组展开在旧 bash 里会报 unbound variable
  if ! python tools/run_unified_ms2_evaluation.py --config "$RULER" \
      --route "$route" --prediction-dir "$pred" --output-dir "$OUT_ROOT/$tag" \
      ${AUDIT[@]+"${AUDIT[@]}"}; then
    echo "!! [$tag] 失败"; failed=1; continue
  fi
  if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
    # audit 查的是 manifest 与 GT，与路线无关，跑一条就够；而且它在建 adapter
    # 之前就返回，下面的探针没有 output_adapter.json 可读
    echo "  audit 与路线无关，其余三条跳过"; break
  fi
  # §13.4 的教训：作业 COMPLETED 不等于开关生效。这两行是探针，不是日志。
  python - "$OUT_ROOT/$tag" <<'PROBE'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
adapter = json.loads((out / "logs" / "output_adapter.json").read_text(encoding="utf-8"))
print(f"  adapter: {adapter['route']} | {adapter['raw_representation_type']} | "
      f"conv={adapter['conversion_to_positive_depth']} | metric={adapter['metric_scale_exists']}")
diagnostics = out / "logs" / "adapter_diagnostics.json"
if diagnostics.is_file():
    print(f"  diagnostics: {json.loads(diagnostics.read_text(encoding='utf-8'))['totals']}")
PROBE
done

echo; echo "=== V1 汇总（image-wise 均值）==="
python - "$OUT_ROOT" <<'TABLE'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
keys = ("aligned_abs_rel", "aligned_rmse_m", "aligned_delta1", "pearson_r", "spearman_r")
print(f"{'route':<14}" + "".join(f"{k:>18}" for k in keys))
for run in sorted(p for p in root.iterdir() if p.is_dir()):
    summary = run / "metrics" / "summary.json"
    if not summary.is_file(): continue
    stats = json.loads(summary.read_text(encoding="utf-8"))["image_wise"]["statistics"]
    row = "".join(f"{stats[k]['mean']:>18.4f}" if k in stats else f"{'-':>18}" for k in keys)
    print(f"{run.name:<14}{row}")
print()
print("⚠️ 门禁：anythermal 的 aligned_abs_rel 应接近它在官方 ssi 下的 0.0821。")
print("   对不上先别信任何一行 —— 说明 adapter 声明或掩码口径有问题。")
print("⚠️ b/c1/c2 在这把尺子下不会等于 ssi_disparity 那组数（0.0884 等），这是换尺子，不是回归。")
print("   Pearson/Spearman 对任何单调变换不变，是唯一不受对齐方式影响的两列。")
TABLE

exit $failed
