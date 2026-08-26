#!/bin/bash
#SBATCH --job-name=metric_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 度量评估的四行，一个作业跑完，机械选点在中间。
#
#   1. 原 checkpoint + raw_inverse        —— 单位自检，不是结果
#   2. 原 checkpoint + train 全局仿射     —— 「不改网络能拿到多少」的基线
#   3. 适配后 checkpoint + 全局归一化     —— 这一阶段要证明的东西
#   4. 原 checkpoint + 逐帧 SSI           —— 旧的仿射不变分数，只作参照
#
# 前三行都不碰 test GT 做任何拟合。第 4 行碰，而且它**不是度量结果**，
# 不能和前三行并排排名 —— 它答的是另一个问题。
#
# ⚠️ 选点只看 val。适配臂的最佳步数由 val 的 metric AbsRel 决定（不是 SSI 的），
#    因为这一阶段优化的就是那个量。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   BASE_CKPT       必填。原 checkpoint（step20000_weights.pt）
#   ADAPT_RUN       适配训练的输出目录名（$IRIS_RUNS/iris_ms2/<ADAPT_RUN>）。
#                   不设＝只跑第 1/2/4 行（适配还没训完时先拿基线）
#   METRIC_NORM     必填。metric_prep 产出的 JSON
#   GLOBAL_AFFINE   必填。metric_prep 产出的 JSON
#   VAL_MANIFEST    必填。官方 3 序列 val 清单
#   TEST_MANIFEST   必填。官方 test 清单（day / rain / night 各投一次）
#   STEPS           适配臂要评的步数，默认 "500 1000 2000 3000 4000"
#   PROMPT          默认 correct（与起点那条臂训练时一致）
#   VAL_STRIDE      默认 4
#
# ── 用法 ────────────────────────────────────────────────────────────
#   BASE_CKPT=... ADAPT_RUN=metric_adapt_lm1.0_lr1e-6 \
#   METRIC_NORM=$IRIS_RUNS/metric_adapt/metric_norm_train.json \
#   GLOBAL_AFFINE=$IRIS_RUNS/metric_adapt/global_affine_train.json \
#   VAL_MANIFEST=$IRIS_MANIFEST_DIR/ms2_val_thermalcap_v3_1_20260819.jsonl \
#   TEST_MANIFEST=$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl \
#     sbatch -J me_day --export=ALL,BASE_CKPT,ADAPT_RUN,METRIC_NORM,GLOBAL_AFFINE,VAL_MANIFEST,TEST_MANIFEST \
#     ~/Iris/slurm/metric_eval.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${BASE_CKPT:?必须指定 BASE_CKPT}"
: "${METRIC_NORM:?必须指定 METRIC_NORM}"
: "${GLOBAL_AFFINE:?必须指定 GLOBAL_AFFINE}"
: "${VAL_MANIFEST:?必须指定 VAL_MANIFEST}"
: "${TEST_MANIFEST:?必须指定 TEST_MANIFEST}"

STEPS="${STEPS:-500 1000 2000 3000 4000}"
PROMPT="${PROMPT:-correct}"
VAL_STRIDE="${VAL_STRIDE:-4}"
ROUTE="${ROUTE:-b_thermal_unet}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$VAL_MANIFEST}"   # 只用于 route_suite 的哈希/校验

for f in "$BASE_CKPT" "$METRIC_NORM" "$GLOBAL_AFFINE" "$VAL_MANIFEST" "$TEST_MANIFEST"; do
  [[ -f "$f" ]] || { echo "!! 找不到 $f"; exit 1; }
done

fingerprint() { sha256sum "$1" | cut -c1-8; }
VAL_FP="$(fingerprint "$VAL_MANIFEST")"
TEST_FP="$(fingerprint "$TEST_MANIFEST")"

cd "$IRIS_REPO"
git log --oneline -1

# tag ckpt manifest stride align source extra...
run_eval() {
  local tag="$1" ckpt="$2" manifest="$3" stride="$4" align="$5" source="$6"; shift 6
  local out="$IRIS_RUNS/eval/$tag"
  mkdir -p "$out"
  local args=(--route "$ROUTE" --output-dir "$out" --ms2-root "$IRIS_MS2_ROOT"
              --train-manifest "$TRAIN_MANIFEST" --eval-checkpoint "$ckpt"
              --val-manifest "$manifest" --val-stride "$stride"
              --val-caption-mode "$PROMPT" --num-inference-steps 1
              --condition-latent mode --local-files-only --align-mode "$align")
  [[ -n "$source" ]] && args+=(--metric-source "$source")
  args+=("$@")
  echo "[eval] $tag  ($align / ${source:-per-frame fit})"
  python tools/train_route_suite.py "${args[@]}"
}

NORM_ARGS=(--metric-norm "$METRIC_NORM")
AFFINE_ARGS=(--global-affine "$GLOBAL_AFFINE")

# --- 行 1/2/4：原 checkpoint，test 上直接评 -------------------------------
BASE_PREFIX="base_test_${TEST_FP}"
run_eval "${BASE_PREFIX}_raw"    "$BASE_CKPT" "$TEST_MANIFEST" 1 none raw_inverse
run_eval "${BASE_PREFIX}_affine" "$BASE_CKPT" "$TEST_MANIFEST" 1 none global_affine "${AFFINE_ARGS[@]}"
run_eval "${BASE_PREFIX}_ssi"    "$BASE_CKPT" "$TEST_MANIFEST" 1 ssi_disparity ""

# --- 行 3：适配臂，先在 val 上选点，再上 test ------------------------------
if [[ -n "${ADAPT_RUN:-}" ]]; then
  RUN_DIR="$IRIS_RUNS/iris_ms2/$ADAPT_RUN"
  CONV_DIR="$RUN_DIR/converted"
  [[ -d "$RUN_DIR" ]] || { echo "!! 找不到 $RUN_DIR"; exit 1; }
  mkdir -p "$CONV_DIR"

  for s in $STEPS; do
    target="$CONV_DIR/step${s}_weights.pt"
    [[ -f "$target" ]] && { echo "[skip] step$s 已转换"; continue; }
    src="$RUN_DIR/checkpoint-$s"
    [[ -d "$src" ]] || { echo "!! 找不到 $src"; exit 1; }
    python tools/convert_iris_ms2_checkpoint.py --source "$src" --output "$target" \
      --caption-mode "$PROMPT" --step "$s"
  done

  for s in $STEPS; do
    run_eval "adapt_${ADAPT_RUN}_val_${VAL_FP}_s${s}" "$CONV_DIR/step${s}_weights.pt" \
      "$VAL_MANIFEST" "$VAL_STRIDE" none global_norm "${NORM_ARGS[@]}"
  done

  # 选点：val 的 **metric** AbsRel 最小。用 SSI 的曲线选一个度量臂的冠军，
  # 挑的是另一门考试的第一名。
  BEST=$(python - "$IRIS_RUNS/eval" "adapt_${ADAPT_RUN}_val_${VAL_FP}" $STEPS <<'PY'
import json, sys
root, prefix, steps = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
scored = {}
for s in steps:
    p = f'{root}/{prefix}_s{s}/eval_eval_metric_no_test_alignment_global_norm.json'
    try:
        scored[s] = json.load(open(p))['abs_rel']
    except Exception:
        pass
if not scored:
    raise SystemExit("no val results")
best = min(scored, key=scored.get)
for s in steps:
    mark = ' <<<' if s == best else ''
    print(f'    step {s:>6}  metric abs_rel={scored.get(s, float("nan")):.5f}{mark}', file=sys.stderr)
if len(scored) > 1 and best in (min(scored), max(scored)):
    print(f'    ⚠️ 最佳落在采样边界 step {best}', file=sys.stderr)
print(best)
PY
)
  echo "=== val 曲线与选点（metric AbsRel）==="
  echo ">>> 选中 step $BEST"

  run_eval "adapt_${ADAPT_RUN}_test_${TEST_FP}_s${BEST}" "$CONV_DIR/step${BEST}_weights.pt" \
    "$TEST_MANIFEST" 1 none global_norm "${NORM_ARGS[@]}"
  # 同一权重的仿射不变分，只为看几何有没有被度量适配弄坏
  run_eval "adapt_${ADAPT_RUN}_test_${TEST_FP}_s${BEST}" "$CONV_DIR/step${BEST}_weights.pt" \
    "$TEST_MANIFEST" 1 ssi_disparity ""
fi

# --- 汇总 -----------------------------------------------------------------
echo
echo "=== test $(basename "$TEST_MANIFEST") / $(wc -l < "$TEST_MANIFEST") 帧 ==="
python - "$IRIS_RUNS/eval" "$BASE_PREFIX" "${ADAPT_RUN:-}" "$TEST_FP" "${BEST:-}" <<'PY'
import json, os, sys
root, base, adapt, fp, best = sys.argv[1:6]
rows = [
    ("original checkpoint", "raw output sanity check", "no",
     f"{root}/{base}_raw/eval_eval_metric_no_test_alignment_raw_inverse.json"),
    ("original checkpoint", "train-only global affine", "no",
     f"{root}/{base}_affine/eval_eval_metric_no_test_alignment_global_affine.json"),
]
if adapt and best:
    rows.append(("GT-adapted checkpoint", "learned metric inverse depth", "no",
                 f"{root}/adapt_{adapt}_test_{fp}_s{best}/eval_eval_metric_no_test_alignment_global_norm.json"))
rows.append(("original checkpoint", "per-frame SSI disparity", "YES (reference only)",
             f"{root}/{base}_ssi/eval_eval.json"))
head = f"| {'model':<22} | {'metric mechanism':<28} | {'test-GT fit':<20} | {'AbsRel':>7} | {'SqRel':>7} | {'RMSE':>7} | {'RMSElog':>8} | {'delta1':>7} |"
print(head); print("|" + "|".join("-" * len(c) for c in head.split("|")[1:-1]) + "|")
for model, mech, fit, path in rows:
    if not os.path.exists(path):
        print(f"| {model:<22} | {mech:<28} | {fit:<20} | {'(missing)':>7} |"); continue
    m = json.load(open(path))
    print(f"| {model:<22} | {mech:<28} | {fit:<20} | {m['abs_rel']:7.4f} | "
          f"{m['sq_rel']:7.4f} | {m['rmse']:7.3f} | {m['rmse_log']:8.4f} | {m['a1']:7.4f} |")
print()
print("⛔ 最后一行不是度量深度结果，不要和前面并排排名。")
PY
