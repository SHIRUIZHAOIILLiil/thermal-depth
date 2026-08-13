#!/bin/bash
#SBATCH --job-name=ims_pipe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 训练之后的全部评估，一个作业跑完，中途不需要人。
#
# 我们之前是「投转换 → 等 → 投 5 个 val → 等 → 读曲线 → 人选最佳 → 投 test → 等 →
# 跑配对」，六轮往返，每轮都要人在场，而且每一步各排一次队。选择这一步其实是
# 机械的（val abs_rel 取最小），没有理由由人来做。
#
# 依赖链（--dependency）解决不了这个：那仍然是每一环各排一次队。一个作业只排一次，
# 而 GPU 被占的那 40 分钟里做完的事，分开投要等好几个窗口。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   RUN          必填。训练输出目录名，如 iris_ms2_ckpt / iris_ms2_ckpt_nocap
#   STEPS        要评的 checkpoint 步数，空格分隔。默认 "1000 2000 3000 5000 8000"
#   SEL_PROMPT   val 选点用的 prompt：empty | correct
#                ⚠️ 必须与该臂**训练时**的输入模式一致 —— 用它没训过的模式挑冠军，
#                挑出来的不是这条臂最好的权重。带 caption 训练的用 correct，不带的用 empty
#   TEST_PROMPTS test 上要评的臂，空格分隔。默认 "empty"；带 caption 的臂给 "empty correct"
#   ROUTE        默认 b_thermal_unet
#   VAL_STRIDE   val 抽帧，默认 4（1453 帧，与历史训练曲线同口径）
#
# ── 用法 ────────────────────────────────────────────────────────────
#   不带 caption 训练的臂：
#     RUN=iris_ms2_ckpt_nocap SEL_PROMPT=empty TEST_PROMPTS="empty" \
#       sbatch -J pipe_nocap --export=ALL,RUN,SEL_PROMPT,TEST_PROMPTS ~/Iris/slurm/iris_ms2_pipeline.sbatch
#
#   带 caption 训练的臂（test 上评两臂，并自动做逐帧配对）：
#     RUN=iris_ms2_ckpt SEL_PROMPT=correct TEST_PROMPTS="empty correct" \
#       sbatch -J pipe_cap --export=ALL,RUN,SEL_PROMPT,TEST_PROMPTS ~/Iris/slurm/iris_ms2_pipeline.sbatch
#
# 已经做过的步骤会跳过，所以被墙钟砍了原样重投即可接着跑。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${RUN:?必须指定 RUN（训练输出目录名）}"
STEPS="${STEPS:-1000 2000 3000 5000 8000}"
SEL_PROMPT="${SEL_PROMPT:?必须指定 SEL_PROMPT（empty 或 correct，须与该臂训练时一致）}"
TEST_PROMPTS="${TEST_PROMPTS:-empty}"
ROUTE="${ROUTE:-b_thermal_unet}"
VAL_STRIDE="${VAL_STRIDE:-4}"

RUN_DIR="$IRIS_RUNS/iris_ms2/$RUN"
CONV_DIR="$RUN_DIR/converted"
TEST_MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"

[[ -d "$RUN_DIR" ]]        || { echo "!! 找不到训练输出目录 $RUN_DIR"; exit 1; }
[[ -f "$TEST_MANIFEST" ]]  || { echo "!! 找不到 test manifest $TEST_MANIFEST"; exit 1; }
case "$SEL_PROMPT" in empty|correct) ;; *) echo "!! SEL_PROMPT 只能是 empty 或 correct"; exit 1 ;; esac

mkdir -p "$CONV_DIR"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 配置 ==="
echo "run          = $RUN"
echo "steps        = $STEPS"
echo "选点 prompt   = $SEL_PROMPT   (须与训练时一致)"
echo "test 臂       = $TEST_PROMPTS"
echo "============"

run_eval() {  # tag ckpt prompt stride [manifest]
  local tag="$1" ckpt="$2" prompt="$3" stride="$4" manifest="${5:-}"
  local out="$IRIS_RUNS/eval/$tag"
  if [[ -f "$out/eval_eval.json" ]]; then
    echo "[skip] $tag 已有结果"
    return 0
  fi
  mkdir -p "$out"
  local args=(--route "$ROUTE" --output-dir "$out" --ms2-root "$IRIS_MS2_ROOT"
              --train-manifest "$TRAIN_MANIFEST" --eval-checkpoint "$ckpt"
              --val-stride "$stride" --val-caption-mode "$prompt"
              --num-inference-steps 1 --condition-latent mode --local-files-only)
  [[ -n "$manifest" ]] && args+=(--val-manifest "$manifest")
  echo "[eval] $tag"
  python tools/train_route_suite.py "${args[@]}"
}

# --- 1. 转换 ---------------------------------------------------------------
for s in $STEPS; do
  target="$CONV_DIR/step${s}_weights.pt"
  if [[ -f "$target" ]]; then echo "[skip] step$s 已转换"; continue; fi
  src="$RUN_DIR/checkpoint-$s"
  [[ -d "$src" ]] || { echo "!! 找不到 $src —— 训练没跑到这一步？"; exit 1; }
  python tools/convert_iris_ms2_checkpoint.py --source "$src" --output "$target" \
    --caption-mode "$SEL_PROMPT" --step "$s"
done

# --- 2. val 逐点评估 -------------------------------------------------------
for s in $STEPS; do
  run_eval "${RUN}_val_s${s}" "$CONV_DIR/step${s}_weights.pt" "$SEL_PROMPT" "$VAL_STRIDE"
done

# --- 3. 按 val 选最佳（机械的一步，不需要人） -------------------------------
BEST=$(python - "$IRIS_RUNS/eval" "$RUN" $STEPS <<'PY'
import json, sys
root, run, steps = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
scored = {}
for s in steps:
    try:
        scored[s] = json.load(open(f'{root}/{run}_val_s{s}/eval_eval.json'))['abs_rel']
    except Exception:
        pass
if not scored:
    raise SystemExit("no val results")
best = min(scored, key=scored.get)
for s in steps:
    mark = ' <<<' if s == best else ''
    print(f'    step {s:>6}  abs_rel={scored.get(s, float("nan")):.5f}{mark}', file=sys.stderr)
if best in (min(scored), max(scored)) and len(scored) > 1:
    print(f'    ⚠️ 最佳落在采样边界 step {best}，最小值可能没被夹住', file=sys.stderr)
print(best)
PY
)
echo "=== val 曲线与选点 ==="
echo ">>> 选中 step $BEST"

# --- 4. test 评估 ----------------------------------------------------------
for p in $TEST_PROMPTS; do
  run_eval "${RUN}_test_s${BEST}_${p}" "$CONV_DIR/step${BEST}_weights.pt" "$p" 1 "$TEST_MANIFEST"
done

# --- 5. 两臂逐帧配对（给了两个 prompt 才有意义） -----------------------------
set -- $TEST_PROMPTS
if [[ $# -ge 2 ]]; then
  echo "=== 逐帧配对：$2 − $1 ==="
  python tools/compare_route_evals.py \
    "$IRIS_RUNS/eval/${RUN}_test_s${BEST}_${1}/eval_eval_per_sample.csv" \
    "$IRIS_RUNS/eval/${RUN}_test_s${BEST}_${2}/eval_eval_per_sample.csv"
fi

# --- 6. 汇总 ---------------------------------------------------------------
echo "=== 最终结果（test 16-08-46 / 2543 帧，官方协议，condition_latent=mode）==="
python - "$IRIS_RUNS/eval" "$RUN" "$BEST" $TEST_PROMPTS <<'PY'
import json, sys
root, run, best, prompts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
for p in prompts:
    m = json.load(open(f'{root}/{run}_test_s{best}_{p}/eval_eval.json'))
    print(f"  {run} / {p:<8} abs_rel={m['abs_rel']:.5f}  rmse={m['rmse']:.3f}  a1={m['a1']:.4f}")
print("\n  对照（同卷，空 prompt）：b 线 0.08843 / 3.956 / 0.9071"
      "；b+天空损失 0.08487 / 3.816 / 0.9142")
PY
