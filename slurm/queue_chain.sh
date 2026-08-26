#!/bin/bash
# 在**登录节点**跑一次，把接下来的作业按依赖排好，然后就可以走人。
#
#   bash ~/Iris/slurm/queue_chain.sh
#
# 每一环都用 `--dependency=afterok`：前一环失败，后面的自动不跑，而不是拿着
# 半成品往下算。所有清单在提交任何作业之前先查一遍存在性 —— 排完队几小时
# 之后才发现路径错，是这个项目今天已经踩过七次的坑。
#
# 三条链，互不依赖，同时排队：
#
#   A  已发表基线，全帧口径          3 个作业，各约 1 小时，无依赖
#   B  seed 43 的 RGB caption 评估    3 个作业，串行，等训练结束
#   C  天空规则（改目标）训练 + 评估  掩码 → 训练 → 3 个评估，串行
#   D  度量微调                      4 次训练，无依赖，可并行
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   SKYMASK_JOB    已经在跑的掩码作业号。不设＝本脚本提交一个新的
#   RGBCAP_JOB     已经在跑的 f8rgb_s43 作业号。不设＝自动从 squeue 找
#   SKIP           跳过某些链，逗号分隔，如 SKIP=A,B
#   DRY_RUN=1      只打印要提交什么，不真提交

set -uo pipefail
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh" >/dev/null

REPO="${IRIS_REPO:-$HOME/Iris}"
SKIP="${SKIP:-}"
DRY="${DRY_RUN:-0}"
# dry 模式的假作业号必须跨子 shell 递增：submit 是在 $( ) 里调用的，子 shell
# 的变量自增传不回来，于是每一环都拿到同一个号，依赖链看起来是错的而实际不是。
# dry 跑存在的意义就是验证这条链，所以计数器落到文件上。
DRY_COUNTER="$(mktemp)"; echo 0 > "$DRY_COUNTER"
trap 'rm -f "$DRY_COUNTER"' EXIT

# 这几个名字在本仓库的不同作业脚本里含义不同（STEPS 尤其：pipeline 里是
# checkpoint 步数、eval 里是去噪步数、metric_adapt 里是训练步数）。登录 shell
# 里只要 export 过一次就会被 --export=ALL 带进来。本脚本自己会显式设它需要的，
# 所以先清干净，别把上一次提交的值带进这一批。
unset STEPS TAG CKPT ROUTE CAPTION_MODE SAVE_RAW VAL_STRIDE       RUN SEL_PROMPT TEST_PROMPTS TEST_MANIFEST VAL_MANIFEST TRAIN_MANIFEST       SEED RUN_TAG PSEUDO_DIR SKY_MASKS MAX_SAMPLES SAVE_MASKS MANIFEST       METRIC_NORM INIT_CKPT SMOKE LIMIT TEST_ENV SAMPLE_STEP OUT_DIR 2>/dev/null || true
skipped() { [[ ",$SKIP," == *",$1,"* ]]; }

M="$IRIS_MANIFEST_DIR"
TRAIN_THERMAL="$M/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
TRAIN_RGB="$M/ms2_train_official8_rgbcap_20260823.jsonl"
VAL_THERMAL="$M/ms2_val_official3_thermalcap_20260821.jsonl"
VAL_RGB="$M/ms2_val_official3_rgbcap_20260823.jsonl"
TEST_T_DAY="$M/ms2_test_day3_common_thermalcap_20260821.jsonl"
TEST_T_NIGHT="$M/ms2_test_night3_thermalcap_20260821.jsonl"
TEST_T_RAIN="$M/ms2_test_rainy3_thermalcap_20260821.jsonl"
TEST_R_DAY="$M/ms2_test_day3_common_rgbcap_20260823.jsonl"
TEST_R_NIGHT="$M/ms2_test_night3_rgbcap_20260823.jsonl"
TEST_R_RAIN="$M/ms2_test_rainy3_rgbcap_20260823.jsonl"
PSEUDO="$IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth"
SKYMASKS="$IRIS_RUNS/sky_masks/skymask_official8/masks"
METRIC_NORM_JSON="$IRIS_RUNS/metric_adapt/metric_norm_train.json"
CONV="$IRIS_RUNS/iris_ms2"
CKPT_NOCAP="$CONV/iris_ms2_full8_nocap/converted/step12000_weights.pt"
CKPT_THERMAL="$CONV/iris_ms2_full8_thermalcap/converted/step20000_weights.pt"
CKPT_RGB="$CONV/iris_ms2_full8_rgbcap/converted/step20000_weights.pt"
TRAIN_NOCAP="$M/ms2_train_official8_nocap_20260821.jsonl"

# ── 提交前的存在性检查 ──────────────────────────────────────────────
missing=0
check() { [[ -e "$1" ]] || { echo "!! 找不到 $1"; missing=1; }; }
for f in "$TRAIN_THERMAL" "$TRAIN_RGB" "$VAL_THERMAL" "$VAL_RGB" \
         "$TEST_T_DAY" "$TEST_T_NIGHT" "$TEST_T_RAIN" \
         "$TEST_R_DAY" "$TEST_R_NIGHT" "$TEST_R_RAIN"; do check "$f"; done
check "$PSEUDO"
if ! skipped D; then
  for f in "$METRIC_NORM_JSON" "$CKPT_NOCAP" "$CKPT_THERMAL" "$CKPT_RGB" "$TRAIN_NOCAP"; do check "$f"; done
fi
(( missing == 0 )) || { echo "!! 有路径不存在，一个作业都没提交。"; exit 1; }
echo "[check] 10 份清单 + 伪 GT 目录都在"

submit() {  # submit <描述> <sbatch 参数...>   -> 打印并回显 job id
  local desc="$1"; shift
  if [[ "$DRY" == "1" ]]; then
    local n=$(( $(cat "$DRY_COUNTER") + 1 )); echo "$n" > "$DRY_COUNTER"
    echo "[dry] $desc" >&2
    echo "      sbatch $*" >&2
    echo "      环境: STEPS=${STEPS:-<未设>}" >&2
    echo "9$(printf '%05d' "$n")"
    return
  fi
  local out
  out=$(sbatch "$@" 2>&1) || { echo "!! 提交失败：$desc"; echo "   $out"; exit 1; }
  local id="${out##* }"
  printf '  %-52s -> %s\n' "$desc" "$id" >&2
  echo "$id"
}

echo
echo "=========== A：已发表基线，全帧口径 ==========="
# 1/10 官方子集已经跑过，和全帧差 ≤0.00025。跑全帧是为了让基线和我们自己的
# 数字站在同一个帧集上，表里就不必再解释两个口径。每条件约 1 小时。
if skipped A; then echo "  (跳过)"; else
  for e in test_day test_night test_rain; do
    submit "基线全帧 $e" -J "bbfull_${e#test_}" --time=03:00:00 \
      --export=ALL,TEST_ENV=$e,SAMPLE_STEP=1,OUT_DIR=$IRIS_RUNS/baseline_bench/full_$e \
      "$REPO/slurm/baseline_bench.sbatch" >/dev/null
  done
fi

echo
echo "=========== B：seed 43 的 RGB caption 评估 ==========="
if skipped B; then echo "  (跳过)"; else
  RGBCAP_JOB="${RGBCAP_JOB:-$(squeue -u "$USER" -h -n f8rgb_s43 -o %i | head -1)}"
  if [[ -z "$RGBCAP_JOB" ]]; then
    echo "  !! 没找到在跑的 f8rgb_s43。若训练已结束，用 RGBCAP_JOB=0 表示无需等待。"
    exit 1
  fi
  export STEPS="2000 4000 8000 12000 20000"
  dep="$RGBCAP_JOB"
  # 串行而非并行：pipeline 会跳过已有结果，所以第一个作业做完 val 选点之后，
  # 后两个直接复用，不重复算那 5 个 checkpoint 的 val。
  for spec in "day $TEST_R_DAY" "night $TEST_R_NIGHT" "rain $TEST_R_RAIN"; do
    set -- $spec
    depflag=(); [[ "$dep" != "0" ]] && depflag=(--dependency=afterok:$dep)
    dep=$(submit "rgbcap_s43 评估 $1" -J "r43_$1" --time=03:00:00 "${depflag[@]}" \
      --export=ALL,STEPS,RUN=iris_ms2_full8_rgbcap_s43,SEL_PROMPT=correct,TEST_PROMPTS=correct,TRAIN_MANIFEST=$TRAIN_RGB,VAL_MANIFEST=$VAL_RGB,TEST_MANIFEST=$2 \
      "$REPO/slurm/iris_ms2_pipeline.sbatch")
  done
fi

echo
echo "=========== C：天空规则（改目标）==========="
if skipped C; then echo "  (跳过)"; else
  SKYMASK_JOB="${SKYMASK_JOB:-$(squeue -u "$USER" -h -n skymask_full -o %i | head -1)}"
  if [[ -z "$SKYMASK_JOB" ]]; then
    if [[ -d "$SKYMASKS" ]]; then
      echo "  掩码已存在（$(ls "$SKYMASKS" | wc -l) 张），不再重出"
      SKYMASK_JOB=0
    else
      SKYMASK_JOB=$(submit "天空掩码全量" -J skymask_full --time=04:00:00 \
        --export=ALL,MANIFEST=$TRAIN_THERMAL,TAG=skymask_official8,MAX_SAMPLES=0,SAVE_MASKS=1 \
        "$REPO/slurm/sky_masks.sbatch")
    fi
  else
    echo "  掩码作业已在队列：$SKYMASK_JOB"
  fi

  depflag=(); [[ "$SKYMASK_JOB" != "0" ]] && depflag=(--dependency=afterok:$SKYMASK_JOB)
  TRAIN_JOB=$(submit "天空规则训练（同种子42，只差天空）" -J f8t_sky "${depflag[@]}" \
    --export=ALL,SEED=42,RUN_TAG=iris_ms2_full8_thermalcap_sky,TRAIN_MANIFEST=$TRAIN_THERMAL,PSEUDO_DIR=$PSEUDO,SKY_MASKS=$SKYMASKS \
    "$REPO/slurm/iris_ms2.sbatch")

  export STEPS="2000 4000 8000 12000 16000 20000"
  dep="$TRAIN_JOB"
  for spec in "day $TEST_T_DAY" "night $TEST_T_NIGHT" "rain $TEST_T_RAIN"; do
    set -- $spec
    dep=$(submit "天空臂评估 $1" -J "sky_$1" --time=03:00:00 --dependency=afterok:$dep \
      --export=ALL,STEPS,RUN=iris_ms2_full8_thermalcap_sky,SEL_PROMPT=correct,TEST_PROMPTS=correct,TRAIN_MANIFEST=$TRAIN_THERMAL,VAL_MANIFEST=$VAL_THERMAL,TEST_MANIFEST=$2 \
      "$REPO/slurm/iris_ms2_pipeline.sbatch")
  done
fi

echo
echo "=========== D：度量微调（三条臂 + 一次 λ 灵敏度）==========="
# 让三条臂都输出米制逆深度，整张表才能建在同一代模型上。
# ⚠️ 三条臂共用同一套 q_lo/q_hi —— 常数是数据集的属性不是模型的属性，各拟合一套
#    会让三条臂的输出落在不同单位里，正好毁掉这次适配要建立的可比性。
# ⚠️ λ 只在 thermalcap 上用 val 选一次然后三条臂共用；每臂各选各的会让目标函数不同。
if skipped D; then echo "  (跳过)"; else
  unset STEPS   # metric_adapt.sbatch 里 STEPS 是训练步数，别把 B/C 的值带进来
  # 臂名 起点checkpoint 训练清单 caption开关 lambda
  while read -r tag ckpt manifest nocap lam; do
    [[ -z "$tag" ]] && continue
    extra=""; [[ "$nocap" == "1" ]] && extra=",NO_CAPTIONS=1"
    submit "度量微调 $tag (lambda=$lam)" -J "ma_$tag" --time=12:00:00       --export=ALL,METRIC_NORM=$METRIC_NORM_JSON,INIT_CKPT=$ckpt,TRAIN_MANIFEST=$manifest,LAMBDA_METRIC=$lam,LR=1e-6,STEPS=4000,RUN_TAG=metric_${tag}$extra       "$REPO/slurm/metric_adapt.sbatch" >/dev/null
  done <<EOF
thermal_lm1   $CKPT_THERMAL  $TRAIN_THERMAL  0  1
thermal_lm20  $CKPT_THERMAL  $TRAIN_THERMAL  0  20
nocap_lm1     $CKPT_NOCAP    $TRAIN_NOCAP    1  1
rgbcap_lm1    $CKPT_RGB      $TRAIN_RGB      0  1
EOF
fi

echo
echo "=========== 队列 ==========="
squeue -u "$USER" -o "%.10i %.16j %.2t %.11L %.20E" 2>/dev/null
echo
echo "看结果：bash $REPO/slurm/status.sh"
echo "afterok 的含义：前一环非零退出，后面全部留在队列里不跑（状态 DependencyNeverSatisfied），"
echo "不会拿着坏结果继续算。要清就 scancel。"
