#!/bin/bash
#SBATCH --job-name=base_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 用他们自己的权重、他们自己的预处理、他们自己的帧集，把 MS2 的四个已发表基线
# 在我们机器上跑一遍，一次前向出三个对齐口径（none / median / ssi）。
#
# 为什么要自己跑：已发表的数字没有一个是「无 test-GT 对齐」的 ——
# DORN/BTS/AdaBins/NeWCRF 拿的是逐图**中位数缩放**（1 参数，深度空间），
# MiDaS 系拿的是逐图 **scale+shift**（2 参数，深度空间），
# 我们拿的是逐图 scale+shift（2 参数，**视差**空间）。三张不同的卷子。
# 自己跑一遍，才能把所有行放到同一个帧集、同一台评估器上。
#
# ⛔ 门禁：median 那一列必须复现 MS2 论文 Table IV(c)：
#      NeWCRF  day 0.071 / night 0.074 / rain 0.095   （AbsRel）
#      BTS     day 0.075 / night 0.079 / rain 0.102
#      AdaBins day 0.079 / night 0.079 / rain 0.104
#      DORN    day 0.101 / night 0.099 / rain 0.127
#    对不上就说明预处理、帧集或加载有出入，**这时候别看其他两列**。
#
# ── 前置（在登录节点做一次，别投作业）─────────────────────────────
#   1. clone 网络定义：
#        git clone https://github.com/UkcheolShin/SupDepth4Thermal $SCRATCH/baselines/SupDepth4Thermal
#   2. 预热权重缓存 —— DORN/BTS 会从 download.pytorch.org 取 ImageNet 初始化，
#      AdaBins 走 torch.hub 去 GitHub 取 EfficientNet。**GPU 节点没有外网**，
#      不预热的话作业会在排队几小时之后倒在下载上：
#        python tools/run_ms2_supdepth_baselines.py --build-only \
#          --bmsd-root $SCRATCH/baselines/SupDepth4Thermal \
#          --ckpt-dir  $SCRATCH/baselines/supdepth4thermal \
#          --ms2-root  $IRIS_MS2_ROOT --test-env test_day \
#          --output-dir /tmp/build_check
#      （这一步同时就是 strict 加载检查：键名对不上会当场报出来。）
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   TEST_ENV     test_day | test_night | test_rain。默认 test_day
#   BMSD_ROOT    默认 $SCRATCH/baselines/SupDepth4Thermal
#   CKPT_DIR     默认 $SCRATCH/baselines/supdepth4thermal
#   MODELS       默认 dorn,bts,adabins,newcrf
#   SAMPLE_STEP  默认 10（官方值）。设 1 = 全帧，贵 10 倍而差值 ≤0.00025
#   EXPECT_FRAMES 断言复刻出的帧数正好是这个值。不设＝只和论文的约数比，容差 2%
#   LIMIT        冒烟用，只跑前 N 帧（会跳过帧数断言）
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟：LIMIT=20 TEST_ENV=test_day sbatch -J bb_smoke --time=00:30:00 \
#           --export=ALL,LIMIT,TEST_ENV ~/Iris/slurm/baseline_bench.sbatch
#   三个条件并行投，各一个作业：
#     for e in test_day test_night test_rain; do
#       TEST_ENV=$e sbatch -J bb_${e#test_} --export=ALL,TEST_ENV ~/Iris/slurm/baseline_bench.sbatch
#     done

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

TEST_ENV="${TEST_ENV:-test_day}"
BMSD_ROOT="${BMSD_ROOT:-$SCRATCH/baselines/SupDepth4Thermal}"
CKPT_DIR="${CKPT_DIR:-$SCRATCH/baselines/supdepth4thermal}"
MODELS="${MODELS:-dorn,bts,adabins,newcrf}"
SAMPLE_STEP="${SAMPLE_STEP:-10}"
OUT_DIR="${OUT_DIR:-$IRIS_RUNS/baseline_bench/$TEST_ENV}"

[[ -d "$BMSD_ROOT/models/network" ]] || { echo "!! 找不到网络定义 $BMSD_ROOT/models/network —— 先 clone SupDepth4Thermal"; exit 1; }
[[ -d "$CKPT_DIR" ]]                 || { echo "!! 找不到权重目录 $CKPT_DIR"; exit 1; }
for m in ${MODELS//,/ }; do
  case "$m" in
    dorn) f=DORN;; bts) f=BTS;; adabins) f=AdaBins;; newcrf) f=NeWCRF;;
    *) echo "!! 未知模型 $m"; exit 1;;
  esac
  [[ -f "$CKPT_DIR/MS2_MD_${f}_THR_ckpt.ckpt" ]] || { echo "!! 缺 MS2_MD_${f}_THR_ckpt.ckpt"; exit 1; }
done

mkdir -p "$OUT_DIR"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 配置 ==="
echo "test env    = $TEST_ENV"
echo "models      = $MODELS"
echo "sample step = $SAMPLE_STEP  (官方 10)"
echo "网络定义    = $BMSD_ROOT"
echo "权重        = $CKPT_DIR"
echo "输出        = $OUT_DIR"
echo "============"

ARGS=(--bmsd-root "$BMSD_ROOT" --ckpt-dir "$CKPT_DIR" --ms2-root "$IRIS_MS2_ROOT"
      --test-env "$TEST_ENV" --output-dir "$OUT_DIR" --models "$MODELS"
      --sample-step "$SAMPLE_STEP")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "$LIMIT")
[[ -n "${EXPECT_FRAMES:-}" ]] && ARGS+=(--expect-frames "$EXPECT_FRAMES")

python tools/run_ms2_supdepth_baselines.py "${ARGS[@]}"
