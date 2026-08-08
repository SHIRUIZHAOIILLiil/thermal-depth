#!/bin/bash
#SBATCH --job-name=route
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 六线训练作业。所有开关走环境变量，配 --export=ALL,<VAR>,... 传入。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   ROUTE            必填，六选一
#   CAPTION=1        caption 臂：同时给 --caption-mode / --val-caption-mode correct
#                    （只给前者会用空 prompt 给 caption 模型选 best，等于用它没训过的
#                     输入模式挑冠军）
#   CAPTION_MODE     覆盖 CAPTION=1：empty | correct | permuted。
#                    permuted = 训练时每帧拿一张**随机另一帧**的 caption（均匀置换，
#                    不是旋转 —— 旋转被证污染，R^2 0.32 仍能预测本帧深度）。
#                    这是权重效应从来没有过的内容对照：注入效应已证与内容无关
#                    （shuffled−correct −0.00006、胜率 49.5%），但「训练时用 caption
#                    有益」究竟是内容还是正则化，没人测过。val 跟着 train 走。
#   RANK_W           caption 对比损失的 λ：L = w_gt·L_correct + λ·relu(m + L_c − L_w)。
#                    不设＝关闭（连第二次前向都不做，与历史逐位一致）。要配 CAPTION=1。
#                    ⚠️ λ 乘的是**未加权**的排序项，数据项带 --gt-loss-weight 5.0，
#                    所以 λ=0.5 才相当于数据项的十分之一。
#   RANK_REFS        正确 caption 要赢过谁：wrong | empty | both（默认 both）
#                    ⚠️ 现在「空」比「对」还好（0.0869 vs 0.0882），empty 那一条
#                    才是真正要关掉的差距。空分支默认 detach，不然模型会靠
#                    把空 prompt 做烂来满足间隔 —— 那正是我们最好的那一档。
#   RANK_M           margin，默认 0（"正确 caption 至少不能输"，无需标定量纲）
#   RANK_DETACH=1    错误分支只作参照、不回传 —— 防退化解的备用开关
#   SMOKE=1          5 个 update 即退出
#   OVERFIT=1        300 步 / 32 样本自检
#   EPOCHS           默认 20
#   SNAPSHOT_EPOCHS  默认 1,2,3,5,10,15（含 e3 —— 本地 caption 臂最好点落在 e3）
#   TRAIN_MANIFEST   默认 clip75_20260728（≤75 token，逐帧同序同路径）
#   VAL_MANIFEST     不设则用脚本自带默认
#   INIT_FROM        跨 route 搬权重（任务 1b 阶段二用 c1 的 best）
#   FREEZE_ADAPTER=1 冻结 adapter 只训 U-Net，必须配 INIT_FROM
#   LOCAL_FILES_ONLY 默认 1 —— 权重缺失立即报错，而不是并发去写同一个 HF 缓存目录
#   GRAD_WEIGHT      MiDaS 多尺度梯度匹配项的权重；不设＝关闭。数据项系数 5.0 时
#                    对应 MiDaS α=0.5 的平衡点是 2.5
#   EXCLUDE_TOP_ROWS 天空实验：最上 N 行不进**训练损失**（验证不变）。不设＝关闭
#   SKY_W            sky loss 权重（Jasmine 附录 C.3）。不设＝关闭。要配 SKY_MASKS
#   SKY_MASKS        天空掩码目录（tools/build_sky_masks.py --save-masks 出的）
#   SKY_MODE         l1（默认，与 Jasmine 同）| hinge（只罚判得过近的）
#   PSEUDO_DIR       calibrated_pseudo_depth 目录（build_anythermal_pseudo_gt.py 产出）
#   PSEUDO_W         补全区权重；真实 lidar 恒为 1.0。不设＝关闭，走原基线代码路径
#   PSEUDO_RANGE     drop（默认，超量程不给目标）| clip（钉到边界＝断言远处都在那里）
#   RUN_TAG          覆盖输出目录名
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟：
#     ROUTE=b_thermal_unet SMOKE=1 sbatch -J smoke_b --time=00:30:00 \
#       --export=ALL,ROUTE,SMOKE ~/Iris/slurm/route_suite.sbatch
#
#   全量（--time 按线报，见 docs/AIRE_HPC_MIGRATION_STATUS_20260729.md §12.5，
#   贵的线直接报 48h，别统一 24h）：
#     ROUTE=c1_vae_adapter sbatch -J route_c1 --time=1-00:00:00 \
#       --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch
#     ROUTE=a_rgb_unet     sbatch -J route_a  --time=2-00:00:00 \
#       --export=ALL,ROUTE ~/Iris/slurm/route_suite.sbatch
#
#   caption 臂：
#     ROUTE=b_thermal_unet CAPTION=1 sbatch -J route_b_cap --time=2-00:00:00 \
#       --export=ALL,ROUTE,CAPTION ~/Iris/slurm/route_suite.sbatch
#
#   caption 内容对照臂（用打乱的 caption 训练，填权重效应那个空格）：
#     ROUTE=b_thermal_unet CAPTION_MODE=permuted EPOCHS=5 SNAPSHOT_EPOCHS=1,3,5
#     RUN_TAG=b_caption_permuted sbatch -J bcap_perm --time=08:00:00
#       --export=ALL,ROUTE,CAPTION_MODE,EPOCHS,SNAPSHOT_EPOCHS,RUN_TAG ~/Iris/slurm/route_suite.sbatch
#     出数后要用 eval.sbatch 以 CAPTION_MODE=empty 在 test 上评，才与 0.0869 同口径
#
#   天空实验（b 线，最上 32 行不进损失，5 个 epoch 就够 —— b 的最佳点在 e5）：
#     ROUTE=b_thermal_unet EXCLUDE_TOP_ROWS=32 EPOCHS=5 SNAPSHOT_EPOCHS=1,3,5 \
#     RUN_TAG=b_skyexcl32 sbatch -J sky32 --time=08:00:00 \
#       --export=ALL,ROUTE,EXCLUDE_TOP_ROWS,EPOCHS,SNAPSHOT_EPOCHS,RUN_TAG \
#       ~/Iris/slurm/route_suite.sbatch
#     # 先冒烟确认掩码真的被削了（日志里会打印掉了多少监督像素）：
#     # ROUTE=b_thermal_unet EXCLUDE_TOP_ROWS=32 SMOKE=1 sbatch -J sky_smoke \
#     #   --time=00:30:00 --export=ALL,ROUTE,EXCLUDE_TOP_ROWS,SMOKE ~/Iris/slurm/route_suite.sbatch
#
#   任务 1b 阶段二（冻结 c1 的 adapter，只训 U-Net）：
#     ROUTE=c2_vae_adapter_unet FREEZE_ADAPTER=1 \
#     INIT_FROM=$SCRATCH/checkpoints/c1_stage1/best_weights.pt \
#       sbatch -J route_c2_stage2 --time=2-00:00:00 \
#       --export=ALL,ROUTE,FREEZE_ADAPTER,INIT_FROM ~/Iris/slurm/route_suite.sbatch
#
# 续跑：同一 ROUTE/TAG 重投即自动 --resume latest.pt。
# latest.pt 只在每个 epoch 末尾写，被墙钟砍最多丢一个 epoch。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${ROUTE:?必须指定 ROUTE，例如 ROUTE=b_thermal_unet}"
EPOCHS="${EPOCHS:-20}"

# caption 臂与 empty 臂必须写到不同目录，否则互相覆盖
TAG="${RUN_TAG:-$ROUTE}"
[[ "${CAPTION:-0}" == "1" && -z "${RUN_TAG:-}" ]] && TAG="${ROUTE}_caption"

# validate_args() 要求：--smoke-updates 的输出目录名必须含 "smoke"，
# --overfit-steps 的必须含 "overfit"。目录名要跟着模式走。
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT="$IRIS_RUNS/route_suite/smoke_$TAG"
elif [[ "${OVERFIT:-0}" == "1" ]]; then
  OUT="$IRIS_RUNS/route_suite/overfit_$TAG"
else
  OUT="$IRIS_RUNS/route_suite/$TAG"
fi
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --route "$ROUTE"
  --output-dir "$OUT"
  --ms2-root "$IRIS_MS2_ROOT"
  --train-manifest "${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
)
[[ -n "${VAL_MANIFEST:-}" ]] && ARGS+=(--val-manifest "$VAL_MANIFEST")

# 预取完成后权重都在 $HF_HOME；缺失就立即失败，而不是六个作业并发重下 4 GB
[[ "${LOCAL_FILES_ONLY:-1}" == "1" ]] && ARGS+=(--local-files-only)

# MiDaS 的多尺度梯度匹配项。不设＝关闭，与既有全部结果同口径。
# 逐点 L1 让"边界抹平"比"锐利但错位"便宜（合成阶跃上 0.0385 vs 0.0729），
# 而 AbsRel/RMSE 同为逐点平均，会奖励同一种投机 —— 这一项给它标价。
[[ -n "${GRAD_WEIGHT:-}" ]] && ARGS+=(--grad-loss-weight "$GRAD_WEIGHT")

# caption 对比约束：同一张热像、同一份种子噪声、同一个 timestep、同一张 valid mask，
# 只换 prompt 跑两次前向，要求「自己的 caption 不能输给随机另一帧的 caption」。
# 现在的损失里，忽略文本是零代价的 —— 这一项给它标价。
# 探针依据：文本进得去（Δ 1.132）、内容分得清（0.279）、传得到输出（0.293），
# 缺的只是方向监督。⚠️ 先知上界只有 +0.0018，别指望数量级变化。
if [[ -n "${RANK_W:-}" ]]; then
  ARGS+=(--caption-rank-weight "$RANK_W" --caption-rank-margin "${RANK_M:-0}")
  ARGS+=(--caption-rank-refs "${RANK_REFS:-both}")
  [[ "${RANK_DETACH:-0}" == "1" ]] && ARGS+=(--caption-rank-detach-wrong)
  [[ "${RANK_EMPTY_LIVE:-0}" == "1" ]] && ARGS+=(--no-caption-rank-detach-empty)
fi

# 天空实验：把最上 N 行从**训练损失**里拿掉（验证不受影响，仍走官方协议全图，
# 所以曲线与历史可比）。那 N 行里 97-98% 的像素没有回波，剩下 2-3% 全是近处物体
# （GT 中位 8-11 m，>89% 在 15 m 内），损失在那里携带的唯一信号就是"近"。
# 删掉它 = 让两种解释各自兑现预测：
#   无监督区自由漂移 ⇒ 天空照样塌；标签系统性偏近 ⇒ 天空保持在 40 m 附近。
# 只拿掉全部监督像素的 0.7-1.0%，AbsRel 基本不该动 —— 判据看**上带中位**。
# 详见 docs/SKY_PROBLEM_SUMMARY_20260805.md §5。
[[ -n "${EXCLUDE_TOP_ROWS:-}" ]] && ARGS+=(--loss-exclude-top-rows "$EXCLUDE_TOP_ROWS")

# Jasmine 附录 C.3 的 sky loss：在分割出的天空掩码上把视差压向 1/max_depth。
# 无监督区唯一的正向约束 —— sky32 证明了「删掉有偏的标签」只会更糟（上带 9.6→4.3 m），
# 要的是加约束而不是减标签。⚠️ AbsRel 不会因此变好（天空本来就不进指标），
# 判据是 tools/analyze_sky_band.py 的上带中位。
if [[ -n "${SKY_W:-}" ]]; then
  : "${SKY_MASKS:?SKY_W 需要配 SKY_MASKS（掩码目录）}"
  [[ -d "$SKY_MASKS" ]] || { echo "找不到掩码目录: $SKY_MASKS"; exit 1; }
  ARGS+=(--sky-loss-weight "$SKY_W" --sky-mask-dir "$SKY_MASKS" --sky-loss-mode "${SKY_MODE:-l1}")
fi

# Dense completion: lidar leaves ~three quarters of a frame with no target at all,
# and calibrated pseudo depth stands in exactly there. Real returns always win where
# they exist, and the alignment is still fitted on them alone.
if [[ -n "${PSEUDO_W:-}" ]]; then
  : "${PSEUDO_DIR:?PSEUDO_W 需要配 PSEUDO_DIR（calibrated_pseudo_depth 目录）}"
  [[ -d "$PSEUDO_DIR" ]] || { echo "找不到 pseudo GT 目录: $PSEUDO_DIR"; exit 1; }
  ARGS+=(--pseudo-gt-dir "$PSEUDO_DIR" --pseudo-weight "$PSEUDO_W"
         --pseudo-range-mode "${PSEUDO_RANGE:-drop}")
fi

# CAPTION_MODE 优先于 CAPTION=1。val 跟着 train 走 —— 否则会用它没训过的输入模式挑 best。
if [[ -n "${CAPTION_MODE:-}" ]]; then
  echo "=== caption 臂：train 与 val 都用 $CAPTION_MODE ==="
  ARGS+=(--caption-mode "$CAPTION_MODE" --val-caption-mode "$CAPTION_MODE")
elif [[ "${CAPTION:-0}" == "1" ]]; then
  echo "=== caption 臂：train 与 val 都用 correct ==="
  ARGS+=(--caption-mode correct --val-caption-mode correct)
fi

# --init-from 起新跑、--resume 接旧跑，脚本层面互斥。已有 latest.pt 时优先续跑并
# 忽略 INIT_FROM —— 否则一个被墙钟砍掉的 48h 作业，按原命令重投会从头再来。
RESUMING=0
if [[ "${SMOKE:-0}" != "1" && "${OVERFIT:-0}" != "1" && -f "$OUT/latest.pt" ]]; then
  RESUMING=1
fi

if [[ -n "${INIT_FROM:-}" && "$RESUMING" == "0" ]]; then
  [[ -f "$INIT_FROM" ]] || { echo "!! 找不到 INIT_FROM=$INIT_FROM"; exit 1; }
  ARGS+=(--init-from "$INIT_FROM")
elif [[ -n "${INIT_FROM:-}" && "$RESUMING" == "1" ]]; then
  echo "=== 已有 latest.pt，忽略 INIT_FROM 转为续跑 ==="
fi
[[ "${FREEZE_ADAPTER:-0}" == "1" ]] && ARGS+=(--freeze-adapter)

if [[ "${SMOKE:-0}" == "1" ]]; then
  echo "=== 冒烟模式：${SMOKE_UPDATES:-5} 个 update，跳过验证 ==="
  ARGS+=(--smoke-updates "${SMOKE_UPDATES:-5}" --skip-val)
elif [[ "${OVERFIT:-0}" == "1" ]]; then
  echo "=== 过拟合自检：${OVERFIT_STEPS:-300} 步 / ${OVERFIT_SAMPLES:-32} 样本 ==="
  ARGS+=(--overfit-steps "${OVERFIT_STEPS:-300}" --overfit-samples "${OVERFIT_SAMPLES:-32}" --skip-val)
else
  ARGS+=(--epochs "$EPOCHS" --snapshot-epochs "${SNAPSHOT_EPOCHS:-1,2,3,5,10,15}")
  if [[ "$RESUMING" == "1" ]]; then
    echo "=== 发现 $OUT/latest.pt，续跑 ==="
    ARGS+=(--resume "$OUT/latest.pt")
  fi
fi

echo "=== python tools/train_route_suite.py ${ARGS[*]} ==="
python tools/train_route_suite.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="

# ── 跑完即清 latest.pt ──────────────────────────────────────────────
# latest.pt 含优化器动量，约 10.4 GB，唯一用途是断点续训。12 条线全投时它占
# 约 125 GB / 1 TB，盘满会让后续作业静默失败（本地踩过：41 个训练攒到 1165 GB）。
# 三道闸（沿用本地清盘纪律）：① 退出码 0 ② 有 summary.json ③ end_weights.pt 幸存。
# 被墙钟砍掉的作业 rc≠0，latest.pt 会保住，续投不受影响。KEEP_LATEST=1 可关闭。
if [[ "$rc" == "0" && "${SMOKE:-0}" != "1" && "${OVERFIT:-0}" != "1" && "${KEEP_LATEST:-0}" != "1" ]]; then
  if [[ -f "$OUT/summary.json" && -f "$OUT/end_weights.pt" && -f "$OUT/latest.pt" ]]; then
    sz=$(du -h "$OUT/latest.pt" | cut -f1)
    rm -f "$OUT/latest.pt"
    echo "=== 清理：已删 latest.pt（$sz）；summary.json 与 end_weights.pt 均在 ==="
  else
    echo "=== 清理：跳过（缺 summary.json / end_weights.pt，或已无 latest.pt）==="
  fi
fi

du -sh "$OUT" 2>/dev/null
ls -la "$OUT"
lfs quota -h -u "$USER" /scratch
exit $rc
