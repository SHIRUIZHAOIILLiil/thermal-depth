# DA V2 两臂 8000 → 20000：续跑与独立验证

用户明确批准完成原定 20000 步预算。8000 步资源 gate 未通过仍是事实，
本次是重新决定预算，不将旧 gate 改判通过，不宣称已经充分收敛或 caption 必然有效。

## 不改原训练代码

原模型、数据、训练器均保持原样，继续使用原 `--resume --stop-step 20000`。
恢复模型、优化器、scheduler、RNG 和消费图像游标；不是仅加载权重。
总调度仍为 20000，8000 后不重启学习率。每臂最终 240000 帧次，约 3.17 轮。
不换 loss，不加 adapter warmup，不重训 text encoder，不改 batch/seed/caption。

续跑原目录 `$IRIS_RUNS/da2_caption_v1/{no-caption,caption}_s43`。
`last.pt` 和 `completed.json` 正常更新，`train.jsonl` 追加。
已有 `step008000.pt`、8000 步验证结果保持不变，记录 SHA256 并在后续验证检查。
8000 消融只读固定快照，与续跑兼容。不要再用旧 pilot 提交入口重复启动同一臂。
新续跑/验证入口用每臂 flock 避免同时写入；旧入口不参与该锁。

## 固定验证方案

- 两臂各自训练完成后，afterok 启动独立 GPU 验证。
- 按原方案：step 2000,4000,...,20000，manifest 顺序 stride4，共3563帧。
- 用 ssi AbsRel 最小选点，完全相等时取更早；不是按 none 或 RMSE 选点。
- 始终跑 step20000 的完整14249帧验证，作为同预算 caption 对照。
- 再跑选中点的完整验证；若选中20000则复用，不重复计算。
- 沿用 pilot 的 official_protocol、ssi/none 分开、原始 thermal-view LiDAR。
  不是论文统一重评阶段；不混用旧路线指标、不跑 test。
- 输出在每臂 `validation_20000/`，包括来源hash、curve_stride4/、full/、
  selection.json 和 summary.json。逐帧CSV包含原ID；不另存模型或稠密预测。
- 若独立验证中断，重新提交该臂验证即可；只复用来源/代码/权重hash完全相同
  且样本数/步数正确的已完成JSON。部分CSV不被当成完成标记。

## 本地检查

新增6项CPU测试：完整状态门禁、8000后学习率延续、完整曲线选点/并列规则、
全20000步配对日志核对、afterok提交规则、独立验证流程/缓存与变更拒绝。
不通过 `from tests...` 导入，避免 Aire 第三方包同名冲突。
已有DA V2测试仍需通过。真实权重/超算GPU运行在Aire完成，非本地实测。

## 投递

先同步新增文件到Aire；本地编辑不等于已经commit/push。激活 iris 后在登录节点：

```bash
(
set -euo pipefail
cd ~/Iris
python tests/test_da2_continuation.py
export SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_v1/smoke_9042340
bash slurm/submit_da2_continue.sh
)
```

9042340 是用户提供的成功冒烟作业；如目录实际名称不同，改为实际原始冒烟目录。
不是重新跑smoke，也不需要重建token缓存。
提交器打印4个作业号：两条续训和各自依赖的验证。若中途sbatch失败，已打印的
作业仍存在，不要盲目重复提交整个脚本。新作业不依赖8000消融。

完成两个验证后，在 iris 环境中：

```bash
cd ~/Iris
python tools/da2_continuation.py report \
  --base /mnt/scratch/dwdh0363/runs/da2_caption_v1
```

报告先检查两臂配置、20000步完成标记、全部步数和每步图像顺序，再分别打印
固定20000点和各自选中点结果。不会自动宣布成功、投递测试或继续训练。

仅某个训练作业失败时，查明原因后重投该臂（不重投另一臂）：

```bash
export ARM=caption  # 或 no-caption
export SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_v1/smoke_9042340
J=$(sbatch --parsable --export=ALL slurm/da2_continue.sbatch)
sbatch --dependency=afterok:${J%%;*} --export=ALL slurm/da2_continue_val.sbatch
```

旧依赖失败作业可能仍在队列等待 DependencyNeverSatisfied，需按实际作业ID清理，
不要使用通配删除/取消其它作业。若仅验证失败，直接重投对应ARM验证脚本。

## 资源

每臂一个单GPU训练作业，预留12小时；验证预留8小时，均非实测耗时承诺。
每臂验证约3.5–4.5次全量验证的推理量（10个stride4点 + 1或2个full点）。
继续保存每2000步完整checkpoint：每臂新增6份，另外last.pt更新。
模型+优化器+冻结文本等都在checkpoint内，正式投前用 `du -h` 和配额工具
确认scratch空间，不应把“纯推理诊断存储小”理解为“续训不占空间”。
