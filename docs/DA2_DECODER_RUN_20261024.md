# 文本条件化 DPT：实现与投递

本轮仅新增独立代码；未提交远程作业。旧模型、训练器及其结果不变。

## 冻结方案

- 起点：`da2_caption_v1/no-caption_s43/step018000.pt`，两臂相同。
- DINOv2 与 SD2 CLIP 冻结且 eval。整个 DPT depth_head 与两个新融合层训练。
- 在四组 patch 特征的索引 2、3 接入融合，class token 不变，不修改外部 DA2 仓库。
- 融合：视觉 768、文本 1024 分别 LayerNorm 后投影到 256；四头 cross-attention，
  视觉 Q、文本 K/V、padding mask、dropout=0；视觉投影与 attention 拼接，经
  Linear(512,256)、GELU、Linear(256,768) 后加回视觉特征。
- 最后一层零初始化；没有 alpha。初始输出复现 B0。并不保证训练后一定利用文本。
- frame 使用本帧 caption；fixed 始终用 `A thermal image of a scene.`。
  两臂同结构、同初始化、同样本顺序，77-token 上限与真实 mask。
- 原版 caption 本来就训练 DPT；此版变化是冻结视觉编码器、更换融合方式，
  从相同已训练视觉底座配对微调。不是“首次解冻 DPT”。
- 真实激光 GT，原 masked_log_l1、原高位深热像转换与几何处理，无伪 GT、新增强或新 loss。
- 各新增 20000 步，seed43，batch4 × accumulation3；DPT lr1e-5、融合 lr5e-5，
  AdamW betas(0.9,0.999)、wd0.01、poly0.9、BF16、clip1。
- 正式完整验证仅在10000、20000步，14249帧；ssi主表、none单独报告。
  不新增测试集推理，不追加机制诊断，不根据中间验证延长预算或改主比较点。

## 一次提交

先将本次新增文件同步到 Aire 的 `~/Iris`，然后：

```bash
cd ~/Iris && bash slurm/submit_da2_decoder.sh
```

提交三个作业：一个双臂各32步冒烟；frame/fixed正式训练分别依赖同一个冒烟
`afterok`。冒烟失败两条正式训练均不会启动，invalid dependency自动取消。
正式训练重新加载B0和同一新模块初始权重，不使用冒烟权重。
冒烟含6项CPU单测、真实初始化复现、冻结权重不变、DPT/融合更新、attention梯度可达，
以及每臂8帧评估。CPU测试不等于真实GPU冒烟通过。

```bash
# 用提交时输出的冒烟JOBID替换下面的数字
grep -E '^\[(config|paired-smoke|completed)\]' da2_dec_smoke-JOBID.out
tail -n 30 da2_dec_smoke-JOBID.err
```

输出目录为 `$IRIS_RUNS/da2_caption_decoder_v1/{frame,fixed}_s43`。
每1000步覆盖last.pt，10000/20000步另存checkpoint；仅保存DPT、融合层及优化器、
调度器、RNG/游标，冻结资产通过路径和hash引用，不反复复制。
验证结果为 `val_step010000.json`、`val_step020000.json` 和逐帧CSV。
不重复跑B0完整验证，B0参考此前同协议的无caption18000步结果。

需要显式恢复时（不重新提交整条链）：

```bash
cd ~/Iris
# 改成这次已通过的冒烟目录；ARM也可为fixed
sbatch -J da2_dec_frame --export=ALL,ACTION=train,ARM=frame,RESUME=1,SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_decoder_v1/smoke_JOBID slurm/da2_decoder.sbatch
```

## 读取两臂结果

```bash
cd ~/Iris && python - <<'PY'
import json, os
from pathlib import Path
b = Path(os.environ.get('IRIS_RUNS', '/mnt/scratch/dwdh0363/runs')) / 'da2_caption_decoder_v1'
for step in (10000, 20000):
    for arm in ('frame', 'fixed'):
        p = b / f'{arm}_s43' / f'val_step{step:06d}.json'
        if not p.exists():
            print(arm, step, 'not ready'); continue
        r = json.loads(p.read_text())
        for mode in ('ssi', 'none'):
            m = r['metrics'][mode]
            print(arm, step, mode, r['samples'],
                  ' '.join(f'{k}={m[k]:.6f}' for k in ('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3')))
        print('sequence AbsRel:', {k: v['abs_rel'] for k,v in r['sequences']['ssi'].items()})
PY
```

## 固定判断标准

主比较为20000步Frame vs Fixed；同时与B0比较。不混用中间点。
相对两个参考分别要求AbsRel降低至少0.0005，RMSE不恶化超过0.5%，a1不下降
超过0.001，并至少2/3验证序列AbsRel改善。通过表示值得继续投入，不等于统计显著。
两臂都改善但彼此接近，只支持微调/模块收益；不能归功于本帧caption。
单种子结果按探索性报告。测试集此前已经看过，本轮不使用它选结构、参数或checkpoint。

## 文件

- `models/da2_decoder.py`：模型。
- `tools/train_da2_decoder.py`：独立训练入口，复用旧数据与官方评估实现。
- `tests/test_da2_decoder.py`：六项必要CPU测试，无外部权重下载。
- `slurm/da2_decoder.sbatch`、`slurm/submit_da2_decoder.sh`：冒烟、依赖与训练验证。
