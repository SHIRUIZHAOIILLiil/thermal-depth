# SiLog thermal：有 caption / 无模块

独立实验，不改旧代码与结果；未向Aire提交作业。复用现有DecoderFusion定义。

## 固定方案

- 两臂均从assets.json指定并校验的官方depth_anything_v2_metric_vkitti_vitb.pth开始；
  不加载任何thermal checkpoint。整个视觉编码器和DPT训练，lr分别5e-6和5e-5。
- caption臂冻结CLIP，第3、4组patch接现有256维四头融合层，lr5e-5；零输出初始化、无alpha。
  no-caption臂没有融合参数，不加载CLIP/tokenizer，不提供文本tensor。
- 官方SiLog：sqrt(mean(e^2)-0.5*mean(e)^2)，e=log(GT)-log(pred)，FP32计算。
  每microbatch汇总有效点，不逐帧平均；三个microbatch损失平均不等价于12帧整体SiLog。
- 真实激光，finite且0.001<GT<80；不插值GT、不用伪GT、不训练对齐、不加其他loss。
  有效位置预测非正/非有限则报错；精确零误差取零次梯度避免sqrt(0)的NaN。
- 两臂seed43、相同初始化视觉权重/图像顺序，batch4累积3，各20000步。
  AdamW betas(0.9,0.999)、wd0.01、poly0.9、BF16、clip1，无额外warmup或增强。
- 数据沿用75688训练帧和14249验证帧，高位深热像转换、几何处理、评估协议不变。
  仅10000/20000完整验证；ssi主表、none单列。没有测试集或额外诊断作业。
- 这是采用官方loss的thermal适配，不是完整复现官方RGB数据、增强与预算。

## 一次投递

先将新增文件同步到Aire的~/Iris，然后：

```bash
cd ~/Iris && bash slurm/submit_da2_silog.sh
```

提交一个双臂冒烟（每臂32步）和两个正式训练作业。冒烟包含CPU测试、真实初始预测复现、
编码器/DPT更新、CLIP冻结、attention梯度可达、每臂8帧评估以及两臂数据顺序核对。
两条正式训练均afterok依赖冒烟，失败不启动，invalid dependency自动取消。
正式训练重新从官方权重开始，不接着冒烟训练。CPU通过不代表GPU冒烟通过。

```bash
# 用实际冒烟ID替换JOBID
grep -E '^\[(config|paired-smoke|completed)\]' da2_silog_smoke-JOBID.out
tail -n 30 da2_silog_smoke-JOBID.err
```

## 输出和恢复

输出：`$IRIS_RUNS/da2_caption_silog_v1/{no-caption,caption}_s43/`。
保存完整可训练core、融合层、optimizer、scheduler、RNG、样本游标，不重复保存CLIP。
每1000步覆盖last.pt，10000/20000另存checkpoint；编码器已解冻，磁盘占用比残差版高。
恢复会移除checkpoint之后尚未提交的日志尾部，防止同一步重复记录。

显式恢复一臂（修改ARM和冒烟目录，不重新提交整条链）：

```bash
cd ~/Iris
sbatch -J da2_silog_caption --export=ALL,ACTION=train,ARM=caption,RESUME=1,SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_silog_v1/smoke_JOBID slurm/da2_silog.sbatch
```

读取结果：

```bash
cd ~/Iris && python - <<'PY'
import json
from pathlib import Path
b=Path('/mnt/scratch/dwdh0363/runs/da2_caption_silog_v1')
for step in (10000,20000):
    for arm in ('no-caption','caption'):
        p=b/f'{arm}_s43'/f'val_step{step:06d}.json'
        if not p.exists():
            print(arm,step,'not ready'); continue
        r=json.loads(p.read_text())
        for mode in ('ssi','none'):
            m=r['metrics'][mode]
            print(arm,step,mode,r['samples'], ' '.join(f'{k}={m[k]:.6f}' for k in
                ('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3')))
        print('sequence AbsRel:',{q:v['abs_rel'] for q,v in r['sequences']['ssi'].items()})
PY
```

## 结论边界

固定比较20000步，主指标ssi AbsRel，七项和三个序列全部报告，不用旧B0替代本轮无模块臂。
不设置0.0005贡献硬门槛；小幅正收益如实报告，但单种子不称稳定/显著。指标取舍如实说明。
不自动选点、续步数、增加seed或跑测试集。

六项CPU测试覆盖官方公式/梯度、mask与零误差、双臂初始化与参数组、视觉权重更新与
CLIP冻结、文本mask/梯度、双臂完整恢复/游标/日志修复、冒烟配对和依赖配置。
另检查两个Shell脚本语法；真实GPU数值及显存在Aire冒烟确认。
