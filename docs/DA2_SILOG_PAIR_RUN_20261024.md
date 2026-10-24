# SiLog：本帧 caption vs 空 caption（独立版本）

本轮新增文件，不修改旧SiLog、decoder或其他实验。旧排队作业不会自动转换为本配置。

## 唯一变量

两臂均从同一官方VKITTI metric ViT-B权重开始，均创建相同CLIP和两个DecoderFusion。
视觉编码器/DPT/融合层训练，CLIP冻结且eval；初始视觉和融合权重完全相同。
frame输入本帧caption，empty输入空字符串`""`，训练和验证使用同一规则。
两臂均由相同tokenizer执行padding=max_length、max_length=77、truncation=True。
空文本保留tokenizer生成的特殊token，不替换成零embedding，不关闭融合层。
保留现有mask：CLIP接收tokenizer attention_mask，cross-attention使用其取反作为
key_padding_mask。空文本mask不能全零。并非逐项复制Iris官方mask行为。

SiLog公式、有效GT范围(0.001,80)、真实激光、热像处理、图像顺序、seed43、
batch4累积3均不变。编码器lr5e-6，DPT/融合lr5e-5，AdamW、poly0.9、BF16、clip1。
各20000步；完整验证仅10000/20000步，ssi主表、none单列，无测试集或额外诊断。
复用现有models/da2_silog.py与models/da2_decoder.py，不改其实现。

## 提交前处理旧队列

用户此前贴出的旧正式作业为9067281、9067282；若尚未取消，且确认仍是这两个旧作业：

```bash
scancel 9067281 9067282
```

旧冒烟9067280可保留，但不能作为本版本正式训练的放行依据。若仍排队且不再需要，
可自行取消以避免重复占卡。本助手没有远程取消或提交任何作业。

## 一次性提交新链

将本次新增文件及其现有依赖同步到Aire的~/Iris后：

```bash
cd ~/Iris && bash slurm/submit_da2_silog_pair.sh
```

三个作业：da2_sp_smoke（依次frame/empty各32步）、da2_sp_frame、da2_sp_empty。
两条正式作业均依赖同一个冒烟afterok，失败不启动，invalid dependency取消。
正式作业重新加载官方权重与新模块初始状态，不从冒烟继续训练。
脚本拒绝覆盖已有正式目录；新旧实验输出不混用。

```bash
# 替换JOBID为新冒烟ID
grep -E '^\[(config|paired-smoke|completed)\]' da2_sp_smoke-JOBID.out
tail -n 30 da2_sp_smoke-JOBID.err
```

新输出为`/mnt/scratch/dwdh0363/runs/da2_caption_silog_pair_v1/{frame,empty}_s43/`。
last.pt每1000步覆盖，10000/20000另存checkpoint；保存完整可训练编码器、DPT、融合层、
优化器、调度器、RNG和样本游标，引用冻结CLIP资产而不反复复制。

恢复单臂时使用本版本脚本和本版本成功冒烟目录：

```bash
cd ~/Iris
sbatch -J da2_sp_frame --export=ALL,ACTION=train,ARM=frame,RESUME=1,SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_silog_pair_v1/smoke_JOBID slurm/da2_silog_pair.sbatch
```

## 读结果

```bash
cd ~/Iris && python - <<'PY'
import json
from pathlib import Path
b=Path('/mnt/scratch/dwdh0363/runs/da2_caption_silog_pair_v1')
for step in (10000,20000):
    for arm in ('frame','empty'):
        p=b/f'{arm}_s43'/f'val_step{step:06d}.json'
        if not p.exists():
            print(arm,step,'not ready'); continue
        r=json.loads(p.read_text())
        for mode in ('ssi','none'):
            m=r['metrics'][mode]
            print(arm,step,mode,r['samples'],' '.join(f'{k}={m[k]:.6f}' for k in
                ('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3')))
        print('sequence AbsRel:',{q:v['abs_rel'] for q,v in r['sequences']['ssi'].items()})
PY
```

固定20000步比较，主指标ssi AbsRel，全部七项和三序列如实报告。
不使用0.0005作为贡献硬门槛，不把单种子微小差异称为稳定或显著收益。
这个对照控制模块容量，检验本帧描述相对于空描述的作用，不保证一定出现正收益。

## 必要检查

复用六项SiLog核心CPU测试，新增三项：空文本77位特殊token/mask保持；双臂同模块、
同初始化和初始预测、空文本梯度通路；严格配对冒烟（融合初始权重也必须相同）。
另外检查两份Shell脚本语法。真实tokenizer、BF16和显存由Aire双臂冒烟确认。
