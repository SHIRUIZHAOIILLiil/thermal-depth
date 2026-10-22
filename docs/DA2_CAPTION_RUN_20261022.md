# DA V2 + 本帧 caption：首次执行

仅两臂：`no-caption` 是原生metric DA V2，无文本参数；`caption` 才有冻结文本编码器
和两层门控adapter。不训练固定文本臂，不读伪GT，不运行test。旧DA2脚本不改。
科学方案见 DA2_CAPTION_PLAN_20261022.md。此处只交付32步冒烟和8000步pilot，
不自动延长至20000，不自动做test或宣称caption增益已经成立。

## 0. 同步及环境

将本轮新文件和修改同步到Aire，激活现有iris环境。需要该环境中可用的torch、
torchvision、transformers、huggingface_hub、safetensors、numpy、Pillow、opencv-python。
不在脚本里自动升级环境或修改已有依赖。BF16要求支持它的GPU。

先运行本地合成单测（不下载权重）：

```bash
cd ~/Iris
PYTHONPATH=$PWD python tests/test_da2_caption.py
bash -n slurm/prepare_da2_caption.sbatch
bash -n slurm/da2_caption.sbatch
```

单测使用小型替身模型，检查真实wrapper/loss/恢复/评估逻辑，不冒充官方权重GPU实测。
输出的合成指标不是科学结果。

## 1. 下载资产 + 双臂冒烟，一次投递

```bash
(
set -euo pipefail
cd ~/Iris
PYTHONPATH=$PWD python tests/test_da2_caption.py
bash -n slurm/prepare_da2_caption.sbatch
bash -n slurm/da2_caption.sbatch
PREP=$(sbatch --parsable slurm/prepare_da2_caption.sbatch)
PREP=${PREP%%;*}
SMOKE=$(sbatch --parsable --dependency=afterok:$PREP --job-name=da2cap_smoke \
  --time=02:00:00 --export=ALL,ACTION=smoke slurm/da2_caption.sbatch)
echo "prepare=$PREP smoke=$SMOKE"
)
```

下载作业没有请求GPU：官方DA V2源码、VKITTI-Base米制权重和现有SD2镜像的
tokenizer/text_encoder。不会下载SD U-Net/VAE。首次准备记录实际源码commit、
权重/tokenizer/config SHA256及许可，后续只校验不自动更新。HF可能需要联网或登录；
如下载失败，修复访问后只重投准备/冒烟，不绕过依赖直接投训练。

默认资产：`$SCRATCH/runs/da2_caption_assets/assets.json`。
默认数据源：`/mnt/scratch/dwdh0363/from_sc23sz`，与之前实验一致。
自定义资产时prepare传DA2_CAPTION_ASSETS_DIR，GPU脚本传DA2_CAPTION_ASSETS。

冒烟顺序执行A32步、B32步，各自独立全新初始化；每步4×3=12图，零门控先与
原DA V2 FP32输出精确比较。B还检查门控有梯度、更新后非零，文本参数无梯度。
验证uint16转换、native GT坐标/掩码、有限loss/梯度。最终核对两臂每步训练ID一致。
token缓存保存77个token和mask，不缓存大体积embedding；冻结文本编码器在线前向。

成功日志应有：

```text
[gate] zero adapters reproduce metric DA V2 exactly
[completed] ... "arm": "no-caption" ...
[completed] ... "arm": "caption" ...
[pair-gate] BOTH GPU smokes completed; image order and frozen configs match
SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_v1/smoke_<jobid>
```

查看：`grep -E '\[gate\]|\[completed\]|\[pair-gate\]|SMOKE_OUTPUT' da2cap_smoke-<jobid>.out`。
真实耗时/显存在completed.json。估算包含训练期checkpoint开销，不含完整验证；
不是严格保证的运行时间。冒烟默认2小时，首次加载和token化也占时间。

## 2. 冒烟通过后，只投两条8000步pilot

替换下面的SMOKE_OUTPUT为实际成功路径；不是冒烟权重作为起点。

```bash
(
set -euo pipefail
cd ~/Iris
export SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_v1/smoke_实际作业号
export RESUME=0
python tools/check_da2_caption_pair.py --a "$SMOKE_OUTPUT/no-caption" --b "$SMOKE_OUTPUT/caption" --smoke
sbatch --job-name=da2_nocap --export=ALL,ACTION=pilot,ARM=no-caption slurm/da2_caption.sbatch
sbatch --job-name=da2_cap --export=ALL,ACTION=pilot,ARM=caption slurm/da2_caption.sbatch
)
```

训练从官方同一权重重新开始。pilot默认48小时；结合冒烟实测决定是否需要更长时限，
不要缩短数据、增减batch或改模型来凑预算。显存不足应停止并修改共同配置后重新冒烟，
不能只给某一臂换batch。无固定文本第三条作业。

输出：`$SCRATCH/runs/da2_caption_v1/no-caption_s43` 和 `caption_s43`。
每1000步更新last.pt，每2000步保存完整快照；6000/8000自动跑完整val，ssi和none
分开记录，并输出逐帧CSV、各序列、深度段指标。没有测试集路径。

检查门槛：

```bash
python tools/check_da2_caption_pair.py \
  --a /mnt/scratch/dwdh0363/runs/da2_caption_v1/no-caption_s43 \
  --b /mnt/scratch/dwdh0363/runs/da2_caption_v1/caption_s43
```

此命令只报告是否值得考虑继续，不提交续训。门槛是B对A，并非原l100门槛；
参数量不同，此时仍只能判断完整caption方案的收益。

## 3. 超时恢复（不是第三个实验臂）

同一SMOKE_OUTPUT、同一代码/资产/清单，设置RESUME=1，仅重投中断臂。例如：

```bash
export SMOKE_OUTPUT=/mnt/scratch/dwdh0363/runs/da2_caption_v1/smoke_实际作业号
sbatch --job-name=da2_cap --export=ALL,ACTION=pilot,ARM=caption,RESUME=1 slurm/da2_caption.sbatch
```

保存模型/optimizer/scheduler/随机状态和已消费图像游标；跨epoch不丢尾部，已预取但
未消费的图片不会影响游标。CPU回放单测逐张量相等；GPU后端的数值非确定性不因此消失。
中断验证会在恢复时根据对应快照补齐。两臂数据ID日志支持审计，恢复重放的步骤取最后一次。
修改代码/输入后会拒绝续训，不得为了绕过检查手动修改contract。

## 验证边界

本地能验证模块接口、数学、梯度、恢复状态、合成数据整条评估链和Shell语法。
本地未下载官方权重、未跑Aire数据、未做真实GPU冒烟；这部分由步骤1显式验证。
本地新增13项单测通过，官方评估器13项独立单测通过，两个sbatch语法检查通过。
额外合跑旧unified评估套件遇到Windows OpenMP运行库冲突，未绕过，不能声称该套件通过；
本实验使用official_protocol而非unified协议。
已有历史记录 docs/PAPER_INTRO_NOTES_20260822.md 确认主线为thermal-derived caption；
本轮配置按此来源声明，并对清单文本做hash记录。不把文件名本身当作生成过程证据。
