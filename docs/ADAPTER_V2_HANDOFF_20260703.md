# AnyThermal → Lotus Adapter V2：新对话交接与任务清单

> 状态日期：2026-07-03  
> 这是后续新对话的首要上下文。开始工作前先读本文件和仓库根目录
> `AGENTS.md`。不要要求用户重新说明 Windows、WSL、Conda、数据路径或
> V1 问题。

## 1. 当前结论（必须先接受）

Lotus line V1 只能作为开发诊断，不得作为最终有效模型结论继续汇报。

根因已经确认：

- 加载的模型是 `jingheya/lotus-depth-g-v2-1-disparity`；
- Lotus 原生训练目标为 inverse-depth / disparity（默认
  `trunc_disparity`）；
- 自定义 MS2 训练代码却在
  `tools/overfit_32_anythermal_lotus.py::load_depth` 中直接对正向 depth 做
  per-image min-max；
- `tools/train_ms2_adapter_v0.py` 复用了该 target；
- 因此 V1 的 target orientation 与 Lotus-G checkpoint 不一致；
- 半稠密 GT 的无效像素还被填零后整张送入 VAE，latent 在 mask 之前已被
  无效区域污染。

这不是 MS2、Iris/Lotus 官方 evaluator 或 thermal uint16 解码的问题，而是
自定义 V1 训练预处理和目标设计的问题。

### 完整 Val loss audit（5810 张）

文件：

- `archive/lotus_line_legacy_20260703/outputs/lotus_line_v1/condition_loss_audit_val_full/summary.json`
- `archive/lotus_line_legacy_20260703/outputs/lotus_line_v1/condition_loss_audit_val_full/per_sample_loss.csv`

同一 target/noise/timestep/mask 下的 legacy depth-oriented latent MSE：

| Route | Image-wise mean latent MSE |
|---|---:|
| Thermal-VAE condition + pretrained U-Net | 1.928592 |
| Direct AnyThermal condition + pretrained U-Net | 0.507357 |
| trained Adapter condition + trained U-Net | 0.234576 |

这组数字证明 trained Joint 拟合了 legacy target，不能用于深度质量排名。
Thermal-VAE 几何输出更合理却得到最高 loss，正是 target representation
不一致的报警信号。

## 2. 哪些 V1 内容仍然有效

仍然有效：

- manifest 和 split；
- MS2 left-thermal 与 thermal-view filtered LiDAR GT 配对；
- thermal `uint16` 解码和 `AnyThermalEncoder._array_to_uint8`；
- 上游 Iris/Lotus evaluator 的调用；
- 已生成 checkpoint 的真实输出及其官方指标（作为 legacy diagnostic）；
- loss audit 对 legacy target 的计算结果。

不得作为最终结论：

- V1 route winner；
- V1 checkpoint 的最终几何优劣；
- V1 Caption 的最终贡献结论；
- “loss 更低所以深度更好”；
- 旧 GT-mask PNG 所呈现的所谓稠密细节。

旧可视化曾执行 `np.where(valid, prediction, 0)`，只显示 GT-valid 区域。
模型数组虽为 dense，但 PNG 的轮廓部分由 GT validity mask 提供。诊断见：

- `archive/lotus_line_legacy_20260703/docs/diagnostics/same_prediction_full_vs_gt_masked.png`
- `archive/lotus_line_legacy_20260703/docs/diagnostics/ANYTHERMAL_VISUALIZATION_GAP.md`

当前 slides 不要作为正式成果发送给老师。线上只简要说明旧可视化和训练
目标已完成复查；正式汇报等待 V2。

## 3. 固定环境信息（不要再次询问用户）

### Windows host

- 仓库：`E:\project\Iris`
- thermal-depth 项目：`E:\project\thermal-depth`
- MS2 数据：`E:\dataset\ms2`
- PowerShell 为 Codex host shell。
- Windows Python 用于轻量审计、JSON/CSV、PPT 脚本；不要用它启动 GPU
  训练。
- Windows 原生 Conda 根目录：`E:\conda`。
- Windows 原生环境：`base`、`E:\conda\envs\pytorch_`。
- 当前 PATH 中 `C:\Windows\System32\conda` 会遮蔽真正的
  `E:\conda\Scripts\conda.exe`；若必须在 Windows 使用 Conda，应显式调用
  `E:\conda\Scripts\conda.exe`。正式 GPU 训练仍使用 WSL 的
  `wsl-pytorch`，不要混用 Windows `pytorch_` 环境。
- 本仓库 `.codex/vendor` 已有 `python-pptx`，用于生成 slides。

### WSL

- Linux 仓库：`/mnt/e/project/Iris`
- thermal-depth：`/mnt/e/project/thermal-depth`
- MS2：`/mnt/e/dataset/ms2`
- 用户 shell prompt：`dawn@DESKTOP-CG61T60`
- Conda 环境：`wsl-pytorch`
- 已验证：`torch 2.7.0+cu128`，`torch.cuda.is_available() == True`
- GPU 显存：32 GB
- 常用启动：

  ```bash
  cd /mnt/e/project/Iris
  conda activate wsl-pytorch
  ```

- Codex app 从 Windows 非交互调用 `wsl` 偶尔会报
  `Wsl/Service/CreateInstance/CreateVm/E_INVALIDARG`，但用户交互式 WSL 可以
  正常运行。遇到这种情况直接给用户 WSL 命令，不要因此改写实验。

### WSL 磁盘保护规则

- 现有 VHDX：`E:\project\Iris\.wsl\UbuntuIris\ext4.vhdx`
- 2026-07-03 检查大小约 33.45 GB。
- 此前发生过掉盘并重新注册现有 VHDX。
- 不要降级 WSL；不要重装 Ubuntu；不要 Reset；不要格式化；不要删除、
  移动或修改原始 VHDX。
- 若再次出现发行版异常，先检查 VHDX 路径和大小，再按“注册信息丢失”
  处理。

## 4. 固定数据协议

- Train manifest：
  `/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl`
- Train 数量：10441。
- Val manifest：
  `/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl`
- Val 数量：5810。
- Val manifest SHA256：
  `c8b63d0a19de7c427a3f46f04355dcd27506a0b9065c8c473a66f8487b77301e`
- Test manifest：
  `/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_test.jsonl`
- Test 数量：9508；在 V2 route selection 完成前不要再次使用 Test 调参。
- 输入：MS2 left thermal。
- GT：thermal-view filtered LiDAR depth。
- 禁止 RGB-view GT 与 thermal-view GT 混比。

## 5. Adapter V2 任务清单

### Phase A：冻结并标记 V1

- [ ] 不删除 `archive/lotus_line_legacy_20260703` 中的 V1 checkpoint、日志、
  官方 evaluator 输出和 loss audit。
- [ ] 新实验输出必须使用新目录，例如 `outputs/lotus_line_v2/...`。
- [ ] 禁止从 V1 checkpoint resume V2。
- [ ] 在 V2 报告中把 V1 标记为 `legacy depth-oriented target`。

### Phase B：先审计 Lotus 官方 target convention

- [ ] 以 `lotus/utils/hypersim_dataset.py`、
  `lotus/utils/vkitti_dataset.py` 和 `lotus/train_iris_g.py` 为准确认
  `trunc_disparity`。
- [ ] 记录 disparity quantile、clip 和映射到 `[-1, 1]` 的具体公式。
- [ ] 单元测试：相同样本中 near pixel 的 normalized disparity 必须大于
  far pixel。
- [ ] 单元测试：invalid/zero depth 不得参与倒数、quantile、normalization。
- [ ] 单元测试：同 seed 的 target latent 和 noise 完全可复现。
- [ ] 禁止继续调用 legacy `overfit.load_depth` 作为 V2 target loader。

### Phase C：解决半稠密 GT 不能直接送入 VAE 的问题

先做设计门槛，不允许直接开大训练：

- [ ] 明确 invalid pixel 在 VAE encode 前的处理方式。
- [ ] 不允许简单填 0 后整张 VAE encode，再假设 latent mask 能消除污染。
- [ ] 对至少 8 张样本记录 raw depth、valid ratio、disparity range、填充前后
  分布和 target latent 统计。
- [ ] 比较至少两个方案：
  1. **推荐首试：condition latent distillation**——让 Adapter 模仿同一张
     thermal image 经 Lotus VAE 得到的 dense condition latent；先不训练
     U-Net；
  2. 若要训练 U-Net，使用经过审计的 dense disparity teacher/depth
     completion target，或提出能避免 VAE invalid contamination 的实现。
- [ ] teacher/pseudo-label 只能来自 Train 输入；不得用 Val/Test GT 生成
  训练监督。

### Phase D：Adapter V2 首轮（优先 condition distillation）

- [ ] AnyThermal frozen。
- [ ] Lotus VAE frozen。
- [ ] Lotus U-Net frozen。
- [ ] Empty Caption；Caption 暂时后置。
- [ ] Adapter 输出目标为 thermal-image Lotus VAE condition latent。
- [ ] 同时记录：latent MSE、channel mean/std、cosine similarity、Pearson
  correlation，不能只看一个 MSE。
- [ ] Adapter 输出和 VAE reference 必须使用相同 spatial resolution。
- [ ] 保存完整 dense prediction；不得在主定性图中套 GT mask。

### Phase E：逐级 smoke gate

- [ ] 1 batch forward/backward，检查 finite、shape、gradient owner。
- [ ] 固定 8 张，检查 thermal min/max/std、condition latent 与预测。
- [ ] 32 张 overfit：要求 condition distillation loss 明显下降，并且完整
  dense prediction 的场景结构不比 Direct 更差。
- [ ] 128 张短训：检查未见样本，不只看训练样本。
- [ ] 任一 gate 若只降低 loss 而完整预测更糟，停止，不进入全量训练。

### Phase F：正式训练

- [ ] Effective batch size 固定为 4，以便与 V1 诊断对照。
- [ ] Train 10441 张，完整一轮约 2610 个 optimizer updates（最后不足 batch
  的处理必须记录）。
- [ ] 从头训练 V2，至少完整遍历一轮；不要把 V1 的 1000 steps 称为完整
  epoch。
- [ ] 保存 step 0/100/500/1000/2000/end checkpoint。
- [ ] 小规模固定 Val 仅用于训练监控；路线选择使用完整 5810 Val。
- [ ] checkpoint selection rule 在看结果前冻结。

### Phase G：统一路线比较

比较顺序：

1. Thermal-VAE condition + pretrained Lotus U-Net（sanity reference）；
2. Direct AnyThermal condition + pretrained Lotus U-Net；
3. Adapter V2 condition + frozen pretrained Lotus U-Net；
4. 只有 Adapter V2 成功后，才考虑 Adapter V2 + U-Net joint。

- [ ] 全部使用根目录 `AGENTS.md` 指定的上游 Iris/Lotus evaluator。
- [ ] 官方 `vis/*.png` 原样保留。
- [ ] 固定范围/GT/error 图只作为单独 diagnostic 或最终统一 MS2 阶段，
  不得混入 route-selection 主表。
- [ ] diffusion/latent loss 与 AbsRel/RMSE/δ 指标分表报告。
- [ ] loss 较低不得替代几何指标或完整预测检查。

### Phase H：Caption（基础深度稳定后再做）

- [ ] Caption 来自 RGB 图像；系统输入应写成
  `thermal + RGB-derived Caption`。
- [ ] Correct/Empty 使用同 checkpoint、同图、同 seed/noise/timestep。
- [ ] Caption 模型训练时加入预先冻结的 dropout 设计，否则 Empty 是未见
  分布。
- [ ] 先 Val，冻结 checkpoint 后只做一次 Test。
- [ ] Attention entropy / response ratio 单独作为机制诊断。

### Phase I：最终论文统一评估

- [ ] 只有最终 checkpoint 进入 `docs/MS2_UNIFIED_EVALUATION_PROTOCOL_V1.md`
  和 `ms2_eval`。
- [ ] 先运行：

  ```powershell
  python -m unittest tests.test_ms2_unified_evaluator -v
  ```

- [ ] 冻结 manifest/config/checkpoint hash。
- [ ] 不得把 route-selection 官方指标与统一论文指标混在同一表。

## 6. V2 必须新增的测试

- [ ] `depth=[2m, 10m, 50m]` 转换后 disparity 单调递减。
- [ ] 改变 invalid GT 像素内容不改变 valid disparity statistics。
- [ ] 空 valid mask 清晰报错。
- [ ] NaN/Inf depth 或 prediction 清晰报错。
- [ ] thermal uint16 转换后不是常数/全白。
- [ ] Adapter V2 输出 shape 与 VAE condition latent 完全一致。
- [ ] 同输入下 Direct、V2、Thermal-VAE 使用同 seed/noise/prompt。
- [ ] 保存的“full prediction”不应用 GT mask。
- [ ] 保存的“prediction sampled on GT mask”必须在文件名和标题明确标注。

## 7. 新对话第一条建议指令

新对话可直接说：

> 读取 `AGENTS.md` 和 `docs/ADAPTER_V2_HANDOFF_20260703.md`，从 Phase B
> 开始。先审计并实现正确的 Lotus `trunc_disparity` target 单元测试，不要
> 启动训练，不要修改或删除 V1 输出。
