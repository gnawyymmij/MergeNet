# MergeNet 论文补充实验执行手册（E1 / E2 / E3 / E6）

本文档是交给实验执行者及其 AI 助手的完整执行合同。请先通读全文，再修改代码或启动任务。实验目标是补齐论文当前保留的四项证据：matched DTEM、统一 accuracy/latency/memory benchmark、两个组件的推理消融，以及被动 routing trace。
**回传数据格式与隐私边界另见 `DATA_HANDOFF.md`；正式执行前也必须通读。**请尽量交付其中的 L1 匿名逐图数值，使作者能在本地复算表格、配对区间和路由图，而不传出 checkpoint 或原始数据；无法外传时须在状态文件明示限制。

正式运行必须遵守以下原则：

- 不修改、覆盖或删除已有 checkpoint；新输出只写入新的 `outputs/` 子目录。
- 不用中途最高精度代替完整终点，不删除失败或负结果，不静默重启换 seed。
- 不把不同 GPU、环境、checkpoint、attention policy 或 token budget 的结果拼成同一 Pareto 点。
- 不把随机初始化 timing 与训练后 accuracy 组合。
- 所有方法都必须实测最终 token 数；“配置目标是 392”不等于实际执行了 392 patches。
- 数据集、checkpoint 和第三方源码不提交到本仓库。输出中不得包含密钥、用户名或未脱敏的内部路径。
- 第三方 DTEM/ToMe/PiToMe 只从固定 commit 的独立 checkout 导入，不复制进本 Apache-2.0 仓库。
- 在 smoke gates 全部通过前，不得启动 300-epoch 正式训练。

## 1. 本轮最终需要交付什么

| ID | 必需结果 | 是否训练 | 主要输出 |
|---|---|---:|---|
| E1 | DeiT-S/8、224px、392-patch 的 matched DTEM | 是，1 个 seed-42 300e run | best/final EMA accuracy、完整 summary、checkpoint receipt、实际 token trajectory |
| E2 | dense、DTEM、ToMe、PiToMe、MergeNet 的统一 accuracy/latency/memory | 否 | 五方法完整 ImageNet-val accuracy；batch 1/64 timing 和 memory；原始 JSON |
| E3 | mass bias 与 recovery 的四行因果消融 | 否 | native、mass-off、recovery-off、both-off 的完整验证集结果与 paired CI |
| E6 | 最终 MergeNet 的被动 routing trace | 否 | trace 等价性凭据、1,000 图聚合统计、固定规则选出的 overlays |

推荐顺序：先完成第 2–5 节的预检；随后启动 E1。E1 训练期间实现并运行 E3 和 E6，搭好 E2；E1 完成后把 DTEM 加入 E2。除 E1 外，其余任务均不得更新权重。

## 2. 必需输入与目录约定

执行者需要自行设置以下路径。不要把真实路径硬编码进提交的源码。

```bash
export MN_DATA_DIR=/path/to/imagenet
export MN_OUTPUT_ROOT=/path/to/mergenet-paper-runs
export MN_DENSE_CKPT=/path/to/deit_s8_300e_checkpoint.pth.tar
export MN_MERGENET_CKPT=/path/to/mergenet_r3_300e_checkpoint.pth.tar
export MN_DTEM_CKPT=/path/to/e1_dtem_checkpoint.pth.tar  # E1 完成后设置
export MN_BENCH_GPU=0
```

ImageNet 必须是标准 ImageFolder：

```text
$MN_DATA_DIR/
  train/<1000 synset dirs>/*.JPEG
  val/<1000 synset dirs>/*.JPEG
```

正式运行前检查：

```bash
test -d "$MN_DATA_DIR/train"
test -d "$MN_DATA_DIR/val"
find "$MN_DATA_DIR/train" -type f | wc -l  # 必须为 1281167
find "$MN_DATA_DIR/val" -type f | wc -l    # 必须为 50000
find "$MN_DATA_DIR/train" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | sort | sha256sum
find "$MN_DATA_DIR/val" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | sort | sha256sum
```

两个 split 的 class-list digest 必须相同。把计数和 digest 写入 `$MN_OUTPUT_ROOT/dataset_receipt.txt`。

checkpoint 约定：

- Dense：论文中的 DeiT-S/8、224px、300e EMA checkpoint。
- MergeNet：论文中的 MN-L2-R3、224px、300e EMA checkpoint。
- E1 DTEM：本轮训练产生的 DTEM EMA checkpoint。
- 优先使用论文数字实际对应的 EMA state；同时记录 checkpoint 文件 SHA-256、state key、epoch、best/final 属性。
- 如果文件是完整 trainer checkpoint，明确实际加载的是 `state_dict_ema`、`state_dict`、`model_ema` 还是其他 key；禁止依赖模糊的“第一个可加载 key”。
- 所有模型必须打印 loaded/missing/unexpected keys。分类头或 positional embedding 缺失时立即失败。

建议输出树：

```text
$MN_OUTPUT_ROOT/
  dataset_receipt.txt
  environment.txt
  e1_dtem/
  e2_benchmark/
    accuracy/
    latency/
    summary.csv
  e3_ablation/
    predictions/
    summary.csv
  e6_routing_trace/
    traces/
    figures/
    summary.csv
  DELIVERY_MANIFEST.json
```

## 3. 环境与代码身份

主仓库环境：

```bash
conda env create -f environment.yml
conda activate mergenet-in1k
python -m pip install -r requirements-lock.txt
export OPENTOME_SKIP_OPTIONAL_NLP=1
```

记录以下内容到 `environment.txt`：

```bash
date -Iseconds
git rev-parse HEAD
git status --porcelain
python --version
python -m pip freeze
nvidia-smi -q
```

正式结果要求主仓库 worktree clean。如果必须改 adapter、评测或 trace 代码，先提交到独立 commit，再记录该 commit。运行过程中不得修改源码。

外部源码固定版本：

| 方法 | 仓库 | commit | 用途 |
|---|---|---|---|
| DTEM | `https://github.com/movinghoon/DTEM.git` | `13f204634725209ab6a562c6e753675f2c430927` | E1 和 E2 |
| ToMe | `https://github.com/facebookresearch/ToMe.git` | `af95e4b1befa172dadccd8c81e223b10090f9579` | E2 |
| PiToMe adapter | `https://github.com/Westlake-AI/OpenToMe.git` | `4b791cbc6bcef55040236c1852ff2f4102d82725` | E2；与已有论文 PiToMe 行保持实现身份一致 |

DTEM 固定 commit 未包含明确 LICENSE，ToMe/PiToMe 相关代码带非商业或其他上游许可约束。因此只能独立 clone 用于科研评测，不得 vendor 到本仓库。

建议 checkout：

```bash
mkdir -p third_party
git clone https://github.com/movinghoon/DTEM.git third_party/DTEM
git -C third_party/DTEM checkout 13f204634725209ab6a562c6e753675f2c430927
git clone https://github.com/facebookresearch/ToMe.git third_party/ToMe
git -C third_party/ToMe checkout af95e4b1befa172dadccd8c81e223b10090f9579
git clone https://github.com/Westlake-AI/OpenToMe.git third_party/OpenToMe
git -C third_party/OpenToMe checkout 4b791cbc6bcef55040236c1852ff2f4102d82725
```

每个 checkout 必须满足 `git status --porcelain` 为空；把实际 HEAD 和入口文件 SHA-256 写入结果 manifest。若为了兼容当前 timm 必须写 adapter，只在本仓库 `experiments/` 中写薄适配层，不修改第三方 checkout；适配行为必须有测试。

## 4. Phase 0：任何正式任务之前的 smoke gates

### 4.1 仓库 CPU gates

```bash
python tests/test_release_contract.py
python tests/test_accumulation_schedule.py
python tests/test_dtem_spatial_mask.py
python tests/test_imagefolder_loader.py
```

四条必须全部以 exit code 0 结束。

### 4.2 MergeNet CUDA gate

在一张可用 GPU 上运行：

```bash
CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python tests/test_dtem_spatial_sparse.py
CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python tests/test_biased_local_attention.py
CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python tests/test_model_smoke.py
```

必须得到有限 logits/gradients、正确 retained-token count 和确定性 eval grouping。该测试只用小模型，不是正式实验。

### 4.3 启动命令 dry-run

下面只检查 ImageNet 目录、global-batch 算术和最终 torchrun 命令，不启动训练：

```bash
DATA_DIR="$MN_DATA_DIR" \
OUTPUT_DIR="$MN_OUTPUT_ROOT/dry_run" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 BATCH_SIZE=64 GLOBAL_BATCH=1024 DRY_RUN=1 \
bash scripts/train_imagenet_300e.sh
```

输出中的 `--update_freq` 必须为 2。若使用每卡 batch 128，则必须为 1。

### 4.4 新实验代码必须提供的 smoke 接口

执行 AI 应在 `experiments/` 下实现 E1–E6 所需入口，并为每个入口提供 `--smoke` 或等价模式：

```text
experiments/e1_train_dtem.py
experiments/e2_evaluate.py
experiments/e2_benchmark.py
experiments/e3_ablate.py
experiments/e6_trace.py
experiments/summarize.py
```

`--smoke` 只允许处理 2–8 张合成图或最多 32 张真实图，必须检查 forward、backward（E1）、checkpoint round-trip、token count 和 JSON schema。smoke 输出必须放在 `$MN_OUTPUT_ROOT/smoke/`，不得混入正式结果。

正式任务的启动 gate：

- E1：DTEM 224/p8 forward + backward 有限；train/eval 两种 token trajectory 已记录；保存后 strict reload 一致。
- E2：dense 和 MergeNet 的 32-image eval 可运行；token hook 在计时前移除；每种方法独立进程。
- E3：四个 variants 从同一 checkpoint strict load；native 重复两次 logits 一致。
- E6：trace-off/on 等价性 gate 在 8 张图上通过。

### 4.5 执行 AI 必须实现的 CLI 合同

当前公开仓库提供 MergeNet 训练入口，但不把尚未验证的实验脚本伪装成可运行实现。接手的 AI 应先按下面的参数名实现第 4.4 节列出的六个入口；实现后，下列命令必须可以原样运行。所有入口都要支持 `--help`，缺失输入时应报清晰错误并以非零状态退出，不能自动搜索 checkpoint 或数据集。

```bash
# E1 smoke：CPU 或单卡、小模型/小 batch，不启动正式 epoch
python experiments/e1_train_dtem.py \
  --dtem-root third_party/DTEM \
  --output "$MN_OUTPUT_ROOT/smoke/e1" \
  --seed 42 --smoke

# E1 正式训练：以下示例是 8 卡 × 64/GPU × accumulation 2 = global batch 1024
torchrun --standalone --nproc_per_node=8 experiments/e1_train_dtem.py \
  --data-dir "$MN_DATA_DIR" \
  --dtem-root third_party/DTEM \
  --output "$MN_OUTPUT_ROOT/e1_dtem" \
  --seed 42 --epochs 300 --batch-size 64 --update-freq 2

# E2 accuracy；每种方法独立进程，防止 patch/global state 相互污染
for method in dense dtem tome pitome mergenet; do
  CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python experiments/e2_evaluate.py \
    --method "$method" --data-dir "$MN_DATA_DIR" \
    --checkpoint-manifest "$MN_OUTPUT_ROOT/e2_benchmark/checkpoint_manifest.json" \
    --output "$MN_OUTPUT_ROOT/e2_benchmark/accuracy/$method.json"
done

# E2 latency + memory；batch 1 和 64 必须各起一个 fresh process
for method in dense dtem tome pitome mergenet; do
  for batch in 1 64; do
    CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python experiments/e2_benchmark.py \
      --method "$method" --batch-size "$batch" --precision fp16 \
      --warmup 50 --rounds 5 --iterations 200 \
      --checkpoint-manifest "$MN_OUTPUT_ROOT/e2_benchmark/checkpoint_manifest.json" \
      --output "$MN_OUTPUT_ROOT/e2_benchmark/latency/${method}_b${batch}.json"
  done
done

# E3：一次进程顺序评测四个 variant，并保存同序逐图片结果
CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python experiments/e3_ablate.py \
  --data-dir "$MN_DATA_DIR" --checkpoint "$MN_MERGENET_CKPT" \
  --variants native,mass_off,recovery_off,both_off \
  --batch-size 64 --bootstrap-samples 10000 --bootstrap-seed 20260923 \
  --output "$MN_OUTPUT_ROOT/e3_ablation"

# E6：先创建并冻结 manifest，再跑 equivalence gate 和完整 trace
CUDA_VISIBLE_DEVICES="$MN_BENCH_GPU" python experiments/e6_trace.py \
  --data-dir "$MN_DATA_DIR" --checkpoint "$MN_MERGENET_CKPT" \
  --sample-rule one-per-class-sha256 --sample-seed 20260923 \
  --num-images 1000 --equivalence-images 8 \
  --output "$MN_OUTPUT_ROOT/e6_routing_trace"

# 只读汇总；不得重新运行或修改模型结果
python experiments/summarize.py \
  --input-root "$MN_OUTPUT_ROOT" \
  --output "$MN_OUTPUT_ROOT/DELIVERY_MANIFEST.json"
```

实现约束：

- 共同的 model/checkpoint/data/receipt 逻辑放在 `experiments/common/`，不要在六个入口中复制并逐渐分叉。
- `--smoke`、正式运行和汇总使用同一代码路径；smoke 只能通过限制样本/epoch/模型规模缩短运行，不能换成空实现。
- E2 的第三方 patch 必须在 fresh process 中应用；禁止在同一 Python 进程依次 patch 五种方法。
- E3 四个 variant 可共用一个进程，但每行评测前都必须从同一原始 checkpoint 重新 strict-load 模型，避免 runtime flag 或 hook 泄漏。
- E6 先写 `sample_manifest.csv` 并计算 SHA-256；若该文件已存在，默认只读并核对，不重新抽样。
- 每个正式入口启动时打印 resolved config，结束时原子写 JSON；异常时保留日志和 `BLOCKED.md`，不要留下看似完整的半截 JSON。

## 5. E1 — Matched DTEM

### 5.1 研究问题与结果身份

E1 回答：在与论文主模型匹配的 ImageNet-1K、DeiT-S/8、300 epochs、最终 392 patch budget 下，DTEM 能达到什么准确率，其评估路径实际执行多少 token。

该结果必须命名为：

> DTEM-p8, our common-recipe adaptation

不要称为官方 DTEM checkpoint 或官方论文复现。官方 DTEM 使用 patch-16、预训练 backbone 和 metric-only modular training；本轮是为了主表公平性而进行的 patch-8/common-recipe adaptation。

### 5.2 模型和预算

- Backbone：DeiT-S/8。
- 输入：224×224。
- Patch size：8；输入为 784 patch tokens + 1 CLS。
- Width：384；depth：12；heads：6；MLP ratio：4。
- DTEM 上游：固定 commit `13f2046...` 的 `dtem.patch`。
- DTEM 参数：`k2=3`、`tau1=0.1`、`tau2=0.1`、默认 `feat_dim`。
- 每层 reduction schedule：`[32,32,32,32,32,32,32,32,32,32,32,40]`。
- 评估路径的实际 token trajectory（含 CLS）应为：

```text
785 -> 753 -> 721 -> 689 -> 657 -> 625 -> 593
    -> 561 -> 529 -> 497 -> 465 -> 433 -> 393
```

最终必须是 392 patches + CLS。不能只相信 schedule，要用 hooks 实测。

DTEM 原生 soft training path 会保留物理 tensor slots，同时让 logical active count 下降；eval hard path 才物理缩短。训练和评估 trajectory 必须分别记录，E2 timing 使用 hard eval path。不得把 soft-train 的逻辑 token 数当作物理加速。

### 5.3 训练协议

| 项目 | 固定值 |
|---|---|
| Dataset | ImageNet-1K，1,281,167 train / 50,000 val |
| Seed | 42 |
| Initialization | 从随机初始化开始；不加载 dense 或 ImageNet pretrained 权重 |
| Trainable parameters | 全部 DTEM + DeiT 参数；不得冻结为 metric-only |
| Epochs | 300，完整 epoch 0–299 |
| Optimizer | AdamW |
| Base LR | `5e-4` |
| Weight decay | `0.05` |
| Warmup | 5 epochs，warmup LR `1e-6` |
| Schedule | cosine，minimum LR `1e-5` |
| Global batch | 1,024 |
| Precision | AMP fp16 |
| Gradient clipping | norm 1.0 |
| EMA | decay `0.99996` |
| Augmentation | RandAugment `rand-m9-mstd0.5-inc1`；Mixup 0.8；CutMix 1.0；random erase 0.25；label smoothing 0.1 |
| Drop path | 0.1 |
| Distillation | 全部关闭 |
| Validation | 224px，crop_pct 0.9，bicubic，无 TTA，完整 50k |

global batch 可以通过不同 microbatch × world size × update frequency 实现，但必须严格等于 1,024，并记录每 epoch optimizer update 数。8 GPUs × 64/GPU 时 `update_freq=2`；8 × 128 时为 1。

### 5.4 Adapter 实现要求

执行 AI 需要写一个薄 model factory，完成以下操作：

1. 用当前 `timm` 创建 `deit_small_patch16_224.fb_in1k`，但显式覆盖 `img_size=224, patch_size=8, pretrained=False, num_classes=1000`。
2. 验证 `num_patches=784`、width=384、depth=12、heads=6。
3. 从固定 checkout 导入 `dtem.patch`；禁止导入本仓库中为 MergeNet 扩展过的 `opentome/timm/dtem.py` 冒充官方 DTEM。
4. 应用上述 DTEM 参数并设置 12 层 schedule。
5. 保持所有参数可训练。断言 trainable parameter names 不只包含 `metric`。
6. 让模型输出标准 `[B,1000]` logits，接入本仓库相同 trainer/data/EMA/checkpoint 路径。
7. 将 model/source/schedule 身份写入 checkpoint args 和结果 manifest。

Adapter 单元测试至少覆盖：224/p8 geometry、有限 forward/backward、metric 参数有梯度、非 metric backbone 参数也有梯度、393 最终 eval tokens、checkpoint strict round-trip。

### 5.5 中止条件

出现以下任一情况必须中止并保留日志，不得静默修补后继续原 run：

- 数据计数或 class mapping 不符；
- 任一 loss/metric/gradient 非有限；
- epoch 序号不连续；
- optimizer 实际更新为 0，或 AMP 大量跳步；
- final eval tokens 不等于 393（含 CLS）；
- checkpoint reload 出现未解释的 missing/unexpected keys；
- world size/global batch/学习率等与协议不一致。

修复后应启动新 run ID，并在 manifest 中关联失败 run；不得覆盖原目录。

### 5.6 E1 必须交付

```text
e1_dtem/
  config.yaml
  resolved_args.yaml
  launch_command.txt
  environment.txt
  source_sha256.json
  train.log
  summary.csv                 # 300 行，epoch 0–299 连续
  best_checkpoint_receipt.json
  final_checkpoint_receipt.json
  token_trajectory_train.json
  token_trajectory_eval.json
  final_eval.json
```

`final_eval.json` 至少包含 raw/EMA 的 final top-1、top-5、loss，以及 best EMA top-1/top-5、best epoch。主表最终使用哪个 checkpoint，要在 E2 中按与 dense/MergeNet 相同的 selection policy 重新评测；不能只抄训练日志。

## 6. E2 — 统一 accuracy / latency / memory benchmark

### 6.1 比较对象

| ID | 权重 | 压缩设置 | 目标 patch 数 |
|---|---|---|---:|
| `dense` | Dense DeiT-S/8 EMA | 无压缩 | 784 |
| `dtem` | E1 DTEM EMA | E1 hard eval schedule | 392 |
| `tome` | 与 dense 完全同一 EMA | ToMe，proportional attention 开启 | 392 |
| `pitome` | 与 dense 完全同一 EMA | PiToMe，proportional attention 开启 | 392 |
| `mergenet` | MN-L2-R3 EMA | native λ=2、R=3 | 392 |

ToMe 和 PiToMe 都是 dense checkpoint 上的 post-training 方法，不允许加载不同 dense 权重。ToMe/PiToMe 都设置总 reduction 为 392 patches，并用分布式每层 reduction 防止单层 clamp；建议起点为 `[32]×11+[40]`，但最终以实测 token trajectory 为准。

PiToMe 使用固定 OpenToMe commit 的 `pitome_apply_patch`，显式设置 `prop_attn=True`、`trace_source=False`。ToMe 同样设置 `prop_attn=True`、`trace_source=False`。accuracy 与 timing 必须使用完全相同的 attention policy。

### 6.2 Checkpoint selection policy

先建立 `checkpoint_manifest.json`，每个训练模型记录：

- 文件 SHA-256；
- 实际加载 state key；
- raw 或 EMA；
- epoch；
- best 或 final；
- loaded/missing/unexpected keys；
- parameter count。

现有论文主表对应重评后的 best EMA checkpoint，因此 E2 主行应对 dense、MergeNet 和 DTEM 使用相同的 best-EMA policy；同时单独保存 final-epoch EMA 结果用于审计。若任一方法没有同语义 checkpoint，不得混填，应先协调统一 policy。

### 6.3 Accuracy 协议

- Dataset：同一 ImageNet `val`，恰好 50,000 张，每张一次，无 padding/重复/drop-last。
- Input：224×224；bicubic；crop percentage 0.9；ImageNet mean/std；无 TTA。
- `model.eval()` + `torch.inference_mode()`。
- FP16 autocast；模型参数按 checkpoint 原 dtype 加载。
- Eval batch：64；同一图片顺序。
- 单 GPU 主评测，避免不同 DDP sampler 语义。若采用多 GPU，必须证明总样本仍恰好 50,000 且无重复。
- 输出：top-1、top-5、mean cross-entropy、样本数、eval 秒数、每张图的 label/top-1/top-5 correctness；空间允许时保存 fp16 logits。

在运行所有未知结果前先做 sanity gate：

- Dense 应在论文记录 82.246% 附近，容差 ±0.10 pp。
- MergeNet 应在论文重评 81.360% 附近，容差 ±0.10 pp。
- 两项任一失败，停止整个 E2，检查 preprocessing、EMA key、position embedding 和 routing config。

每个方法在正式 accuracy 前用 hooks 实测每层 attention input、MLP input 和 reduction output 的 patch/CLS 数。记录逻辑 tokens 与物理 tensor tokens；hooks 收集完成后必须移除，再进行 timing。

### 6.4 Latency 协议

只能在一张完全空闲、独占的物理 GPU 上运行五个方法。不得在共享 GPU、MIG 或可见外部 compute process 的设备上记录正式数据。

固定并记录：GPU 名称/UUID、driver、CUDA、PyTorch、cuDNN、timm、FlashAttention、TF32、attention backend、power/clocks、代码 commit。所有方法必须同 GPU、同 fresh-process policy、同 precision。

测量设置：

- synthetic device-resident `torch.randn(B,3,224,224)`，不含 dataloader、JPEG decode 和 H2D；
- `model.eval()`、`torch.inference_mode()`、fp16 autocast；
- `torch.compile` 关闭；profiler、trace、token hooks、debug logging 全部关闭；
- 主表 batch 64；在线场景补充 batch 1；
- 每个 batch size/方法使用新进程；
- 每个进程 warmup 50 iterations；
- 5 rounds × 200 timed iterations；
- 使用 CUDA events；每个 round 末尾同步；
- 五方法顺序在 rounds 间轮换或随机化并保存顺序；
- 不删除 outliers；保存全部 1,000 个 raw latency samples。

报告：每 batch latency median、IQR、P10/P90、images/s、五个 round medians 和 round CV。CV > 3% 或运行中出现外部进程/明显 throttling 时，该 cell 无效，整 cell 重跑并保留无效记录。

### 6.5 Memory 协议

每个方法和 batch size 在干净新进程中：

1. 加载模型并完成 warmup；
2. `torch.cuda.empty_cache()`；
3. `torch.cuda.reset_peak_memory_stats()`；
4. 执行一次与 timing 相同的完整 forward；
5. 同步；
6. 记录 peak allocated MiB 与 peak reserved MiB；
7. 另记加载模型后、输入分配前的 baseline allocated/reserved。

不要把 allocated 与 reserved 混成一个 memory 数字。若任一方法 batch 64 OOM，应把所有方法的主 throughput batch 统一降到共同可运行值，同时保留 batch 1。

### 6.6 E2 必须交付

每个 cell 一个自描述 JSON，至少包含：

```json
{
  "schema_version": "mergenet.e2.v1",
  "method": "mergenet",
  "checkpoint_sha256": "...",
  "checkpoint_state_key": "...",
  "checkpoint_epoch": 291,
  "ema": true,
  "requested_patch_tokens": 392,
  "achieved_patch_tokens": 392,
  "token_trajectory": [],
  "top1": 0.0,
  "top5": 0.0,
  "cross_entropy": 0.0,
  "n_images": 50000,
  "batch": 64,
  "warmup": 50,
  "rounds": 5,
  "iterations_per_round": 200,
  "raw_latency_ms": [],
  "peak_allocated_mib": 0.0,
  "peak_reserved_mib": 0.0,
  "environment": {}
}
```

最后只从 JSON 自动生成 `summary.csv`，禁止手工复制数字。accuracy、latency、memory 只有在 method/checkpoint/budget/attention policy 完全一致时才能位于同一行。

## 7. E3 — Mass bias 与 recovery 的 inference-only 消融

### 7.1 共同设置

四个 variants 使用同一个 `$MN_MERGENET_CKPT`，不训练、不更新权重：

| Variant | Mass bias | Recovery cross-attention |
|---|---:|---:|
| `native` | 开 | 开 |
| `mass_off` | 关 | 开 |
| `recovery_off` | 开 | 关 |
| `both_off` | 关 | 关 |

评测设置与 E2 MergeNet accuracy 完全相同：224px、crop 0.9、batch 64、fp16、无 TTA、完整 50k、同样本顺序。E3 的 `native` 结果必须与 E2 MergeNet 在数值精度内一致，否则停止。

### 7.2 Mass-bias-off 的唯一允许定义

保留以下内容原样：

- local routing scores；
- transported mass；
- top-K indices；
- carrier features；
- recovery cross-attention；
- token budget 和 latent depth。

只在进入 latent self-attention 时，将 `size_trace` 替换为同 shape 的 ones，使 proportional attention 中的 additive `log(mass)` key bias 精确变为 0。

禁止在 routing 前把 mass 设为 1，因为那会同时改变 routing、carrier selection 和 transport，不再是单因素消融。实现后应断言 `topk_indices(native) == topk_indices(mass_off)`，并比较 recovery 输入在 mass 替换前完全一致。

### 7.3 Recovery-off 的唯一允许定义

保持相同 local output、top-K indices、carrier features、mass 和 latent encoder，只把：

```python
x_trace = encode_cross_attention(topk_x, x_embed, mask=bias) + topk_x
```

替换为：

```python
x_trace = topk_x
```

不要删除或重新初始化 recovery 参数，不要改变 checkpoint load。最安全做法是实例化 native `mergenet_small_cls`、strict load 同一 checkpoint，然后在 forward 中通过 runtime flag 跳过 residual recovery。本仓库的 `disable_encode_cross_attention` 路径可作实现参考，但 E3 不应因为构造另一种模型而产生 missing/unexpected keys。

`both_off` 同时应用上述两个单因素干预，其余全部相同。

### 7.4 正确性 gates

每个 variant 必须满足：

- 输出恰好 392 patch carriers + CLS；
- logits、loss、mass 均无 NaN/Inf；
- checkpoint strict load 无 missing/unexpected keys；
- 相同 variant、相同 batch 连续运行两次的 predicted class 一致；
- trace/debug 开关不改变 native logits；
- `mass_off` 不改变 top-K indices；
- `recovery_off` 不改变 top-K indices 和 `size_trace`；
- 所有 variants 处理完全相同的 50,000 个 image IDs。

### 7.5 统计

保存每张图：relative path 或稳定 image ID、label、top-1 prediction、top-5 predictions、cross-entropy、correctness。报告：

- top-1、top-5、mean cross-entropy；
- 相对 native 的 Δtop-1、Δtop-5、Δloss；
- prediction flip count；
- 单因素和双关闭行。

对每个 variant-native 的逐图片 top-1 correctness difference 做 paired bootstrap：10,000 resamples、bootstrap seed `20260923`，每次对 50,000 image indices 有放回抽样，报告 percentile 95% CI。bootstrap 衡量固定 checkpoint 上的图像抽样不确定性，不能替代训练 seed 方差；论文中必须这样说明。

### 7.6 E3 必须交付

```text
e3_ablation/
  config.json
  native.json
  mass_off.json
  recovery_off.json
  both_off.json
  predictions/native.npz
  predictions/mass_off.npz
  predictions/recovery_off.npz
  predictions/both_off.npz
  paired_bootstrap.json
  integrity_checks.json
  summary.csv
```

## 8. E6 — 被动 routing trace 与可视化

### 8.1 E6 回答什么

E6 用已有最终 MergeNet checkpoint 展示模型实际把哪些 patch 信息路由到哪些 carrier、路由距离如何分布、mass 是否集中以及 carrier 在空间上的覆盖。它不是 accuracy 或 latency 证据，不得从图片推导“保持语义边界”等未经标注验证的结论。

E6 不训练，不改变模型输出。Trace hooks 必须与 E2 timing 完全分离。

### 8.1a 论文图的预注册设计与可分享边界

本轮 E6 的目标成图是**一张正文定性图 + 一张附录总体统计图**，不得事后根据哪一层、哪张图“最好看”来改选择规则。
版式预览见 `docs/visualization_preview.png`；其生成器 `docs/render_visualization_preview.py` 只使用合成数据，不能当作任何实验结果或复用其中的数值。

- 正文图采用两行两列：上行为固定正确案例，下行为固定错误案例；左列画**第 3 个 local routing step** 的实际 donor→receiver 边，右列画最终 **392/784 个 carrier 的原始 28×28 位置**。每个案例的左、右列使用同一张输入图、同一 checkpoint、同一次 eval 协议。图下注明这只是 routing 行为，非语义分割或因果证据。
- 正确/错误案例均从下述 1,000 张预先冻结的 manifest 中选；各取其组内 `digest` 字典序最小的一张。若没有错误案例，不编造，改为前两张正确例并在图注解释。预测正确性取 E2 native MergeNet 同一 checkpoint/预处理下的 top-1 与 ground truth 比较，不能为作图另换 checkpoint 或 crop。
- 左列原图背景必须是**实际送入模型的 224×224 center crop**（反归一化仅用于显示），不是未裁剪原图。四个 14×14 象限中，各取最终应用于 transport 的 `assign_postmask_ij > 0` 且最高的 8 条边；它是 soft gate/row-softmax 经 donor 归一化及最终 physical mask 后的实际权重，**不能直接用未归一化的 `g_i q_ij` 代替**。同一 donor/receiver 重复边保留其 step-3 实际记录，按 `(weight desc, donor index asc, receiver index asc)` 排序破同分，最多 32 条。箭头从 donor 原始 patch 中心指向 receiver 原始 patch 中心；线宽/透明度按全图统一固定范围编码实际 transport weight，并附标尺。不得将 cosine similarity、eligible mask 或 source center 的连线冒充实际 transport 边。
- 右列在 28×28 原始格点上标记最终 top-K **selected slot index**，非 center-of-mass；dot 大小按最终 carrier mass 编码。正确/错误两行使用同一 mass 尺度与颜色图例；未选中的位置用低对比灰点，避免把空白误认为图像边界。注明 `K=392, CLS excluded`，核对 unique count=392。右列可叠加与左列相同的裁剪图作浅背景；若数据/图片授权不允许对外传原图，就只导出网格版，绝不从无图底板猜测语义边界。
- 图中文字最小 8 pt，最终文件同时给矢量 PDF（照片嵌入可以是 raster）与 300-dpi PNG；用色盲友好的深蓝/橙/灰配色，灰度打印仍以线型/大小区分。任何尚无真实数据的预览必须显著标注 `SYNTHETIC MOCKUP / NOT AN EXPERIMENTAL RESULT`，不得放入投稿稿。
- 附录总体图只用全部 1,000 张预选图片：左为六步每图 routing-distance p50/p90 的分布（单位：patch），右为 28×28 最终 carrier selection frequency 热图（分母始终为 1,000）；可在图注另报每图 coverage mean/p95 与 top-10% mass share 的均值和分位数。正确/错误分组只作描述性附表，不以图中差异推断因果或语义保留。

学长/CC 机器是原始数据、checkpoint、原始 ImageNet 图片与完整内部 trace 的权威存放地。**建议回传 `DATA_HANDOFF.md` 定义的 L1 包**：成图/总表之外还有匿名逐图评测、1,000 图分层指标/最终 carrier，以及固定案例的实际路由边，使作者可在本地复算和重画。**不需要传出 checkpoint、原始逐图 logits、1000 份原始 NPZ 或含内部绝对路径的 manifest。** 图片背景是否能传出须先按公司与数据集政策确认；不获批准时使用 grid-only 版本。论文图中的任何数值必须由真实结果包产生，不能从预览图抄写。

### 8.2 样本集固定规则

主统计集固定为 1,000 张 ImageNet validation images，每个 ground-truth class 一张。为避免挑图：

1. 对每个 class 的每个 relative path 计算 `sha256("20260923:" + relative_path)`；
2. 每类选择 digest 字典序最小的一张；
3. 保存完整 `sample_manifest.csv`，包含 class、relative path、digest；
4. 在看模型预测和 trace 之前冻结该 manifest。

聚合统计必须使用全部 1,000 张。正文图固定展示上述一个正确和一个错误案例。另可导出 12 张审查联系表：前 8 张取全体 manifest digest 最小者；另外 2 个正确案例和 2 个错误案例分别按 digest 最小规则选择。正文案例不得从这 12 张中人工改选“最好看”的样本。若图像不能离开 CC，则联系表仅供 CC 本地审阅，外传 grid-only 图。

### 8.3 Trace-off/on 等价性 gate

在正式 1,000 图 trace 前，对固定 8 张图运行：

1. trace 完全关闭的 native forward；
2. 打开 detached/passive hooks 的 forward；
3. 比较 logits、predictions、top-K indices、carrier mass。

要求：

- predicted class 完全一致；
- top-K indices 完全一致；
- fp32 路径优先要求 logits 逐元素一致；
- fp16 若底层 kernel 存在可重复的微小误差，预先固定 `atol=5e-4, rtol=5e-4`，并保存实际 max-abs/max-rel；
- 每层 patch mass 总和相对初始 784 的误差要记录；fp32 目标绝对误差 ≤ `1e-3`，fp16 若超过则切换 trace 统计到 fp32，不得放宽后静默接受；
- trace 只能 `detach()`/copy 已产生的张量，不能改写共享 `_tome_info`、RNG、grouping 或 forward inputs。

任一 gate 失败时，不得生成论文图；先修正 trace 实现。

### 8.4 每层必须采集的原始字段

对 6 个 local routing steps 和最终 gather，至少保存：

- image ID、layer index；
- 原始 28×28 patch 坐标；
- donor indices、receiver indices；
- eligible mask；
- 实际 transport weight；
- donor/receiver 的 pre/post mass；
- 每条实际 routing edge 的 Euclidean grid distance；
- 当前 token mass vector；
- final top-K carrier indices、排序前后 indices；
- carrier center/center-of-mass；
- local 和 latent token counts。

为成图，必须在**实际 sparse transport 调用之前、`assign = assign * physical_mask` 之后**增加只读 trace tap，记录 step-3 的 `(donor_original_index, receiver_original_index, assign_postmask_ij)`，并保留六步同定义的聚合计数。普通 module forward hook 不一定看得到函数内部的边权；不能只依赖 `_tome_info` 中的 eligible mask。每条边的两个 index 都必须经一次小网格手工核验是原始 28×28 patch slot，而不是局部 A/B 排序后的临时下标。原始边权与最终 mass 记录只在 CC 机器本地保存；导出的正文案例边数组如获批准也仅含匿名 `image_id` 和网格坐标。

若当前实现只能提供 flattened center 而没有完整 source membership，必须在 metadata 标成 `center-only`，不能据此绘制完整成员归属图。若新增 full membership trace，必须先通过 trace-off/on gate。

推荐每图存压缩 NPZ，聚合结果存 Parquet/CSV/JSON；不要为 1,000 图保存无必要的全量 attention matrix。

### 8.5 聚合指标

对每层和总体报告：

- donor→receiver distance histogram，以及 mean/median/p90/p95；
- carrier mass 的 mean/std/p50/p90/p99、top-10% mass share、Gini coefficient；
- 392 个 carrier 在 28×28 网格上的 occupancy heatmap；
- 每图 carrier coverage：每个原始 patch 到最近 carrier 的距离 mean/p95；
- 7×7 等面积 spatial bins 的 occupied-bin ratio；
- boundary patches 与 center patches 的 carrier retention rate。Boundary 预定义为距图像网格边界 ≤2 patches；center 为其余 patches；
- top-K unique count、mass conservation error、NaN/Inf count；
- 按模型预测正确/错误分组的描述性统计。该分组只作诊断，不做因果结论。

overlay 至少包含：原图、28×28 grid、carrier 位置（点大小按 mass）、选定 routing edges 或中心迁移。颜色范围在所有图片间固定，caption 写清 layer 和统计定义。

附录的 carrier selection-frequency 热图按每个原始 slot 在 1,000 图中进入最终 top-K 的次数除以 1,000 计算；不使用 carrier center-of-mass 或 mass 加权。routing-distance 分布先在每张图、每一步对 `assign_postmask_ij>0` 的实际边计算 p50/p90，再按 1,000 张图显示分布，不将全部边拼接成一个“典型图片”。图稿必须附 `figure_data.csv`、`figure_config.json`、原始输出 digest 和渲染脚本 commit，以便仅凭可分享的数字重建附录图。

为避免不同实现给出不可比统计，统一使用以下定义：

- patch 坐标为原始 28×28 网格的整数 `(row, col)`；routing distance 是 donor 与 receiver 原始坐标间的欧氏距离，单位为 patch，不除以 28。
- occupancy 使用最终 top-K carrier 的原始选中位置；同一位置若异常重复只计一次，并单独报告 `topk_unique_count`。
- coverage 对 784 个原始 patch 分别计算到最近最终 carrier 原始位置的欧氏距离，再在图内求 mean/p95；不要使用学习到的 feature distance 或 carrier center-of-mass 代替。
- 7×7 spatial bin 将每个 4×4 patch block 作为一格；occupied-bin ratio 为至少含一个最终 carrier 的 bin 数除以 49。
- boundary retention rate 为边界集合中被选为最终 carrier 的 patch 数除以边界 patch 总数；center retention 同理。两者报告原始比率，不因集合大小不同做二次重权。
- 对非负 mass 向量排序为 `m_(1) <= ... <= m_(K)`，Gini 定义为 `sum_i (2i-K-1)m_(i) / (K * sum_i m_(i))`；若总 mass 非正则立即失败。
- top-10% mass share 使用 mass 最大的 `ceil(0.1*K)` 个 carrier 之和除以总 mass；K=392 时取 40 个。
- histogram 的 bin edges 必须在看分组结果之前固定并写入 `trace_config.json`；正确/错误分组共用完全相同的 edges。
- 所有跨图片汇总以 image 为统计单位：先计算每图指标，再报告 1,000 图的 mean/median/p10/p90；不得把不同图片的 edge 或 carrier 直接拼接后只给一个微平均值。

### 8.6 E6 必须交付

```text
e6_routing_trace/
  sample_manifest.csv
  trace_config.json
  equivalence_gate.json
  traces/<image_id>.npz
  layer_summary.csv
  image_summary.csv
  routing_distance_histogram.csv
  carrier_mass_histogram.csv
  figures/aggregate_*.pdf
  figures/overlay_<image_id>.png
  figures/paper_routing_main.pdf
  figures/paper_routing_main.png
  figures/paper_routing_population.pdf
  figures/paper_routing_population.png
  README.md
```

`README.md` 必须写清样本选择规则、trace 模式、字段定义、聚合公式、checkpoint hash 和限制。
**正式回传另建 `DATA_HANDOFF.md` 定义的根级 `share_packet/`**，其中包括匿名逐图指标、逐层直方图、最终 carrier 数组及固定案例的真实边数组。内部 `sample_manifest.csv` 的 relative paths、`traces/*.npz`、原图、checkpoint 和内部机器路径默认不外传。图中的匿名 `image_id` 与 E2/E3 共用，按 `DATA_HANDOFF.md` 的私钥 HMAC 规则在 CC 侧生成；密钥和映射不外传。回传包内的 `figures/caption_draft.md` 必须写明模型、数据 split、1,000 图选择规则、step=3、最多 32 条边的确定性取法、carrier/mass 编码、trace 等价性、图像授权状态及“不是语义分割/因果证据”的限制。若图片授权不通过，`paper_routing_main.*` 必须为 grid-only，不得因为缺图片就省去其它统计和收据。

### 8.7 可直接发给 CC 的可视化任务摘要

> 请用公司侧现有最终 MergeNet EMA checkpoint 按 E6 全文做一次**不训练的被动 trace**。先在固定 8 张图上证明 trace 开/关的 logits、top-K 和 mass 等价，再按固定 hash 规则从 ImageNet-val 每类选 1 张，共 1,000 张，输出六步真实 routing 边/距离、最终 392 个 carrier/mass 的聚合统计。正文图只画预先规定的一个正确例与一个错误例：第 3 步每象限权重最高 8 条实际 transport 边 + 最终 carrier 网格；附录图汇总全部 1,000 张。请在公司侧完成原始 trace 和绘图，并按 `DATA_HANDOFF.md` 尽量给我们 L1 匿名逐图数值包，使我们能在本地复算、重画，而不只是收到成图。不要发送 checkpoint、ImageNet 原始图、逐图 logits、原始 NPZ 或内部路径。如果图片背景不可分享，请交 grid-only 图并注明。预览 PNG 只是版式示意，里面没有真实实验数据。

## 9. 最终自动汇总与论文可用表

`experiments/summarize.py` 应只读取上述 JSON/CSV/NPZ，不重新运行模型，生成：

1. `e1_dtem_endpoint.csv`：DTEM best/final EMA 与 token trajectory；
2. `e2_main_table.csv`：五方法 top-1/top-5、batch-1 latency、batch-64 throughput、allocated/reserved memory；
3. `e3_component_ablation.csv`：四 variants 与相对 native delta/CI；
4. `e6_routing_summary.csv`：层级 routing distance、mass、coverage；
5. `DELIVERY_MANIFEST.json`：所有交付文件的相对路径、大小和 SHA-256。

主表中的数值精度：accuracy 保留三位小数，latency 保留两位毫秒，memory 保留一位 MiB 或两位 GiB；原始文件保留全精度。不要在原始 JSON 中预先四舍五入。

## 10. 完成定义

只有同时满足以下条件才算完成：

- E1 300 epochs 完整结束，DTEM hard eval 达到实际 392 patches；
- E2 五方法 accuracy gate、token gate、独占 GPU timing 和 memory 全部完成；
- E3 四行均使用同一 MergeNet checkpoint，单因素定义通过 integrity checks；
- E6 trace-off/on 等价，1,000 图 manifest、正文/附录两张真实结果图和按 `DATA_HANDOFF.md` 逐项标记 L0/L1/L2 状态的脱敏 `share_packet/` 完整；
- 所有结果都能从原始输出自动重建；
- 所有 checkpoint/source/environment/data identities 有 receipt；
- 失败、重跑和协议偏差均在 manifest 中保留，不覆盖；
- 返回仓库代码 commit 与结果目录，不需要上传 checkpoint 或 ImageNet 数据。

若某个任务因源码兼容、checkpoint 缺失、GPU 不独占或数据权限无法完成，应在对应输出目录写 `BLOCKED.md`，记录已执行命令、完整错误、环境和下一步；不得用旧数据、估算值或其他设置的结果填空。
