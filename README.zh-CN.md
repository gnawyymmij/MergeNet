# MergeNet

MergeNet 是一个带可微空间路由和真实 token 瓶颈的视觉 Transformer。模型在
原始二维 patch 网格上学习局部 token 传输，精确收集固定数量的 carrier tokens，
再由 latent Transformer 处理压缩后的序列。

本仓库仅保留 MergeNet 模型、ImageNet-1K 训练器、论文采用的 MN-L2-R3 配置和
必要测试。预训练权重、内部实验 campaign、集群调度及协作交接材料不属于公开
仓库内容。

## 模型结构

默认模型采用 DeiT-S 规模：

```text
224x224 图像
  -> 28x28 patch 网格（patch size 8）+ CLS
  -> 6 个 full-grid local blocks
  -> 原始二维网格上的 R3 空间路由
  -> 精确 gather：784 patches -> 392 carriers
  -> full-grid recovery cross-attention
  -> 6 个 latent Transformer blocks
  -> 分类头
```

详细路由约束见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 环境安装

```bash
conda env create -f environment.yml
conda activate mergenet-in1k
python -m pip install -r requirements-lock.txt
export OPENTOME_SKIP_OPTIONAL_NLP=1
```

锁定环境为 Python 3.10、PyTorch 2.6.0 + CUDA 12.4、torchvision 0.21.0、
timm 0.9.11、Triton 3.2.0 和 FlashAttention 2.7.4.post1。

## 无数据 smoke test

CPU 测试：

```bash
python tests/test_release_contract.py
python tests/test_accumulation_schedule.py
python tests/test_dtem_spatial_mask.py
python tests/test_imagefolder_loader.py
```

CUDA 和 FlashAttention 可用时再运行：

```bash
python tests/test_dtem_spatial_sparse.py
python tests/test_biased_local_attention.py
python tests/test_model_smoke.py
```

## ImageNet-1K 训练

数据应采用标准 ImageFolder 布局：

```text
/path/to/imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

单机训练入口：

```bash
DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=./outputs \
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/train_imagenet_300e.sh
```

启动脚本会通过梯度累积保持 effective global batch 为 1,024。也可以通过环境
变量设置 `BATCH_SIZE`、`GLOBAL_BATCH`、`RUN_NAME` 和 `RESUME`。完整配置位于
[`configs/mergenet_l2_spatial_r3.yaml`](configs/mergenet_l2_spatial_r3.yaml)。

论文仍需补充的 DTEM、统一效率评测、组件消融和路由可视化实验见
[`EXPERIMENTS.md`](EXPERIMENTS.md)。如只想检查启动参数而不开始训练，可在上述
命令前增加 `DRY_RUN=1`。

本仓库不提供预训练 checkpoint；checkpoint 会在训练过程中由 trainer 正常生成。

## License

代码基于 OpenToMe，使用 Apache License 2.0。
