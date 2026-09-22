# MergeNet

MergeNet is a vision transformer with differentiable spatial routing and a
physical token bottleneck. It preserves the original two-dimensional patch
geometry while learning local token transport, gathers an exact number of
carrier tokens, and processes the compressed sequence with a latent
Transformer.

This repository contains the MergeNet model, its ImageNet-1K trainer, the
reported MN-L2-R3 configuration, and focused implementation tests. Pretrained
checkpoints, internal experiment campaigns, and cluster-specific launch files
are intentionally not included.

## Architecture

The default model uses a DeiT-S-scale encoder:

```text
224x224 image
  -> 28x28 patch grid (patch size 8) + CLS
  -> 6 full-grid local blocks
  -> spatial routing on radius-3 neighborhoods
  -> exact gather: 784 patches -> 392 carriers
  -> full-grid recovery cross-attention
  -> 6 latent Transformer blocks
  -> classifier
```

Routing candidates are restricted by Euclidean distance on the original patch
grid. The sparse forward path scores only eligible neighbors; a dense reference
path is retained for numerical parity tests. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed routing contract.

## Installation

The pinned environment targets Linux, Python 3.10, PyTorch 2.6.0 with CUDA
12.4, torchvision 0.21.0, timm 0.9.11, Triton 3.2.0, and FlashAttention
2.7.4.post1.

```bash
conda env create -f environment.yml
conda activate mergenet-in1k
python -m pip install -r requirements-lock.txt
export OPENTOME_SKIP_OPTIONAL_NLP=1
```

## Smoke tests

The CPU checks do not require a dataset or checkpoint:

```bash
python tests/test_release_contract.py
python tests/test_accumulation_schedule.py
python tests/test_dtem_spatial_mask.py
python tests/test_imagefolder_loader.py
```

With a CUDA GPU and FlashAttention installed:

```bash
python tests/test_dtem_spatial_sparse.py
python tests/test_biased_local_attention.py
python tests/test_model_smoke.py
```

## ImageNet-1K training

Prepare ImageNet in standard ImageFolder form:

```text
/path/to/imagenet/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

Launch the reported 300-epoch MN-L2-R3 recipe on one node:

```bash
DATA_DIR=/path/to/imagenet \
OUTPUT_DIR=./outputs \
GPUS=0,1,2,3,4,5,6,7 \
bash scripts/train_imagenet_300e.sh
```

The launcher derives gradient accumulation to keep the effective global batch
at 1,024. `BATCH_SIZE`, `GLOBAL_BATCH`, `RUN_NAME`, and `RESUME` may be supplied
as environment variables. The full scientific configuration is in
[`configs/mergenet_l2_spatial_r3.yaml`](configs/mergenet_l2_spatial_r3.yaml).

The remaining paper experiments (matched DTEM, common accuracy/latency/memory,
component interventions, and routing traces) are specified in
[`EXPERIMENTS.md`](EXPERIMENTS.md). To validate launcher arguments without
starting training, add `DRY_RUN=1` to the command above.

For direct control, invoke the trainer with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=8 \
  trainer/classification/in1k_trainer.py \
  --config configs/mergenet_l2_spatial_r3.yaml \
  --data_dir /path/to/imagenet \
  --batch_size 64 --update_freq 2 \
  --output ./outputs --experiment mergenet_l2_r3
```

## License

The code is derived from OpenToMe and is released under the Apache License 2.0.
