# Architecture and routing contract

## MN-L2-R3 at 224×224

| Component | Paper setting |
|---|---|
| Learnable parameters | 23,097,064 (matched DeiT-S/8: 22,055,272) |
| Patch embedding | 8×8 stride-8, 28×28 grid, width 384 |
| Local encoder | 6 LocalBlocks, 6 heads, MLP ratio 4 |
| LocalBlock attention | sequence-local window 16; CLS global |
| Routing metric | learned 64-D donor/receiver features |
| Partition | train: random per sample; eval: alternating per layer |
| Merge budget | λ=2, 784→392 patch carriers |
| Routing geometry | original-grid Euclidean distance ≤3 |
| Production backend | sparse indexed score and fused transport |
| Recovery | pre-norm cross-attention residual, zero-init output projection |
| Latent encoder | 6 global Transformer blocks |
| Classifier | final CLS token, 1,000-way linear head |

The model has 12 Transformer blocks in total (6 local + 6 latent). Compression
occurs in the local stage; `total_merge_latent=0`, so the latent sequence is not
further merged.

## True-grid routing

After removing the prefix token, patch slot `p` is mapped using the actual grid
width:

```text
row(p) = p // grid_width
col(p) = p % grid_width
```

For radius `R`, a donor/receiver candidate is valid iff its Euclidean distance
is at most `R`. Eligibility is applied before every operation that can affect
the selected route: ranking, donor selection, row aggregation, log-sum-exp,
and receiver softmax. Boundary patches naturally have fewer candidates.

The historical `dtem_window_size=8` compares flattened indices. It therefore
admits row-wrap edges and changes its physical meaning with image width. The
paper model sets `dtem_window_size=0` and uses `dtem_spatial_radius=3.0`.

## Sparse and dense paths

The dense backend materializes the full donor×receiver score and masks invalid
edges. It is the mathematical oracle used in parity tests.

The sparse backend:

1. precomputes a padded neighbor table from the 2-D grid;
2. maps receivers into eligible neighbor slots;
3. computes indexed dot products using Triton when supported;
4. keeps assignment as `[batch, donors, valid-neighbor-slots]`;
5. performs fused indexed transport/scatter without a dense score or
   assignment tensor during forward.

Initialization currently constructs a dense boolean adjacency once and
converts it into the padded neighbor table. The forward-path claim above is
therefore intentionally narrower than a claim of fully sparse initialization.

Geometry buffers are non-persistent, so dense and sparse models retain the
same learnable state schema and old checkpoints load strictly.

## Empty rows and fixed-budget behavior

Invalid candidates have zero assignment mass. Rows with no eligible receiver
remain finite and unmerged; they never fall back to an out-of-radius token. If
the legal graph cannot satisfy the fixed merge target, execution fails closed.

## What is and is not two-dimensional

DTEM merge routing uses real `(row, col)` coordinates. `LocalBlock` attention
still uses its independent `local_block_window=16` sequence-local kernel. The
completed ablation changes only routing geometry. Converting LocalBlock
attention to a 2-D kernel would be a different model and requires new training.

## Evaluation modes

Training uses `random_per_sample` grouping to vary donor/receiver partitions.
Evaluation uses deterministic `alternating_per_layer_fast` grouping. The
dense/sparse numerical parity tests cover the routing implementation separately.
