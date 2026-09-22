# ------ jinxin modified ------ #
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import math
import operator
from timm.layers import Mlp, DropPath, use_fused_attn
from timm.models.vision_transformer import VisionTransformer, LayerScale
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from opentome.utils.thetopk import ThreTopK
from functools import partial

from opentome.tome.tome import (
    bipartite_soft_matching,
    merge_source_matrix,
    merge_source_map,
    merge_wavg,
    parse_r,
)
from opentome.timm import Attention, Block
from opentome.timm.bias_local_attn import biased_local_attention

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the dependency-free fallback
    triton = None
    tl = None


DTEM_EVAL_GROUPING_MODES = (
    "alternating",
    "alternating_per_layer",
    "alternating_per_layer_fast",
    "seeded",
)

DTEM_TRAIN_GROUPING_MODES = (
    "random_per_sample",
    *DTEM_EVAL_GROUPING_MODES,
)


DTEM_SPATIAL_METRICS = ("euclidean",)
DTEM_SPATIAL_BACKENDS = ("dense", "sparse")


def build_patch_spatial_adjacency(
    grid_size,
    radius,
    metric="euclidean",
    device=None,
):
    """Return the allowed patch-pair matrix for a true 2-D distance radius.

    ``radius`` is measured between patch centres in patch-grid units.  This is
    deliberately separate from DTEM's historical ``window_size``, which is a
    one-dimensional window after row-major flattening.
    """
    if not isinstance(grid_size, (tuple, list, torch.Size)) or len(grid_size) != 2:
        raise ValueError(f"grid_size must be a (height, width) pair, got {grid_size!r}")
    if any(isinstance(value, bool) for value in grid_size):
        raise ValueError(f"grid_size values must be integers, got {grid_size!r}")
    try:
        grid_h, grid_w = (operator.index(grid_size[0]), operator.index(grid_size[1]))
    except TypeError as exc:
        raise ValueError(
            f"grid_size values must be integers, got {grid_size!r}"
        ) from exc
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"grid_size values must be positive, got {(grid_h, grid_w)!r}")
    if isinstance(radius, bool):
        raise ValueError("radius must be a positive finite number, not bool")
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be a positive finite number, got {radius!r}")
    if metric not in DTEM_SPATIAL_METRICS:
        raise ValueError(
            f"Unsupported DTEM spatial metric {metric!r}; expected one of "
            f"{DTEM_SPATIAL_METRICS}"
        )

    patch_index = torch.arange(grid_h * grid_w, device=device, dtype=torch.long)
    rows = torch.div(patch_index, grid_w, rounding_mode="floor")
    cols = patch_index.remainder(grid_w)
    delta_row = rows[:, None] - rows[None, :]
    delta_col = cols[:, None] - cols[None, :]
    # Squared Euclidean distance avoids a sqrt and makes the inclusive boundary
    # exact for integer radii.
    return delta_row.square() + delta_col.square() <= radius * radius


def adjacency_to_padded_neighbors(adjacency):
    """Convert a square adjacency matrix to a row-major padded neighbor list.

    The returned ``indices`` and ``valid`` tensors both have shape ``[P, K]``,
    where ``K`` is the largest neighborhood in the grid.  Valid neighbors stay
    in ascending patch-index order.  That order matches the subsequence of the
    dense implementation because DTEM sorts each receiver partition by its
    absolute token index before scoring.
    """
    if not isinstance(adjacency, torch.Tensor) or adjacency.ndim != 2:
        raise ValueError("adjacency must be a rank-2 tensor")
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] <= 0:
        raise ValueError(f"adjacency must be non-empty and square, got {adjacency.shape}")
    if adjacency.dtype != torch.bool:
        raise ValueError(f"adjacency must have bool dtype, got {adjacency.dtype}")

    counts = adjacency.sum(dim=-1, dtype=torch.long)
    max_neighbors = int(counts.max().item())
    if max_neighbors <= 0:
        raise ValueError("every patch must have at least one spatial neighbor")

    rows, cols = adjacency.nonzero(as_tuple=True)
    row_starts = counts.cumsum(dim=0) - counts
    slots = torch.arange(rows.numel(), device=adjacency.device) - torch.repeat_interleave(
        row_starts, counts
    )
    indices = torch.zeros(
        adjacency.shape[0], max_neighbors, device=adjacency.device, dtype=torch.long
    )
    valid = torch.zeros_like(indices, dtype=torch.bool)
    indices[rows, slots] = cols
    valid[rows, slots] = True
    return indices, valid


def build_patch_spatial_neighbors(
    grid_size,
    radius,
    metric="euclidean",
    device=None,
):
    """Build the padded local-neighborhood representation of a 2-D radius."""
    adjacency = build_patch_spatial_adjacency(
        grid_size=grid_size,
        radius=radius,
        metric=metric,
        device=device,
    )
    return adjacency_to_padded_neighbors(adjacency)


def gather_patch_spatial_neighbors(
    neighbor_patch_indices,
    neighbor_patch_valid,
    a_orig_idx,
    b_orig_idx,
    num_prefix_tokens=1,
    *,
    receivers_are_sorted=False,
    validate=True,
):
    """Map static patch neighborhoods to a batch's receiver-slot indices.

    Unlike :func:`gather_patch_spatial_mask`, this never creates a donor by
    receiver matrix.  It builds a ``[B, Na, K]`` index list, with ``K`` bounded
    by the number of grid points inside the 2-D radius, and a matching validity
    mask.  Invalid entries use receiver slot zero only as a safe gather target;
    callers must keep them masked with the returned boolean tensor.
    """
    if (
        not isinstance(neighbor_patch_indices, torch.Tensor)
        or not isinstance(neighbor_patch_valid, torch.Tensor)
        or neighbor_patch_indices.ndim != 2
        or neighbor_patch_valid.ndim != 2
        or neighbor_patch_indices.shape != neighbor_patch_valid.shape
    ):
        raise ValueError("neighbor indices and validity must share shape [P, K]")
    if neighbor_patch_indices.shape[0] <= 0 or neighbor_patch_indices.shape[1] <= 0:
        raise ValueError("neighbor table must be non-empty")
    if neighbor_patch_indices.dtype != torch.long:
        raise ValueError("neighbor_patch_indices must have torch.long dtype")
    if neighbor_patch_valid.dtype != torch.bool:
        raise ValueError("neighbor_patch_valid must have torch.bool dtype")
    if not isinstance(a_orig_idx, torch.Tensor) or not isinstance(b_orig_idx, torch.Tensor):
        raise TypeError("a_orig_idx and b_orig_idx must be tensors")
    if a_orig_idx.ndim != 2 or b_orig_idx.ndim != 2:
        raise ValueError("a_orig_idx and b_orig_idx must both have shape [B, N]")
    if a_orig_idx.shape[0] != b_orig_idx.shape[0]:
        raise ValueError("a_orig_idx and b_orig_idx must have the same batch size")
    devices = {
        neighbor_patch_indices.device,
        neighbor_patch_valid.device,
        a_orig_idx.device,
        b_orig_idx.device,
    }
    if len(devices) != 1:
        raise ValueError("neighbor tables and token indices must be on the same device")
    if isinstance(num_prefix_tokens, bool):
        raise ValueError("num_prefix_tokens must be a non-negative integer")
    try:
        prefix = operator.index(num_prefix_tokens)
    except TypeError as exc:
        raise ValueError(
            "num_prefix_tokens must be a non-negative integer"
        ) from exc
    if prefix < 0:
        raise ValueError("num_prefix_tokens must be a non-negative integer")

    patch_count = neighbor_patch_indices.shape[0]
    a_patch = a_orig_idx.long() - prefix
    b_patch = b_orig_idx.long() - prefix
    if validate:
        for name, patch_idx in (("a_orig_idx", a_patch), ("b_orig_idx", b_patch)):
            if patch_idx.numel() and (
                int(patch_idx.min().item()) < 0
                or int(patch_idx.max().item()) >= patch_count
            ):
                raise ValueError(f"{name} contains a non-patch or out-of-grid token index")
        if b_patch.numel():
            sorted_b = b_patch.sort(dim=-1).values
            if bool((sorted_b[:, 1:] == sorted_b[:, :-1]).any().item()):
                raise ValueError("b_orig_idx contains duplicate receiver positions")

    batch, num_receivers = b_patch.shape
    if receivers_are_sorted and num_receivers > 0:
        # _merge_train sorts receiver slots by absolute patch position. Batched
        # searchsorted avoids constructing/scattering a [B, P] inverse lookup
        # and is measurably faster for the formal ImageNet 28x28 (224/p8) grid.
        patch_neighbors = neighbor_patch_indices[a_patch]
        static_valid = neighbor_patch_valid[a_patch]
        flat_neighbors = patch_neighbors.flatten(1).contiguous()
        positions = torch.searchsorted(
            b_patch.contiguous(), flat_neighbors
        ).view_as(patch_neighbors)
        safe_positions = positions.clamp_max(num_receivers - 1)
        found = b_patch.gather(
            dim=1, index=safe_positions.flatten(1)
        ).view_as(patch_neighbors).eq(patch_neighbors)
        receiver_valid = static_valid & positions.lt(num_receivers) & found
        return safe_positions, receiver_valid

    receiver_lookup = torch.full(
        (batch, patch_count), -1, device=b_patch.device, dtype=torch.long
    )
    receiver_slots = torch.arange(
        num_receivers, device=b_patch.device, dtype=torch.long
    ).unsqueeze(0).expand(batch, -1)
    receiver_lookup.scatter_(dim=1, index=b_patch, src=receiver_slots)

    patch_neighbors = neighbor_patch_indices[a_patch]
    static_valid = neighbor_patch_valid[a_patch]
    receiver_indices = receiver_lookup.gather(
        dim=1, index=patch_neighbors.flatten(1)
    ).view_as(patch_neighbors)
    receiver_valid = static_valid & receiver_indices.ge(0)
    return receiver_indices.clamp_min(0), receiver_valid


def densify_sparse_assignment(
    assignment,
    receiver_indices,
    receiver_valid,
    num_receivers,
):
    """Reference/debug conversion from a narrow assignment to a dense matrix."""
    if assignment.ndim != 3:
        raise ValueError("assignment must have shape [B, Na, K]")
    if receiver_indices.shape != assignment.shape or receiver_valid.shape != assignment.shape:
        raise ValueError("receiver indices/validity must match assignment shape")
    if receiver_indices.dtype != torch.long or receiver_valid.dtype != torch.bool:
        raise ValueError("receiver indices must be long and validity must be bool")
    if isinstance(num_receivers, bool):
        raise ValueError("num_receivers must be a positive integer")
    try:
        num_receivers = operator.index(num_receivers)
    except TypeError as exc:
        raise ValueError("num_receivers must be a positive integer") from exc
    if num_receivers <= 0:
        raise ValueError("num_receivers must be a positive integer")
    if receiver_indices.numel() and int(receiver_indices.max().item()) >= num_receivers:
        raise ValueError("receiver index is out of range")
    dense = assignment.new_zeros(*assignment.shape[:2], num_receivers)
    dense.scatter_add_(
        dim=-1,
        index=receiver_indices,
        src=assignment * receiver_valid.to(assignment.dtype),
    )
    return dense


if triton is not None:
    @triton.jit
    def _sparse_neighbor_dot_forward_kernel(
        a_ptr,
        b_ptr,
        receiver_ptr,
        valid_ptr,
        output_ptr,
        total_edges,
        num_donors: tl.constexpr,
        num_receivers: tl.constexpr,
        neighbors: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
        in_bounds = edge < total_edges
        batch = edge // (num_donors * neighbors)
        donor = (edge % (num_donors * neighbors)) // neighbors
        receiver = tl.load(receiver_ptr + edge, mask=in_bounds, other=0)
        edge_valid = in_bounds & tl.load(valid_ptr + edge, mask=in_bounds, other=0)
        channel = tl.arange(0, BLOCK_CHANNELS)
        channel_valid = channel < channels
        matrix_mask = edge_valid[:, None] & channel_valid[None, :]
        a_offset = (batch * num_donors + donor)[:, None] * channels + channel[None, :]
        b_offset = (batch * num_receivers + receiver)[:, None] * channels + channel[None, :]
        a_value = tl.load(a_ptr + a_offset, mask=matrix_mask, other=0.0)
        b_value = tl.load(b_ptr + b_offset, mask=matrix_mask, other=0.0)
        score = tl.sum(a_value * b_value, axis=1)
        tl.store(output_ptr + edge, score, mask=in_bounds)


    @triton.jit
    def _sparse_neighbor_dot_backward_kernel(
        a_ptr,
        b_ptr,
        receiver_ptr,
        valid_ptr,
        grad_score_ptr,
        grad_a_ptr,
        grad_b_ptr,
        total_edges,
        num_donors: tl.constexpr,
        num_receivers: tl.constexpr,
        neighbors: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
        in_bounds = edge < total_edges
        batch = edge // (num_donors * neighbors)
        donor = (edge % (num_donors * neighbors)) // neighbors
        receiver = tl.load(receiver_ptr + edge, mask=in_bounds, other=0)
        edge_valid = in_bounds & tl.load(valid_ptr + edge, mask=in_bounds, other=0)
        grad_score = tl.load(grad_score_ptr + edge, mask=edge_valid, other=0.0)
        channel = tl.arange(0, BLOCK_CHANNELS)
        channel_valid = channel < channels
        matrix_mask = edge_valid[:, None] & channel_valid[None, :]
        a_offset = (batch * num_donors + donor)[:, None] * channels + channel[None, :]
        b_offset = (batch * num_receivers + receiver)[:, None] * channels + channel[None, :]
        a_value = tl.load(a_ptr + a_offset, mask=matrix_mask, other=0.0)
        b_value = tl.load(b_ptr + b_offset, mask=matrix_mask, other=0.0)
        tl.atomic_add(
            grad_a_ptr + a_offset,
            grad_score[:, None] * b_value,
            mask=matrix_mask,
        )
        tl.atomic_add(
            grad_b_ptr + b_offset,
            grad_score[:, None] * a_value,
            mask=matrix_mask,
        )


    @triton.jit
    def _sparse_spatial_score_forward_kernel(
        a_ptr,
        b_ptr,
        neighbor_patch_ptr,
        neighbor_valid_ptr,
        donor_patch_ptr,
        receiver_patch_ptr,
        score_ptr,
        receiver_index_ptr,
        receiver_valid_ptr,
        total_edges,
        num_donors: tl.constexpr,
        num_receivers: tl.constexpr,
        neighbors: tl.constexpr,
        channels: tl.constexpr,
        BINARY_STEPS: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
        in_bounds = edge < total_edges
        batch = edge // (num_donors * neighbors)
        donor = (edge % (num_donors * neighbors)) // neighbors
        neighbor_slot = edge % neighbors
        donor_patch = tl.load(
            donor_patch_ptr + batch * num_donors + donor,
            mask=in_bounds,
            other=0,
        )
        table_offset = donor_patch * neighbors + neighbor_slot
        neighbor_patch = tl.load(
            neighbor_patch_ptr + table_offset, mask=in_bounds, other=0
        )
        static_valid = in_bounds & tl.load(
            neighbor_valid_ptr + table_offset, mask=in_bounds, other=0
        )

        # b_orig_idx is sorted by _merge_train. Find the receiver slot for the
        # static neighbor in-register, fusing mapping and indexed dot-product
        # into one kernel without a [B, P] lookup or candidate-index gather.
        lower = tl.zeros((BLOCK_EDGES,), dtype=tl.int32)
        upper = tl.full((BLOCK_EDGES,), num_receivers, dtype=tl.int32)
        for _ in range(BINARY_STEPS):
            middle = (lower + upper) // 2
            safe_middle = tl.minimum(middle, num_receivers - 1)
            middle_patch = tl.load(
                receiver_patch_ptr + batch * num_receivers + safe_middle,
                mask=static_valid,
                other=num_receivers,
            )
            move_right = middle_patch < neighbor_patch
            lower = tl.where(move_right, middle + 1, lower)
            upper = tl.where(move_right, upper, middle)

        safe_receiver = tl.minimum(lower, num_receivers - 1)
        matched_patch = tl.load(
            receiver_patch_ptr + batch * num_receivers + safe_receiver,
            mask=static_valid,
            other=-1,
        )
        edge_valid = static_valid & (lower < num_receivers) & (matched_patch == neighbor_patch)
        channel = tl.arange(0, BLOCK_CHANNELS)
        channel_valid = channel < channels
        matrix_mask = edge_valid[:, None] & channel_valid[None, :]
        a_offset = (batch * num_donors + donor)[:, None] * channels + channel[None, :]
        b_offset = (batch * num_receivers + safe_receiver)[:, None] * channels + channel[None, :]
        a_value = tl.load(a_ptr + a_offset, mask=matrix_mask, other=0.0)
        b_value = tl.load(b_ptr + b_offset, mask=matrix_mask, other=0.0)
        score = tl.sum(a_value * b_value, axis=1)
        tl.store(score_ptr + edge, score, mask=in_bounds)
        tl.store(receiver_index_ptr + edge, safe_receiver, mask=in_bounds)
        tl.store(receiver_valid_ptr + edge, edge_valid, mask=in_bounds)


    @triton.jit
    def _sparse_transport_forward_kernel(
        assignment_ptr,
        value_ptr,
        receiver_ptr,
        valid_ptr,
        output_ptr,
        total_edges,
        num_donors: tl.constexpr,
        num_receivers: tl.constexpr,
        neighbors: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
        in_bounds = edge < total_edges
        batch = edge // (num_donors * neighbors)
        donor = (edge % (num_donors * neighbors)) // neighbors
        receiver = tl.load(receiver_ptr + edge, mask=in_bounds, other=0)
        edge_valid = in_bounds & tl.load(valid_ptr + edge, mask=in_bounds, other=0)
        weight = tl.load(assignment_ptr + edge, mask=edge_valid, other=0.0)
        channel = tl.arange(0, BLOCK_CHANNELS)
        channel_valid = channel < channels
        matrix_mask = edge_valid[:, None] & channel_valid[None, :]
        value_offset = (batch * num_donors + donor)[:, None] * channels + channel[None, :]
        output_offset = (batch * num_receivers + receiver)[:, None] * channels + channel[None, :]
        value = tl.load(value_ptr + value_offset, mask=matrix_mask, other=0.0)
        tl.atomic_add(
            output_ptr + output_offset,
            weight[:, None] * value,
            mask=matrix_mask,
        )


    @triton.jit
    def _sparse_transport_backward_kernel(
        assignment_ptr,
        value_ptr,
        receiver_ptr,
        valid_ptr,
        grad_output_ptr,
        grad_assignment_ptr,
        grad_value_ptr,
        total_edges,
        num_donors: tl.constexpr,
        num_receivers: tl.constexpr,
        neighbors: tl.constexpr,
        channels: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK_EDGES + tl.arange(0, BLOCK_EDGES)
        in_bounds = edge < total_edges
        batch = edge // (num_donors * neighbors)
        donor = (edge % (num_donors * neighbors)) // neighbors
        receiver = tl.load(receiver_ptr + edge, mask=in_bounds, other=0)
        edge_valid = in_bounds & tl.load(valid_ptr + edge, mask=in_bounds, other=0)
        weight = tl.load(assignment_ptr + edge, mask=edge_valid, other=0.0)
        channel = tl.arange(0, BLOCK_CHANNELS)
        channel_valid = channel < channels
        matrix_mask = edge_valid[:, None] & channel_valid[None, :]
        value_offset = (batch * num_donors + donor)[:, None] * channels + channel[None, :]
        output_offset = (batch * num_receivers + receiver)[:, None] * channels + channel[None, :]
        value = tl.load(value_ptr + value_offset, mask=matrix_mask, other=0.0)
        grad_output = tl.load(
            grad_output_ptr + output_offset, mask=matrix_mask, other=0.0
        )
        grad_weight = tl.sum(value * grad_output, axis=1)
        tl.store(grad_assignment_ptr + edge, grad_weight, mask=in_bounds)
        tl.atomic_add(
            grad_value_ptr + value_offset,
            weight[:, None] * grad_output,
            mask=matrix_mask,
        )


def _can_use_triton_sparse_dot(a, b, receiver_indices, receiver_valid):
    return (
        triton is not None
        and a.is_cuda
        # Triton's CUDA backend does not provide bf16 atomic_add on the
        # supported deployment stack; bf16 uses the portable fallback.
        and a.dtype in (torch.float16, torch.float32)
        and a.is_contiguous()
        and b.is_contiguous()
        and receiver_indices.is_contiguous()
        and receiver_valid.is_contiguous()
    )


def _launch_triton_sparse_dot(a, b, receiver_indices, receiver_valid):
    batch, num_donors, channels = a.shape
    num_receivers = b.shape[1]
    neighbors = receiver_indices.shape[-1]
    total_edges = receiver_indices.numel()
    scores = torch.empty(receiver_indices.shape, device=a.device, dtype=a.dtype)
    # A tile of 64 edges amortizes program overhead at the ImageNet 28x28 / R3
    # geometry while keeping a 64-channel metric tile within A100 registers.
    block_edges = 64
    block_channels = triton.next_power_of_2(channels)
    _sparse_neighbor_dot_forward_kernel[(triton.cdiv(total_edges, block_edges),)](
        a,
        b,
        receiver_indices,
        receiver_valid,
        scores,
        total_edges,
        num_donors,
        num_receivers,
        neighbors,
        channels,
        BLOCK_EDGES=block_edges,
        BLOCK_CHANNELS=block_channels,
    )
    return scores


def _launch_triton_sparse_spatial_score(
    a,
    b,
    neighbor_patch_indices,
    neighbor_patch_valid,
    donor_patch_indices,
    receiver_patch_indices,
):
    batch, num_donors, channels = a.shape
    num_receivers = b.shape[1]
    neighbors = neighbor_patch_indices.shape[-1]
    total_edges = batch * num_donors * neighbors
    shape = (batch, num_donors, neighbors)
    scores = torch.empty(shape, device=a.device, dtype=a.dtype)
    receiver_indices = torch.empty(shape, device=a.device, dtype=torch.long)
    receiver_valid = torch.empty(shape, device=a.device, dtype=torch.bool)
    block_edges = 64
    block_channels = triton.next_power_of_2(channels)
    # One more comparison is required when num_receivers is an exact power of
    # two (e.g. 8 receivers); int.bit_length gives that safe fixed trip count.
    binary_steps = max(num_receivers.bit_length(), 1)
    _sparse_spatial_score_forward_kernel[(triton.cdiv(total_edges, block_edges),)](
        a,
        b,
        neighbor_patch_indices,
        neighbor_patch_valid,
        donor_patch_indices,
        receiver_patch_indices,
        scores,
        receiver_indices,
        receiver_valid,
        total_edges,
        num_donors,
        num_receivers,
        neighbors,
        channels,
        BINARY_STEPS=binary_steps,
        BLOCK_EDGES=block_edges,
        BLOCK_CHANNELS=block_channels,
    )
    return scores, receiver_indices, receiver_valid


class _SparseNeighborDot(torch.autograd.Function):
    """Autograd wrapper for exact indexed donor/receiver dot products.

    The PyTorch implementation is intentionally the portable reference.  It is
    kept in a custom Function so the large ``[B, Na, K, C]`` gathered tensor is
    not retained by autograd; backward recomputes it and scatters gradients.
    """

    @staticmethod
    def forward(ctx, a, b, receiver_indices, receiver_valid):
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("a and b must have shape [B, N, C]")
        if a.shape[0] != b.shape[0] or a.shape[-1] != b.shape[-1]:
            raise ValueError("a and b batch/channel dimensions must match")
        expected = (a.shape[0], a.shape[1])
        if receiver_indices.shape[:2] != expected or receiver_valid.shape != receiver_indices.shape:
            raise ValueError("receiver neighbor tensors must have shape [B, Na, K]")
        if receiver_indices.dtype != torch.long or receiver_valid.dtype != torch.bool:
            raise ValueError("receiver indices must be long and validity must be bool")
        if receiver_indices.device != a.device or receiver_valid.device != a.device or b.device != a.device:
            raise ValueError("all sparse scorer inputs must share a device")
        use_triton = _can_use_triton_sparse_dot(
            a, b, receiver_indices, receiver_valid
        )
        if use_triton:
            scores = _launch_triton_sparse_dot(
                a, b, receiver_indices, receiver_valid
            )
        else:
            batch_index = torch.arange(a.shape[0], device=a.device)[:, None, None]
            candidates = b[batch_index, receiver_indices]
            scores = (a.unsqueeze(2) * candidates).sum(dim=-1)
            scores = scores * receiver_valid.to(scores.dtype)
        ctx.use_triton = use_triton
        ctx.save_for_backward(a, b, receiver_indices, receiver_valid)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        a, b, receiver_indices, receiver_valid = ctx.saved_tensors
        if ctx.use_triton:
            grad_scores = grad_scores.contiguous()
            batch, num_donors, channels = a.shape
            num_receivers = b.shape[1]
            neighbors = receiver_indices.shape[-1]
            total_edges = receiver_indices.numel()
            grad_a = torch.zeros_like(a)
            grad_b = torch.zeros_like(b)
            # Backward performs two atomic reductions per edge/channel. Small
            # edge tiles reduce intra-program address contention.
            block_edges = 4
            block_channels = triton.next_power_of_2(channels)
            _sparse_neighbor_dot_backward_kernel[(triton.cdiv(total_edges, block_edges),)](
                a,
                b,
                receiver_indices,
                receiver_valid,
                grad_scores,
                grad_a,
                grad_b,
                total_edges,
                num_donors,
                num_receivers,
                neighbors,
                channels,
                BLOCK_EDGES=block_edges,
                BLOCK_CHANNELS=block_channels,
            )
        else:
            grad_scores = grad_scores * receiver_valid.to(grad_scores.dtype)
            batch_index = torch.arange(a.shape[0], device=a.device)[:, None, None]
            candidates = b[batch_index, receiver_indices]
            grad_a = (grad_scores.unsqueeze(-1) * candidates).sum(dim=2)
            grad_b = torch.zeros_like(b)
            contribution = grad_scores.unsqueeze(-1) * a.unsqueeze(2)
            grad_b.scatter_add_(
                dim=1,
                index=receiver_indices.flatten(1).unsqueeze(-1).expand(-1, -1, b.shape[-1]),
                src=contribution.flatten(1, 2),
            )
        return grad_a, grad_b, None, None


def sparse_neighbor_dot(a, b, receiver_indices, receiver_valid):
    """Compute only the indexed spatial-neighborhood similarities."""
    if (
        not torch.is_grad_enabled()
        and _can_use_triton_sparse_dot(a, b, receiver_indices, receiver_valid)
    ):
        return _launch_triton_sparse_dot(a, b, receiver_indices, receiver_valid)
    return _SparseNeighborDot.apply(a, b, receiver_indices, receiver_valid)


def _can_use_triton_sparse_spatial_score(
    a,
    b,
    neighbor_patch_indices,
    neighbor_patch_valid,
    donor_patch_indices,
    receiver_patch_indices,
):
    return (
        triton is not None
        and a.is_cuda
        and a.dtype in (torch.float16, torch.float32)
        and all(
            tensor.is_contiguous()
            for tensor in (
                a,
                b,
                neighbor_patch_indices,
                neighbor_patch_valid,
                donor_patch_indices,
                receiver_patch_indices,
            )
        )
    )


class _SparseSpatialScore(torch.autograd.Function):
    """Fused sorted-receiver mapping and sparse dot-product on CUDA."""

    @staticmethod
    def forward(
        ctx,
        a,
        b,
        neighbor_patch_indices,
        neighbor_patch_valid,
        donor_patch_indices,
        receiver_patch_indices,
    ):
        use_triton = _can_use_triton_sparse_spatial_score(
            a,
            b,
            neighbor_patch_indices,
            neighbor_patch_valid,
            donor_patch_indices,
            receiver_patch_indices,
        )
        if use_triton:
            scores, receiver_indices, receiver_valid = (
                _launch_triton_sparse_spatial_score(
                    a,
                    b,
                    neighbor_patch_indices,
                    neighbor_patch_valid,
                    donor_patch_indices,
                    receiver_patch_indices,
                )
            )
        else:
            receiver_indices, receiver_valid = gather_patch_spatial_neighbors(
                neighbor_patch_indices,
                neighbor_patch_valid,
                donor_patch_indices,
                receiver_patch_indices,
                num_prefix_tokens=0,
                receivers_are_sorted=True,
                validate=False,
            )
            batch_index = torch.arange(a.shape[0], device=a.device)[:, None, None]
            candidates = b[batch_index, receiver_indices]
            scores = (a.unsqueeze(2) * candidates).sum(dim=-1)
            scores = scores * receiver_valid.to(scores.dtype)
        ctx.use_triton = use_triton
        ctx.save_for_backward(a, b, receiver_indices, receiver_valid)
        ctx.mark_non_differentiable(receiver_indices, receiver_valid)
        return scores, receiver_indices, receiver_valid

    @staticmethod
    def backward(ctx, grad_scores, _grad_indices, _grad_valid):
        a, b, receiver_indices, receiver_valid = ctx.saved_tensors
        if ctx.use_triton:
            grad_scores = grad_scores.contiguous()
            batch, num_donors, channels = a.shape
            num_receivers = b.shape[1]
            neighbors = receiver_indices.shape[-1]
            total_edges = receiver_indices.numel()
            grad_a = torch.zeros_like(a)
            grad_b = torch.zeros_like(b)
            block_edges = 4
            block_channels = triton.next_power_of_2(channels)
            _sparse_neighbor_dot_backward_kernel[(triton.cdiv(total_edges, block_edges),)](
                a,
                b,
                receiver_indices,
                receiver_valid,
                grad_scores,
                grad_a,
                grad_b,
                total_edges,
                num_donors,
                num_receivers,
                neighbors,
                channels,
                BLOCK_EDGES=block_edges,
                BLOCK_CHANNELS=block_channels,
            )
        else:
            grad_scores = grad_scores * receiver_valid.to(grad_scores.dtype)
            batch_index = torch.arange(a.shape[0], device=a.device)[:, None, None]
            candidates = b[batch_index, receiver_indices]
            grad_a = (grad_scores.unsqueeze(-1) * candidates).sum(dim=2)
            grad_b = torch.zeros_like(b)
            contribution = grad_scores.unsqueeze(-1) * a.unsqueeze(2)
            grad_b.scatter_add_(
                dim=1,
                index=receiver_indices.flatten(1).unsqueeze(-1).expand(
                    -1, -1, b.shape[-1]
                ),
                src=contribution.flatten(1, 2),
            )
        return grad_a, grad_b, None, None, None, None


def sparse_spatial_score(
    a,
    b,
    neighbor_patch_indices,
    neighbor_patch_valid,
    donor_patch_indices,
    receiver_patch_indices,
):
    """Return sparse scores and receiver slots for sorted DTEM partitions."""
    if (
        not torch.is_grad_enabled()
        and _can_use_triton_sparse_spatial_score(
            a,
            b,
            neighbor_patch_indices,
            neighbor_patch_valid,
            donor_patch_indices,
            receiver_patch_indices,
        )
    ):
        return _launch_triton_sparse_spatial_score(
            a,
            b,
            neighbor_patch_indices,
            neighbor_patch_valid,
            donor_patch_indices,
            receiver_patch_indices,
        )
    return _SparseSpatialScore.apply(
        a,
        b,
        neighbor_patch_indices,
        neighbor_patch_valid,
        donor_patch_indices,
        receiver_patch_indices,
    )


def _can_use_triton_sparse_transport(
    assignment, values, receiver_indices, receiver_valid
):
    return (
        triton is not None
        and assignment.is_cuda
        and assignment.dtype in (torch.float16, torch.float32)
        and values.dtype == assignment.dtype
        and all(
            tensor.is_contiguous()
            for tensor in (assignment, values, receiver_indices, receiver_valid)
        )
    )


def _launch_triton_sparse_transport(
    assignment, values, receiver_indices, receiver_valid, num_receivers
):
    batch, num_donors, neighbors = assignment.shape
    channels = values.shape[-1]
    total_edges = assignment.numel()
    output = values.new_zeros(batch, num_receivers, channels)
    # The scalar mass path is launch-bound, while the 384-channel feature path
    # is register/atomic-bound. Use separate tiles for the two formal calls.
    block_edges = 128 if channels <= 4 else 8
    block_channels = triton.next_power_of_2(channels)
    _sparse_transport_forward_kernel[(triton.cdiv(total_edges, block_edges),)](
        assignment,
        values,
        receiver_indices,
        receiver_valid,
        output,
        total_edges,
        num_donors,
        num_receivers,
        neighbors,
        channels,
        BLOCK_EDGES=block_edges,
        BLOCK_CHANNELS=block_channels,
    )
    return output


class _SparseTransport(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, assignment, values, receiver_indices, receiver_valid, num_receivers
    ):
        num_receivers = int(num_receivers)
        use_triton = _can_use_triton_sparse_transport(
            assignment, values, receiver_indices, receiver_valid
        )
        if use_triton:
            output = _launch_triton_sparse_transport(
                assignment,
                values,
                receiver_indices,
                receiver_valid,
                num_receivers,
            )
        else:
            contribution = (
                assignment.unsqueeze(-1)
                * values.unsqueeze(2)
                * receiver_valid.unsqueeze(-1).to(values.dtype)
            )
            output = values.new_zeros(
                assignment.shape[0], num_receivers, values.shape[-1]
            )
            output.scatter_add_(
                dim=1,
                index=receiver_indices.flatten(1).unsqueeze(-1).expand(
                    -1, -1, values.shape[-1]
                ),
                src=contribution.flatten(1, 2),
            )
        ctx.use_triton = use_triton
        ctx.num_receivers = num_receivers
        ctx.save_for_backward(
            assignment, values, receiver_indices, receiver_valid
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assignment, values, receiver_indices, receiver_valid = ctx.saved_tensors
        if ctx.use_triton:
            grad_output = grad_output.contiguous()
            batch, num_donors, neighbors = assignment.shape
            channels = values.shape[-1]
            total_edges = assignment.numel()
            grad_assignment = torch.empty_like(assignment)
            grad_values = torch.zeros_like(values)
            block_edges = 128 if channels <= 4 else 8
            block_channels = triton.next_power_of_2(channels)
            _sparse_transport_backward_kernel[(triton.cdiv(total_edges, block_edges),)](
                assignment,
                values,
                receiver_indices,
                receiver_valid,
                grad_output,
                grad_assignment,
                grad_values,
                total_edges,
                num_donors,
                ctx.num_receivers,
                neighbors,
                channels,
                BLOCK_EDGES=block_edges,
                BLOCK_CHANNELS=block_channels,
            )
        else:
            batch_index = torch.arange(
                assignment.shape[0], device=assignment.device
            )[:, None, None]
            gathered_grad = grad_output[batch_index, receiver_indices]
            gathered_grad = gathered_grad * receiver_valid.unsqueeze(-1).to(
                gathered_grad.dtype
            )
            grad_assignment = (gathered_grad * values.unsqueeze(2)).sum(dim=-1)
            grad_values = (
                gathered_grad * assignment.unsqueeze(-1)
            ).sum(dim=2)
        return grad_assignment, grad_values, None, None, None


def sparse_weighted_transport(
    assignment, values, receiver_indices, receiver_valid, num_receivers
):
    """Scatter weighted donor values without materializing [B, Na, K, C]."""
    if (
        not torch.is_grad_enabled()
        and _can_use_triton_sparse_transport(
            assignment, values, receiver_indices, receiver_valid
        )
    ):
        return _launch_triton_sparse_transport(
            assignment,
            values,
            receiver_indices,
            receiver_valid,
            int(num_receivers),
        )
    return _SparseTransport.apply(
        assignment,
        values,
        receiver_indices,
        receiver_valid,
        int(num_receivers),
    )


def gather_patch_spatial_mask(
    adjacency,
    a_orig_idx,
    b_orig_idx,
    num_prefix_tokens=1,
    *,
    validate=True,
):
    """Gather a batched donor/receiver mask from a patch adjacency matrix.

    ``a_orig_idx`` and ``b_orig_idx`` are absolute token-slot indices and thus
    include any CLS/prefix offset.  DTEM soft merge keeps every local token slot,
    so these coordinates remain stable across local merge layers.
    """
    if not isinstance(adjacency, torch.Tensor) or adjacency.ndim != 2:
        raise ValueError("adjacency must be a rank-2 tensor")
    if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] <= 0:
        raise ValueError(f"adjacency must be non-empty and square, got {adjacency.shape}")
    if adjacency.dtype != torch.bool:
        raise ValueError(f"adjacency must have bool dtype, got {adjacency.dtype}")
    if not isinstance(a_orig_idx, torch.Tensor) or not isinstance(b_orig_idx, torch.Tensor):
        raise TypeError("a_orig_idx and b_orig_idx must be tensors")
    if a_orig_idx.ndim != 2 or b_orig_idx.ndim != 2:
        raise ValueError("a_orig_idx and b_orig_idx must both have shape [B, N]")
    if a_orig_idx.shape[0] != b_orig_idx.shape[0]:
        raise ValueError("a_orig_idx and b_orig_idx must have the same batch size")
    if adjacency.device != a_orig_idx.device or adjacency.device != b_orig_idx.device:
        raise ValueError("adjacency and token indices must be on the same device")
    if isinstance(num_prefix_tokens, bool):
        raise ValueError("num_prefix_tokens must be a non-negative integer")
    try:
        prefix = operator.index(num_prefix_tokens)
    except TypeError as exc:
        raise ValueError(
            "num_prefix_tokens must be a non-negative integer"
        ) from exc
    if prefix < 0:
        raise ValueError("num_prefix_tokens must be a non-negative integer")
    a_patch = a_orig_idx.long() - prefix
    b_patch = b_orig_idx.long() - prefix

    if validate:
        patch_count = adjacency.shape[0]
        if a_patch.numel() and (
            int(a_patch.min().item()) < 0 or int(a_patch.max().item()) >= patch_count
        ):
            raise ValueError("a_orig_idx contains a non-patch or out-of-grid token index")
        if b_patch.numel() and (
            int(b_patch.min().item()) < 0 or int(b_patch.max().item()) >= patch_count
        ):
            raise ValueError("b_orig_idx contains a non-patch or out-of-grid token index")

    return adjacency[a_patch.unsqueeze(-1), b_patch.unsqueeze(-2)]


def build_dtem_eval_group_indices(
    n_tokens,
    batch_size,
    device,
    mode="alternating_per_layer",
    seed=0,
    layer_index=0,
):
    """Build deterministic donor/receiver groups for DTEM evaluation.

    ``alternating`` keeps the same donor parity at every layer. On an even-width
    row-major patch grid this can create a strong stripe prior in accumulated
    token mass, so it is retained only for checkpoint diagnosis.

    ``alternating_per_layer`` is the default deterministic inference policy and
    swaps the donor parity on odd local layers. ``alternating_per_layer_fast``
    builds the exact same partition, but lets the merge implementation use a
    strided slice/interleave path instead of the generic sort/gather/restore
    path. It changes no learned parameter and is intended for checkpoint-
    compatible efficiency probes.
    ``seeded`` uses a fixed pseudo-random permutation that changes by local
    layer. The same permutation is shared across the batch so the prediction of
    an example does not depend on its batch slot. The permutation is generated
    on CPU to make the grouping independent of the CUDA device in use.

    Historical training still draws an independent random partition for every
    sample and local layer. Explicit deterministic training protocols reuse
    this helper so their train/eval partition is exactly identical.
    """
    n_tokens = int(n_tokens)
    batch_size = int(batch_size)
    layer_index = int(layer_index)
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be non-negative, got {n_tokens}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if mode not in DTEM_EVAL_GROUPING_MODES:
        raise ValueError(
            f"Unsupported DTEM eval grouping {mode!r}; "
            f"expected one of {DTEM_EVAL_GROUPING_MODES}"
        )

    base_idx = torch.arange(n_tokens, device=device, dtype=torch.long)
    if mode == "alternating":
        permutation = torch.cat([base_idx[1::2], base_idx[0::2]], dim=0)
    elif mode in ("alternating_per_layer", "alternating_per_layer_fast"):
        if layer_index % 2:
            permutation = torch.cat([base_idx[0::2], base_idx[1::2]], dim=0)
        else:
            permutation = torch.cat([base_idx[1::2], base_idx[0::2]], dim=0)
    else:
        # Use a large odd stride so adjacent layers do not reuse a nearby RNG
        # state. Modulo keeps manual_seed in its documented signed-64-bit range.
        mixed_seed = (int(seed) + 1_000_003 * layer_index) % (2**63 - 1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        permutation = torch.randperm(
            n_tokens, generator=generator, device="cpu"
        ).to(device=device)

    n_donors = n_tokens // 2
    donor_idx = permutation[:n_donors].unsqueeze(0).expand(batch_size, -1)
    receiver_idx = permutation[n_donors:].unsqueeze(0).expand(batch_size, -1)
    return donor_idx, receiver_idx


# ---------------------- dtem 专用通用 map/trace 工具 ---------------------- #
# def merge_source_map_for_dtem(current_level_map, source_map):
#     if source_map is None:
#         return current_level_map
#     return torch.gather(current_level_map, dim=1, index=source_map)

@torch.no_grad()
def trace_token_merge(x, size, token_map):
    """
    将 token_map 所指定的分组进行加权聚合，返回合并后的 x/size 以及压缩后的映射：
    - x: [B, L, C]
    - size: [B, L] 或 [B, L, 1] 或 None（None 表示等权重）
    - token_map: [B, L]，每个原 token 合并到的目标 group 索引
    返回 (x_merged [B, M_max, C], size_merged [B, M_max, 1], compact_token_map [B, L])
    """
    B, L, C = x.shape
    device = x.device
    dtype = x.dtype

    if size is None:
        size_local = torch.ones(B, L, device=device, dtype=dtype)
    else:
        size_local = size.squeeze(-1) if size.ndim == 3 else size

    offset = torch.arange(B, device=device, dtype=torch.long).view(B, 1) * L
    global_token_map = token_map + offset

    unique_global_indices, inverse_indices = torch.unique(global_token_map.flatten(), return_inverse=True)
    batch_indices = torch.div(unique_global_indices, L, rounding_mode='floor')

    perm = torch.argsort(unique_global_indices)
    sorted_batch_indices = batch_indices[perm]
    is_new_batch = torch.cat([torch.tensor([True], device=device), sorted_batch_indices[1:] != sorted_batch_indices[:-1]])
    batch_group_ids = torch.cumsum(is_new_batch.long(), dim=0) - 1
    segment_starts = torch.where(is_new_batch)[0]
    local_rank_sorted = torch.arange(len(unique_global_indices), device=device) - segment_starts[batch_group_ids]
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(unique_global_indices), device=device)
    local_rank_unsorted = local_rank_sorted[inv_perm]
    compact_token_map = local_rank_unsorted[inverse_indices].view(B, L)

    counts_per_batch = torch.bincount(batch_indices, minlength=B)
    M_max = counts_per_batch.max().item()

    index = compact_token_map.unsqueeze(-1).expand_as(x)
    x_w = x * size_local.unsqueeze(-1)

    num = torch.zeros(B, M_max, C, device=device, dtype=dtype)
    den = torch.zeros(B, M_max, device=device, dtype=dtype)

    num.scatter_add_(dim=1, index=index, src=x_w)
    den.scatter_add_(dim=1, index=compact_token_map, src=size_local)

    den_clamped = torch.clamp(den, min=torch.finfo(den.dtype).eps)
    x_merged = num / den_clamped.unsqueeze(-1)
    size_merged = den.unsqueeze(-1)

    return x_merged, size_merged, compact_token_map


def token_unmerge_from_map_for_dtem(merged_tokens, token_map_compact, T_full):
    """
    将长度 L-k 的 merged_tokens 还原到原长度 L：
    - merged_tokens: [B, L-k, C]
    - token_map_compact: [B, L]，每个原 token 在合并后序列中的位置
    - T_full: 原长度 L
    返回: [B, L, C]
    """
    B, Lk, C = merged_tokens.shape
    device = merged_tokens.device
    b_idx = torch.arange(B, device=device)[:, None].expand(B, T_full)
    return merged_tokens[b_idx, token_map_compact]


class DTEMLinear(nn.Linear):
    def __init__(self, qkv_layer, feat_dim):
        super().__init__(in_features=qkv_layer.weight.shape[1], out_features=qkv_layer.weight.shape[0] + feat_dim, bias=True)
        # qkv
        self.qkv_layer = qkv_layer

        # metric
        self.feat_dim = feat_dim
        self.metric_layer = nn.Linear(qkv_layer.weight.shape[-1], feat_dim)

        # During training, we use qkv_layer and metric_layer directly
        # So the inherited weight/bias are only used in eval mode
        # Set them to not require gradients to avoid DDP unused parameter issues
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        # copy
        self.update()

    @torch.no_grad()
    def update(self):
        # Ensure weights are on the same device
        device = self.qkv_layer.weight.device

        # Move metric_layer to the same device as qkv_layer
        if self.metric_layer.weight.device != device:
            self.metric_layer = self.metric_layer.to(device)

        if self.weight.device != device:
            self.weight.data = self.weight.data.to(device)
            self.bias.data = self.bias.data.to(device)

        # qkv -> self
        self.weight.data[:-self.feat_dim].copy_(self.qkv_layer.weight.data)
        self.bias.data[:-self.feat_dim].copy_(self.qkv_layer.bias.data)

        # metric_layer -> self
        self.weight.data[-self.feat_dim:].copy_(self.metric_layer.weight.data)
        self.bias.data[-self.feat_dim:].copy_(self.metric_layer.bias.data)

    def train(self, mode=True):
        if mode is False:   # if eval
            self.update()
        return super().train(mode)

    def forward(self, input: torch.Tensor):
        if not self.training:
            # Ensure weights are on the same device as input
            if self.weight.device != input.device:
                self.update()
            out = F.linear(input, self.weight, self.bias)
            return out[..., :-self.feat_dim], out[..., -self.feat_dim:]

        # training
        # Ensure metric_layer is on the same device as input
        if self.metric_layer.weight.device != input.device:
            self.metric_layer = self.metric_layer.to(input.device)

        out1 = self.qkv_layer(input)  # Shape: (B, N, 3 * num_heads * head_dim)
        # Allow 10% gradient to flow through metric for end-to-end optimization
        metric_input = input * 0.1 + input.detach() * 0.9
        out2 = self.metric_layer(metric_input)  # Shape: (B, N, feat_dim)
        return out1, out2

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False


"""
    timm - deit patch
"""
class DTEMAttention(Attention):
    def patch(self, feat_dim):
        if feat_dim is not None:
            out_dim = feat_dim
        else:
            dim = self.head_dim * self.num_heads
            out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim
        # add metric_layer
        self.qkv = DTEMLinear(self.qkv, out_dim)

    def forward(self, x, size=None):    # x:(B, N, C), size:(B, N) or (B, N, 1)
        B, N, C = x.shape
        out1, out2 = self.qkv(x)
        qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # B, H, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        window_size = self._tome_info.get("swa_size")
        if window_size is None:
            window_size = self._tome_info.get("window_size")
        if window_size is not None and window_size > 0:
            if not FLASH_ATTN_AVAILABLE:
                raise RuntimeError(
                    "DTEM local attention requires flash-attn. "
                    "Please install flash-attn or set dtem_window_size=None."
                )
            if size is None:
                raise RuntimeError(
                    "DTEM local attention requires size for bias. "
                    "Got size=None while local window is enabled."
                )
            size_log = size.squeeze(-1) if size.ndim == 3 else size
            size_log = size_log.float().clamp_min(1e-6).log()

            x_out = biased_local_attention(
                q, k, v,
                bias=size_log,
                local_window=window_size,
                dropout_p=self.attn_drop.p,
                training=self.training,
                x_dtype=x.dtype,
                input_layout="BHND",
            )
            x_bnc = x_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        else:
            # Compute softmax in fp32 for numerical stability, keep everything else in AMP dtype
            # The FlashAttention helper applies this same scale internally.
            # Keep q unscaled at the call boundary so both branches apply
            # head_dim**-0.5 exactly once.
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if size is None or (not self._tome_info["r"]):
                attn = attn.float().softmax(dim=-1).to(v.dtype)
            else:
                attn_f = attn.float()
                _attn = attn_f - torch.max(attn_f, dim=-1, keepdim=True)[0]
                size_ = size if size is not None else torch.ones(B, N, 1, device=x.device, dtype=torch.float32)
                _attn = _attn.exp_() * size_[:, None, None, :, 0].float()
                attn_f = _attn / _attn.sum(dim=-1, keepdim=True)
                attn = attn_f.to(v.dtype)
            attn = self.attn_drop(attn)
            x_bnc = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.head_dim)

        x = x_bnc
        x = self.proj(x)
        x = self.proj_drop(x)
        metric = dict(
            q = q,
            k = k,
            v = v,
            x = x,
            metric = out2
        )
        return x, metric

    def _naive_local_attention(self, q, k, v, size, window_size, x_dtype=torch.float32):
        """
        无依赖的局部窗口注意力参考实现（不在主路径中使用）。
        q,k,v: (B, H, N, D)
        size: (B, N, 1) or (B, N) or None
        返回: (B, N, C)
        """
        B, H, N, D = q.shape
        BH = B * H

        q_flat = q.transpose(1, 2).reshape(BH, N, D)
        k_flat = k.transpose(1, 2).reshape(BH, N, D)
        v_flat = v.transpose(1, 2).reshape(BH, N, D)

        padded_k = F.pad(k_flat, (0, 0, window_size, window_size))
        padded_v = F.pad(v_flat, (0, 0, window_size, window_size))

        k_windows = padded_k.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
        v_windows = padded_v.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
        k_windows = k_windows.transpose(-1, -2)
        v_windows = v_windows.transpose(-1, -2)
        q_reshaped = q_flat.unsqueeze(2)  # (BH, N, 1, D)
        attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2) * self.scale  # (BH, N, win)

        if size is not None:
            size_log = size.log().squeeze(-1) if size.ndim == 3 else size.log()
            size_bias_bh = size_log.unsqueeze(1).expand(-1, H, -1).reshape(BH, N)
            padded_size_bias = F.pad(size_bias_bh, (window_size, window_size), mode='constant', value=-1e9)
            size_bias_windows = padded_size_bias.unfold(dimension=1, size=2 * window_size + 1, step=1)
            attn = attn + size_bias_windows

        win_indices = torch.arange(-window_size, window_size + 1, device=q.device).view(1, -1)
        q_indices = torch.arange(N, device=q.device).view(-1, 1)
        abs_k_pos = q_indices + win_indices  # (N, win)
        mask = (abs_k_pos >= 0) & (abs_k_pos < N)
        attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
        attn_out = attn_out_flat.view(B, H, N, D).transpose(1, 2)  # (B, N, H, D)
        x_processed = attn_out.reshape(B, N, H * D).to(x_dtype)
        return x_processed




class DTEMBlock(Block):

    def _select(self, k, a, b, a_orig_idx=None, b_orig_idx=None):
        """
        统一的选择逻辑：支持全局或局部窗口匹配，并支持跨层屏蔽。
        a: (B, Na, C), b: (B, Nb, C)
        a_orig_idx: (B, Na) - a tokens 的原始位置索引
        b_orig_idx: (B, Nb) - b tokens 的原始位置索引
        返回 assign:
            - 全局 dense 模式 (window_size <= 0): (B, Na, Nb)
            - 2-D spatial sparse 模式: (B, Na, K), K 为 padded 邻域宽度
            - 局部窗口模式 (window_size > 0): (B, Na, 2*window_size+1) 狭长矩阵
        """
        EPSILON = torch.finfo(torch.float32).eps
        B, Na, C = a.shape
        _, Nb, _ = b.shape
        device = a.device

        window_size = self._tome_info.get("window_size")
        if window_size is None or window_size <= 0:
            spatial_radius = self._tome_info.get("dtem_spatial_radius")
            # Missing backend means a historical R3 checkpoint/config. Those
            # runs used dense score-then-mask, so fail backward-compatibly to
            # dense; the sparse backend is selected explicitly.
            spatial_backend = self._tome_info.get("dtem_spatial_backend", "dense")
            if spatial_backend not in DTEM_SPATIAL_BACKENDS:
                raise RuntimeError(
                    f"Unsupported DTEM spatial backend {spatial_backend!r}; "
                    f"expected one of {DTEM_SPATIAL_BACKENDS}"
                )
            allowed = None
            valid_row = None
            sparse_spatial = spatial_radius is not None and spatial_backend == "sparse"
            if spatial_radius is not None:
                if a_orig_idx is None or b_orig_idx is None:
                    raise RuntimeError(
                        "DTEM spatial masking requires absolute donor/receiver token indices"
                    )
                if sparse_spatial:
                    neighbor_indices = self._tome_info.get("spatial_neighbor_indices")
                    neighbor_valid = self._tome_info.get("spatial_neighbor_valid")
                    if neighbor_indices is None or neighbor_valid is None:
                        raise RuntimeError(
                            "sparse DTEM spatial scoring is enabled but the neighbor "
                            "table is unavailable"
                        )
                    prefix = self._tome_info.get("num_prefix_tokens", 1)
                    scores, receiver_indices, allowed = sparse_spatial_score(
                        a,
                        b,
                        neighbor_indices,
                        neighbor_valid,
                        a_orig_idx.long() - prefix,
                        b_orig_idx.long() - prefix,
                    )
                    self._tome_info["assign_layout"] = "spatial_sparse"
                    self._tome_info["assign_b_indices"] = receiver_indices
                    self._tome_info["assign_valid_mask"] = allowed
                else:
                    adjacency = self._tome_info.get("spatial_adjacency")
                    if adjacency is None:
                        raise RuntimeError(
                            "dense DTEM spatial scoring is enabled but spatial_adjacency "
                            "is unavailable"
                        )
                    scores = a @ b.transpose(-1, -2)
                    allowed = gather_patch_spatial_mask(
                        adjacency,
                        a_orig_idx,
                        b_orig_idx,
                        num_prefix_tokens=self._tome_info.get("num_prefix_tokens", 1),
                        validate=False,
                    )
                    self._tome_info["assign_layout"] = "dense"
                valid_row = allowed.any(dim=-1)
                enough_rows = valid_row.sum(dim=-1) >= int(k)
                message = (
                    "DTEM spatial radius leaves fewer eligible donor rows than the "
                    f"required merge budget k={int(k)}; radius={spatial_radius}, "
                    f"grid={self._tome_info.get('patch_grid_size')}"
                )
                if hasattr(torch, "_assert_async"):
                    torch._assert_async(enough_rows.all(), message)
                elif not bool(enough_rows.all().item()):
                    raise RuntimeError(message)
            else:
                scores = a @ b.transpose(-1, -2)
                self._tome_info["assign_layout"] = "dense"

            tau1 = self._tome_info.get("tau1", 1.0)
            if self._tome_info.get("use_softkmax") is not False:
                # Sorting is a pure permutation for the soft-top-k formulation:
                # row energy and row-wise softmax are permutation invariant.  Work
                # directly on the dense score matrix to avoid a large argsort/index
                # tensor in the no-window experiments.
                logits = scores / tau1
                if allowed is not None:
                    logits = logits.masked_fill(~allowed, -torch.inf)
                    safe_logits = torch.where(
                        valid_row.unsqueeze(-1), logits, torch.zeros_like(logits)
                    )
                    # Compute empty rows from finite placeholders. Taking
                    # logsumexp over an all--inf row produces a NaN backward
                    # even if a later torch.where zeros the forward value.
                    row_energy = torch.logsumexp(safe_logits, dim=-1).exp()
                    row_energy = torch.where(valid_row, row_energy, torch.zeros_like(row_energy))
                    row_khot = ThreTopK(row_energy, k, self._tome_info["tau2"])
                    row_khot = row_khot * valid_row.to(row_khot.dtype)
                    probabilities = F.softmax(safe_logits, dim=-1)
                    probabilities = probabilities * allowed.to(probabilities.dtype)
                    khot = probabilities * row_khot.unsqueeze(-1)
                else:
                    row_energy = torch.logsumexp(logits, dim=-1).exp()
                    row_khot = ThreTopK(row_energy, k, self._tome_info["tau2"])
                    khot = F.softmax(logits, dim=-1) * row_khot.unsqueeze(-1)

                tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
                nkhot = (khot / tmp).to(scores.dtype)
                assign = nkhot
            else:
                _idx = scores.argsort(dim=-1, descending=True)
                _x = scores.gather(dim=-1, index=_idx) / tau1
                sorted_allowed = None
                if allowed is not None:
                    sorted_allowed = allowed.gather(dim=-1, index=_idx)
                    _x = _x.masked_fill(~sorted_allowed, -torch.inf)
                    sample_has_candidate = sorted_allowed.flatten(1).any(dim=-1)
                    safe_x = torch.where(
                        sample_has_candidate[:, None, None], _x, torch.zeros_like(_x)
                    )
                else:
                    safe_x = _x
                khot = torch.zeros_like(_x)
                _x_iter = safe_x.clone()  # Clone to avoid inplace modification
                for _ in range(k):
                    # The sparse spatial score width is K rather than Nb. Use
                    # the actual score shape so the non-softkmax compatibility
                    # path remains dense-reference equivalent for K != Nb.
                    onehot_approx = F.softmax(
                        _x_iter.view(B, -1)
                        * self._tome_info.get("tau2", 30.0),
                        dim=-1,
                    ).view_as(_x_iter)
                    if sorted_allowed is not None:
                        onehot_approx = onehot_approx * sorted_allowed.to(onehot_approx.dtype)
                    khot += onehot_approx
                    khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                    _x_iter = _x_iter + torch.log(khot_mask)

                tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
                nkhot = (khot / tmp).to(scores.dtype)
                assign = torch.zeros_like(scores).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        else:
            # 某些情况下 Na 与 Nb 可能相差 1（例如边界填充后 unfold），先对齐长度避免广播错误
            if Na != Nb:
                new_len = min(Na, Nb)
                a = a[:, :new_len, :]
                b = b[:, :new_len, :]
                if a_orig_idx is not None:
                    a_orig_idx = a_orig_idx[:, :new_len]
                if b_orig_idx is not None:
                    b_orig_idx = b_orig_idx[:, :new_len]
                Na = Nb = new_len

            padded_b = F.pad(b, (0, 0, window_size, window_size))
            b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
            b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
            a_reshaped = a.unsqueeze(2)
            local_scores = (a_reshaped * b_windows).sum(dim=-1) / (self._tome_info.get("tau1", 1.0))

            # Mask out invalid positions (outside [0, Nb) range)
            a_indices = torch.arange(Na, device=device).view(1, -1, 1)
            window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
            b_indices = a_indices + window_offsets
            valid_mask = (b_indices >= 0) & (b_indices < Nb)
            neg_inf = torch.finfo(local_scores.dtype).min
            local_scores = local_scores.masked_fill(~valid_mask, neg_inf)

            if self._tome_info.get("use_softkmax")==False:
                khot = torch.zeros_like(local_scores)
                local_scores_iter = local_scores.clone()  # Clone to avoid inplace modification
                for _ in range(k):
                    onehot_approx = F.softmax(local_scores_iter * self._tome_info.get("tau2", 30.0), dim=-1)
                    khot += onehot_approx
                    khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                    local_scores_iter = local_scores_iter + torch.log(khot_mask)
            else:
                local_scores_energy = torch.logsumexp(local_scores, dim=-1).exp()
                row_khot = ThreTopK(local_scores_energy, k, self._tome_info["tau2"])
                khot = F.softmax(local_scores, dim=-1) * row_khot.unsqueeze(-1)

            tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
            nkhot = khot / tmp

            # Keep assign as narrow matrix (B, Na, 2*window_size+1) for local window
            # Store b_indices to map window positions to actual b token indices
            a_indices = torch.arange(Na, device=device).view(1, -1, 1)
            window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
            b_indices = a_indices + window_offsets  # (1, Na, 2*window_size+1)
            valid_mask = (b_indices >= 0) & (b_indices < Nb)  # (1, Na, 2*window_size+1)

            # assign stays as narrow matrix with masked values
            assign = torch.where(valid_mask.expand(B, -1, -1), nkhot, 0.0)  # (B, Na, 2*window_size+1)

            # Store b_indices for later use in merging
            self._tome_info['assign_b_indices'] = b_indices  # (1, Na, 2*window_size+1)
            self._tome_info['assign_valid_mask'] = valid_mask  # (1, Na, 2*window_size+1)

            # Store original indices for physical locality check
            self._tome_info['assign_a_orig_idx'] = a_orig_idx  # (B, Na) or None
            self._tome_info['assign_b_orig_idx'] = b_orig_idx  # (B, Nb) or None
            self._tome_info["assign_layout"] = "flat_window"
        # These diagnostics are not consumed by MergeNet. Each ``.item()``
        # forces a device synchronization, so keep them behind an explicit
        # compatibility flag instead of stalling every local layer.
        if self._tome_info.get("collect_merge_stats", True):
            with torch.no_grad():
                out_dict = {
                    'num': (nkhot if 'nkhot' in locals() else khot).sum().item(),
                    'max': (khot if 'khot' in locals() else nkhot).view(B, -1).max(dim=-1)[0].sum().item(),
                }
        else:
            out_dict = None
        return assign, out_dict

    def _merge_train(self, x, size, r, n, metric, source_matrix=None):

        metric = metric['metric']
        norm = metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        metric = metric / norm

        r = min(r, (n - 1) // 2)
        if r <= 0:
            return x, size, n, metric, source_matrix

        # 保证 size 为 [B, N, 1]
        if size is not None and size.ndim == 2:
            size = size.unsqueeze(-1)

        source_trace_mode = self._tome_info.get("source_trace_mode", None)
        if source_trace_mode is None:
            trace_source = self._tome_info.get("trace_source", True)
            source_tracking_mode = self._tome_info.get("source_tracking_mode", "none")
            source_trace_mode = "matrix" if trace_source and source_tracking_mode == "matrix" else "none"
        if source_trace_mode not in ("matrix", "detached", "center", "none"):
            raise ValueError(f"Unsupported source_trace_mode: {source_trace_mode!r}")
        use_source_matrix = source_trace_mode in ("matrix", "detached")
        if not use_source_matrix:
            source_matrix = None

        # Initialize source_matrix only for modes that need full source
        # distributions. "center" and "none" avoid this dense trace entirely.
        if source_matrix is None and use_source_matrix:
            B, N = x.shape[0], x.shape[1]
            window_size = self._tome_info.get("window_size", 0)
            local_depth = self._tome_info.get("local_depth", 1)
            if window_size is not None and window_size > 0:
                width = 2 * window_size * local_depth + 1
                center = window_size * local_depth
            else:
                # Global matching can move source mass between any two token
                # positions in one layer, so the exact relative trace must cover
                # the whole sequence rather than the local-window radius.
                width = 2 * N - 1
                center = N - 1

            source_matrix = torch.zeros(B, N, width, device=x.device, dtype=x.dtype)
            source_matrix[:, :, center] = 1.0  # Identity: each token 100% from itself
            # Store metadata only, not the matrix itself
            self._tome_info["source_matrix_center"] = center
            self._tome_info["source_matrix_width"] = width

        # xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        # a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        # wa = size[..., 1:n:2, 0]
        # wb = size[..., 2:n:2, 0]

        # Select donor/receiver groups according to the train/eval policy.
        # =================================================================================== #
        B, T, C = x.shape
        device = x.device

        # Random grouping: split tokens (excluding cls if present) into two groups
        has_cls = self._tome_info.get("class_token", True)
        if has_cls:
            n_tokens = n - 1  # exclude cls token at position 0
            offset = 1
        else:
            n_tokens = n
            offset = 0

        Na = n_tokens // 2
        Nb = n_tokens - Na

        # ``random_per_sample`` is the historical training behavior and remains
        # the default. Deterministic training modes deliberately reuse the eval
        # helper so an explicitly matched train/inference protocol cannot drift
        # through two separate grouping implementations.
        use_strided_fast_path = False
        donor_patch_parity = None
        if (
            self.training
            and self._tome_info.get("train_grouping", "random_per_sample")
            == "random_per_sample"
        ):
            rand_vals = torch.rand(B, n_tokens, device=device)
            rand_perm = torch.argsort(rand_vals, dim=1)  # [B, n_tokens]
            a_idx = rand_perm[:, :Na] + offset  # +offset to skip cls if present
            b_idx = rand_perm[:, Na:] + offset
        else:
            if self.training:
                grouping = self._tome_info.get(
                    "train_grouping", "random_per_sample"
                )
                grouping_seed = self._tome_info.get(
                    "train_grouping_seed", 0
                )
                if grouping == "random_per_sample":
                    raise RuntimeError(
                        "random_per_sample must use the stochastic training branch"
                    )
            else:
                grouping = self._tome_info.get(
                    "eval_grouping", "alternating_per_layer"
                )
                grouping_seed = self._tome_info.get(
                    "eval_grouping_seed", 0
                )
            merge_layer_index = self._tome_info.get("merge_layer_index", 0)
            a_idx, b_idx = build_dtem_eval_group_indices(
                n_tokens=n_tokens,
                batch_size=B,
                device=device,
                mode=grouping,
                seed=grouping_seed,
                layer_index=merge_layer_index,
            )
            a_idx = a_idx + offset
            b_idx = b_idx + offset

            # The fast policy is mathematically identical to
            # ``alternating_per_layer``. Restrict it to the even-token,
            # no-dense-source case so the specialized interleave below remains
            # simple and auditable; all other geometries use the generic path.
            use_strided_fast_path = (
                grouping == "alternating_per_layer_fast"
                and n_tokens % 2 == 0
                and source_matrix is None
            )
            if use_strided_fast_path:
                donor_patch_parity = 0 if merge_layer_index % 2 else 1

        if use_strided_fast_path:
            # Both parity groups are already monotonically increasing.
            a_idx_sorted = a_idx
            b_idx_sorted = b_idx
        else:
            # Sort indices within each group for efficient gathering.
            a_idx_sorted, _ = torch.sort(a_idx, dim=1)
            b_idx_sorted, _ = torch.sort(b_idx, dim=1)

        # Extract tokens using sorted indices
        # Note: x and metric have different feature dimensions
        D_metric = metric.shape[-1]

        if use_strided_fast_path:
            active_end = offset + n_tokens
            receiver_patch_parity = 1 - donor_patch_parity
            donor_slice = slice(
                offset + donor_patch_parity, active_end, 2
            )
            receiver_slice = slice(
                offset + receiver_patch_parity, active_end, 2
            )
            # ``contiguous`` preserves the layout produced by gather, while
            # avoiding index expansion and four gather kernels.
            xa = x[:, donor_slice, :].contiguous()
            xb = x[:, receiver_slice, :].contiguous()
            a = metric[:, donor_slice, :].contiguous()
            b = metric[:, receiver_slice, :].contiguous()
            wa = size[:, donor_slice, 0].contiguous()
            wb = size[:, receiver_slice, 0].contiguous()
        else:
            a_idx_expanded_x = a_idx_sorted.unsqueeze(-1).expand(-1, -1, C)
            b_idx_expanded_x = b_idx_sorted.unsqueeze(-1).expand(-1, -1, C)
            a_idx_expanded_metric = a_idx_sorted.unsqueeze(-1).expand(-1, -1, D_metric)
            b_idx_expanded_metric = b_idx_sorted.unsqueeze(-1).expand(-1, -1, D_metric)

            xa = torch.gather(x, dim=1, index=a_idx_expanded_x)
            xb = torch.gather(x, dim=1, index=b_idx_expanded_x)
            a = torch.gather(metric, dim=1, index=a_idx_expanded_metric)
            b = torch.gather(metric, dim=1, index=b_idx_expanded_metric)
            wa = torch.gather(size[..., 0], dim=1, index=a_idx_sorted)
            wb = torch.gather(size[..., 0], dim=1, index=b_idx_sorted)
        wa_before_merge = wa.detach()
        wb_before_merge = wb.detach()

        if source_trace_mode == "center":
            token_center = self._tome_info.get("token_center", None)
            if token_center is None:
                token_center = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1)
            if use_strided_fast_path:
                center_a = token_center[:, donor_slice].contiguous()
                center_b = token_center[:, receiver_slice].contiguous()
            else:
                center_a = torch.gather(token_center, dim=1, index=a_idx_sorted)
                center_b = torch.gather(token_center, dim=1, index=b_idx_sorted)

        # Extract source matrix for a and b tokens
        if source_matrix is not None:
            width = self._tome_info["source_matrix_width"]
            a_idx_expanded_source = a_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            b_idx_expanded_source = b_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
            source_a = torch.gather(source_matrix, dim=1, index=a_idx_expanded_source)
            source_b = torch.gather(source_matrix, dim=1, index=b_idx_expanded_source)

            # Save old weights for source update
            wa_old = wa.clone()
            wb_old = wb.clone()

        # =================================================================================== #

        # 对齐合并粒度 t
        t = max(int(self._tome_info.get("t", 1)), 1)
        r = (r // t) * t
        # print(f"r: {r}")
        # print(f"a: {a.shape}")
        # print(f"b: {b.shape}")
        assign, _out = self._select(k=r, a=a, b=b,
                                    a_orig_idx=a_idx_sorted,
                                    b_orig_idx=b_idx_sorted)

        # Both the historical flat window and the true 2-D sparse scorer return
        # a narrow assignment plus receiver indices.  Keep that representation
        # through transport/source tracing; densifying here would forfeit the
        # scorer's memory and latency gains.
        window_size = self._tome_info.get("window_size")
        assign_layout = self._tome_info.get("assign_layout", "dense")
        narrow_assignment = assign_layout in ("flat_window", "spatial_sparse")
        if narrow_assignment:
            B_cur, Na_cur = assign.shape[0], assign.shape[1]
            Nb_cur = xb.shape[1]
            C_cur = xa.shape[-1]

            # ``b_indices`` is [1, Na, K] for the flat window and [B, Na, K]
            # for sample-dependent sparse spatial neighborhoods.
            b_indices = self._tome_info['assign_b_indices']
            valid_mask = self._tome_info['assign_valid_mask']

            # Get original indices for physical locality check
            a_orig_idx = self._tome_info.get('assign_a_orig_idx')  # (B, Na) or None
            b_orig_idx = self._tome_info.get('assign_b_orig_idx')  # (B, Nb) or None

            # The flat-index window needs its legacy second physical-distance
            # check. Spatial neighborhoods were constructed directly from the
            # exact 2-D adjacency, so their validity mask is already final.
            physical_mask = valid_mask.clone()  # Start with valid_mask
            if (
                assign_layout == "flat_window"
                and a_orig_idx is not None
                and b_orig_idx is not None
            ):
                # For each a[i], check if |a_orig_idx[i] - b_orig_idx[b_indices[i,j]]| <= window_size
                # NOTE: a_orig_idx and b_orig_idx may have been aligned in _select if Na != Nb
                # So their shape[1] may be min(Na, Nb), not the original Na/Nb
                # a_orig_idx: (B, Na_aligned) -> (B, Na_aligned, 1)
                # b_indices: (1, Na, 2*window_size+1)
                # b_orig_idx: (B, Nb_aligned)

                # Use actual shapes from orig_idx tensors
                Na_aligned = a_orig_idx.shape[1]
                Nb_aligned = b_orig_idx.shape[1]

                # Only proceed if aligned sizes match current assign shape
                if Na_aligned == Na_cur:
                    a_orig_expanded = a_orig_idx.unsqueeze(2)  # (B, Na_aligned, 1)
                    # Clamp b_indices to valid range of b_orig_idx
                    b_indices_clamped = b_indices.clamp(0, Nb_aligned - 1)  # (1, Na, 2*window_size+1)
                    b_indices_expanded = b_indices_clamped.expand(B_cur, -1, -1)  # (B, Na, 2*window_size+1)

                    # If Na from b_indices doesn't match Na_aligned, truncate
                    if b_indices_expanded.shape[1] > Na_aligned:
                        b_indices_expanded = b_indices_expanded[:, :Na_aligned, :]

                    # Ensure indices are within bounds
                    b_indices_expanded = b_indices_expanded.clamp(0, Nb_aligned - 1)

                    # Gather b's original positions at window indices
                    b_orig_expanded = b_orig_idx.unsqueeze(1).expand(-1, Na_aligned, -1)  # (B, Na_aligned, Nb_aligned)
                    b_orig_at_window = torch.gather(
                        b_orig_expanded,  # (B, Na_aligned, Nb_aligned)
                        dim=2,
                        index=b_indices_expanded  # (B, Na_aligned, 2*window_size+1)
                    )  # (B, Na_aligned, 2*w+1)

                    # Check physical distance
                    physical_distance = torch.abs(a_orig_expanded - b_orig_at_window)  # (B, Na_aligned, 2*w+1)
                    physical_local_mask = physical_distance <= window_size  # (B, Na_aligned, 2*w+1)

                    # Combine with valid_mask
                    physical_mask = valid_mask & physical_local_mask  # (B, Na, 2*w+1)

            # Apply physical mask to assign (use same dtype as assign)
            assign = assign * physical_mask.to(assign.dtype)

            # Clamp indices for scatter operation
            b_indices_clamped = b_indices.clamp(0, Nb_cur - 1)  # (1, Na, 2*window_size+1)

            # Compute weighted xa: (B, Na, C)
            weighted_xa = wa[..., None] * xa  # (B, Na, C)
            if assign_layout == "spatial_sparse":
                # Fused indexed scatter: never materialize the prohibitive
                # [B, Na, K, C_embed] contribution tensor (about 0.55 GiB for
                # one fp16 MN-L2 layer at B=64, 224/p8, R3).
                xb_contrib = sparse_weighted_transport(
                    assign,
                    weighted_xa,
                    b_indices_clamped,
                    valid_mask,
                    Nb_cur,
                )
                wb_contrib = sparse_weighted_transport(
                    assign,
                    wa.unsqueeze(-1),
                    b_indices_clamped,
                    valid_mask,
                    Nb_cur,
                ).squeeze(-1)
                xb = wb[..., None] * xb + xb_contrib
                wb = wb + wb_contrib
            else:
                # Historical flat-window compatibility path.
                assign_expanded = assign.unsqueeze(-1)
                weighted_xa_expanded = weighted_xa.unsqueeze(2)
                contribution = assign_expanded * weighted_xa_expanded
                b_indices_expanded = b_indices_clamped.expand(
                    B_cur, -1, -1
                ).unsqueeze(-1).expand(-1, -1, -1, C_cur)
                xb_contrib = torch.zeros(
                    B_cur, Nb_cur, C_cur, device=xb.device, dtype=xb.dtype
                )
                xb_contrib.scatter_add_(
                    dim=1,
                    index=b_indices_expanded.reshape(B_cur, -1, C_cur),
                    src=contribution.reshape(B_cur, -1, C_cur),
                )
                xb = wb[..., None] * xb + xb_contrib

                wa_expanded = wa.unsqueeze(2)
                wb_contribution = assign_expanded[..., 0] * wa_expanded
                b_indices_expanded_1d = b_indices_clamped.expand(B_cur, -1, -1)
                wb_contrib = torch.zeros(
                    B_cur, Nb_cur, device=wb.device, dtype=wb.dtype
                )
                wb_contrib.scatter_add_(
                    dim=1,
                    index=b_indices_expanded_1d.reshape(B_cur, -1),
                    src=wb_contribution.reshape(B_cur, -1),
                )
                wb = wb + wb_contrib

            # Compute tmp: 1 - sum of assign over last dim
            tmp = 1 - assign.sum(dim=-1)  # (B, Na)
        else:
            # Dense global/dense-reference spatial case: [B, Na, Nb].
            xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)
            wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]
            tmp = 1 - assign.sum(dim=-1)

        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())
        wb_safe = torch.clamp(wb, min=torch.finfo(wb.dtype).eps)
        xb = xb / wb_safe[..., None]

        if source_trace_mode == "center":
            with torch.no_grad():
                if narrow_assignment:
                    center_contribution = assign * (wa_before_merge * center_a).unsqueeze(2)
                    center_b_contrib = torch.zeros(B_cur, Nb_cur, device=device, dtype=torch.float32)
                    center_b_contrib.scatter_add_(
                        dim=1,
                        index=b_indices_clamped.expand(B_cur, -1, -1).reshape(B_cur, -1),
                        src=center_contribution.reshape(B_cur, -1).float(),
                    )
                    center_b_new = (wb_before_merge.float() * center_b.float() + center_b_contrib) / wb_safe.float()
                else:
                    center_b_num = (
                        wb_before_merge.float() * center_b.float()
                        + (assign.transpose(-1, -2).float() @ (wa_before_merge.float() * center_a.float()).unsqueeze(-1))[..., 0]
                    )
                    center_b_new = center_b_num / wb_safe.float()
                center_a_new = center_a.float()

        # Update source_matrix
        if source_matrix is not None:
            trace_ctx = torch.no_grad() if source_trace_mode == "detached" else contextlib.nullcontext()
            with trace_ctx:
                width = self._tome_info["source_matrix_width"]
                center = self._tome_info["source_matrix_center"]

                if narrow_assignment:
                    # Indexed narrow case: update source_matrix without creating
                    # a donor-by-receiver assignment.

                    # Get stored indices
                    b_indices = self._tome_info['assign_b_indices']  # (1, Na, 2*window_size+1)
                    b_indices_clamped = b_indices.clamp(0, Nb_cur - 1).expand(B_cur, -1, -1)  # (B, Na, 2*w+1)

                    # Compute delta: physical distance from a to b
                    a_orig_expanded = a_idx_sorted.unsqueeze(2)  # (B, Na, 1)
                    batch_idx = torch.arange(B_cur, device=b_idx_sorted.device).view(-1, 1, 1)  # (B, 1, 1)
                    b_orig_at_window = b_idx_sorted[batch_idx, b_indices_clamped]  # (B, Na, 2*w+1)
                    delta = a_orig_expanded - b_orig_at_window  # (B, Na, 2*w+1)

                    # Update source_matrix with correct logic:
                    # When a[i] contributes weight w to b[j]:
                    # 1. b[j] receives a[i]'s all sources scaled by w (with position shift)
                    # 2. a[i] loses those contributions (scaled by w)

                    # Prepare output tensors
                    source_b_new = source_b.clone()

                    # OPTIMIZED: Compute transferred contributions once and reuse
                    # Transferred contributions: source_a[i, k] * assign[i, j_local]
                    transferred = source_a.unsqueeze(2) * assign.unsqueeze(-1)  # (B, Na, 2*w+1, width)

                    # Compute target positions in b's reference frame
                    k_range = torch.arange(width, device=x.device).view(1, 1, 1, -1)
                    delta_4d = delta.unsqueeze(-1)  # (B, Na, 2*w+1, 1)

                    # New position in b's frame: k + delta
                    target_k = k_range + delta_4d  # (B, Na, 2*w+1, width)
                    valid_target = (target_k >= 0) & (target_k < width)
                    target_k_clamped = target_k.clamp(0, width - 1)

                    # Apply valid mask (use same dtype as transferred)
                    transferred_masked = transferred * valid_target.to(transferred.dtype)

                    # Scatter to source_b using b_indices
                    batch_idx = torch.arange(B_cur, device=x.device).view(-1, 1, 1, 1)
                    b_idx_4d = b_indices_clamped.unsqueeze(-1)  # (B, Na, 2*w+1, 1)

                    # Linear index: batch * (Nb * width) + b_idx * width + target_k
                    linear_idx = batch_idx * Nb_cur * width + b_idx_4d * width + target_k_clamped

                    # Flatten and scatter_add
                    source_b_flat = source_b_new.reshape(-1)
                    source_b_flat.scatter_add_(0, linear_idx.reshape(-1), transferred_masked.reshape(-1))
                    source_b_new = source_b_flat.reshape(B_cur, Nb_cur, width)

                    # Update source_a with SAME logic as wa update
                    # wa uses: wa * (tmp + (clamp(tmp, 0, 1) - tmp).detach())
                    # source_a should use the SAME scaling factor
                    # Note: source_a.sum(dim=-1) should equal wa after this operation
                    tmp_source = tmp.unsqueeze(-1)  # (B, Na, 1) - broadcast over width dimension
                    source_a_new = source_a * (tmp_source + (torch.clamp(tmp_source, min=0., max=1.) - tmp_source).detach())
                else:
                    # Global case: exact source propagation with full assign.
                    # If a[i] contributes to b[j], a[i]'s source distribution is
                    # shifted from a's relative frame into b's relative frame by
                    # delta = pos(a) - pos(b), then accumulated into b[j].
                    source_b_new = source_b.clone()
                    tmp_source = tmp.unsqueeze(-1)
                    source_a_new = source_a * (
                        tmp_source + (torch.clamp(tmp_source, min=0., max=1.) - tmp_source).detach()
                    )

                    k_range = torch.arange(width, device=x.device).view(1, 1, 1, -1)
                    j_range = torch.arange(Nb, device=x.device).view(1, 1, Nb, 1)
                    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
                    source_b_flat = source_b_new.reshape(-1)

                    # Chunk over a-tokens to avoid materializing [B, Na, Nb, width]
                    # for the global path. This path is intentionally exact, not
                    # optimized as the default fast training route.
                    chunk = int(self._tome_info.get("global_source_chunk_size", 1))
                    chunk = max(chunk, 1)
                    for start in range(0, Na, chunk):
                        end = min(start + chunk, Na)
                        assign_chunk = assign[:, start:end, :]  # (B, ch, Nb)
                        source_chunk = source_a[:, start:end, :]  # (B, ch, width)
                        delta = (
                            a_idx_sorted[:, start:end].unsqueeze(2)
                            - b_idx_sorted.unsqueeze(1)
                        )  # (B, ch, Nb)

                        target_k = k_range[:, :end - start] + delta.unsqueeze(-1)
                        valid_target = (target_k >= 0) & (target_k < width)
                        target_k_clamped = target_k.clamp(0, width - 1)
                        transferred = (
                            assign_chunk.unsqueeze(-1)
                            * source_chunk.unsqueeze(2)
                            * valid_target.to(source_a.dtype)
                        )
                        linear_idx = batch_idx * Nb * width + j_range * width + target_k_clamped
                        source_b_flat.scatter_add_(0, linear_idx.reshape(-1), transferred.reshape(-1))

                    source_b_new = source_b_flat.reshape(B, Nb, width)

                # Write back to full source_matrix
                source_matrix_new = source_matrix.clone()

                # Scatter a's sources back
                a_idx_expanded_source = a_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
                source_matrix_new.scatter_(dim=1, index=a_idx_expanded_source, src=source_a_new)

                # Scatter b's sources back
                b_idx_expanded_source = b_idx_sorted.unsqueeze(-1).expand(-1, -1, width)
                source_matrix_new.scatter_(dim=1, index=b_idx_expanded_source, src=source_b_new)

                # Optional: normalize to ensure sum = 1 (compensate for floating point errors)
                if self._tome_info.get("normalize_source_matrix", False):
                    source_sum = source_matrix_new.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    source_matrix_new = source_matrix_new / source_sum

                source_matrix = source_matrix_new

        # =================================================================================== #
        # w = torch.cat([wa, wb], dim=-1)
        # nx = torch.cat([xa, xb], dim=1)
        # nidxs = w.argsort(dim=-1, descending=True)
        # w = w.gather(dim=-1, index=nidxs)
        # nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))
        # =================================================================================== #

        # =================================================================================== #

        # Restore original spatial order based on the random grouping
        # SOFT MERGE: Keep ALL tokens (both a and b), only weights change
        # wa becomes smaller for merged tokens, wb becomes larger for receiving tokens
        Na_curr = xa.shape[1]  # Should equal Na (all a tokens kept)
        Nb_curr = xb.shape[1]  # Should equal Nb (all b tokens kept)
        n_total = Na_curr + Nb_curr  # Should equal Na + Nb

        if use_strided_fast_path:
            # Recreate row-major patch order directly. For donor parity 0 the
            # order is [a0,b0,a1,b1,...]; for donor parity 1 it is reversed.
            if donor_patch_parity == 0:
                even_tokens, odd_tokens = xa, xb
                even_weights, odd_weights = wa, wb
            else:
                even_tokens, odd_tokens = xb, xa
                even_weights, odd_weights = wb, wa
            nx = torch.stack((even_tokens, odd_tokens), dim=2).reshape(
                B, n_tokens, C
            )
            w = torch.stack((even_weights, odd_weights), dim=2).reshape(
                B, n_tokens
            )
            if source_trace_mode == "center":
                with torch.no_grad():
                    if donor_patch_parity == 0:
                        even_centers, odd_centers = center_a_new, center_b_new
                    else:
                        even_centers, odd_centers = center_b_new, center_a_new
                    center_sorted = torch.stack(
                        (even_centers, odd_centers), dim=2
                    ).reshape(B, n_tokens)
        else:
            # Combine all donor and receiver tokens with their original
            # positions, then restore the generic random/seeded order.
            all_orig_pos = torch.cat([a_idx_sorted, b_idx_sorted], dim=1)
            all_tokens = torch.cat([xa, xb], dim=1)
            all_weights = torch.cat([wa, wb], dim=1)
            sort_indices = torch.argsort(all_orig_pos, dim=1)
            sort_indices_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, C)
            nx = torch.gather(all_tokens, dim=1, index=sort_indices_expanded)
            w = torch.gather(all_weights, dim=1, index=sort_indices)

            if source_trace_mode == "center":
                with torch.no_grad():
                    all_centers = torch.cat([center_a_new, center_b_new], dim=1)
                    center_sorted = torch.gather(
                        all_centers, dim=1, index=sort_indices
                    )

        # =================================================================================== #

        # Handle cls token if present
        has_cls = self._tome_info.get("class_token", True)
        if has_cls:
            x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
            size_output = torch.cat([size[:, :1, 0], w, size[:, n:, 0]], dim=-1).unsqueeze(-1)
            if source_trace_mode == "center":
                cls_center = self._tome_info.get("token_center", None)
                if cls_center is None:
                    cls_center = torch.zeros(B, 1, device=device, dtype=torch.float32)
                else:
                    cls_center = cls_center[:, :1].float()
                tail_center = self._tome_info.get("token_center", None)
                tail_center = tail_center[:, n:].float() if tail_center is not None and n < tail_center.shape[1] else center_sorted[:, :0]
                self._tome_info["token_center"] = torch.cat([cls_center, center_sorted, tail_center], dim=1)
        else:
            x_output = torch.cat([nx, x[:, n:]], dim=1)
            size_output = torch.cat([w, size[:, n:, 0]], dim=-1).unsqueeze(-1)
            if source_trace_mode == "center":
                tail_center = self._tome_info.get("token_center", None)
                tail_center = tail_center[:, n:].float() if tail_center is not None and n < tail_center.shape[1] else center_sorted[:, :0]
                self._tome_info["token_center"] = torch.cat([center_sorted, tail_center], dim=1)
        return x_output, size_output, n , _out, source_matrix

    def _merge_eval(self, x, size, r, n, metric, source_matrix=None):
        metric = metric['metric']
        norm = metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        metric = metric / norm

        r = min(r, (n - 1) // 2)
        if r <= 0:
            return x, size, n, None, source_matrix

        merge, _, current_level_map = bipartite_soft_matching(metric,
                                           r,
                                           self._tome_info["class_token"],
                                           self._tome_info["distill_token"],
                                           )
        if self._tome_info["trace_source"]:
            if "source_tracking_mode" in self._tome_info:
                if self._tome_info["source_tracking_mode"] == 'map':
                    source_map = self._tome_info["source_map"]
                    if source_map is None:
                        b, t, _ = x.shape
                        source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
                    self._tome_info["source_map"] = merge_source_map(current_level_map, x, source_map)
                else:
                    source_matrix = self._tome_info["source_matrix"]
                    self._tome_info["source_matrix"] = merge_source_matrix(merge, x, source_matrix)

        x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
        return x, self._tome_info["size"], x.size(1), None, source_matrix

    def merge(self, x, size, r, n, metric, source_matrix=None):
        if self.training:
            return self._merge_train(x, size, r, n, metric, source_matrix)
        return self._merge_eval(x, size, r, n, metric, source_matrix)

    def forward(self, x, size, n=None, source_matrix=None):
        if size is None:
            size=torch.ones_like(x[..., 0, None])
        tmp, metric = self.attn(self.norm1(x), size=size)
        assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
        x = x + self.drop_path1(self.ls1(tmp))
        # Merging
        r = self._tome_info["r"].pop(0)
        if size is not None and r > 0 and n > 0:
            x, size, n, metric, source_matrix = self.merge(x, size, r, n, metric, source_matrix)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        return x, size, n, metric, source_matrix


def make_tome_class(transformer_class):
    class DTEMVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward_features(self, x):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)

            n = x.size(1)
            self._tome_info["r"] = parse_r(
                len(self.blocks), self.r, self._tome_info["total_merge"]
            )
            self._tome_info["size"] = torch.ones_like(x[..., 0, None])
            self._tome_info["source_map"] = None

            out_dicts = []
            source_matrix = None  # Initialize, will be created in first block
            for block in self.blocks:
                x, size, n, out_dict, source_matrix = block(x, self._tome_info["size"], n=n, source_matrix=source_matrix)
                out_dicts.append(out_dict)

            x = self.norm(x)
            return x, out_dicts

        def forward(self, x, return_out_dicts=False):
            x, out_dicts = self.forward_features(x)
            x = self.forward_head(x)
            if return_out_dicts:
                return x, out_dicts
            return x

    return DTEMVisionTransformer



""""
Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers, NIPS'2024
    - paper (https://openreview.net/forum?id=pVPyCgXv57)
    - code  (https://github.com/movinghoon/DTEM)
"""
def dtem_apply_patch(
    model: VisionTransformer,
    feat_dim=None,
    trace_source=True,
    prop_attn=True,
    source_tracking_mode: str = 'map',
    default_r: int = 2,
    window_size: int = None,
    t: int = 1,
    use_softkmax: bool = False,
    swa_size: int = None,
):
    """
    扩展：
    - default_r: 默认合并强度（用于初始化 model.r）
    - window_size: 局部注意力/局部匹配窗口半径（None 或 <=0 表示禁用）
    - t: 合并粒度，使每层实际 r 对齐到 t 的整数倍
    """
    DTEMVisionTransformer = make_tome_class(model.__class__)
    model.__class__ = DTEMVisionTransformer

    model.r = default_r
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source_map": None,
        "source_matrix": None,
        "total_merge": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": getattr(model, 'cls_token', None) is not None,
        "distill_token": getattr(model, 'dist_token', None) is not None,
        "source_tracking_mode": source_tracking_mode,
        "source_trace_mode": "matrix" if trace_source and source_tracking_mode == "matrix" else "none",
        "k2": None,
        "tau1": 1.0,
            "tau2": 30.0,  # Temperature for softmax/ThreTopK (higher = smoother)
        "feat_dim": feat_dim,
        "window_size": window_size,
        "t": t,
        "use_softkmax": use_softkmax,
        "swa_size": swa_size,
    }
    for module in model.modules():
        if isinstance(module, (Block, TimmBlock)):
            module.__class__ = DTEMBlock
            module._tome_info = model._tome_info
        elif isinstance(module, (Attention, TimmAttention)):
            module.__class__ = DTEMAttention
            module._tome_info = model._tome_info
            module.patch(model._tome_info["feat_dim"])


# Archieve Version #

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.layers import Mlp, DropPath, use_fused_attn
# from timm.models.vision_transformer import VisionTransformer, LayerScale
# from timm.models.vision_transformer import Attention as TimmAttention
# from timm.models.vision_transformer import Block as TimmBlock
# from functools import partial

# from opentome.tome.tome import (
#     bipartite_soft_matching,
#     merge_source_matrix,
#     merge_source_map,
#     merge_wavg,
#     parse_r,
# )
# from opentome.timm import Attention, Block


# # ---------------------- dtem 专用通用 map/trace 工具 ---------------------- #
# def merge_source_map_for_dtem(current_level_map, source_map):
#     if source_map is None:
#         return current_level_map
#     return torch.gather(current_level_map, dim=1, index=source_map)

# @torch.no_grad()
# def trace_token_merge(x, size, token_map):
#     """
#     将 token_map 所指定的分组进行加权聚合，返回合并后的 x/size 以及压缩后的映射：
#     - x: [B, L, C]
#     - size: [B, L] 或 [B, L, 1] 或 None（None 表示等权重）
#     - token_map: [B, L]，每个原 token 合并到的目标 group 索引
#     返回 (x_merged [B, M_max, C], size_merged [B, M_max, 1], compact_token_map [B, L])
#     """
#     B, L, C = x.shape
#     device = x.device
#     dtype = x.dtype

#     if size is None:
#         size_local = torch.ones(B, L, device=device, dtype=dtype)
#     else:
#         size_local = size.squeeze(-1) if size.ndim == 3 else size

#     offset = torch.arange(B, device=device, dtype=torch.long).view(B, 1) * L
#     global_token_map = token_map + offset

#     unique_global_indices, inverse_indices = torch.unique(global_token_map.flatten(), return_inverse=True)
#     batch_indices = torch.div(unique_global_indices, L, rounding_mode='floor')

#     perm = torch.argsort(unique_global_indices)
#     sorted_batch_indices = batch_indices[perm]
#     is_new_batch = torch.cat([torch.tensor([True], device=device), sorted_batch_indices[1:] != sorted_batch_indices[:-1]])
#     batch_group_ids = torch.cumsum(is_new_batch.long(), dim=0) - 1
#     segment_starts = torch.where(is_new_batch)[0]
#     local_rank_sorted = torch.arange(len(unique_global_indices), device=device) - segment_starts[batch_group_ids]
#     inv_perm = torch.empty_like(perm)
#     inv_perm[perm] = torch.arange(len(unique_global_indices), device=device)
#     local_rank_unsorted = local_rank_sorted[inv_perm]
#     compact_token_map = local_rank_unsorted[inverse_indices].view(B, L)

#     counts_per_batch = torch.bincount(batch_indices, minlength=B)
#     M_max = counts_per_batch.max().item()

#     index = compact_token_map.unsqueeze(-1).expand_as(x)
#     x_w = x * size_local.unsqueeze(-1)

#     num = torch.zeros(B, M_max, C, device=device, dtype=dtype)
#     den = torch.zeros(B, M_max, device=device, dtype=dtype)

#     num.scatter_add_(dim=1, index=index, src=x_w)
#     den.scatter_add_(dim=1, index=compact_token_map, src=size_local)

#     den_clamped = torch.clamp(den, min=torch.finfo(den.dtype).eps)
#     x_merged = num / den_clamped.unsqueeze(-1)
#     size_merged = den.unsqueeze(-1)

#     return x_merged, size_merged, compact_token_map

# def token_unmerge_from_map_for_dtem(merged_tokens, token_map_compact, T_full):
#     """
#     将长度 L-k 的 merged_tokens 还原到原长度 L：
#     - merged_tokens: [B, L-k, C]
#     - token_map_compact: [B, L]，每个原 token 在合并后序列中的位置
#     - T_full: 原长度 L
#     返回: [B, L, C]
#     """
#     B, Lk, C = merged_tokens.shape
#     device = merged_tokens.device
#     b_idx = torch.arange(B, device=device)[:, None].expand(B, T_full)
#     return merged_tokens[b_idx, token_map_compact]

# class DTEMLinear(nn.Linear):
#     def __init__(self, qkv_layer, feat_dim):
#         super().__init__(in_features=qkv_layer.weight.shape[1], out_features=qkv_layer.weight.shape[0] + feat_dim, bias=True)
#         # qkv
#         self.qkv_layer = qkv_layer

#         # metric
#         self.feat_dim = feat_dim
#         self.metric_layer = nn.Linear(qkv_layer.weight.shape[-1], feat_dim)

#         # copy
#         self.update()

#     @torch.no_grad()
#     def update(self):
#         # qkv -> self
#         self.weight[:-self.feat_dim].copy_(self.qkv_layer.weight)
#         self.bias[:-self.feat_dim].copy_(self.qkv_layer.bias)

#         # metric_layer -> self
#         self.weight[-self.feat_dim:].copy_(self.metric_layer.weight)
#         self.bias[-self.feat_dim:].copy_(self.metric_layer.bias)

#     def train(self, mode=True):
#         if mode is False:   # if eval
#             self.update()
#         return super().train(mode)

#     def forward(self, input: torch.Tensor):
#         if not self.training:
#             out = F.linear(input, self.weight, self.bias)
#             return out[..., :-self.feat_dim], out[..., -self.feat_dim:]

#         # training
#         out1 = self.qkv_layer(input)  # Shape: (B, N, 3 * num_heads * head_dim)
#         out2 = self.metric_layer(input.detach())  # Shape: (B, N, feat_dim)
#         return out1, out2


# try:
#     from flash_attn import flash_attn_func
#     FLASH_ATTN_AVAILABLE = True
# except Exception:
#     FLASH_ATTN_AVAILABLE = False


# """
#     timm - deit patch
# """
# class DTEMAttention(Attention):
#     def patch(self, feat_dim):
#         if feat_dim is not None:
#             out_dim = feat_dim
#         else:
#             dim = self.head_dim * self.num_heads
#             out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim
#         # add metric_layer
#         self.qkv = DTEMLinear(self.qkv, out_dim)

#     def forward(self, x, size=None):    # x:(B, N, C), size:(B, N) or (B, N, 1)
#         B, N, C = x.shape
#         out1, out2 = self.qkv(x)
#         qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)     # B, H, N, head_dim
#         q, k = self.q_norm(q), self.k_norm(k)

#         # fp32 for softmax computation
#         q, k, v = q.type(torch.float32), k.type(torch.float32), v.type(torch.float32)
#         with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
#             q = q * self.scale
#             window_size = self._tome_info.get("window_size")
#             if window_size is not None and window_size > 0:
#                 # 尝试使用 triton 实现；失败回退到 naive/flash
#                 x_bnc = None
#                 try:
#                     from .local_attn_triton import local_attn_sizebias_bhnd
#                     x_bnhd = local_attn_sizebias_bhnd(q, k, v, size, window_size, softmax_scale=1.0)  # (B, N, H, D)
#                     x_bnc = x_bnhd.reshape(B, N, self.num_heads * self.head_dim)
#                     x_bnc = self.attn_drop(x_bnc)
#                 except Exception:
#                     # 先试 flash，再退回 naive
#                     x_bnc = self._flash_local_attention(q, k, v, size, window_size, x_dtype=x.dtype) if FLASH_ATTN_AVAILABLE else None
#                     if x_bnc is None:
#                         x_bnc = self._naive_local_attention(q, k, v, size, window_size, x_dtype=x.dtype)
#             else:
#                 # 全局注意力路径
#                 attn = q @ k.transpose(-2, -1)
#                 if size is None or (not self._tome_info["r"]): # for MAE
#                     attn = attn.softmax(dim=-1)
#                 else:   # as in DynamicViT
#                     _attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
#                     size_ = size if size is not None else torch.ones(B, N, 1, device=x.device, dtype=torch.float32)
#                     _attn = _attn.exp_() * size_[:, None, None, :, 0].type(torch.float32)
#                     attn = _attn / _attn.sum(dim=-1, keepdim=True)
#                 attn = self.attn_drop(attn)
#                 # (B, H, N, D) -> (B, N, C)
#                 x_bnc = (attn @ v).permute(0, 2, 1, 3).contiguous().view(B, N, self.num_heads * self.head_dim)

#         x = x_bnc
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         metric = dict(
#             q = q,
#             k = k,
#             v = v,
#             x = x,
#             metric = out2
#         )
#         return x, metric

#     def _naive_local_attention(self, q, k, v, size, window_size, x_dtype=torch.float32):
#         """
#         无依赖的局部窗口注意力回退实现。
#         q,k,v: (B, H, N, D)
#         size: (B, N, 1) or (B, N) or None
#         返回: (B, N, C)
#         """
#         B, H, N, D = q.shape
#         BH = B * H

#         q_flat = q.transpose(1, 2).reshape(BH, N, D)
#         k_flat = k.transpose(1, 2).reshape(BH, N, D)
#         v_flat = v.transpose(1, 2).reshape(BH, N, D)

#         padded_k = F.pad(k_flat, (0, 0, window_size, window_size))
#         padded_v = F.pad(v_flat, (0, 0, window_size, window_size))

#         k_windows = padded_k.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
#         v_windows = padded_v.unfold(dimension=1, size=2 * window_size + 1, step=1)  # (BH, N, win, D)
#         k_windows = k_windows.transpose(-1, -2)
#         v_windows = v_windows.transpose(-1, -2)
#         q_reshaped = q_flat.unsqueeze(2)  # (BH, N, 1, D)
#         attn = (q_reshaped @ k_windows.transpose(-1, -2)).squeeze(2)  # (BH, N, win)

#         if size is not None:
#             size_log = size.log().squeeze(-1) if size.ndim == 3 else size.log()
#             size_bias_bh = size_log.unsqueeze(1).expand(-1, H, -1).reshape(BH, N)
#             padded_size_bias = F.pad(size_bias_bh, (window_size, window_size), mode='constant', value=-1e9)
#             size_bias_windows = padded_size_bias.unfold(dimension=1, size=2 * window_size + 1, step=1)
#             attn = attn + size_bias_windows

#         win_indices = torch.arange(-window_size, window_size + 1, device=q.device).view(1, -1)
#         q_indices = torch.arange(N, device=q.device).view(-1, 1)
#         abs_k_pos = q_indices + win_indices  # (N, win)
#         mask = (abs_k_pos >= 0) & (abs_k_pos < N)
#         attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         attn_out_flat = (attn.unsqueeze(2) @ v_windows).squeeze(2)  # (BH, N, D)
#         attn_out = attn_out_flat.view(B, H, N, D).transpose(1, 2)  # (B, N, H, D)
#         x_processed = attn_out.reshape(B, N, H * D).to(x_dtype)
#         return x_processed

#     def _flash_local_attention(self, q, k, v, size, window_size, x_dtype=torch.float32):
#         """
#         使用 FlashAttention 的局部窗口注意力；失败返回 None。
#         """
#         B, H, N, D = q.shape
#         try:
#             q_nhw = q.permute(0, 2, 1, 3).contiguous()
#             k_nhw = k.permute(0, 2, 1, 3).contiguous()
#             v_nhw = v.permute(0, 2, 1, 3).contiguous()

#             if q_nhw.dtype not in (torch.float16, torch.bfloat16):
#                 q_nhw = q_nhw.to(torch.float16)
#                 k_nhw = k_nhw.to(torch.float16)
#                 v_nhw = v_nhw.to(torch.float16)

#             out = flash_attn_func(
#                 q_nhw, k_nhw, v_nhw,
#                 dropout_p=self.attn_drop.p if self.training else 0.0,
#                 softmax_scale=1.0,
#                 causal=False,
#                 window_size=(window_size, window_size),
#                 deterministic=True,
#             )

#             if out.ndim == 3 and out.shape[-1] == H * D:
#                 out = out.view(B, N, H, D)
#             elif out.ndim != 4 or out.shape[:3] != (B, N, H):
#                 out = out.reshape(B, N, H, D)

#             attn_out_bhnd = out.permute(0, 2, 1, 3).contiguous()
#             x_bnc = attn_out_bhnd.transpose(1, 2).reshape(B, N, H * D).to(x_dtype)
#             return x_bnc
#         except Exception:
#             return None



# class DTEMBlock(Block):

#     def _select(self, k, a, b):
#         """
#         统一的选择逻辑：支持全局或局部窗口匹配，并支持跨层屏蔽。
#         a: (B, Na, C), b: (B, Nb, C)
#         返回 assign: (B, Na, Nb)
#         """
#         EPSILON = torch.finfo(torch.float32).eps
#         B, Na, C = a.shape
#         _, Nb, _ = b.shape
#         device = a.device

#         window_size = self._tome_info.get("window_size")
#         if window_size is None or window_size <= 0:
#             scores = a @ b.transpose(-1, -2)
#             token_mask = self._tome_info.get("token_mask_for_dtem", None)
#             if token_mask is not None:
#                 Na_cur = a.shape[1]
#                 Nb_cur = b.shape[1]
#                 mask_a = token_mask[:, 1:1 + 2 * Na_cur:2]
#                 mask_b = token_mask[:, 2:2 + 2 * Nb_cur:2]
#                 neg_inf = torch.finfo(scores.dtype).min
#                 scores = scores.masked_fill(mask_a.unsqueeze(-1), neg_inf)
#                 scores = scores.masked_fill(mask_b.unsqueeze(1), neg_inf)
#             _idx = scores.argsort(dim=-1, descending=True)
#             _x = scores.gather(dim=-1, index=_idx)
#             _x = _x / (self._tome_info.get("tau1", 1.0))
#             khot = torch.zeros_like(_x)
#             for _ in range(k):
#                 onehot_approx = F.softmax(_x.view(B, -1) / (self._tome_info.get("tau2", 0.1)), dim=-1).view(B, Na, Nb)
#                 khot += onehot_approx
#                 khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
#                 _x = _x + torch.log(khot_mask)
#             tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
#             nkhot = khot / tmp
#             nkhot = nkhot.to(scores.dtype)
#             assign = torch.zeros_like(scores).scatter_reduce(-1, _idx, nkhot, reduce='sum')
#         else:
#             padded_b = F.pad(b, (0, 0, window_size, window_size))
#             b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
#             b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
#             a_reshaped = a.unsqueeze(2)
#             local_scores = (a_reshaped * b_windows).sum(dim=-1) / (self._tome_info.get("tau1", 1.0))

#             token_mask = self._tome_info.get("token_mask_for_dtem", None)
#             if token_mask is not None:
#                 Na_cur = a.shape[1]
#                 Nb_cur = b.shape[1]
#                 mask_a = token_mask[:, 1:1 + 2 * Na_cur:2]
#                 mask_b = token_mask[:, 2:2 + 2 * Nb_cur:2]
#                 neg_inf_local = torch.finfo(local_scores.dtype).min
#                 local_scores = local_scores.masked_fill(mask_a.unsqueeze(-1), neg_inf_local)
#                 padded_mask_b = F.pad(mask_b, (window_size, window_size))
#                 mask_windows_unfolded = padded_mask_b.unfold(dimension=1, size=2*window_size+1, step=1)
#                 local_scores = local_scores.masked_fill(mask_windows_unfolded, neg_inf_local)

#             khot = torch.zeros_like(local_scores)
#             for _ in range(k):
#                 onehot_approx = F.softmax(local_scores / (self._tome_info.get("tau2", 0.1)), dim=-1)
#                 khot += onehot_approx
#                 khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
#                 local_scores = local_scores + torch.log(khot_mask)

#             tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
#             nkhot = khot / tmp

#             a_indices = torch.arange(Na, device=device).view(1, -1, 1)
#             window_offsets = torch.arange(-window_size, window_size + 1, device=device).view(1, 1, -1)
#             b_indices = a_indices + window_offsets
#             valid_mask = (b_indices >= 0) & (b_indices < Nb)
#             assign = torch.zeros(B, Na, Nb, device=device, dtype=a.dtype)
#             nkhot_masked = torch.where(valid_mask.expand(B, -1, -1), nkhot, 0.0)
#             b_indices_clamped = b_indices.clamp(0, Nb - 1)
#             b_indices_expanded = b_indices_clamped.expand(B, -1, -1)
#             assign.scatter_add_(dim=2, index=b_indices_expanded, src=nkhot_masked)

#         with torch.no_grad():
#             out_dict = {
#                 'num': (nkhot if 'nkhot' in locals() else khot).sum().item(),
#                 'max': (khot if 'khot' in locals() else nkhot).view(B, -1).max(dim=-1)[0].sum().item(),
#             }
#         return assign, out_dict

#     @torch.no_grad()
#     def _build_token_map_hard(self, metric_a, metric_b, r, n, t_orig, class_token, distill_token):
#         B, Na, D = metric_a.shape
#         _, Nb, _ = metric_b.shape
#         device = metric_a.device

#         if r == 0:
#             token_map = torch.arange(t_orig, device=device).unsqueeze(0).expand(B, -1)
#             src_mask = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
#             return token_map, src_mask

#         window_size = self._tome_info.get("window_size")
#         if window_size is not None and window_size > 0:
#             padded_b = F.pad(metric_b, (0, 0, window_size, window_size if Na == Nb else window_size + 1))
#             b_windows_unfolded = padded_b.unfold(dimension=1, size=2 * window_size + 1, step=1)
#             b_windows = b_windows_unfolded.permute(0, 1, 3, 2)
#             a_reshaped = metric_a.unsqueeze(2)
#             local_scores = (a_reshaped * b_windows).sum(dim=-1)
#             node_max, local_node_idx = local_scores.max(dim=-1)
#             node_idx = torch.arange(Na, device=device).view(1, -1) + local_node_idx - window_size
#         else:
#             scores = metric_a @ metric_b.transpose(-1, -2)
#             node_max, node_idx = scores.max(dim=-1)
#         edge_idx = node_max.argsort(dim=-1, descending=True)
#         src_a_idx = edge_idx[:, :r]
#         unm_a_idx = edge_idx[:, r:]
#         dst_b_idx = torch.gather(node_idx, dim=1, index=src_a_idx)
#         src_orig_idx = 1 + 2 * src_a_idx
#         unm_orig_idx = 1 + 2 * unm_a_idx
#         dst_orig_idx = 2 + 2 * dst_b_idx
#         b_all_orig_idx = torch.arange(2, 2 * Nb + 2, 2, device=device).unsqueeze(0).expand(B, -1)

#         num_unm = Na - r
#         unm_new_idx = torch.arange(1, num_unm + 1, device=device).unsqueeze(0).expand(B, -1)
#         b_all_new_idx = torch.arange(num_unm + 1, num_unm + 1 + Nb, device=device).unsqueeze(0).expand(B, -1)
#         dst_new_idx = torch.gather(b_all_new_idx, dim=1, index=dst_b_idx)

#         token_map_batch = torch.arange(t_orig, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
#         src_mask_batch = torch.zeros((B, t_orig), device=device, dtype=torch.bool)
#         map_builder = token_map_batch.clone()
#         if unm_orig_idx.numel() > 0:
#             map_builder.scatter_(dim=1, index=unm_orig_idx, src=unm_new_idx)
#         map_builder.scatter_(dim=1, index=b_all_orig_idx, src=b_all_new_idx)
#         map_builder.scatter_(dim=1, index=src_orig_idx, src=dst_new_idx)
#         src_mask_batch.scatter_(dim=1, index=src_orig_idx, value=True)
#         return map_builder, src_mask_batch

#     def _merge_train(self, x, size, r, n, metric):
#         metric = metric['metric']
#         metric = metric / metric.norm(dim=-1, keepdim=True)

#         n = n if self.training else x.size()[1]
#         r = min(r, (n - 1) // 2)
#         if r <= 0:
#             return x, size, n, metric

#         # 保证 size 为 [B, N, 1]
#         if size is not None and size.ndim == 2:
#             size = size.unsqueeze(-1)

#         xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
#         a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
#         wa = size[..., 1:n:2, 0]
#         wb = size[..., 2:n:2, 0]

#         # 对齐合并粒度 t
#         t = max(int(self._tome_info.get("t", 1)), 1)
#         r = (r // t) * t
#         assign, _out = self._select(k=r, a=a, b=b)

#         # 层间屏蔽保护（如果存在）
#         token_mask = self._tome_info.get("token_mask_for_dtem", None)
#         if token_mask is not None:
#             mask_a = token_mask[:, 1:n:2]
#             mask_b = token_mask[:, 2:n:2]
#             combined_mask = mask_a.unsqueeze(2) | mask_b.unsqueeze(1)
#             assign.masked_fill_(combined_mask, 0)

#         with torch.no_grad():
#             B = a.shape[0]
#             T_full = x.shape[1]
#             current_level_map, src_mask_batch = self._build_token_map_hard(
#                 a, b, r, B, T_full,
#                 self._tome_info["class_token"],
#                 self._tome_info["distill_token"],
#             )
#             if self._tome_info.get("token_map_for_dtem", None) is None:
#                 self._tome_info["token_map_for_dtem"] = torch.arange(T_full, device=x.device)[None, :].expand(B, -1).clone()
#             if self._tome_info.get("token_mask_for_dtem", None) is None:
#                 self._tome_info["token_mask_for_dtem"] = torch.zeros(B, T_full, dtype=torch.bool, device=x.device)
#             token_map_global = merge_source_map_for_dtem(current_level_map, self._tome_info["token_map_for_dtem"])
#             self._tome_info["token_map_for_dtem"] = token_map_global
#             self._tome_info["token_mask_for_dtem"] = self._tome_info["token_mask_for_dtem"] | src_mask_batch

#         xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)
#         wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]
#         tmp = 1 - assign.sum(dim=-1)
#         wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())
#         wb_safe = torch.clamp(wb, min=torch.finfo(wb.dtype).eps)
#         xb = xb / wb_safe[..., None]

#         w = torch.cat([wa, wb], dim=-1)
#         nx = torch.cat([xa, xb], dim=1)
#         nidxs = w.argsort(dim=-1, descending=True)
#         w = w.gather(dim=-1, index=nidxs)
#         nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

#         x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
#         size_output = torch.cat([size[:, :1, 0], w, size[:, n:, 0]], dim=-1).unsqueeze(-1)
#         return x_output, size_output, n - r, _out

#     def _merge_eval(self, x, size, r, metric):    # eval：保留 ToMe 行为并补充 token_map_for_dtem 追踪
#         metric = metric['metric']
#         metric = metric / metric.norm(dim=-1, keepdim=True)

#         merge, _, current_level_map = bipartite_soft_matching(metric,
#                                            r,
#                                            self._tome_info["class_token"],
#                                            self._tome_info["distill_token"],
#                                            )
#         if self._tome_info["trace_source"]:
#             # 兼容旧 map/matrix 追踪
#             if "source_tracking_mode" in self._tome_info:
#                 if self._tome_info["source_tracking_mode"] == 'map':
#                     source_map = self._tome_info["source_map"]
#                     if source_map is None:
#                         b, t, _ = x.shape
#                         source_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
#                     self._tome_info["source_map"] = merge_source_map(current_level_map, x, source_map)
#                 else: # 'matrix'
#                     source_matrix = self._tome_info["source_matrix"]
#                     self._tome_info["source_matrix"] = merge_source_matrix(merge, x, source_matrix)
#             # 新 token_map_for_dtem 追踪（总是尝试维护）
#             token_map = self._tome_info.get("token_map_for_dtem", None)
#             b, t, _ = x.shape
#             if token_map is None:
#                 token_map = torch.arange(t, device=x.device, dtype=torch.long).expand(b, -1)
#             self._tome_info["token_map_for_dtem"] = merge_source_map_for_dtem(current_level_map, token_map)

#         x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
#         return x, self._tome_info["size"], x.size(1), None

#     def merge(self, x, size, r, n, metric):
#         return self._merge_train(x, size, r, n, metric) if self.training else self._merge_eval(x, size, r, metric)

#     def forward(self, x, size, n=None):
#         if size is None or n is None:
#             tmp, _ = self.attn(self.norm1(x))
#             x = x + self.drop_path1(self.ls1(tmp))
#             x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#             return x
#         else:
#             tmp, metric = self.attn(self.norm1(x), size=size)
#             assert isinstance(metric['metric'], (float, torch.Tensor)), "metric not a float or torch.Tensor"
#             x = x + self.drop_path1(self.ls1(tmp))
#             # Merging
#             r = self._tome_info["r"].pop(0)
#             if size is not None and r > 0 and n > 0:
#                 x, size, n, metric = self.merge(x, size, r, n, metric)
#             x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

#             return x, size, n, metric


# def make_tome_class(transformer_class):
#     class DTEMVisionTransformer(transformer_class):
#         """
#         Modifications:
#         - Initialize r, token size, and token sources.
#         """

#         def forward_features(self, x):
#             x = self.patch_embed(x)
#             x = self._pos_embed(x)
#             x = self.patch_drop(x)
#             x = self.norm_pre(x)

#             n = x.size(1)
#             self._tome_info["r"] = parse_r(
#                 len(self.blocks), self.r, self._tome_info["total_merge"]
#             )
#             self._tome_info["size"] = torch.ones_like(x[..., 0, None])
#             self._tome_info["source_map"] = None
#             self._tome_info["source_matrix"] = None

#             out_dicts = []
#             for block in self.blocks:
#                 x, size, n, out_dict = block(x, self._tome_info["size"], n=n)
#                 out_dicts.append(out_dict)

#             x = self.norm(x)
#             return x, out_dicts

#         def forward(self, x, return_out_dicts=False):
#             x, out_dicts = self.forward_features(x)
#             x = self.forward_head(x)
#             if return_out_dicts:
#                 return x, out_dicts
#             return x

#     return DTEMVisionTransformer



# """"
# Learning to Merge Tokens via Decoupled Embedding for Efficient Vision Transformers, NIPS'2024
#     - paper (https://openreview.net/forum?id=pVPyCgXv57)
#     - code  (https://github.com/movinghoon/DTEM)
# """
# def dtem_apply_patch(
#     model: VisionTransformer,
#     feat_dim=None,
#     trace_source=True,
#     prop_attn=True,
#     source_tracking_mode: str = 'map',
#     default_r: int = 2,
#     window_size: int = None,
#     t: int = 1,
# ):
#     """
#     扩展：
#     - default_r: 默认合并强度（用于初始化 model.r）
#     - window_size: 局部注意力/局部匹配窗口半径（None 或 <=0 表示禁用）
#     - t: 合并粒度，使每层实际 r 对齐到 t 的整数倍
#     """
#     DTEMVisionTransformer = make_tome_class(model.__class__)
#     model.__class__ = DTEMVisionTransformer

#     model.r = default_r
#     model._tome_info = {
#         "r": model.r,
#         "size": None,
#         # 旧版追踪结构：保留兼容
#         "source_map": None,
#         "source_matrix": None,
#         "total_merge": None,
#         "trace_source": trace_source,
#         "prop_attn": prop_attn,
#         "class_token": getattr(model, 'cls_token', None) is not None,
#         "distill_token": getattr(model, 'dist_token', None) is not None,
#         "source_tracking_mode": source_tracking_mode,
#         # DTEM 超参
#         "k2": None,
#         "tau1": 1.0,
#         "tau2": 0.1,
#         "feat_dim": feat_dim,
#         # 新的持续 token 追踪与屏蔽
#         "token_map_for_dtem": None,   # [B, T]
#         "token_mask_for_dtem": None,  # [B, T] bool
#         # 局部窗口与合并粒度
#         "window_size": window_size,
#         "t": t,
#     }

#     for module in model.modules():
#         if isinstance(module, (Block, TimmBlock)):
#             module.__class__ = DTEMBlock
#             module._tome_info = model._tome_info
#         elif isinstance(module, (Attention, TimmAttention)):
#             module.__class__ = DTEMAttention
#             module._tome_info = model._tome_info
#             module.patch(model._tome_info["feat_dim"])
