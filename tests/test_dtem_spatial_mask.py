#!/usr/bin/env python3
"""CPU-only regression checks for DTEM's 2-D patch-distance mask."""

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opentome.timm.dtem import (
    DTEMBlock,
    build_patch_spatial_adjacency,
    gather_patch_spatial_mask,
)
from opentome.utils.thetopk import ThreTopK


def expect_failure(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except (IndexError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(
        f"expected {getattr(call, '__name__', repr(call))} to reject "
        f"args={args!r}, kwargs={kwargs!r}"
    )


def test_euclidean_row_boundary_geometry():
    adjacency = build_patch_spatial_adjacency(
        grid_size=(2, 3), radius=1, metric="euclidean", device="cpu"
    )
    assert adjacency.shape == (6, 6)
    assert adjacency.dtype == torch.bool
    assert adjacency.device.type == "cpu"

    # Row-major patch 2 is at (0, 2), while patch 3 is at (1, 0).
    # Their flattened indices differ by one, but their real 2-D distance is
    # sqrt(5), so an R=1 spatial mask must reject this row-wrap edge.
    assert not bool(adjacency[2, 3])
    assert not bool(adjacency[3, 2])

    # Patch 0=(0, 0) and patch 3=(1, 0) are vertical neighbours even though
    # their flattened indices differ by the full grid width.
    assert bool(adjacency[0, 3])
    assert bool(adjacency[3, 0])
    assert bool(torch.diagonal(adjacency).all())
    assert torch.equal(adjacency, adjacency.transpose(0, 1))


def test_large_radius_is_global():
    grid_size = (3, 5)
    diagonal = math.hypot(grid_size[0] - 1, grid_size[1] - 1)
    adjacency = build_patch_spatial_adjacency(
        grid_size=grid_size,
        radius=diagonal,
        metric="euclidean",
    )
    assert adjacency.shape == (15, 15)
    assert bool(adjacency.all())


def test_cls_offset_and_batched_gather():
    adjacency = build_patch_spatial_adjacency(
        grid_size=(2, 3), radius=1, metric="euclidean"
    )

    # Indices are absolute token positions, including a single CLS prefix.
    # The two samples intentionally use different donor/receiver layouts.
    a_orig_idx = torch.tensor([[1, 3], [4, 6]], dtype=torch.long)
    b_orig_idx = torch.tensor([[4, 6, 2], [1, 3, 5]], dtype=torch.long)
    actual = gather_patch_spatial_mask(
        adjacency,
        a_orig_idx,
        b_orig_idx,
        num_prefix_tokens=1,
    )

    expected = torch.stack(
        [
            adjacency[a_orig_idx[batch, :, None] - 1,
                      b_orig_idx[batch, None, :] - 1]
            for batch in range(a_orig_idx.shape[0])
        ],
        dim=0,
    )
    assert actual.shape == (2, 2, 3)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)

    # Explicitly retain the two geometry counterexamples after the CLS shift.
    wrapped = gather_patch_spatial_mask(
        adjacency,
        torch.tensor([[3]]),  # patch 2
        torch.tensor([[4]]),  # patch 3
        num_prefix_tokens=1,
    )
    vertical = gather_patch_spatial_mask(
        adjacency,
        torch.tensor([[1]]),  # patch 0
        torch.tensor([[4]]),  # patch 3
        num_prefix_tokens=1,
    )
    assert not bool(wrapped.item())
    assert bool(vertical.item())


def test_invalid_parameters_and_indices_fail_closed():
    for grid_size in (0, -1, (0, 3), (2, 0), (2.0, 3), (2,), (2, 3, 4)):
        expect_failure(
            build_patch_spatial_adjacency,
            grid_size=grid_size,
            radius=1,
            metric="euclidean",
        )
    expect_failure(
        build_patch_spatial_adjacency,
        grid_size=(2, 3),
        radius=-1,
        metric="euclidean",
    )
    expect_failure(
        build_patch_spatial_adjacency,
        grid_size=(2, 3),
        radius=float("nan"),
        metric="euclidean",
    )
    expect_failure(
        build_patch_spatial_adjacency,
        grid_size=(2, 3),
        radius=1,
        metric="flat_index",
    )

    adjacency = build_patch_spatial_adjacency((2, 3), 1)
    valid = torch.tensor([[1]], dtype=torch.long)
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        torch.tensor([[0]], dtype=torch.long),  # points into the CLS prefix
        valid,
        num_prefix_tokens=1,
    )
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        torch.tensor([[7]], dtype=torch.long),  # patch index 6 is out of range
        valid,
        num_prefix_tokens=1,
    )
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        valid,
        torch.tensor([[7]], dtype=torch.long),
        num_prefix_tokens=1,
    )
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        valid,
        valid,
        num_prefix_tokens=-1,
    )
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        valid,
        valid,
        num_prefix_tokens=1.0,
    )
    expect_failure(
        gather_patch_spatial_mask,
        adjacency,
        torch.tensor([[1], [2]], dtype=torch.long),
        torch.tensor([[1]], dtype=torch.long),
        num_prefix_tokens=1,
    )


def make_select_block(adjacency, grid_size, radius=1):
    # _select only consumes _tome_info, so bypass the parameter-heavy timm Block
    # constructor to keep this regression test CPU-only and focused on routing.
    block = DTEMBlock.__new__(DTEMBlock)
    torch.nn.Module.__init__(block)
    block._tome_info = {
        "window_size": 0,
        "dtem_spatial_radius": radius,
        "dtem_spatial_metric": "euclidean",
        "dtem_spatial_backend": "dense",
        "patch_grid_size": grid_size,
        "num_prefix_tokens": 1,
        "spatial_adjacency": adjacency,
        "tau1": 1.0,
        "tau2": 30.0,
        "use_softkmax": True,
        "collect_merge_stats": False,
    }
    return block


def test_select_masks_edges_before_assignment_and_handles_empty_rows():
    adjacency = build_patch_spatial_adjacency((2, 3), 1)
    block = make_select_block(adjacency, (2, 3))

    # Patch 0 has no R=1 receiver in this partition. Patches 1 and 3 do, so
    # k=1 remains feasible and the isolated donor must simply retain its mass.
    a_orig_idx = torch.tensor([[1, 2, 4]], dtype=torch.long)  # patches 0, 1, 3
    b_orig_idx = torch.tensor([[3, 5, 6]], dtype=torch.long)  # patches 2, 4, 5
    a = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]], requires_grad=True
    )
    b = torch.tensor(
        [[[0.2, 0.8], [0.0, 1.0], [1.0, 0.0]]], requires_grad=True
    )

    allowed = gather_patch_spatial_mask(
        adjacency, a_orig_idx, b_orig_idx, num_prefix_tokens=1
    )
    assert not bool(allowed[0, 0].any())
    assert int(allowed.any(dim=-1).sum()) >= 1

    assign, _ = block._select(
        k=1,
        a=a,
        b=b,
        a_orig_idx=a_orig_idx,
        b_orig_idx=b_orig_idx,
    )
    assert assign.shape == allowed.shape
    assert bool(torch.isfinite(assign).all())
    assert int(torch.count_nonzero(assign.masked_select(~allowed))) == 0
    assert int(torch.count_nonzero(assign[0, 0])) == 0
    assign.square().sum().backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad).all())
    assert b.grad is not None and bool(torch.isfinite(b.grad).all())


def test_select_rejects_an_infeasible_spatial_budget():
    # A diagonal-only adjacency gives the disjoint donor/receiver groups no
    # legal edge. The implementation must fail closed before softmax rather
    # than emit NaNs or silently route beyond the spatial threshold.
    adjacency = torch.eye(2, dtype=torch.bool)
    block = make_select_block(adjacency, (1, 2), radius=0.5)
    expect_failure(
        block._select,
        k=1,
        a=torch.tensor([[[1.0, 0.0]]]),
        b=torch.tensor([[[0.0, 1.0]]]),
        a_orig_idx=torch.tensor([[1]], dtype=torch.long),
        b_orig_idx=torch.tensor([[2]], dtype=torch.long),
    )


def test_global_softkmax_matches_legacy_sorted_formulation():
    torch.manual_seed(20260814)
    batch, donors, receivers, channels, k = 2, 5, 6, 4, 2
    a = torch.randn(batch, donors, channels)
    b = torch.randn(batch, receivers, channels)
    block = make_select_block(
        build_patch_spatial_adjacency((2, 3), 8),
        (2, 3),
        radius=8,
    )
    block._tome_info["dtem_spatial_radius"] = None
    block._tome_info["spatial_adjacency"] = None
    actual, _ = block._select(k=k, a=a, b=b)

    scores = a @ b.transpose(-1, -2)
    sorted_index = scores.argsort(dim=-1, descending=True)
    sorted_scores = scores.gather(dim=-1, index=sorted_index)
    energy = torch.logsumexp(sorted_scores, dim=-1).exp()
    selected_rows = ThreTopK(energy, k, block._tome_info["tau2"])
    khot = torch.softmax(sorted_scores, dim=-1) * selected_rows.unsqueeze(-1)
    normalizer = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0) + 1
    legacy_sorted = khot / normalizer
    expected = torch.zeros_like(scores).scatter_reduce(
        -1, sorted_index, legacy_sorted, reduce="sum"
    )
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def main():
    test_euclidean_row_boundary_geometry()
    test_large_radius_is_global()
    test_cls_offset_and_batched_gather()
    test_invalid_parameters_and_indices_fail_closed()
    test_select_masks_edges_before_assignment_and_handles_empty_rows()
    test_select_rejects_an_infeasible_spatial_budget()
    test_global_softkmax_matches_legacy_sorted_formulation()
    print("DTEM_SPATIAL_MASK_TEST_PASS")


if __name__ == "__main__":
    main()
