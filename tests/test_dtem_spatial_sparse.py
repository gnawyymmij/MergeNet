#!/usr/bin/env python3
"""Dense-reference parity gates for DTEM's true sparse 2-D scorer."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opentome.timm.dtem as dtem_module
from opentome.timm.dtem import (
    DTEMBlock,
    adjacency_to_padded_neighbors,
    build_patch_spatial_adjacency,
    build_patch_spatial_neighbors,
    densify_sparse_assignment,
    gather_patch_spatial_mask,
    gather_patch_spatial_neighbors,
    sparse_neighbor_dot,
    sparse_weighted_transport,
)


def make_select_block(grid_size, radius, backend, use_softkmax=True):
    adjacency = build_patch_spatial_adjacency(grid_size, radius)
    neighbor_indices, neighbor_valid = adjacency_to_padded_neighbors(adjacency)
    block = DTEMBlock.__new__(DTEMBlock)
    torch.nn.Module.__init__(block)
    block._tome_info = {
        "window_size": 0,
        "dtem_spatial_radius": radius,
        "dtem_spatial_metric": "euclidean",
        "dtem_spatial_backend": backend,
        "patch_grid_size": grid_size,
        "num_prefix_tokens": 1,
        "spatial_adjacency": adjacency,
        "spatial_neighbor_indices": neighbor_indices,
        "spatial_neighbor_valid": neighbor_valid,
        "tau1": 1.0,
        "tau2": 30.0,
        "use_softkmax": use_softkmax,
        "collect_merge_stats": False,
    }
    return block


def test_rectangular_neighbor_table_and_boundaries():
    adjacency = build_patch_spatial_adjacency((3, 5), radius=1.5)
    indices, valid = build_patch_spatial_neighbors((3, 5), radius=1.5)
    assert indices.shape == valid.shape == (15, 9)

    reconstructed_count = torch.zeros_like(adjacency, dtype=torch.long)
    reconstructed_count.scatter_add_(1, indices, valid.long())
    reconstructed = reconstructed_count.bool()
    assert torch.equal(reconstructed, adjacency)

    # patch 4=(0,4), patch 5=(1,0): adjacent after flattening but far in 2-D.
    assert not bool(reconstructed[4, 5])
    # patch 4=(0,4), patch 9=(1,4): a real vertical neighbor.
    assert bool(reconstructed[4, 9])
    # Padded entries at a corner are invalid and never become candidate edges.
    assert int(valid[0].sum()) == 4


def test_partition_mapping_matches_dense_mask_for_batches():
    adjacency = build_patch_spatial_adjacency((3, 5), radius=2)
    neighbor_indices, neighbor_valid = adjacency_to_padded_neighbors(adjacency)
    a_idx = torch.tensor(
        [[1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14]],
        dtype=torch.long,
    )
    b_idx = torch.tensor(
        [[2, 4, 6, 8, 10, 12, 14, 15], [1, 3, 5, 7, 9, 11, 13, 15]],
        dtype=torch.long,
    )
    receiver_indices, receiver_valid = gather_patch_spatial_neighbors(
        neighbor_indices, neighbor_valid, a_idx, b_idx, num_prefix_tokens=1
    )
    sparse_mask_dense = densify_sparse_assignment(
        receiver_valid.to(torch.float32),
        receiver_indices,
        receiver_valid,
        num_receivers=b_idx.shape[1],
    ).bool()
    dense_mask = gather_patch_spatial_mask(
        adjacency, a_idx, b_idx, num_prefix_tokens=1
    )
    assert torch.equal(sparse_mask_dense, dense_mask)


def dot_parity_case(device, dtype=torch.float32):
    torch.manual_seed(20260820)
    batch, donors, receivers, neighbors, channels = 2, 7, 8, 6, 11
    a = torch.randn(batch, donors, channels, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(batch, receivers, channels, device=device, dtype=dtype, requires_grad=True)
    receiver_indices = torch.randint(
        receivers, (batch, donors, neighbors), device=device
    )
    receiver_valid = torch.rand(batch, donors, neighbors, device=device) > 0.25

    actual = sparse_neighbor_dot(a, b, receiver_indices, receiver_valid)
    batch_index = torch.arange(batch, device=device)[:, None, None]
    expected = (
        a.unsqueeze(2) * b[batch_index, receiver_indices]
    ).sum(dim=-1) * receiver_valid
    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-6
    assert torch.allclose(actual, expected, atol=atol, rtol=atol)

    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(), (a, b), retain_graph=True
    )
    expected_grad = torch.autograd.grad((expected * probe).sum(), (a, b))
    grad_atol = 8e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    for got, want in zip(actual_grad, expected_grad):
        assert bool(torch.isfinite(got).all())
        assert torch.allclose(got, want, atol=grad_atol, rtol=grad_atol)


def test_sparse_dot_cpu_forward_backward():
    dot_parity_case(torch.device("cpu"), torch.float64)


def transport_parity_case(device, dtype):
    torch.manual_seed(20260820)
    batch, donors, receivers, neighbors, channels = 2, 7, 8, 6, 11
    assignment = torch.rand(
        batch, donors, neighbors, device=device, dtype=dtype, requires_grad=True
    )
    values = torch.randn(
        batch, donors, channels, device=device, dtype=dtype, requires_grad=True
    )
    receiver_indices = torch.randint(
        receivers, (batch, donors, neighbors), device=device
    )
    receiver_valid = torch.rand(batch, donors, neighbors, device=device) > 0.25

    actual = sparse_weighted_transport(
        assignment, values, receiver_indices, receiver_valid, receivers
    )
    expected = torch.zeros(
        batch, receivers, channels, device=device, dtype=dtype
    )
    expected.scatter_add_(
        dim=1,
        index=receiver_indices.flatten(1).unsqueeze(-1).expand(
            -1, -1, channels
        ),
        src=(
            assignment.unsqueeze(-1)
            * values.unsqueeze(2)
            * receiver_valid.unsqueeze(-1)
        ).flatten(1, 2),
    )
    atol = 1e-2 if dtype == torch.float16 else 1e-5
    assert torch.allclose(actual, expected, atol=atol, rtol=atol)

    probe = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(
        (actual * probe).sum(), (assignment, values), retain_graph=True
    )
    expected_grad = torch.autograd.grad(
        (expected * probe).sum(), (assignment, values)
    )
    grad_atol = 1e-2 if dtype == torch.float16 else 1e-5
    for got, want in zip(actual_grad, expected_grad):
        assert bool(torch.isfinite(got).all())
        assert torch.allclose(got, want, atol=grad_atol, rtol=grad_atol)
    assert int(torch.count_nonzero(actual_grad[0].masked_select(~receiver_valid))) == 0


def test_sparse_transport_cpu_forward_backward():
    transport_parity_case(torch.device("cpu"), torch.float64)


def select_parity_case(device, use_softkmax=True):
    torch.manual_seed(20260820)
    grid_size, radius = (3, 5), 2
    # Same sorted donor/receiver partitions used by _merge_train.
    a_idx = torch.tensor(
        [[1, 3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12, 14]],
        device=device,
        dtype=torch.long,
    )
    b_idx = torch.tensor(
        [[2, 4, 6, 8, 10, 12, 14, 15], [1, 3, 5, 7, 9, 11, 13, 15]],
        device=device,
        dtype=torch.long,
    )
    dense = make_select_block(
        grid_size, radius, "dense", use_softkmax=use_softkmax
    ).to(device)
    sparse = make_select_block(
        grid_size, radius, "sparse", use_softkmax=use_softkmax
    ).to(device)

    # Bare DTEMBlock test fixtures do not own registered buffers, so move the
    # shared geometry in _tome_info explicitly.
    for block in (dense, sparse):
        for key in (
            "spatial_adjacency",
            "spatial_neighbor_indices",
            "spatial_neighbor_valid",
        ):
            block._tome_info[key] = block._tome_info[key].to(device)

    dtype = torch.float64 if device.type == "cpu" else torch.float32
    a_dense = torch.randn(2, 7, 9, device=device, dtype=dtype, requires_grad=True)
    b_dense = torch.randn(2, 8, 9, device=device, dtype=dtype, requires_grad=True)
    a_sparse = a_dense.detach().clone().requires_grad_()
    b_sparse = b_dense.detach().clone().requires_grad_()
    dense_assignment, _ = dense._select(3, a_dense, b_dense, a_idx, b_idx)
    sparse_assignment, _ = sparse._select(3, a_sparse, b_sparse, a_idx, b_idx)
    sparse_dense = densify_sparse_assignment(
        sparse_assignment,
        sparse._tome_info["assign_b_indices"],
        sparse._tome_info["assign_valid_mask"],
        b_idx.shape[1],
    )
    atol = 3e-10 if dtype == torch.float64 else 2e-5
    assert torch.allclose(sparse_dense, dense_assignment, atol=atol, rtol=atol)

    probe = torch.randn_like(dense_assignment)
    dense_grad = torch.autograd.grad(
        (dense_assignment * probe).sum(), (a_dense, b_dense)
    )
    sparse_grad = torch.autograd.grad(
        (sparse_dense * probe).sum(), (a_sparse, b_sparse)
    )
    grad_atol = 1e-8 if dtype == torch.float64 else 8e-5
    for got, want in zip(sparse_grad, dense_grad):
        assert bool(torch.isfinite(got).all())
        assert torch.allclose(got, want, atol=grad_atol, rtol=grad_atol)


def test_select_cpu_dense_sparse_forward_backward():
    select_parity_case(torch.device("cpu"))


def test_select_nonsoft_cpu_dense_sparse_forward_backward():
    # Compatibility path: iterative flattened soft top-k. In sparse mode its
    # score width is K, deliberately different from Nb for this geometry.
    select_parity_case(torch.device("cpu"), use_softkmax=False)


def test_all_invalid_sparse_rows_and_budget_fail_closed():
    # Radius < 1 leaves only self edges, while donor/receiver partitions are
    # disjoint. Every sparse row is therefore invalid.
    block = make_select_block((1, 4), radius=0.5, backend="sparse")
    a_idx = torch.tensor([[1, 3]], dtype=torch.long)
    b_idx = torch.tensor([[2, 4]], dtype=torch.long)
    receiver_indices, receiver_valid = gather_patch_spatial_neighbors(
        block._tome_info["spatial_neighbor_indices"],
        block._tome_info["spatial_neighbor_valid"],
        a_idx,
        b_idx,
        num_prefix_tokens=1,
    )
    assert receiver_indices.shape == receiver_valid.shape
    assert not bool(receiver_valid.any())
    try:
        block._select(
            1,
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 4),
            a_idx,
            b_idx,
        )
    except (AssertionError, RuntimeError):
        pass
    else:
        raise AssertionError("all-invalid spatial budget must fail closed")


def make_merge_block(grid_size, radius, backend, device):
    block = make_select_block(grid_size, radius, backend).to(device)
    for key in (
        "spatial_adjacency",
        "spatial_neighbor_indices",
        "spatial_neighbor_valid",
    ):
        block._tome_info[key] = block._tome_info[key].to(device)
    block._tome_info.update(
        {
            "class_token": True,
            "distill_token": False,
            "source_trace_mode": "center",
            "trace_source": False,
            "source_tracking_mode": "none",
            "local_depth": 1,
            "train_grouping": "alternating_per_layer",
            "train_grouping_seed": 0,
            "merge_layer_index": 0,
            "t": 1,
        }
    )
    block.train()
    return block


def test_merge_transport_dense_sparse_cpu_parity():
    torch.manual_seed(20260820)
    device = torch.device("cpu")
    dense = make_merge_block((3, 5), 2, "dense", device)
    sparse = make_merge_block((3, 5), 2, "sparse", device)
    x_dense = torch.randn(2, 16, 7, dtype=torch.float64, requires_grad=True)
    metric_dense = torch.randn(2, 16, 5, dtype=torch.float64, requires_grad=True)
    x_sparse = x_dense.detach().clone().requires_grad_()
    metric_sparse = metric_dense.detach().clone().requires_grad_()
    size_dense = torch.ones(2, 16, 1, dtype=torch.float64)
    size_sparse = size_dense.clone()

    dense_out = dense._merge_train(
        x_dense, size_dense, 3, 16, {"metric": metric_dense}
    )
    sparse_out = sparse._merge_train(
        x_sparse, size_sparse, 3, 16, {"metric": metric_sparse}
    )
    assert torch.allclose(sparse_out[0], dense_out[0], atol=1e-9, rtol=1e-9)
    assert torch.allclose(sparse_out[1], dense_out[1], atol=1e-9, rtol=1e-9)
    assert torch.allclose(
        sparse._tome_info["token_center"],
        dense._tome_info["token_center"],
        atol=1e-9,
        rtol=1e-9,
    )

    probe = torch.randn_like(dense_out[0])
    dense_grad = torch.autograd.grad(
        (dense_out[0] * probe).sum(), (x_dense, metric_dense)
    )
    sparse_grad = torch.autograd.grad(
        (sparse_out[0] * probe).sum(), (x_sparse, metric_sparse)
    )
    for got, want in zip(sparse_grad, dense_grad):
        assert torch.allclose(got, want, atol=2e-8, rtol=2e-8)


def test_checkpoint_state_is_backend_independent():
    from opentome.models.mergenet.model import LocalEncoder

    common = dict(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2,
        local_depth=2,
        dtem_feat_dim=16,
        dtem_window_size=0,
        dtem_spatial_radius=2,
        dtem_spatial_metric="euclidean",
        total_merge_local=4,
        use_softkmax=True,
        local_block_window=2,
        source_trace_mode="center",
    )
    dense = LocalEncoder(**common, dtem_spatial_backend="dense")
    sparse = LocalEncoder(**common, dtem_spatial_backend="sparse")
    legacy = LocalEncoder(**common)
    assert legacy.dtem_spatial_backend == "dense"
    assert legacy._tome_info["dtem_spatial_backend"] == "dense"
    dense_keys = set(dense.state_dict())
    sparse_keys = set(sparse.state_dict())
    assert dense_keys == sparse_keys
    assert not any("_dtem_spatial_" in key for key in dense_keys)
    incompatible = sparse.load_state_dict(dense.state_dict(), strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    legacy_incompatible = legacy.load_state_dict(dense.state_dict(), strict=True)
    assert (
        not legacy_incompatible.missing_keys
        and not legacy_incompatible.unexpected_keys
    )


def test_local_encoder_cuda_checkpoint_forward_backward_if_available():
    if not torch.cuda.is_available():
        return
    from opentome.models.mergenet.model import LocalEncoder

    common = dict(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2,
        local_depth=2,
        dtem_feat_dim=16,
        dtem_window_size=0,
        dtem_spatial_radius=2,
        dtem_spatial_metric="euclidean",
        total_merge_local=4,
        use_softkmax=True,
        local_block_window=2,
        source_trace_mode="center",
    )
    dense = LocalEncoder(**common, dtem_spatial_backend="dense").cuda().eval()
    sparse = LocalEncoder(**common, dtem_spatial_backend="sparse").cuda().eval()
    sparse.load_state_dict(dense.state_dict(), strict=True)
    dense.set_eval_grouping("alternating_per_layer", seed=0)
    sparse.set_eval_grouping("alternating_per_layer", seed=0)

    torch.manual_seed(20260820)
    dense_input = torch.randn(2, 3, 32, 32, device="cuda", requires_grad=True)
    sparse_input = dense_input.detach().clone().requires_grad_()
    dense_out = dense(dense_input)
    sparse_out = sparse(sparse_input)
    for got, want in zip(sparse_out[:3], dense_out[:3]):
        assert torch.allclose(got, want, atol=2e-6, rtol=2e-6)
    assert sparse._tome_info["assign_layout"] == "spatial_sparse"
    assert (
        sparse._tome_info["assign_b_indices"].shape[-1]
        == sparse._dtem_spatial_neighbor_indices.shape[-1]
    )

    probe = torch.randn_like(dense_out[0])
    dense_targets = (dense_input, *tuple(dense.metric_layers.parameters()))
    sparse_targets = (sparse_input, *tuple(sparse.metric_layers.parameters()))
    dense_grad = torch.autograd.grad(
        (dense_out[0] * probe).sum(), dense_targets, allow_unused=True
    )
    sparse_grad = torch.autograd.grad(
        (sparse_out[0] * probe).sum(), sparse_targets, allow_unused=True
    )
    for got, want in zip(sparse_grad, dense_grad):
        if got is None or want is None:
            assert got is None and want is None
            continue
        assert bool(torch.isfinite(got).all())
        assert torch.allclose(got, want, atol=4e-5, rtol=4e-5)


def test_cuda_forward_backward_if_available():
    if not torch.cuda.is_available():
        return
    assert dtem_module.triton is not None, "CUDA sparse gate must exercise Triton"
    probe_a = torch.randn(1, 2, 4, device="cuda")
    probe_b = torch.randn(1, 3, 4, device="cuda")
    probe_indices = torch.zeros(1, 2, 2, device="cuda", dtype=torch.long)
    probe_valid = torch.ones_like(probe_indices, dtype=torch.bool)
    assert dtem_module._can_use_triton_sparse_dot(
        probe_a, probe_b, probe_indices, probe_valid
    )
    assert dtem_module._can_use_triton_sparse_transport(
        torch.ones(1, 2, 2, device="cuda"),
        probe_a,
        probe_indices,
        probe_valid,
    )
    dot_parity_case(torch.device("cuda"), torch.float32)
    dot_parity_case(torch.device("cuda"), torch.float16)
    dot_parity_case(torch.device("cuda"), torch.bfloat16)
    transport_parity_case(torch.device("cuda"), torch.float32)
    transport_parity_case(torch.device("cuda"), torch.float16)
    select_parity_case(torch.device("cuda"))
    select_parity_case(torch.device("cuda"), use_softkmax=False)
    test_local_encoder_cuda_checkpoint_forward_backward_if_available()


def main():
    test_rectangular_neighbor_table_and_boundaries()
    test_partition_mapping_matches_dense_mask_for_batches()
    test_sparse_dot_cpu_forward_backward()
    test_sparse_transport_cpu_forward_backward()
    test_select_cpu_dense_sparse_forward_backward()
    test_select_nonsoft_cpu_dense_sparse_forward_backward()
    test_all_invalid_sparse_rows_and_budget_fail_closed()
    test_merge_transport_dense_sparse_cpu_parity()
    test_checkpoint_state_is_backend_independent()
    test_cuda_forward_backward_if_available()
    print("DTEM_SPATIAL_SPARSE_TEST_PASS")


if __name__ == "__main__":
    main()
