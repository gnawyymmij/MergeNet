#!/usr/bin/env python3
"""Small CUDA forward/backward and deterministic-eval smoke for MergeNet."""

from pathlib import Path
import sys

import torch
from timm import create_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import registers the delivered model factory with timm.
import opentome.models.mergenet.model  # noqa: F401
import opentome.timm.dtem as dtem_module
from opentome.timm.dtem import DTEMAttention, build_dtem_eval_group_indices


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MergeNet model smoke test")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("flash-attn is required for the MergeNet model smoke test") from exc

    torch.manual_seed(20260809)
    model = create_model(
        "mergenet_small_cls",
        pretrained=False,
        img_size=32,
        patch_size=8,
        num_classes=10,
        dtem_window_size=0,
        dtem_spatial_radius=3.0,
        dtem_spatial_metric="euclidean",
        dtem_spatial_backend="sparse",
        dtem_feat_dim=16,
        dtem_train_grouping="random_per_sample",
        dtem_eval_grouping="alternating_per_layer_fast",
        lambda_local=2.0,
        total_merge_latent=0,
        local_depth=2,
        latent_depth=2,
        local_block_window=2,
        local_cls_global=True,
        swa_size=2,
        source_trace_mode="center",
    ).cuda()
    if any(isinstance(module, DTEMAttention) for module in model.modules()):
        raise AssertionError(
            "canonical mergenet_small_cls must use LocalBlock + DTEMMergeOnly, "
            "not the legacy DTEMAttention compatibility path"
        )
    if model.local._tome_info.get("dtem_spatial_radius") != 3.0:
        raise AssertionError("model smoke did not construct the canonical true-2D radius")
    if model.local._tome_info.get("dtem_spatial_backend") != "sparse":
        raise AssertionError("model smoke did not construct the sparse spatial backend")

    # Historical canonical results use unbiased LocalBlock attention. Make the
    # separation executable: canonical forward must remain healthy even if the
    # legacy biased-attention entrypoint is made fatal.
    def unexpected_biased_attention(*args, **kwargs):
        raise AssertionError("canonical model unexpectedly called biased_local_attention")

    dtem_module.biased_local_attention = unexpected_biased_attention

    model.train()
    images = torch.randn(2, 3, 32, 32, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits, aux = model(images)
        loss = logits.float().square().mean()
    if logits.shape != (2, 10):
        raise AssertionError(f"unexpected logits shape: {tuple(logits.shape)}")
    if aux.get("retained_tokens") != 8:
        raise AssertionError(f"unexpected retained-token count: {aux.get('retained_tokens')}")
    if model.local._tome_info.get("assign_layout") != "spatial_sparse":
        raise AssertionError("canonical forward did not execute sparse spatial DTEM matching")
    if not torch.isfinite(loss):
        raise AssertionError(f"non-finite training loss: {loss.item()}")
    loss.backward()
    grad_params = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not grad_params:
        raise AssertionError("no finite parameter gradients were produced")
    nonfinite_grad_names = [
        name for name, gradient in grad_params
        if not torch.isfinite(gradient).all()
    ]
    if nonfinite_grad_names:
        raise AssertionError(
            "non-finite gradients in: " + ", ".join(nonfinite_grad_names[:10])
        )
    finite_grad_params = len(grad_params)

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        first, first_aux = model(images)
        second, second_aux = model(images)
    # The grouping contract is discrete and must be bitwise stable.  Check it
    # directly instead of conflating it with bitwise reproducibility of
    # FlashAttention/Triton FP16 reductions, which can differ by one ULP across
    # otherwise identical launches on A100.
    for layer_index in range(2):
        group_a_1, group_b_1 = build_dtem_eval_group_indices(
            n_tokens=16,
            batch_size=2,
            device=images.device,
            mode="alternating_per_layer_fast",
            seed=0,
            layer_index=layer_index,
        )
        group_a_2, group_b_2 = build_dtem_eval_group_indices(
            n_tokens=16,
            batch_size=2,
            device=images.device,
            mode="alternating_per_layer_fast",
            seed=0,
            layer_index=layer_index,
        )
        if not torch.equal(group_a_1, group_a_2) or not torch.equal(group_b_1, group_b_2):
            raise AssertionError(f"deterministic eval grouping drifted at layer {layer_index}")

    max_abs = (first.float() - second.float()).abs().max().item()
    if max_abs > 5.0e-4:
        raise AssertionError(
            "repeated FP16 eval drift exceeded the audited one-ULP envelope: "
            f"max_abs={max_abs}"
        )
    if first_aux["retained_tokens"] != 8 or second_aux["retained_tokens"] != 8:
        raise AssertionError("eval retained-token metadata drifted")

    print(
        "MERGENET_MODEL_SMOKE_PASS",
        {"params": sum(p.numel() for p in model.parameters()),
         "finite_grad_params": finite_grad_params,
         "retained_tokens": first_aux["retained_tokens"]},
    )


if __name__ == "__main__":
    main()
