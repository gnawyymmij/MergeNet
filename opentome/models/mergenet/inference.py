"""Optional steady-state inference compilation for MergeNet.

Load a checkpoint, move the model to its inference device, and call ``eval()``
before enabling this path. Compilation occurs on the first forward. It leaves
parameter names and checkpoint keys unchanged because it wraps ``forward``
methods rather than replacing modules.
"""

from __future__ import annotations

import torch


def compile_transformer_blocks(
    model, *, mode: str = "default", compile_local_encoder: bool = False
):
    """Compile inference hotspots for repeated fixed-shape inputs.

    ``compile_local_encoder`` can lower latency further but increases peak
    memory substantially at large batch sizes.
    """
    if model.training:
        raise ValueError("Call model.eval() before compiling inference blocks")
    for blocks in (model.local.vit.blocks, model.latent.vit.blocks):
        for block in blocks:
            if getattr(block, "_mergenet_inference_compiled", False):
                continue
            block.forward = torch.compile(block.forward, mode=mode)
            block._mergenet_inference_compiled = True
    if hasattr(model, "encode_cross_attention"):
        attention = model.encode_cross_attention
        if not getattr(attention, "_mergenet_inference_compiled", False):
            attention.forward = torch.compile(attention.forward, mode=mode)
            attention._mergenet_inference_compiled = True
    selector = model.local.merge_block
    if not getattr(selector, "_mergenet_select_compiled", False):
        selector._select = torch.compile(selector._select, mode=mode)
        selector._mergenet_select_compiled = True
    if not getattr(selector, "_mergenet_merge_compiled", False):
        selector._merge_train = torch.compile(selector._merge_train, mode=mode)
        selector._mergenet_merge_compiled = True
    if compile_local_encoder and not getattr(model.local, "_mergenet_inference_compiled", False):
        model.local.forward = torch.compile(model.local.forward, mode=mode)
        model.local._mergenet_inference_compiled = True
    return model
