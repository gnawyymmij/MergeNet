#!/usr/bin/env python3
"""Checks for the public MergeNet model and training configuration."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def assert_r3(value: dict) -> None:
    expected = {
        "model": "mergenet_small_cls",
        "img_size": 224,
        "patch_size": 8,
        "local_depth": 6,
        "latent_depth": 6,
        "lambda_local": 2.0,
        "total_merge_latent": 0,
        "dtem_window_size": 0,
        "dtem_spatial_radius": 3.0,
        "dtem_spatial_metric": "euclidean",
        "dtem_spatial_backend": "sparse",
        "local_block_window": 16,
        "pretrained": False,
        "epochs": 300,
        "model_ema": True,
        "lr": 0.00075,
    }
    wrong = {key: (value.get(key), wanted) for key, wanted in expected.items() if value.get(key) != wanted}
    assert not wrong, wrong
    for key in (
        "distill_weight", "routing_distill_weight", "feat_distill_weight",
        "feat_distill_token_weight",
    ):
        assert float(value.get(key, 0.0)) == 0.0, key


def main() -> None:
    primary = load_yaml(ROOT / "configs" / "mergenet_l2_spatial_r3.yaml")
    assert_r3(primary)

    module_path = ROOT / "opentome" / "models" / "mergenet" / "model.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    defaults = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "MERGENET_SMALL_CANONICAL_KWARGS" for target in targets):
                defaults = ast.literal_eval(node.value)
                break
    assert isinstance(defaults, dict), "canonical model defaults not found"
    assert_r3({**primary, **defaults})
    for key in (
        "img_size", "patch_size", "local_depth", "latent_depth", "lambda_local",
        "total_merge_latent", "dtem_window_size", "dtem_spatial_radius",
        "dtem_spatial_metric", "dtem_spatial_backend", "local_block_window",
    ):
        assert defaults[key] == primary[key], (key, defaults[key], primary[key])
    model_source = module_path.read_text(encoding="utf-8")
    assert "if key not in resolved:" in model_source
    assert "resolved.get(key, None) is None" not in model_source
    assert 'key == "dtem_spatial_radius" and "dtem_window_size" in explicit_keys' in model_source

    # The public trainer must resolve a bare ``--model mergenet_small_cls`` to
    # the same true-grid R3 model.  Config files can still explicitly select
    # global or legacy flat-window ablations.
    trainer_path = ROOT / "trainer" / "classification" / "in1k_trainer.py"
    trainer_source = trainer_path.read_text(encoding="utf-8")
    for required in (
        "args.dtem_window_size = 0",
        "args.dtem_spatial_radius = 3.0",
        "args.lambda_local = 2.0",
        "args.local_depth = 6",
        "args.latent_depth = 6",
        "args.local_cls_global = True",
        "args.soft_topk = True",
        "dtem_window_size = 0",
    ):
        assert required in trainer_source, required
    for forbidden in (
        "args.dtem_window_size = 8",
        "args.lambda_local = 4.0",
        "dtem_window_size = 8",
    ):
        assert forbidden not in trainer_source, forbidden

    print("RELEASE_CONTRACT_PASS")


if __name__ == "__main__":
    main()
