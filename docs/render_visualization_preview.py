#!/usr/bin/env python3
"""Render a layout-only routing figure with deterministic synthetic data.

This script NEVER loads ImageNet, checkpoints, or experimental outputs. Its PNG
is a design preview and must not be used as a paper result.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


OUT = Path(__file__).with_name("visualization_preview.png")
NAVY = "#163a52"
TEAL = "#087e8b"
ORANGE = "#df7a32"
GRAY = "#aab7bf"


def synthetic_image(kind: int) -> np.ndarray:
    y, x = np.mgrid[0:224, 0:224]
    base = np.stack(
        [0.85 - 0.30 * y / 224, 0.89 - 0.35 * y / 224, 0.90 - 0.38 * y / 224],
        axis=-1,
    )
    if kind == 0:
        ellipse = ((x - 111) / 64) ** 2 + ((y - 119) / 73) ** 2 < 1
        stripe = (np.sin(x / 11 + y / 17) > 0.3) & ellipse
        base[ellipse] = [0.45, 0.58, 0.58]
        base[stripe] = [0.34, 0.47, 0.48]
    else:
        left = ((x - 78) / 45) ** 2 + ((y - 120) / 75) ** 2 < 1
        right = ((x - 151) / 49) ** 2 + ((y - 122) / 69) ** 2 < 1
        base[left] = [0.65, 0.57, 0.46]
        base[right] = [0.46, 0.52, 0.59]
    return np.clip(base, 0, 1)


def draw_base(ax, img, title):
    ax.imshow(img, extent=(0, 28, 28, 0), interpolation="bilinear")
    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_xticks(np.arange(0, 29, 4))
    ax.set_yticks(np.arange(0, 29, 4))
    ax.grid(color="white", alpha=0.37, linewidth=0.45)
    ax.tick_params(labelsize=7, length=0, colors="#71808a")
    for spine in ax.spines.values():
        spine.set_color("#aebdc6")
    ax.set_title(title, loc="left", fontsize=11, fontweight="semibold", color=NAVY, pad=9)


def draw_edges(ax, seed):
    rng = np.random.default_rng(seed)
    for quadrant in range(4):
        row0 = 0 if quadrant < 2 else 14
        col0 = 0 if quadrant % 2 == 0 else 14
        for _ in range(8):
            row = int(rng.integers(row0 + 2, row0 + 12))
            col = int(rng.integers(col0 + 2, col0 + 12))
            dr = int(rng.integers(-2, 3))
            dc = int(rng.integers(-2, 3))
            if dr == dc == 0:
                dc = 1
            weight = float(rng.uniform(0.38, 0.95))
            arrow = FancyArrowPatch(
                (col + 0.5, row + 0.5),
                (col + dc + 0.5, row + dr + 0.5),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=0.8 + 1.7 * weight,
                color=ORANGE,
                alpha=0.55 + 0.35 * weight,
                zorder=4,
            )
            ax.add_patch(arrow)


def draw_carriers(ax, seed):
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[0:28, 0:28]
    scores = rng.random((28, 28)) + 0.15 * np.cos(cc / 4) - 0.08 * np.sin(rr / 5)
    selected = np.zeros(784, dtype=bool)
    selected[np.argpartition(scores.ravel(), -392)[-392:]] = True
    x = cc.ravel() + 0.5
    y = rr.ravel() + 0.5
    ax.scatter(x[~selected], y[~selected], s=4, c="#5e6d78", alpha=0.37, linewidths=0)
    mass = 1.0 + 3.0 * rng.beta(1.3, 4.0, selected.sum())
    ax.scatter(x[selected], y[selected], s=9 + 22 * mass, c=TEAL,
               alpha=0.83, edgecolors="white", linewidths=0.19)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.1), constrained_layout=False)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.115,
                        wspace=0.22, hspace=0.37)
    fig.text(0.08, 0.955, "MergeNet routing: proposed paper figure", fontsize=17,
             fontweight="bold", color=NAVY)
    fig.text(0.08, 0.919, "STYLE PREVIEW  ·  SYNTHETIC MOCKUP  ·  NOT AN EXPERIMENTAL RESULT",
             fontsize=10.5, color="#a64220", fontweight="bold")
    fig.text(0.08, 0.878, "One fixed correct case and one fixed error case; same checkpoint and evaluation crop",
             fontsize=10, color="#4e606c")

    for i, (row_name, seed) in enumerate((("Correct prediction", 1301), ("Incorrect prediction", 2602))):
        img = synthetic_image(i)
        ax_edge, ax_carrier = axes[i]
        draw_base(ax_edge, img, f"{chr(65 + i * 2)}  {row_name} · local routing step 3")
        draw_edges(ax_edge, seed)
        draw_base(ax_carrier, img * 0.44 + 0.56, f"{chr(66 + i * 2)}  Final carrier slots · 392 / 784")
        draw_carriers(ax_carrier, seed + 1)
        if i == 0:
            ax_edge.text(0.02, 0.965, "mock: up to 8 edges / quadrant",
                         transform=ax_edge.transAxes, fontsize=8.2, va="top", color=NAVY,
                         bbox={"facecolor": "white", "alpha": 0.83, "edgecolor": "none", "pad": 4})
            ax_carrier.text(0.02, 0.965, "dot area ∝ carrier mass; original-grid positions",
                            transform=ax_carrier.transAxes, fontsize=8.2, va="top", color=NAVY,
                            bbox={"facecolor": "white", "alpha": 0.83, "edgecolor": "none", "pad": 4})

    fig.text(0.08, 0.052, "Orange arrows: mock donor → receiver edges     •     Teal dots: mock carrier slots",
             fontsize=9, color=NAVY)
    fig.text(0.08, 0.026, "Real paper figure will be rendered on the checkpoint holder's machine; no checkpoint or raw trace transfer is required.",
             fontsize=8.5, color="#63747f")
    fig.savefig(OUT, dpi=180)
    print(OUT)


if __name__ == "__main__":
    main()
