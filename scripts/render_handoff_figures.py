#!/usr/bin/env python3
"""Render grid-only or approved-crop paper figures from an E6 L1 share_packet.

No model, ImageNet root, checkpoint, or private path mapping is needed.
Run validate_handoff.py on the packet first. Output files use *_local names and
will not overwrite existing files.
"""

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


TEAL = "#087e8b"
ORANGE = "#df7a32"
NAVY = "#163a52"


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def main_cases(root):
    cases = {}
    for row in rows(root / "e6/case_metadata.csv"):
        if row["figure_use"] == "main":
            cases[row["image_id"]] = row
    if len(cases) != 2:
        raise ValueError("Exactly two figure_use=main case IDs are required")
    return sorted(cases.values(), key=lambda item: (-int(item["top1_correct"]), item["image_id"]))


def load_carriers(root, ids):
    carriers = defaultdict(list)
    for row in rows(root / "e6/final_carriers.csv.gz"):
        if row["image_id"] in ids:
            carriers[row["image_id"]].append((int(row["slot_index_0based"]), float(row["carrier_mass"])))
    if any(len(carriers[image_id]) != 392 for image_id in ids):
        raise ValueError("Each main case needs exactly 392 final carriers")
    return carriers


def load_step3_edges(root, ids):
    edges = defaultdict(list)
    for row in rows(root / "e6/case_edges.csv.gz"):
        if row["image_id"] in ids and int(row["step"]) == 3:
            edges[row["image_id"]].append((int(row["donor_slot_0based"]),
                                           int(row["receiver_slot_0based"]),
                                           float(row["weight_postmask"])))
    return edges


def fixed_edges_by_quadrant(edges):
    groups = defaultdict(list)
    for donor, receiver, weight in edges:
        row, col = divmod(donor, 28)
        quadrant = 2 * (row >= 14) + (col >= 14)
        groups[quadrant].append((donor, receiver, weight))
    return [edge for quadrant in range(4)
            for edge in sorted(groups[quadrant], key=lambda e: (-e[2], e[0], e[1]))[:8]]


def setup_grid(ax, root, image_id, title):
    crop = root / "e6/images" / f"{image_id}.png"
    if crop.is_file():
        ax.imshow(mpimg.imread(crop), extent=(0, 28, 28, 0), interpolation="bilinear")
    else:
        ax.set_facecolor("#f4f7f8")
    ax.set_xlim(0, 28)
    ax.set_ylim(28, 0)
    ax.set_xticks(np.arange(0, 29, 4))
    ax.set_yticks(np.arange(0, 29, 4))
    ax.tick_params(labelsize=7, colors="#667681", length=0)
    ax.grid(color="#9fb0b8", alpha=0.48, linewidth=0.45)
    ax.set_title(title, loc="left", color=NAVY, fontsize=11, fontweight="semibold")
    for spine in ax.spines.values():
        spine.set_color("#aebdc6")


def render_main(root, out, cases, carriers, edges):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.1))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.11, wspace=0.22, hspace=0.35)
    fig.text(0.08, 0.965, "MergeNet routing on ImageNet-1K validation", color=NAVY,
             fontsize=17, fontweight="bold")
    fig.text(0.08, 0.93, "Fixed correct / error cases · 224px, patch 8 · real result export",
             color="#4e606c", fontsize=10)
    all_mass = [mass for case in cases for _, mass in carriers[case["image_id"]]]
    max_mass = max(np.quantile(all_mass, 0.95), 1e-8)
    all_weight = [weight for case in cases for _, _, weight in edges[case["image_id"]]]
    max_weight = max(np.quantile(all_weight, 0.95), 1e-8) if all_weight else 1.0

    for i, case in enumerate(cases):
        image_id = case["image_id"]
        status = "Correct" if int(case["top1_correct"]) else "Incorrect"
        ax_edge, ax_carrier = axes[i]
        setup_grid(ax_edge, root, image_id, f"{chr(65 + i * 2)}  {status}: routing step 3")
        for donor, receiver, weight in fixed_edges_by_quadrant(edges[image_id]):
            dr, dc = divmod(donor, 28)
            rr, rc = divmod(receiver, 28)
            scale = min(weight / max_weight, 1.0)
            ax_edge.add_patch(FancyArrowPatch(
                (dc + 0.5, dr + 0.5), (rc + 0.5, rr + 0.5),
                arrowstyle="-|>", mutation_scale=6, linewidth=0.65 + 1.9 * scale,
                color=ORANGE, alpha=0.43 + 0.52 * scale, zorder=5))

        setup_grid(ax_carrier, root, image_id, f"{chr(66 + i * 2)}  Final carriers: 392 / 784")
        selected = {slot for slot, _ in carriers[image_id]}
        other = [slot for slot in range(784) if slot not in selected]
        ax_carrier.scatter([slot % 28 + 0.5 for slot in other],
                           [slot // 28 + 0.5 for slot in other],
                           s=4, c="#5e6d78", alpha=0.38, linewidths=0)
        for slot, mass in carriers[image_id]:
            size = 10 + 31 * min(mass / max_mass, 1.0)
            ax_carrier.scatter(slot % 28 + 0.5, slot // 28 + 0.5,
                               s=size, c=TEAL, alpha=0.82, edgecolors="white",
                               linewidths=0.2)

    fig.text(0.08, 0.055, "Orange: actual post-mask donor → receiver weight (up to 8 edges / quadrant) · "
             "Teal: selected original-grid carrier; dot area scales with mass", fontsize=8.5, color=NAVY)
    fig.text(0.08, 0.028, "Cases are deterministic, not manually curated. Routing arrows do not show semantic boundaries.",
             fontsize=8.3, color="#63747f")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"paper_routing_main_local.{ext}", dpi=300)
    plt.close(fig)


def render_population(root, out):
    p50 = [[] for _ in range(6)]
    p90 = [[] for _ in range(6)]
    for row in rows(root / "e6/per_image_layer.csv.gz"):
        step = int(row["step"]) - 1
        p50[step].append(float(row["distance_p50_patch"]))
        p90[step].append(float(row["distance_p90_patch"]))
    if any(len(values) != 1000 for values in p50 + p90):
        raise ValueError("Population figure needs 1000 image-level distances for each of six steps")
    frequency = np.zeros(784, dtype=float)
    for row in rows(root / "e6/selection_frequency_28x28.csv"):
        frequency[int(row["slot_index_0based"])] = float(row["frequency"])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.82, bottom=0.17, wspace=0.28)
    fig.text(0.08, 0.96, "Population routing statistics · 1,000 fixed validation images",
             fontsize=15, fontweight="bold", color=NAVY)
    ax = axes[0]
    box1 = ax.boxplot(p50, positions=np.arange(1, 7) - 0.16, widths=0.27,
                      patch_artist=True, showfliers=False, manage_ticks=False)
    box2 = ax.boxplot(p90, positions=np.arange(1, 7) + 0.16, widths=0.27,
                      patch_artist=True, showfliers=False, manage_ticks=False)
    for patch in box1["boxes"]:
        patch.set_facecolor(TEAL)
        patch.set_alpha(0.75)
    for patch in box2["boxes"]:
        patch.set_facecolor(ORANGE)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Local routing step")
    ax.set_ylabel("Actual edge distance (patch units)")
    ax.set_title("Per-image distance p50 (teal) and p90 (orange)", fontsize=10)
    ax.grid(axis="y", color="#d9e1e5", linewidth=0.6)

    ax = axes[1]
    heat = ax.imshow(frequency.reshape(28, 28), cmap="YlGnBu", vmin=0, vmax=1,
                     interpolation="nearest")
    ax.set_title("Final carrier selection frequency", fontsize=10)
    ax.set_xlabel("Original-grid column")
    ax.set_ylabel("Original-grid row")
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="images selected / 1000")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"paper_routing_population_local.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="unpacked share_packet/ directory")
    parser.add_argument("--output", type=Path, required=True, help="new/empty output directory")
    args = parser.parse_args()
    root = args.packet.resolve()
    out = args.output.resolve()
    if not root.is_dir():
        parser.error(f"packet directory not found: {root}")
    names = [f"paper_routing_{kind}_local.{ext}"
             for kind in ("main", "population") for ext in ("pdf", "png")]
    if any((out / name).exists() for name in names):
        parser.error("output figure already exists; choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    cases = main_cases(root)
    ids = {case["image_id"] for case in cases}
    carriers = load_carriers(root, ids)
    edges = load_step3_edges(root, ids)
    render_main(root, out, cases, carriers, edges)
    render_population(root, out)
    print(f"Wrote four result figures to {out}")


if __name__ == "__main__":
    main()
