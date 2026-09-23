#!/usr/bin/env python3
"""Synthetic result-only smoke test for the local figure renderer."""

import csv
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    with tempfile.TemporaryDirectory(prefix="mergenet-render-test-") as temp:
        packet = Path(temp) / "share_packet"
        out = Path(temp) / "figures"
        ids = ["a" * 32, "b" * 32]
        write_csv(packet / "e6/case_metadata.csv",
                  ["image_id", "figure_use", "top1_correct"],
                  [(ids[0], "main", 1), (ids[1], "main", 0)])
        write_csv(packet / "e6/final_carriers.csv.gz",
                  ["image_id", "slot_index_0based", "carrier_mass"],
                  ((image_id, slot, 1.0 + slot / 784)
                   for image_id in ids for slot in range(0, 784, 2)))
        write_csv(packet / "e6/case_edges.csv.gz",
                  ["image_id", "step", "donor_slot_0based", "receiver_slot_0based",
                   "weight_postmask"],
                  ((image_id, 3, slot, slot + 1, 0.5)
                   for image_id in ids for slot in range(0, 784, 28)))
        write_csv(packet / "e6/per_image_layer.csv.gz",
                  ["step", "distance_p50_patch", "distance_p90_patch"],
                  ((step, 1.0 + 0.1 * step, 2.0 + 0.1 * step)
                   for _ in range(1000) for step in range(1, 7)))
        write_csv(packet / "e6/selection_frequency_28x28.csv",
                  ["slot_index_0based", "frequency"],
                  ((slot, 1.0 if slot % 2 == 0 else 0.0) for slot in range(784)))
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/render_handoff_figures.py"),
             str(packet), "--output", str(out)],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        for kind in ("main", "population"):
            for ext in ("pdf", "png"):
                path = out / f"paper_routing_{kind}_local.{ext}"
                assert path.is_file() and path.stat().st_size > 1000, path
    print("handoff renderer self-test passed")


if __name__ == "__main__":
    main()
