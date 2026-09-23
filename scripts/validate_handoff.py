#!/usr/bin/env python3
"""Read-only structural validator for DATA_HANDOFF.md share_packet v1.0.

This validates exported result files, not model correctness or privacy approval.
It never imports torch or loads checkpoints.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


METHODS = {"dense", "dtem", "tome", "pitome", "mergenet"}
VARIANTS = {"native", "mass_off", "recovery_off", "both_off"}
L0_FILES = {
    "README.md", "SCHEMA_VERSION.txt", "run_receipt.json",
    "dataset_receipt_redacted.json", "STATUS.json", "e1/epoch_metrics.csv.gz",
    "e1/checkpoint_receipts.json", "e1/endpoint_eval.json",
    "e2/eval_summary.csv", "e2/token_trajectory.csv",
    "e2/latency_raw.csv.gz", "e2/memory_raw.csv",
    "e3/ablation_summary.csv", "e3/integrity_gate.json",
    "e6/trace_config.json", "e6/equivalence_gate_redacted.json",
    "e6/aggregate_summary.json", "e6/figures/paper_routing_main.pdf",
    "e6/figures/paper_routing_population.pdf",
}
L1_FILES = {
    "e2/per_image.csv.gz", "e3/per_image.csv.gz",
    "e6/per_image.csv.gz", "e6/per_image_layer.csv.gz",
    "e6/distance_hist_per_image_layer.csv.gz",
    "e6/final_carriers.csv.gz", "e6/case_metadata.csv",
    "e6/case_edges.csv.gz", "e6/selection_frequency_28x28.csv",
}


def read_rows(path, required):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(required) - fields
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        yield from reader


def require(condition, message):
    if not condition:
        raise ValueError(message)


def as_int(value, label):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected integer, got {value!r}") from exc


def as_float(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected float, got {value!r}") from exc
    require(math.isfinite(number), f"{label}: non-finite value")
    return number


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(root):
    manifest_path = root / "file_manifest.json"
    require(manifest_path.is_file(), "missing file_manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(data, dict) and isinstance(data.get("files"), list),
            "file_manifest.json must be {'files': [...]} ")
    listed = set()
    for entry in data["files"]:
        rel = Path(entry["path"])
        require(not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() != "file_manifest.json",
                f"unsafe or invalid manifest path: {rel}")
        require(rel.as_posix() not in listed, f"duplicate manifest entry: {rel}")
        listed.add(rel.as_posix())
        target = root / rel
        require(target.is_file() and not target.is_symlink(), f"missing/symlink file: {rel}")
        require(target.stat().st_size == as_int(entry["bytes"], f"{rel} bytes"),
                f"byte-size mismatch: {rel}")
        require(sha256(target) == entry["sha256"], f"SHA-256 mismatch: {rel}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual - {"file_manifest.json"} == listed,
            f"manifest file-set mismatch: missing={sorted(actual - listed - {'file_manifest.json'})}, "
            f"extra={sorted(listed - actual)}")
    return len(listed)


def check_l1_eval(path, key_name, expected_keys):
    counts = Counter()
    ids_by_key = defaultdict(set)
    correct_by_key = Counter()
    for row in read_rows(path, [key_name, "image_id", "label_index", "pred_index",
                                "top1_correct", "top5_correct", "nll_loss"]):
        key = row[key_name]
        require(key in expected_keys, f"{path.name}: unexpected {key_name}={key}")
        image_id = row["image_id"]
        require(len(image_id) == 32 and all(c in "0123456789abcdef" for c in image_id),
                f"{path.name}: invalid image_id")
        require(image_id not in ids_by_key[key], f"{path.name}: duplicate {key}, {image_id}")
        ids_by_key[key].add(image_id)
        label = as_int(row["label_index"], "label_index")
        pred = as_int(row["pred_index"], "pred_index")
        require(0 <= label < 1000 and 0 <= pred < 1000, "invalid class index")
        correct = as_int(row["top1_correct"], "top1_correct")
        require(correct in (0, 1) and correct == int(label == pred),
                "top1_correct disagrees with label/pred")
        require(as_int(row["top5_correct"], "top5_correct") in (0, 1),
                "invalid top5_correct")
        require(as_float(row["nll_loss"], "nll_loss") >= 0, "negative NLL")
        counts[key] += 1
        correct_by_key[key] += correct
    require(set(counts) == expected_keys, f"{path.name}: missing methods/variants")
    require(all(n == 50000 for n in counts.values()), f"{path.name}: expected 50000 rows per arm: {counts}")
    first = ids_by_key[next(iter(expected_keys))]
    require(all(ids == first for ids in ids_by_key.values()),
            f"{path.name}: image_id sets differ between arms")
    return first, {key: 100 * correct_by_key[key] / counts[key] for key in expected_keys}


def check_e6(root, eval_ids):
    image_ids = set()
    for row in read_rows(root / "e6/per_image.csv.gz", ["image_id", "top1_correct", "final_k", "final_unique_k"]):
        image_id = row["image_id"]
        require(image_id not in image_ids, "e6 per_image duplicate image_id")
        image_ids.add(image_id)
        require(as_int(row["final_k"], "final_k") == 392, "e6 final_k != 392")
        require(as_int(row["final_unique_k"], "final_unique_k") == 392,
                "e6 final_unique_k != 392")
    require(len(image_ids) == 1000 and image_ids <= eval_ids,
            "e6 must contain 1000 images drawn from E2 val IDs")

    layer_keys = set()
    edge_counts = {}
    for row in read_rows(root / "e6/per_image_layer.csv.gz",
                         ["image_id", "step", "edge_count_positive", "mass_conservation_abs_error"]):
        key = (row["image_id"], as_int(row["step"], "step"))
        require(key[0] in image_ids and 1 <= key[1] <= 6 and key not in layer_keys,
                "invalid/duplicate e6 image-step")
        layer_keys.add(key)
        edge_counts[key] = as_int(row["edge_count_positive"], "edge_count_positive")
        require(as_float(row["mass_conservation_abs_error"], "mass error") >= 0,
                "negative mass error")
    require(len(layer_keys) == 6000, "e6 must have 1000 × 6 layer rows")

    hist_counts = Counter()
    for row in read_rows(root / "e6/distance_hist_per_image_layer.csv.gz",
                         ["image_id", "step", "bin_left_patch", "bin_right_patch", "edge_count"]):
        key = (row["image_id"], as_int(row["step"], "step"))
        require(key in layer_keys, "histogram has unknown image-step")
        left = as_float(row["bin_left_patch"], "bin_left_patch")
        right = as_float(row["bin_right_patch"], "bin_right_patch")
        require(0 <= left < right, "invalid histogram bin")
        hist_counts[key] += as_int(row["edge_count"], "edge_count")
    require(set(hist_counts) == layer_keys, "missing histogram image-step")
    require(all(hist_counts[key] == edge_counts[key] for key in layer_keys),
            "histogram edge counts disagree with layer summary")

    carrier_counts = Counter()
    frequency_counts = Counter()
    seen = set()
    seen_ranks = set()
    mass_sums = defaultdict(float)
    for row in read_rows(root / "e6/final_carriers.csv.gz",
                         ["image_id", "slot_index_0based", "grid_row_0based",
                          "grid_col_0based", "carrier_mass", "selection_rank"]):
        image_id = row["image_id"]
        slot = as_int(row["slot_index_0based"], "slot_index")
        require(image_id in image_ids and 0 <= slot < 784, "invalid carrier image/slot")
        require((image_id, slot) not in seen, "duplicate carrier slot per image")
        seen.add((image_id, slot))
        rank = as_int(row["selection_rank"], "selection_rank")
        require(0 <= rank < 392 and (image_id, rank) not in seen_ranks,
                "invalid/duplicate selection_rank")
        seen_ranks.add((image_id, rank))
        require(as_int(row["grid_row_0based"], "grid_row") == slot // 28 and
                as_int(row["grid_col_0based"], "grid_col") == slot % 28,
                "carrier grid/slot mismatch")
        mass = as_float(row["carrier_mass"], "carrier_mass")
        require(mass >= 0, "negative carrier mass")
        mass_sums[image_id] += mass
        carrier_counts[image_id] += 1
        frequency_counts[slot] += 1
    require(set(carrier_counts) == image_ids and all(n == 392 for n in carrier_counts.values()),
            "expected 392 final carriers per E6 image")
    require(all(total > 0 for total in mass_sums.values()), "non-positive final carrier mass sum")

    frequency_slots = set()
    for row in read_rows(root / "e6/selection_frequency_28x28.csv",
                         ["slot_index_0based", "count_selected", "frequency"]):
        slot = as_int(row["slot_index_0based"], "frequency slot")
        require(0 <= slot < 784 and slot not in frequency_slots, "invalid frequency slot")
        frequency_slots.add(slot)
        count = as_int(row["count_selected"], "count_selected")
        require(count == frequency_counts[slot], "frequency count disagrees with carriers")
        require(abs(as_float(row["frequency"], "frequency") - count / 1000) < 1e-8,
                "frequency rate disagrees with count")
    require(len(frequency_slots) == 784, "selection_frequency must have 784 rows")

    case_ids = set()
    main_ids = set()
    for row in read_rows(root / "e6/case_metadata.csv", ["image_id", "role", "figure_use"]):
        image_id = row["image_id"]
        require(image_id in image_ids, "case image not in E6 sample")
        case_ids.add(image_id)
        if row["figure_use"] == "main":
            main_ids.add(image_id)
    require(len(main_ids) == 2, "expected exactly two fixed main-figure cases")
    case_counts = Counter()
    for row in read_rows(root / "e6/case_edges.csv.gz",
                         ["image_id", "step", "donor_slot_0based", "receiver_slot_0based",
                          "weight_postmask", "donor_mass_pre", "receiver_mass_pre"]):
        image_id = row["image_id"]
        step = as_int(row["step"], "case edge step")
        donor = as_int(row["donor_slot_0based"], "donor_slot")
        receiver = as_int(row["receiver_slot_0based"], "receiver_slot")
        require(image_id in case_ids and 1 <= step <= 6, "unknown case/step")
        require(0 <= donor < 784 and 0 <= receiver < 784 and donor != receiver,
                "invalid case edge slot")
        distance = math.hypot(donor // 28 - receiver // 28, donor % 28 - receiver % 28)
        require(distance <= 3 + 1e-6, "case edge exceeds radius 3")
        weight = as_float(row["weight_postmask"], "weight_postmask")
        require(0 < weight <= 1 + 1e-6, "invalid post-mask edge weight")
        require(as_float(row["donor_mass_pre"], "donor_mass_pre") >= 0 and
                as_float(row["receiver_mass_pre"], "receiver_mass_pre") >= 0,
                "negative pre-transport mass")
        case_counts[(image_id, step)] += 1
    edge_image_ids = {image_id for image_id, _ in case_counts}
    require(main_ids <= edge_image_ids, "missing edge arrays for main-figure cases")
    for image_id in edge_image_ids:
        for step in range(1, 7):
            key = (image_id, step)
            require(case_counts[key] == edge_counts[key],
                    f"case edges incomplete or inconsistent: {key}")
    return image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, help="unpacked share_packet/ directory")
    parser.add_argument("--level", choices=("L0", "L1"), default="L1")
    args = parser.parse_args()
    root = args.packet.resolve()
    require(root.is_dir(), f"packet directory not found: {root}")
    require((root / "SCHEMA_VERSION.txt").read_text(encoding="utf-8").strip() == "1.0",
            "schema version must be 1.0")
    required = L0_FILES | (L1_FILES if args.level == "L1" else set())
    missing = sorted(rel for rel in required if not (root / rel).is_file())
    require(not missing, f"missing {args.level} files: {missing}")
    verified_files = check_manifest(root)
    print(f"PASS manifest: {verified_files} files")
    if args.level == "L1":
        e2_ids, e2_top1 = check_l1_eval(root / "e2/per_image.csv.gz", "method", METHODS)
        e3_ids, e3_top1 = check_l1_eval(root / "e3/per_image.csv.gz", "variant", VARIANTS)
        require(e2_ids == e3_ids, "E2/E3 image_id sets differ")
        e6_ids = check_e6(root, e2_ids)
        print(f"PASS E2/E3: {len(e2_ids)} paired images, top1={e2_top1}, ablations={e3_top1}")
        print(f"PASS E6: {len(e6_ids)} images, 6000 image-step rows, 392000 carriers")
    print(f"PASS {args.level} structural checks; model-result provenance still requires audit")


if __name__ == "__main__":
    main()
