#!/usr/bin/env python3
"""Freeze validation/test pairs from completed selection-only runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_pairs(path):
    with np.load(path, allow_pickle=False) as archive:
        pairs = np.asarray(archive["pairs"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int32).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise ValueError(f"Malformed prediction artifact: {path}")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError(f"Non-binary labels in {path}")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError(f"Duplicate evaluation pairs in {path}")
    return pairs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("results/main_none"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generated = 0
    for dataset in ("B-dataset", "C-dataset", "F-dataset"):
        dataset_manifest = json.loads(
            (args.data_root / dataset / "folds" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        num_drugs = int(dataset_manifest["num_drugs"])
        num_diseases = int(dataset_manifest["num_diseases"])
        for fold in range(10):
            source_dir = args.source_root / dataset / f"fold_{fold:02d}"
            target_dir = (
                args.data_root / dataset / "folds" / f"fold_{fold:02d}" /
                "fixed_eval"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = target_dir / "manifest.json"
            if manifest_path.exists() and not args.overwrite:
                print(f"[SKIP] {target_dir}")
                continue

            split_data = {}
            source_hashes = {}
            output_hashes = {}
            loaded = {}
            for split in ("validation", "test"):
                source_path = source_dir / f"{split}_predictions.npz"
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                pairs, labels = load_pairs(source_path)
                local_diseases = pairs[:, 1] - num_drugs
                if (
                    np.any(pairs[:, 0] < 0) or
                    np.any(pairs[:, 0] >= num_drugs) or
                    np.any(local_diseases < 0) or
                    np.any(local_diseases >= num_diseases)
                ):
                    raise ValueError(f"Out-of-range pairs in {source_path}")
                output_path = target_dir / f"{split}_pairs.npz"
                np.savez_compressed(output_path, pairs=pairs, labels=labels)
                source_hashes[split] = sha256_file(source_path)
                output_hashes[split] = sha256_file(output_path)
                split_data[split] = {
                    "pairs": int(len(pairs)),
                    "positives": int(labels.sum()),
                    "negatives": int((labels == 0).sum()),
                    "pairs_sha256": sha256_array(pairs),
                    "labels_sha256": sha256_array(labels),
                    "negative_pairs_sha256": sha256_array(pairs[labels == 0]),
                }
                loaded[split] = (pairs, labels)

            val_pairs, val_labels = loaded["validation"]
            test_pairs, test_labels = loaded["test"]
            val_negative = {tuple(pair) for pair in val_pairs[val_labels == 0]}
            test_negative = {tuple(pair) for pair in test_pairs[test_labels == 0]}
            if val_negative & test_negative:
                raise ValueError(
                    f"Validation/test negative overlap in {dataset} fold {fold}"
                )
            manifest = {
                "dataset": dataset,
                "outer_fold": fold,
                "protocol": "versioned_fixed_evaluation_pairs",
                "source_run_tag": args.source_root.name,
                "source_result_dir": str(source_dir.resolve()),
                "pair_format": "drug index and globally offset disease node index",
                "selection_and_refit_reuse_identical_pairs": True,
                "validation_test_negative_overlap": 0,
                "splits": split_data,
                "source_prediction_sha256": source_hashes,
                "output_file_sha256": output_hashes,
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            generated += 1
            print(f"[WROTE] {manifest_path}")
    print(f"Generated {generated} fixed-evaluation manifests")


if __name__ == "__main__":
    main()
