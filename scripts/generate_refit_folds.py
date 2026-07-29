"""Create leakage-safe 90% outer-training artifacts after inner validation."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from generate_cv_folds import (
    compute_gip_kernels,
    load_associations,
    pair_count,
    save_pickle,
    save_symmetric_csr,
    sha256,
    write_pairs,
)


def load_mapping(path: Path) -> dict[int, list[int]]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a drug-to-diseases dictionary")
    return {
        int(drug): sorted({int(disease) for disease in diseases})
        for drug, diseases in value.items()
    }


def merge_mappings(
    train: dict[int, list[int]],
    validation: dict[int, list[int]],
    num_drugs: int,
) -> dict[int, list[int]]:
    return {
        drug: sorted(
            set(train.get(drug, ())) | set(validation.get(drug, ()))
        )
        for drug in range(num_drugs)
    }


def verify_refit(
    complete: list[list[int]],
    train: dict[int, list[int]],
    test: dict[int, list[int]],
) -> None:
    for drug, all_diseases in enumerate(complete):
        train_set = set(train.get(drug, ()))
        test_set = set(test.get(drug, ()))
        if train_set & test_set:
            raise AssertionError(
                f"Drug {drug} has overlapping refit-training and test positives"
            )
        if train_set | test_set != set(all_diseases):
            raise AssertionError(
                f"Drug {drug} refit-training and test do not reconstruct the dataset"
            )


def generate_dataset_refits(dataset_dir: Path, folds: int, force: bool) -> None:
    records = load_associations(dataset_dir / "drug_disease_list.pkl")
    num_drugs = len(records)
    num_diseases = max(max(values) for values in records if values) + 1
    total_pairs = sum(len(values) for values in records)
    fold_root = dataset_dir / "folds"

    for fold in range(folds):
        fold_dir = fold_root / f"fold_{fold:02d}"
        selection_manifest_path = fold_dir / "manifest.json"
        required = [
            fold_dir / "train.pkl",
            fold_dir / "val.pkl",
            fold_dir / "test.pkl",
            selection_manifest_path,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing selection-fold artifacts for {dataset_dir.name} "
                f"fold {fold}: {missing}"
            )

        refit_dir = fold_dir / "refit"
        manifest_path = refit_dir / "manifest.json"
        if manifest_path.exists() and not force:
            print(f"[SKIP] {dataset_dir.name} fold_{fold:02d} refit exists")
            continue

        selection_train = load_mapping(fold_dir / "train.pkl")
        selection_validation = load_mapping(fold_dir / "val.pkl")
        test = load_mapping(fold_dir / "test.pkl")
        refit_train = merge_mappings(
            selection_train, selection_validation, num_drugs
        )
        empty_validation = {drug: [] for drug in range(num_drugs)}
        verify_refit(records, refit_train, test)

        refit_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "train": refit_dir / "train.pkl",
            "validation": refit_dir / "val.pkl",
            "test": refit_dir / "test.pkl",
            "adjacency": refit_dir / "adj_csr.npz",
            "drug_gip": refit_dir / "DrugGIP.npy",
            "disease_gip": refit_dir / "DiseaseGIP.npy",
        }
        drug_gip, disease_gip, drug_gamma, disease_gamma = compute_gip_kernels(
            refit_train, num_drugs, num_diseases
        )
        save_pickle(files["train"], refit_train)
        save_pickle(files["validation"], empty_validation)
        save_pickle(files["test"], test)
        save_symmetric_csr(
            files["adjacency"], refit_train, num_drugs, num_diseases
        )
        np.save(files["drug_gip"], drug_gip, allow_pickle=False)
        np.save(files["disease_gip"], disease_gip, allow_pickle=False)
        write_pairs(refit_dir / "train_pairs.csv", refit_train)
        write_pairs(
            refit_dir / "selection_val_pairs.csv", selection_validation
        )
        write_pairs(refit_dir / "test_pairs.csv", test)

        manifest = {
            "dataset": dataset_dir.name,
            "outer_fold": fold,
            "protocol": "refit_after_inner_validation",
            "training_sources": ["selection_train", "selection_validation"],
            "threshold_source": "selection_validation_f1",
            "gip_source": "refit_training_associations_only",
            "num_drugs": num_drugs,
            "num_diseases": num_diseases,
            "complete_positive_pairs": total_pairs,
            "selection_train_positive_pairs": pair_count(selection_train),
            "selection_validation_positive_pairs": pair_count(
                selection_validation
            ),
            "refit_train_positive_pairs": pair_count(refit_train),
            "test_positive_pairs": pair_count(test),
            "refit_train_ratio": pair_count(refit_train) / total_pairs,
            "test_ratio": pair_count(test) / total_pairs,
            "drug_gip_gamma": drug_gamma,
            "disease_gip_gamma": disease_gamma,
            "selection_manifest_sha256": sha256(selection_manifest_path),
            "sha256": {name: sha256(path) for name, path in files.items()},
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[REFIT] {dataset_dir.name} fold_{fold:02d}: "
            f"train={manifest['refit_train_positive_pairs']}, "
            f"test={manifest['test_positive_pairs']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 90% refit artifacts for existing outer folds"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["B-dataset", "C-dataset", "F-dataset"],
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        generate_dataset_refits(
            args.data_root / dataset, args.folds, args.force
        )


if __name__ == "__main__":
    main()
