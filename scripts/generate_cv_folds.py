"""为 HGCLDR 生成可复现、按药物分层的外层十折及内部验证集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_associations(path: Path) -> list[list[int]]:
    with path.open("rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, (list, tuple)):
        raise TypeError(f"{path} 必须保存按药物排列的疾病列表")
    cleaned = [sorted({int(disease) for disease in diseases}) for diseases in records]
    if not cleaned or not any(cleaned):
        raise ValueError(f"{path} 中没有有效关联")
    return cleaned


def outer_fold_assignments(
    records: list[list[int]], n_folds: int, seed: int
) -> list[dict[int, list[int]]]:
    """按药物分层分配正关联，每条关联恰好进入一个外层测试折。"""
    folds = [{drug: [] for drug in range(len(records))} for _ in range(n_folds)]
    rng = np.random.default_rng(seed)
    for drug, diseases in enumerate(records):
        shuffled = np.asarray(diseases, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        offset = int(rng.integers(0, n_folds))
        for position, disease in enumerate(shuffled.tolist()):
            folds[(offset + position) % n_folds][drug].append(int(disease))
    return folds


def compute_gip_kernels(
    train: dict[int, list[int]], num_drugs: int, num_diseases: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    association = np.zeros((num_drugs, num_diseases), dtype=np.float32)
    for drug, diseases in train.items():
        if diseases:
            association[drug, diseases] = 1.0

    def kernel(profiles: np.ndarray) -> tuple[np.ndarray, float]:
        squared_norms = np.einsum('ij,ij->i', profiles, profiles)
        denominator = float(squared_norms.sum())
        gamma = float(len(profiles) / denominator) if denominator > 0 else 1.0
        squared_distances = (
            squared_norms[:, None]
            + squared_norms[None, :]
            - 2.0 * (profiles @ profiles.T)
        )
        np.maximum(squared_distances, 0.0, out=squared_distances)
        return np.exp(-gamma * squared_distances).astype(np.float32), gamma

    drug_gip, drug_gamma = kernel(association)
    disease_gip, disease_gamma = kernel(association.T)
    return drug_gip, disease_gip, drug_gamma, disease_gamma


def save_pickle(path: Path, value) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)


def save_symmetric_csr(
    path: Path, train: dict[int, list[int]], num_drugs: int, num_diseases: int
) -> None:
    """不依赖 SciPy，写出可被 scipy.sparse.load_npz 读取的 CSR 文件。"""
    size = num_drugs + num_diseases
    neighbours = [[] for _ in range(size)]
    for drug, diseases in train.items():
        for disease in diseases:
            disease_node = num_drugs + disease
            neighbours[drug].append(disease_node)
            neighbours[disease_node].append(drug)

    indices = []
    indptr = [0]
    for row in neighbours:
        indices.extend(sorted(set(row)))
        indptr.append(len(indices))
    np.savez_compressed(
        path,
        indices=np.asarray(indices, dtype=np.int32),
        indptr=np.asarray(indptr, dtype=np.int32),
        format=np.asarray(b"csr"),
        shape=np.asarray([size, size], dtype=np.int64),
        data=np.ones(len(indices), dtype=np.int8),
    )


def pair_count(mapping: dict[int, list[int]]) -> int:
    return sum(len(values) for values in mapping.values())


def write_pairs(path: Path, mapping: dict[int, list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["drug_id", "disease_id"])
        for drug in sorted(mapping):
            for disease in sorted(mapping[drug]):
                writer.writerow([drug, disease])


def verify_partition(
    records: list[list[int]], train: dict[int, list[int]],
    validation: dict[int, list[int]], test: dict[int, list[int]]
) -> None:
    for drug, complete in enumerate(records):
        train_set = set(train[drug])
        validation_set = set(validation[drug])
        test_set = set(test[drug])
        if train_set & validation_set or train_set & test_set or validation_set & test_set:
            raise AssertionError(f"药物 {drug} 的训练、验证和测试关联发生重叠")
        if train_set | validation_set | test_set != set(complete):
            raise AssertionError(f"药物 {drug} 的三部分关联不能还原完整数据")


def generate_dataset(dataset_dir: Path, n_folds: int, seed: int) -> None:
    source = dataset_dir / "drug_disease_list.pkl"
    records = load_associations(source)
    num_drugs = len(records)
    num_diseases = max(max(values) for values in records if values) + 1
    total_pairs = sum(len(values) for values in records)
    output_root = dataset_dir / "folds"
    output_root.mkdir(parents=True, exist_ok=True)

    outer_tests = outer_fold_assignments(records, n_folds, seed)
    assignment_rows = []
    for fold, test in enumerate(outer_tests):
        for drug in sorted(test):
            for disease in sorted(test[drug]):
                assignment_rows.append((drug, disease, fold))

    assignment_path = output_root / "outer_fold_assignments.csv"
    with assignment_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["drug_id", "disease_id", "outer_test_fold"])
        writer.writerows(sorted(assignment_rows))

    fold_summaries = []
    complete_sets = [set(values) for values in records]
    for fold, test in enumerate(outer_tests):
        validation_fold = (fold + 1) % n_folds
        validation = outer_tests[validation_fold]
        train = {
            drug: sorted(
                complete_sets[drug]
                - set(test[drug])
                - set(validation[drug])
            )
            for drug in range(num_drugs)
        }
        verify_partition(records, train, validation, test)

        fold_dir = output_root / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "train": fold_dir / "train.pkl",
            "validation": fold_dir / "val.pkl",
            "test": fold_dir / "test.pkl",
            "adjacency": fold_dir / "adj_csr.npz",
            "drug_gip": fold_dir / "DrugGIP.npy",
            "disease_gip": fold_dir / "DiseaseGIP.npy",
        }
        drug_gip, disease_gip, drug_gamma, disease_gamma = compute_gip_kernels(
            train, num_drugs, num_diseases
        )
        np.save(files["drug_gip"], drug_gip, allow_pickle=False)
        np.save(files["disease_gip"], disease_gip, allow_pickle=False)
        save_pickle(files["train"], train)
        save_pickle(files["validation"], validation)
        save_pickle(files["test"], test)
        save_symmetric_csr(files["adjacency"], train, num_drugs, num_diseases)
        write_pairs(fold_dir / "train_pairs.csv", train)
        write_pairs(fold_dir / "val_pairs.csv", validation)
        write_pairs(fold_dir / "test_pairs.csv", test)

        manifest = {
            "dataset": dataset_dir.name,
            "outer_fold": fold,
            "outer_folds": n_folds,
            "split_seed": seed,
            "validation_strategy": "rotating_outer_fold",
            "validation_fold": validation_fold,
            "validation_ratio": 1.0 / n_folds,
            "num_drugs": num_drugs,
            "num_diseases": num_diseases,
            "complete_positive_pairs": total_pairs,
            "train_positive_pairs": pair_count(train),
            "validation_positive_pairs": pair_count(validation),
            "test_positive_pairs": pair_count(test),
            "zero_degree_drugs_in_train": sum(not train[d] for d in train),
            "zero_degree_diseases_in_train": int(
                num_diseases - len({i for values in train.values() for i in values})
            ),
            "drug_gip_gamma": drug_gamma,
            "disease_gip_gamma": disease_gamma,
            "sha256": {name: sha256(path) for name, path in files.items()},
        }
        manifest_path = fold_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(manifest)

    dataset_manifest = {
        "dataset": dataset_dir.name,
        "source_file": source.name,
        "source_sha256": sha256(source),
        "outer_assignment_sha256": sha256(assignment_path),
        "outer_folds": n_folds,
        "split_seed": seed,
        "validation_strategy": "rotating_outer_fold",
        "validation_ratio": 1.0 / n_folds,
        "num_drugs": num_drugs,
        "num_diseases": num_diseases,
        "complete_positive_pairs": total_pairs,
        "folds": fold_summaries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{dataset_dir.name}: 已生成 {n_folds} 折，共 {total_pairs} 条正关联")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 HGCLDR 十折交叉验证文件")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--datasets", nargs="+", default=["B-dataset", "C-dataset", "F-dataset"]
    )
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("折数至少为2")
    for dataset in args.datasets:
        generate_dataset(args.data_root / dataset, args.folds, args.seed)


if __name__ == "__main__":
    main()
