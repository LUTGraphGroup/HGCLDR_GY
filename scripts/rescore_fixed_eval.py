#!/usr/bin/env python3
"""Rescore saved selection/refit checkpoints on versioned fixed pairs."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run as training


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_pairs(path, num_drugs, num_diseases):
    with np.load(path, allow_pickle=False) as archive:
        pairs = np.asarray(archive["pairs"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int32).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise ValueError(f"Malformed fixed evaluation artifact: {path}")
    local_diseases = pairs[:, 1] - num_drugs
    if (
        np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= num_drugs) or
        np.any(local_diseases < 0) or np.any(local_diseases >= num_diseases)
    ):
        raise ValueError(f"Out-of-range fixed evaluation pairs: {path}")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError(f"Duplicate fixed evaluation pairs: {path}")
    return pairs, labels


def local_negative_pairs(pairs, labels, num_drugs):
    negatives = pairs[labels == 0].copy()
    negatives[:, 1] -= num_drugs
    return negatives


def validate_positive_pairs(pairs, labels, mapping, num_drugs, split_name):
    actual = {
        (int(drug), int(disease - num_drugs))
        for drug, disease in pairs[labels == 1]
    }
    expected = training._dict_pairs(mapping)
    if actual != expected:
        raise ValueError(f"Fixed {split_name} positives do not match fold assignments")


def load_state(path, device):
    return torch.load(path, map_location=device)


def rescore_one(source_dir, output_dir, data_root, device):
    source_dir = source_dir.resolve()
    source_metrics_path = source_dir / "metrics.json"
    source_config_path = source_dir / "config.json"
    selection_state_path = source_dir / "selection_best_model.pt"
    final_state_path = source_dir / "best_model.pt"
    required = (
        source_metrics_path, source_config_path,
        selection_state_path, final_state_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_report = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    dataset = str(source_report["dataset"])
    fold = int(source_report["fold"])
    for key, value in source_config.get("effective_environment", {}).items():
        if value is not None:
            os.environ[key] = str(value)

    source_config["device"] = device
    source_config["dataset"] = dataset
    source_config["fold"] = fold
    source_config["pseudo_mode"] = (
        source_report.get("pseudo_supervision", {}).get("mode") or "none"
    )
    training.args = SimpleNamespace(**source_config)
    training.set_seed(int(source_config["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    selection_data = training.Data(
        dataset,
        source_config["norm_adj"],
        int(source_config["seed"]),
        float(source_config["test_ratio"]),
        float(source_config["ssl_ratio"]),
        fold=fold,
        split_variant="selection",
    )
    refit_seed = source_report.get("seeds", {}).get("refit_training")
    if refit_seed is None:
        raise ValueError(f"Source is not a completed refit result: {source_dir}")
    final_data = training.Data(
        dataset,
        source_config["norm_adj"],
        int(refit_seed),
        float(source_config["test_ratio"]),
        float(source_config["ssl_ratio"]),
        fold=fold,
        split_variant="refit",
    )

    fixed_dir = data_root / dataset / "folds" / f"fold_{fold:02d}" / "fixed_eval"
    manifest_path = fixed_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    validation_pairs, validation_labels = load_fixed_pairs(
        fixed_dir / "validation_pairs.npz",
        selection_data.num_drugs,
        selection_data.num_diseases,
    )
    test_pairs, test_labels = load_fixed_pairs(
        fixed_dir / "test_pairs.npz",
        selection_data.num_drugs,
        selection_data.num_diseases,
    )
    validate_positive_pairs(
        validation_pairs, validation_labels, selection_data.val_dict,
        selection_data.num_drugs, "validation",
    )
    validate_positive_pairs(
        test_pairs, test_labels, selection_data.test_dict,
        selection_data.num_drugs, "test",
    )
    validation_negatives = local_negative_pairs(
        validation_pairs, validation_labels, selection_data.num_drugs
    )
    test_negatives = local_negative_pairs(
        test_pairs, test_labels, selection_data.num_drugs
    )
    validation_negative_set = {tuple(map(int, pair)) for pair in validation_negatives}
    test_negative_set = {tuple(map(int, pair)) for pair in test_negatives}
    if validation_negative_set & test_negative_set:
        raise ValueError("Fixed validation/test negatives overlap")

    selection_state = load_state(selection_state_path, device)
    final_state = load_state(final_state_path, device)
    selection_model = training._build_model(selection_data)
    final_model = training._build_model(final_data)
    selection_model.load_state_dict(selection_state)
    final_model.load_state_dict(final_state)
    selection_model.eval()
    final_model.eval()
    with torch.no_grad():
        selection_embeddings = selection_model.encode4eval(
            selection_data.adj_train_norm
        )
        validation_scores = training._score_pairs(
            selection_model, selection_embeddings, validation_pairs
        )
        threshold, _, _, _ = training._pick_best_threshold(
            validation_labels, validation_scores
        )
        final_embeddings = final_model.encode4eval(final_data.adj_train_norm)
        test_scores = training._score_pairs(final_model, final_embeddings, test_pairs)
    validation_metrics = training._classification_metrics(
        validation_labels, validation_scores, threshold
    )
    test_metrics = training._classification_metrics(
        test_labels, test_scores, threshold
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "best_model.pt", "selection_best_model.pt",
        "pseudo_candidates.npz", "selection_pseudo_candidates.npz",
    ):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        pairs=validation_pairs, labels=validation_labels, scores=validation_scores,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        pairs=test_pairs, labels=test_labels, scores=test_scores,
    )

    report = copy.deepcopy(source_report)
    report["threshold_source"] = "versioned_fixed_validation_f1"
    report["locked_threshold"] = float(threshold)
    report["validation"] = validation_metrics
    report["test"] = test_metrics
    report["negative_sampling"].update({
        "evaluation_strategy": "versioned fixed fold pairs",
        "validation_test_negative_overlap": 0,
        "validation_negative_count": int(len(validation_negatives)),
        "test_negative_count": int(len(test_negatives)),
        "validation_negative_sha256": training._pairs_sha256(validation_negatives),
        "test_negative_sha256": training._pairs_sha256(test_negatives),
        "fixed_eval_manifest_sha256": sha256_file(manifest_path),
    })
    report["rescore_provenance"] = {
        "source_result_dir": str(source_dir),
        "source_metrics_sha256": sha256_file(source_metrics_path),
        "source_locked_threshold": source_report["locked_threshold"],
        "source_test_negative_sha256": source_report["negative_sampling"][
            "test_negative_sha256"
        ],
        "training_reused_without_update": True,
        "rescored_validation_and_test": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_config = copy.deepcopy(source_config)
    output_config["output_root"] = str(output_dir.parents[2])
    output_config["run_tag"] = output_dir.parents[1].name
    output_config["fixed_eval_manifest"] = str(manifest_path.resolve())
    output_config["rescore_from"] = str(source_dir)
    (output_dir / "config.json").write_text(
        json.dumps(output_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[FIXED-EVAL] {dataset} fold={fold:02d} "
        f"AUC={test_metrics['auc']:.5f} AUPR={test_metrics['aupr']:.5f} "
        f"F1={test_metrics['f1']:.5f} threshold={threshold:.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("results/refit_none_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-tag", default="refit_none_fixed_eval_v1")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["B-dataset", "C-dataset", "F-dataset"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    for dataset in args.datasets:
        for fold in args.folds:
            source_dir = args.source_root / dataset / f"fold_{fold:02d}"
            output_dir = args.output_root / args.run_tag / dataset / f"fold_{fold:02d}"
            rescore_one(source_dir, output_dir, args.data_root, args.device)


if __name__ == "__main__":
    main()
