#!/usr/bin/env python3
"""Audit and summarize the fixed-pair pseudo-supervision experiment."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = ("B-dataset", "C-dataset", "F-dataset")
MODES = ("none", "hard", "weighted")
METRICS = ("auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc")
PRIMARY_METRICS = ("auc", "aupr", "f1")
MODE_TAGS = {
    "none": "refit_none_fixed_eval_v1",
    "hard": "refit_pseudo_hard_fixed_eval_v1",
    "weighted": "refit_pseudo_weighted_fixed_eval_v1",
}
FIXED_PAIR_KEYS = (
    "validation_negative_sha256",
    "test_negative_sha256",
    "fixed_eval_manifest_sha256",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Verify the 90 fold-level reports and regenerate pseudo-mode "
            "summary and paired exact-test tables."
        )
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/pseudo_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=50000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return float(np.mean(values))


def sample_std(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def exact_wilcoxon(differences):
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.abs(differences) > 1e-15]
    if len(differences) == 0:
        return 1.0, 0.0, 0
    ranks = average_ranks(np.abs(differences))
    observed = abs(float(np.sum(np.sign(differences) * ranks)))
    extreme = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.sum(np.asarray(signs) * ranks)))
        if statistic >= observed - 1e-12:
            extreme += 1
    p_value = extreme / total
    rank_biserial = float(np.sum(np.sign(differences) * ranks) / np.sum(ranks))
    return p_value, rank_biserial, len(differences)


def bootstrap_mean_ci(differences, seed, replicates):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p_raw"])
    adjusted = [None] * len(rows)
    running = 0.0
    count = len(rows)
    for position, (original_index, row) in enumerate(ordered):
        value = min(1.0, row["p_raw"] * (count - position))
        running = max(running, value)
        adjusted[original_index] = running
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value


def friedman_three(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    ranked = np.vstack([average_ranks(row) for row in matrix])
    rank_sums = ranked.sum(axis=0)
    statistic = (
        12.0 / (n * k * (k + 1.0)) * np.sum(rank_sums ** 2)
        - 3.0 * n * (k + 1.0)
    )
    p_value = math.exp(-statistic / 2.0)
    kendall_w = statistic / (n * (k - 1.0))
    return float(statistic), float(p_value), float(kendall_w)


def load_reports(root):
    reports = {}
    input_hashes = {}
    for mode in MODES:
        for dataset in DATASETS:
            for fold in range(10):
                relative = (
                    Path(MODE_TAGS[mode]) / dataset / "fold_{:02d}".format(fold)
                    / "metrics.json"
                )
                path = root / relative
                if not path.is_file():
                    raise FileNotFoundError(path)
                report = json.loads(path.read_text(encoding="utf-8"))
                reports[(mode, dataset, fold)] = report
                input_hashes[str(relative).replace("\\", "/")] = file_sha256(path)
    return reports, input_hashes


def audit_reports(reports):
    issues = []
    for dataset in DATASETS:
        for fold in range(10):
            reference = reports[("none", dataset, fold)]
            reference_negative = reference.get("negative_sampling", {})
            for mode in MODES:
                report = reports[(mode, dataset, fold)]
                if report.get("dataset") != dataset or int(report.get("fold", -1)) != fold:
                    issues.append("{} fold {}: identity mismatch for {}".format(dataset, fold, mode))
                pseudo = report.get("pseudo_supervision", {})
                if (pseudo.get("mode") or "none").lower() != mode:
                    issues.append("{} fold {}: pseudo mode mismatch for {}".format(dataset, fold, mode))
                if "training-only" not in pseudo.get("candidate_source", ""):
                    issues.append("{} fold {}: candidate provenance missing for {}".format(dataset, fold, mode))
                if report.get("training_protocol", {}).get("name") != "inner_validation_then_outer_refit":
                    issues.append("{} fold {}: protocol mismatch for {}".format(dataset, fold, mode))
                threshold_source = report.get("threshold_source", "").lower()
                if "validation" not in threshold_source or "f1" not in threshold_source:
                    issues.append("{} fold {}: threshold source mismatch for {}".format(dataset, fold, mode))
                for key in FIXED_PAIR_KEYS:
                    if report.get("negative_sampling", {}).get(key) != reference_negative.get(key):
                        issues.append("{} fold {}: {} differs for {}".format(dataset, fold, key, mode))
                for metric in METRICS:
                    if report.get("test", {}).get(metric) is None:
                        issues.append("{} fold {}: missing test {} for {}".format(dataset, fold, metric, mode))
    return issues


def summarize_modes(reports):
    rows = []
    for dataset in DATASETS:
        for mode in MODES:
            for metric in METRICS:
                values = [reports[(mode, dataset, fold)]["test"][metric] for fold in range(10)]
                rows.append({
                    "dataset": dataset,
                    "mode": mode,
                    "metric": metric,
                    "mean": mean(values),
                    "sd": sample_std(values),
                })
    return rows


def calculate_statistics(reports, bootstrap_seed, bootstrap_replicates):
    omnibus = []
    pairwise = []
    seed_index = 0
    for dataset in DATASETS:
        for metric in PRIMARY_METRICS:
            matrix = np.asarray([
                [reports[(mode, dataset, fold)]["test"][metric] for mode in MODES]
                for fold in range(10)
            ])
            statistic, p_value, kendall_w = friedman_three(matrix)
            omnibus.append({
                "dataset": dataset,
                "metric": metric,
                "friedman_q": statistic,
                "df": 2,
                "p_value": p_value,
                "kendall_w": kendall_w,
            })
            family = []
            baseline = matrix[:, 0]
            for candidate_index, candidate in ((1, "hard"), (2, "weighted")):
                differences = matrix[:, candidate_index] - baseline
                p_raw, rank_biserial, nonzero_pairs = exact_wilcoxon(differences)
                seed_index += 1
                ci_low, ci_high = bootstrap_mean_ci(
                    differences,
                    bootstrap_seed + seed_index,
                    bootstrap_replicates,
                )
                family.append({
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": "{}-minus-none".format(candidate),
                    "mean_difference": mean(differences),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw": p_raw,
                    "rank_biserial": rank_biserial,
                    "nonzero_pairs": nonzero_pairs,
                })
            holm_adjust(family)
            pairwise.extend(family)
    return omnibus, pairwise


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise SystemExit("bootstrap-replicates must be positive")
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    reports, input_hashes = load_reports(results_root)
    issues = audit_reports(reports)
    if issues:
        raise ValueError("Pseudo-supervision audit failed:\n" + "\n".join(issues))

    summary = summarize_modes(reports)
    omnibus, pairwise = calculate_statistics(
        reports, args.bootstrap_seed, args.bootstrap_replicates
    )
    write_csv(output_dir / "pseudo_mode_summary.csv", summary)
    write_csv(output_dir / "pseudo_friedman.csv", omnibus)
    write_csv(output_dir / "pseudo_pairwise_exact.csv", pairwise)
    manifest = {
        "datasets": list(DATASETS),
        "modes": list(MODES),
        "folds_per_dataset_mode": 10,
        "fold_reports": len(reports),
        "audit_issues": issues,
        "fixed_pair_keys": list(FIXED_PAIR_KEYS),
        "statistical_method": {
            "omnibus": "Friedman test; chi-square df=2",
            "pairwise": "exact paired Wilcoxon signed-rank",
            "multiplicity": "Holm within dataset and metric",
            "confidence_interval": "paired bootstrap mean difference",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "input_metrics_sha256": input_hashes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pseudo_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "PSEUDO ANALYSIS PASSED: 3 datasets, 3 modes, 10 folds, "
        "90 reports; wrote {}".format(output_dir)
    )


if __name__ == "__main__":
    main()
