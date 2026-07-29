#!/usr/bin/env python3
"""Validate and aggregate the complete three-fold paper parameter experiments."""

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = ("B-dataset", "C-dataset", "F-dataset")
FOLDS = (0, 1, 2)
METRICS = ("auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc")
T_VALUES = (1, 2, 3, 4, 5)
L_VALUES = (2, 3, 4, 5, 6)
RSSL_VALUES = (0.05, 0.10, 0.15, 0.20)
REG_TOKENS = ("0.001", "0.005", "0.01", "0.05", "0.10")
TEMP_TOKENS = ("0.01", "0.05", "0.10", "0.20", "0.50")
PAPER_EPOCHS = {"B-dataset": 1000, "C-dataset": 2000, "F-dataset": 1000}
PAPER_REG = {"B-dataset": 0.01, "C-dataset": 0.10, "F-dataset": 0.10}
PAPER_SINGLE = {"T": 3.0, "L": 4.0, "r_ssl": 0.05}
PAPER_GRID = {
    "B-dataset": (0.01, 0.05),
    "C-dataset": (0.10, 0.05),
    "F-dataset": (0.10, 0.05),
}


def value_id(value):
    return str(value).replace(".", "p")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.fmean(values)


def sd(values):
    return statistics.stdev(values)


def sensitivity_specs():
    for dataset in DATASETS:
        for family, values in (("T", T_VALUES), ("L", L_VALUES), ("r_ssl", RSSL_VALUES)):
            for value in values:
                if family == "T":
                    tag = f"tune_T_{int(value)}"
                elif family == "L":
                    tag = f"tune_L_{int(value)}"
                else:
                    tag = f"tune_rssl_{value_id(f'{value:.2f}')}"
                yield dataset, family, float(value), tag


def grid_specs():
    for dataset in DATASETS:
        for reg_token in REG_TOKENS:
            for temp_token in TEMP_TOKENS:
                tag = (
                    f"tune_grid_reg_{value_id(reg_token)}"
                    f"_temp_{value_id(temp_token)}"
                )
                yield dataset, float(reg_token), float(temp_token), tag


def validate_common(report, config, dataset, fold, path):
    errors = []
    if report.get("dataset") != dataset or int(report.get("fold", -1)) != fold:
        errors.append("dataset/fold mismatch")
    if report.get("training_protocol", {}).get("name") != "inner_validation_selection_only":
        errors.append("training protocol is not selection-only")
    if report.get("pseudo_supervision", {}).get("mode") != "none":
        errors.append("pseudo supervision is not none")
    boundary = report.get("test_boundary", {})
    if any(
        boundary.get(key) is not False
        for key in (
            "outer_test_labels_used_for_selection",
            "outer_test_predictions_generated",
            "outer_test_metrics_computed",
        )
    ):
        errors.append("outer test boundary was crossed")
    if int(config.get("epochs", -1)) != PAPER_EPOCHS[dataset]:
        errors.append(f"epoch cap differs from paper setting {PAPER_EPOCHS[dataset]}")
    expected_config = {
        "selection_only": 1,
        "use_fixed_validation_pairs": 1,
        "refit_after_selection": 0,
        "seed": 1234,
        "batch_size": 512,
        "num_neg": 8,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(f"{key}={config.get(key)!r}, expected {expected!r}")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))


def load_triplet(root, dataset, tag):
    reports = []
    pair_hashes = []
    manifest_hashes = []
    for fold in FOLDS:
        directory = root / tag / dataset / f"fold_{fold:02d}"
        metrics_path = directory / "selection_metrics.json"
        config_path = directory / "config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Missing final paper result: {dataset} {tag} fold {fold}"
            )
        report = read_json(metrics_path)
        config = read_json(config_path)
        validate_common(report, config, dataset, fold, metrics_path)
        reports.append((report, config))
        pair_hashes.append(report["negative_sampling"]["validation_negative_sha256"])
        manifest_hashes.append(
            report["training_protocol"]["fixed_validation_manifest_sha256"]
        )
    return reports, pair_hashes, manifest_hashes


def aggregate(dataset, family, value, tag, triplet):
    row = {
        "dataset": dataset,
        "family": family,
        "value": value,
        "tag": tag,
        "folds": 3,
        "fold_ids": "0,1,2",
        "epoch_cap": PAPER_EPOCHS[dataset],
        "best_epoch_mean": mean([r["best_epoch"] for r, _ in triplet]),
    }
    for metric in METRICS:
        values = [r["validation"][metric] for r, _ in triplet]
        row[f"validation_{metric}_mean"] = mean(values)
        row[f"validation_{metric}_sd"] = sd(values)
    return row


def audit_fixed_pairs(hash_records):
    errors = []
    grouped = defaultdict(lambda: {"pair": set(), "manifest": set()})
    for dataset, fold, pair_hash, manifest_hash in hash_records:
        grouped[(dataset, fold)]["pair"].add(pair_hash)
        grouped[(dataset, fold)]["manifest"].add(manifest_hash)
    for key, values in grouped.items():
        if len(values["pair"]) != 1 or len(values["manifest"]) != 1:
            errors.append(f"{key}: fixed validation identity differs across candidates")
    if errors:
        raise ValueError("\n".join(errors))


def build_recommendations(sensitivity_rows, grid_rows):
    rows = []
    for dataset in DATASETS:
        for family in ("T", "L", "r_ssl"):
            candidates = [
                row for row in sensitivity_rows
                if row["dataset"] == dataset and row["family"] == family
            ]
            best = sorted(
                candidates,
                key=lambda row: (-row["validation_aupr_mean"], row["value"]),
            )[0]
            paper = next(
                row for row in candidates
                if abs(float(row["value"]) - PAPER_SINGLE[family]) < 1e-12
            )
            rows.append({
                "dataset": dataset,
                "experiment": family,
                "best_value": best["value"],
                "paper_value": paper["value"],
                "best_validation_aupr_mean": best["validation_aupr_mean"],
                "best_validation_aupr_sd": best["validation_aupr_sd"],
                "paper_validation_aupr_mean": paper["validation_aupr_mean"],
                "paper_validation_aupr_sd": paper["validation_aupr_sd"],
                "delta_best_minus_paper": (
                    best["validation_aupr_mean"] - paper["validation_aupr_mean"]
                ),
                "selection_rule": "highest 3-fold mean validation AUPR",
            })
        candidates = [row for row in grid_rows if row["dataset"] == dataset]
        best = sorted(
            candidates,
            key=lambda row: (
                -row["validation_aupr_mean"],
                row["lambda_ssl"],
                row["tau"],
            ),
        )[0]
        paper_reg, paper_temp = PAPER_GRID[dataset]
        paper = next(
            row for row in candidates
            if abs(row["lambda_ssl"] - paper_reg) < 1e-12
            and abs(row["tau"] - paper_temp) < 1e-12
        )
        rows.append({
            "dataset": dataset,
            "experiment": "lambda_ssl_x_tau",
            "best_value": f"{best['lambda_ssl']:g},{best['tau']:g}",
            "paper_value": f"{paper_reg:g},{paper_temp:g}",
            "best_validation_aupr_mean": best["validation_aupr_mean"],
            "best_validation_aupr_sd": best["validation_aupr_sd"],
            "paper_validation_aupr_mean": paper["validation_aupr_mean"],
            "paper_validation_aupr_sd": paper["validation_aupr_sd"],
            "delta_best_minus_paper": (
                best["validation_aupr_mean"] - paper["validation_aupr_mean"]
            ),
            "selection_rule": "highest 3-fold mean validation AUPR",
        })
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("tuning-results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper-parameter-final")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()

    sensitivity_rows = []
    grid_rows = []
    hash_records = []
    for dataset, family, value, tag in sensitivity_specs():
        triplet, pair_hashes, manifest_hashes = load_triplet(root, dataset, tag)
        sensitivity_rows.append(aggregate(dataset, family, value, tag, triplet))
        for fold, pair_hash, manifest_hash in zip(
            FOLDS, pair_hashes, manifest_hashes
        ):
            hash_records.append((dataset, fold, pair_hash, manifest_hash))

    for dataset, reg, temp, tag in grid_specs():
        triplet, pair_hashes, manifest_hashes = load_triplet(root, dataset, tag)
        row = aggregate(dataset, "lambda_ssl_x_tau", 0.0, tag, triplet)
        row.pop("value")
        row["lambda_ssl"] = reg
        row["tau"] = temp
        grid_rows.append(row)
        for fold, pair_hash, manifest_hash in zip(
            FOLDS, pair_hashes, manifest_hashes
        ):
            hash_records.append((dataset, fold, pair_hash, manifest_hash))

    if len(sensitivity_rows) != 42 or len(grid_rows) != 75:
        raise ValueError("Unexpected aggregate row count")
    audit_fixed_pairs(hash_records)
    recommendations = build_recommendations(sensitivity_rows, grid_rows)

    output.mkdir(parents=True, exist_ok=True)
    sensitivity_path = output / "parameter_sensitivity_3fold.csv"
    grid_path = output / "lambda_tau_grid_3fold.csv"
    recommendation_path = output / "best_parameter_report.csv"
    write_csv(sensitivity_path, sensitivity_rows)
    write_csv(grid_path, grid_rows)
    write_csv(recommendation_path, recommendations)

    lines = [
        "# 最终论文参数实验汇总",
        "",
        "- 单参数敏感性：126份报告，汇总为42个三折均值。",
        "- lambda_ssl × tau：225份报告，汇总为75个三折均值。",
        "- 所有结果均为固定内层验证集，未访问外层测试集。",
        "- 第一选择指标：三折平均验证AUPR；AUROC和F1用于辅助判断。",
        "",
        "| 数据集 | 实验 | 最佳值 | 论文值 | 最佳AUPR | 论文值AUPR | 差值 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in recommendations:
        lines.append(
            f"| {row['dataset'][0]} | {row['experiment']} | "
            f"{row['best_value']} | {row['paper_value']} | "
            f"{row['best_validation_aupr_mean']:.5f} | "
            f"{row['paper_validation_aupr_mean']:.5f} | "
            f"{row['delta_best_minus_paper']:+.5f} |"
        )
    report_path = output / "final_parameter_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "complete",
        "protocol": "three-fold fixed inner validation; outer test untouched",
        "folds": [0, 1, 2],
        "sensitivity_reports": 126,
        "sensitivity_aggregate_rows": 42,
        "cross_grid_reports": 225,
        "cross_grid_aggregate_rows": 75,
        "fixed_pair_audit_issues": 0,
        "selection_metric": "mean validation AUPR",
        "output_sha256": {
            sensitivity_path.name: sha256(sensitivity_path),
            grid_path.name: sha256(grid_path),
            recommendation_path.name: sha256(recommendation_path),
            report_path.name: sha256(report_path),
        },
    }
    (output / "final_parameter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n".join(lines))
    print(f"\nWrote final paper parameter data to {output}")


if __name__ == "__main__":
    main()