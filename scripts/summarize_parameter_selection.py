#!/usr/bin/env python3
"""Summarize validation-only parameter searches without reading test metrics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = ("auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc")


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return (
        sum((value - center) ** 2 for value in values) / (len(values) - 1)
    ) ** 0.5


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("tuning-results"))
    args = parser.parse_args()

    records = []
    for metrics_path in sorted(
        args.root.glob("*/*/fold_*/selection_metrics.json")
    ):
        config_path = metrics_path.with_name("config.json")
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        boundary = report.get("test_boundary", {})
        boundary_ok = (
            boundary.get("outer_test_labels_used_for_selection") is False
            and boundary.get("outer_test_predictions_generated") is False
            and boundary.get("outer_test_metrics_computed") is False
        )
        if not boundary_ok:
            raise ValueError(
                f"Test-boundary audit failed for {metrics_path}"
            )
        validation = report.get("validation") or {}
        if validation.get("aupr") is None:
            raise ValueError(
                f"Missing validation AUPR in {metrics_path}"
            )
        effective_env = config.get("effective_environment") or {}
        records.append({
            "tag": metrics_path.parents[2].name,
            "dataset": report["dataset"],
            "fold": int(report["fold"]),
            "best_epoch": int(report["best_epoch"]),
            "epoch_cap": int(config["epochs"]),
            "diffusion_steps_T": int(effective_env["PROJ_LGCN_K"]),
            "encoder_depth_L": int(config["num_layers"]),
            "ssl_ratio": float(config["ssl_ratio"]),
            "ssl_reg": float(config["ssl_reg"]),
            "ssl_temp": float(config["ssl_temp"]),
            "learning_rate": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
            "momentum": float(config["momentum"]),
            "batch_size": int(config["batch_size"]),
            "num_neg": int(config["num_neg"]),
            "seed": int(config["seed"]),
            "fixed_validation_manifest_sha256": report[
                "training_protocol"
            ]["fixed_validation_manifest_sha256"],
            "validation_negative_sha256": report[
                "negative_sampling"
            ]["validation_negative_sha256"],
            "test_boundary_ok": boundary_ok,
            **{
                f"validation_{metric}": validation.get(metric)
                for metric in METRICS
            },
        })

    if not records:
        print(f"No selection_metrics.json files found under {args.root}")
        return

    fold_fields = list(records[0].keys())
    write_csv(
        args.root / "parameter_selection_folds.csv",
        records,
        fold_fields,
    )

    groups = defaultdict(list)
    for record in records:
        groups[(record["tag"], record["dataset"])].append(record)

    summaries = []
    parameter_fields = (
        "epoch_cap",
        "diffusion_steps_T",
        "encoder_depth_L",
        "ssl_ratio",
        "ssl_reg",
        "ssl_temp",
        "learning_rate",
        "weight_decay",
        "momentum",
        "batch_size",
        "num_neg",
        "seed",
    )
    for (tag, dataset), rows in groups.items():
        rows = sorted(rows, key=lambda row: row["fold"])
        summary = {
            "tag": tag,
            "dataset": dataset,
            "folds": len(rows),
            "fold_ids": ",".join(str(row["fold"]) for row in rows),
        }
        for field in parameter_fields:
            values = {row[field] for row in rows}
            summary[field] = (
                next(iter(values)) if len(values) == 1 else "MIXED"
            )
        summary["best_epoch_mean"] = mean(
            [row["best_epoch"] for row in rows]
        )
        for metric in METRICS:
            values = [
                float(row[f"validation_{metric}"])
                for row in rows
                if row[f"validation_{metric}"] is not None
            ]
            summary[f"validation_{metric}_mean"] = mean(values)
            summary[f"validation_{metric}_sd"] = sample_std(values)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            row["dataset"],
            -row["validation_aupr_mean"],
            row["tag"],
        )
    )
    summary_fields = list(summaries[0].keys())
    write_csv(
        args.root / "parameter_selection_summary.csv",
        summaries,
        summary_fields,
    )
    ranking_rows = []
    for dataset in ("B-dataset", "C-dataset", "F-dataset"):
        rank = 0
        for summary in [
            row for row in summaries if row["dataset"] == dataset
        ]:
            rank += 1
            ranking_rows.append({"validation_aupr_rank": rank, **summary})
    write_csv(
        args.root / "parameter_selection_ranking.csv",
        ranking_rows,
        ["validation_aupr_rank", *summary_fields],
    )

    print(
        "dataset\trank\ttag\tfolds\tvalidation AUPR(mean+/-sd)"
    )
    for row in ranking_rows:
        print(
            f"{row['dataset']}\t{row['validation_aupr_rank']}\t"
            f"{row['tag']}\t{row['folds']}\t"
            f"{row['validation_aupr_mean']:.5f}+/-"
            f"{row['validation_aupr_sd']:.5f}"
        )
    print(
        "Wrote validation-only fold, summary, and ranking CSV files to "
        f"{args.root}"
    )


if __name__ == "__main__":
    main()
