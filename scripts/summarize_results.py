#!/usr/bin/env python3
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
    return (sum((value - center) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for path in sorted(root.glob("*/*/fold_*/metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        test = report.get("test", {})
        records.append({
            "tag": path.parents[2].name,
            "dataset": report.get("dataset", path.parents[1].name),
            "fold": report.get("fold"),
            **{metric: test.get(metric) for metric in METRICS},
        })

    if not records:
        print(f"No completed metrics found under {root}")
        return

    output = Path(args.output) if args.output else root / "experiment_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("tag", "dataset", "fold", *METRICS))
        writer.writeheader()
        writer.writerows(records)

    groups = defaultdict(list)
    for record in records:
        groups[(record["tag"], record["dataset"])].append(record)

    print("tag\tdataset\tfolds\tAUROC(mean+/-sd)\tAUPR(mean+/-sd)\tF1(mean+/-sd)")
    for (tag, dataset), rows in sorted(groups.items()):
        summaries = []
        for metric in ("auc", "aupr", "f1"):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summaries.append("NA" if not values else
                             f"{mean(values):.5f}+/-{sample_std(values):.5f}")
        print(f"{tag}\t{dataset}\t{len(rows)}\t" + "\t".join(summaries))
    print(f"Wrote fold-level summary to {output}")


if __name__ == "__main__":
    main()
