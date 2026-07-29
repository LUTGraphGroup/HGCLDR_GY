#!/usr/bin/env python3
"""Run the fixed-pair none/hard/weighted pseudo-supervision experiment."""

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ("B-dataset", "C-dataset", "F-dataset")
MODES = ("none", "hard", "weighted")
MODE_TAGS = {
    "none": "refit_none_fixed_eval_v1",
    "hard": "refit_pseudo_hard_fixed_eval_v1",
    "weighted": "refit_pseudo_weighted_fixed_eval_v1",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run all pseudo-supervision modes with identical deterministic "
            "folds and fixed validation/test pairs."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pseudo-pos-fraction", type=float, default=0.2)
    parser.add_argument("--pseudo-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--analysis-output", default="results/pseudo_analysis")
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Do not summarize after training; required for partial dataset/fold runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact run.py commands without training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    invalid_folds = sorted(set(args.folds) - set(range(10)))
    if invalid_folds:
        raise SystemExit("fold indices must be in 0..9: {}".format(invalid_folds))
    if args.epochs <= 0 or args.eval_freq <= 0:
        raise SystemExit("epochs and eval-freq must be positive")
    if not 0.0 <= args.pseudo_pos_fraction:
        raise SystemExit("pseudo-pos-fraction must be non-negative")

    root = Path(__file__).resolve().parents[1]
    jobs = []
    for mode in args.modes:
        for dataset in args.datasets:
            for fold in args.folds:
                jobs.append([
                    sys.executable,
                    "run.py",
                    "--dataset", dataset,
                    "--fold", str(fold),
                    "--device", args.device,
                    "--epochs", str(args.epochs),
                    "--eval-freq", str(args.eval_freq),
                    "--seed", str(args.seed),
                    "--refit_after_selection", "1",
                    "--use_fixed_validation_pairs", "1",
                    "--pseudo_mode", mode,
                    "--pseudo_pos_fraction", str(args.pseudo_pos_fraction),
                    "--pseudo_confidence_threshold",
                    str(args.pseudo_confidence_threshold),
                    "--output_root", args.output_root,
                    "--run_tag", MODE_TAGS[mode],
                ])

    print("PLANNED_JOBS={}".format(len(jobs)), flush=True)
    for index, command in enumerate(jobs, 1):
        print("[{}/{}] {}".format(index, len(jobs), " ".join(command)), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=str(root), check=True)

    if args.dry_run or args.skip_analysis:
        return

    complete_design = (
        set(args.datasets) == set(DATASETS)
        and set(args.folds) == set(range(10))
        and set(args.modes) == set(MODES)
    )
    if not complete_design:
        raise SystemExit(
            "automatic analysis requires all 3 datasets, 10 folds, and 3 modes; "
            "use --skip-analysis for a partial run"
        )

    analysis_command = [
        sys.executable,
        "scripts/analyze_pseudo_ablation.py",
        "--results-root", args.output_root,
        "--output-dir", args.analysis_output,
    ]
    print("ANALYSIS {}".format(" ".join(analysis_command)), flush=True)
    subprocess.run(analysis_command, cwd=str(root), check=True)


if __name__ == "__main__":
    main()
