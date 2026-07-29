# Public R2 result evidence

This directory contains the formal corrected main outputs and the response-required pseudo-supervision summaries.

## Formal main evaluation

`main_results/refit_none_fixed_eval_v1/` contains the complete 30-fold output tree (three datasets by ten outer folds) used for the corrected main results. Each fold retains its `metrics.json`, `config.json`, validation/test predictions, model checkpoints, and pseudo-candidate audit artifacts. The reported main mode is `none`; projected candidates are retained as audit artifacts but are not appended to the supervised-positive set.

`main_results/main_results_fold_metrics.csv` extracts the six manuscript endpoints, locked threshold, threshold source, and pseudo mode from every fold. `main_results/main_results_aggregate.csv` reports the arithmetic mean and sample standard deviation (n-1 denominator) over the ten outer folds. Its four-decimal `*_paper` fields map directly to Table 3 and the corrected main-results table in the Response Letter.

## Response-only pseudo-supervision evidence

`pseudo/pseudo_mode_summary.csv` contains the fixed-pair none/hard/weighted summaries. `pseudo/pseudo_pairwise_exact.csv` contains the paired exact tests used in the Response Letter. These optional modes are response evidence and are not presented as components of the reported manuscript method.

The distributed archive contains the response tables but not the large
hard/weighted checkpoint trees. Reproduce all three modes and regenerate these
tables with `python scripts/run_pseudo_ablation.py --device cuda:0 --epochs
2000`. The runner uses the same fixed folds and evaluation pairs for every mode;
its analysis script refuses to produce the formal comparison if any of the 90
reports or fixed-pair hashes is missing or inconsistent.

## Integrity

`SHA256SUMS.txt` covers every file under `evidence/` except itself. The repository-level `SHA256SUMS.txt` independently covers the complete public release tree, including this evidence directory.
