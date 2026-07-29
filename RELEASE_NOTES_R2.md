# HGCLDR R2 release notes

This release supersedes `HGCLDR-R2-GitHub-lean-20260724.zip`. The verified package corrections and the subsequent documentation and runner additions do not change the model architecture, datasets, fixed folds, evaluation pairs, training objective, hyperparameters, or archived result values.

## Verified package corrections

1. `utils/data.py` initializes and validates the supplied outer fold before constructing fold paths. This corrects the initialization defect that prevented `run.py` from starting in the earlier archive.
2. Both artifact-writing paths serialize the resolved pseudo-supervision mode. When the parser value is null, `config.json` records `"none"`, matching the executed behavior and `metrics.json`.
3. `README.md` and `REPRODUCIBILITY.md` state that the implementation sums the drug- and disease-side contrastive terms rather than mean-reducing them.

## Reproducibility additions

- `scripts/run_pseudo_ablation.py` runs the `none`, `hard`, and `weighted` pseudo-supervision modes over the same three datasets and ten fixed folds.
- `scripts/analyze_pseudo_ablation.py` verifies mode labels, the selection/refit protocol, validation-derived thresholds, fixed-pair hashes, and metric completeness before regenerating the descriptive and paired statistical tables used in the reviewer response.
- The README distinguishes newly generated runtime outputs and server logs from the formal 30-fold outputs distributed under `evidence/main_results/`.

## Verification

Run `python scripts/verify_package.py --root . --recompute-gip` before use. The release includes the formal fold-level outputs and their checksum inventory under `evidence/`.
