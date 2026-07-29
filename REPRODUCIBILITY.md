# Archived protocol and reviewer issue map

1. Ten-fold workflow: exact assignments and seeds are under each dataset's `folds/`; `scripts/run_cv.py` is the one-command runner.
2. GIP leakage: selection GIP uses only eight training folds; refit GIP uses the merged 90% outer training set; validation/test positives are absent. `verify_package.py --recompute-gip` independently proves equality.
3. Objective/code consistency: `models/base_models.py` is authoritative and implements squared Lorentz-distance logits, learnable scale/bias, negative-count positive weighting, sample-weight normalization, and summed drug-plus-disease contrastive loss.
4. Pseudo positives: main results disable them (`none`). `hard` and `weighted` are explicit ablations; candidates and confidence are saved in result artifacts and are generated from training-only projections.
5. Threshold: validation F1 selects the threshold; it is locked before test scoring. AUROC/AUPR are reported separately from threshold-dependent endpoints.
6. Repository: code, raw inputs, preprocessing, exact folds, refit folds, evaluation scripts, exact dependency pins, and the SHA-256 inventory are included. Formal outputs and response-only statistical summaries are included under `evidence/`.

## Environment scope

- Original manuscript environment: Python 3.8.20, PyTorch 1.13.0+cu116, NVIDIA Tesla P100.
- Current RTX 3090 deployment record: NVIDIA driver 550.163.01, host CUDA 12.4, PyTorch 2.4.1+cu121, NumPy 1.24.3, SciPy 1.10.1, scikit-learn 1.2.2, and Geoopt 0.5.1.
- Standardized clean-release target: Python 3.8.18, PyTorch 2.1.2 with the CUDA 12.1 wheel on a compatible NVIDIA host, plus the exact remaining pins in `requirements.txt`.

The hardware/software records describe compatible execution environments. They do not change the fixed folds, evaluation pairs, validation-only checkpoint and threshold selection, model definition, or paper hyperparameters.

## Three-mode pseudo-supervision reproduction

Run the complete fixed-pair `none`, `hard`, and `weighted` design with:

```bash
python scripts/run_pseudo_ablation.py --device cuda:0 --epochs 2000
```

This produces 90 fold-level reports and automatically runs
`scripts/analyze_pseudo_ablation.py`. Before generating statistics, the analysis
checks the declared mode, training-only candidate provenance, selection/refit
protocol, validation-derived threshold source, and equality of the fixed
validation/test pair hashes across the three modes. It regenerates the
seven-metric descriptive summary and the AUROC/AUPR/F1 paired exact Wilcoxon,
Holm-adjusted, paired-bootstrap tables used in the reviewer response.

## Public result evidence

The formal 30-fold main outputs are provided under `evidence/main_results/refit_none_fixed_eval_v1/`, with an aggregate six-metric table in `evidence/main_results/main_results_aggregate.csv`. The response-only pseudo-supervision summaries and paired exact tests are under `evidence/pseudo/`. `evidence/SHA256SUMS.txt` covers every public evidence file except itself.
