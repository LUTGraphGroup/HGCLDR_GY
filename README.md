# HGCLDR R2 reproducibility release

This repository is the auditable code-and-data archive for the R2 revision. The evaluation uses deterministic, drug-stratified 10-fold cross-validation.

## Leakage-safe protocol

For outer fold k, fold k is test and fold (k+1) mod 10 is validation. The other eight folds train model selection. Epoch and F1 threshold are selected only on validation labels. A fresh model is then trained for the selected epoch count on the merged 90% outer-training data; the validation-derived threshold is locked before the outer test is opened. Fold-specific adjacency and drug/disease GIP kernels are computed only from the active training associations. Manifests contain counts, seeds, gamma values, and SHA-256 hashes.

Main paper runs use `--pseudo_mode none`. The disclosed ablations use `hard` and confidence-weighted `weighted`; projected candidates are built from training-only graph information and held-out positives are masked. The supervised loss uses squared Lorentz distance logits with learnable scale/bias, BCE-with-logits, `pos_weight=num_neg`, sample-weight normalization, and a summed drug-plus-disease contrastive term (see `models/base_models.py`).

## Install

Python 3.8.18 is the archived target.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip==23.3.2
python -m pip install -r requirements.txt
```

For CUDA, install the matching PyTorch 2.1.2 wheel for the host before installing the remaining pins.

### Hardware and environment provenance

The implementation is not tied to a single GPU generation. The original model development and primary experiments reported in the manuscript used Python 3.8.20, PyTorch 1.13.0+cu116, and an NVIDIA Tesla P100 GPU. The corrected R2 workflow was subsequently deployed on an NVIDIA GeForce RTX 3090 without changing the fixed folds, evaluation protocol, model definition, or reported hyperparameter settings. The archived RTX 3090 execution record used NVIDIA driver 550.163.01, host CUDA 12.4, PyTorch 2.4.1+cu121, NumPy 1.24.3, SciPy 1.10.1, scikit-learn 1.2.2, and Geoopt 0.5.1.

For a clean and portable installation, this release standardizes on Python 3.8.18, PyTorch 2.1.2 (use the CUDA 12.1 wheel on a compatible NVIDIA host), and the remaining exact versions in `requirements.txt`. These release pins define a reproducibility deployment target; they do not replace the provenance statement for the original P100 experiments. Both P100-class and RTX 3090-class CUDA devices can execute the archived implementation when a compatible PyTorch/CUDA stack is installed.

## Verify before running

```bash
python scripts/verify_package.py --root . --recompute-gip
```

This validates all 60 selection/refit partitions, manifest hashes, fold exclusivity, fold-specific GIP recomputation, fixed evaluation pairs, configuration, and release checksums. Success ends with `VERIFICATION PASSED`.

## Reproduce the corrected main experiment

One fold:

```bash
python run.py --dataset B-dataset --fold 0 --device cuda:0 --epochs 2000 --eval-freq 10 --refit_after_selection 1 --use_fixed_validation_pairs 1 --pseudo_mode none --run_tag table3_main
```

All datasets and folds (one command):

```bash
python scripts/run_cv.py --device cuda:0 --epochs 2000 --run-tag table3_main
python scripts/summarize_results.py --root results --output results/table3_main_summary.csv
```

The formal paper commands always pass `--epochs 2000`. The generic parser default in `config.py` is not the formal R2 training budget; Table 2 reports the explicitly supplied maximum selection-training budget, while validation AUPR selects the actual checkpoint epoch within that budget.

`metrics.json` records the validation-derived locked threshold, threshold-free AUROC/AUPR, threshold-dependent Accuracy/Precision/Recall/F1/MCC, seeds, effective environment, pair hashes, GIP source, and pseudo-supervision mode. `config.json` is the exact effective configuration.

## Reproduce the three pseudo-supervision modes

Run `none`, `hard`, and `weighted` on the same three datasets, ten folds,
fixed validation/test pairs, seeds, selection/refit protocol, and 2000-epoch
selection budget:

```bash
python scripts/run_pseudo_ablation.py --device cuda:0 --epochs 2000
```

The runner schedules 90 fold-level jobs and then calls
`scripts/analyze_pseudo_ablation.py`. The analysis first audits mode labels,
training-only candidate provenance, validation-derived threshold selection, and
fixed-pair hashes. It then writes `pseudo_mode_summary.csv`,
`pseudo_friedman.csv`, `pseudo_pairwise_exact.csv`, and an input-hash manifest
under `results/pseudo_analysis/`.

Use `--dry-run` to print all commands without training. Partial dataset/fold
runs are supported with `--datasets`, `--folds`, and `--modes`; add
`--skip-analysis` because the formal paired analysis requires all 90 reports.
The pseudo fraction and confidence threshold are explicitly fixed at 0.2 and
0.0 by default and can be overridden through the documented runner options.

## Rebuild preprocessing artifacts

The release already includes the exact folds. To regenerate deterministically from each `drug_disease_list.pkl`:

```bash
python scripts/generate_cv_folds.py --seed 1234
python scripts/generate_refit_folds.py --force
python scripts/materialize_fixed_eval_pairs.py
python scripts/verify_package.py --root . --recompute-gip
```


## Output-to-paper map

- Table 3: `results/table3_main_summary.csv` from `pseudo_mode=none`.
- Pseudo-positive ablation: repeat `run_cv.py` with `--pseudo-mode hard` and `weighted`; compare against `none` using identical fixed evaluation pairs.
- One-command pseudo-supervision reproduction: `scripts/run_pseudo_ablation.py`; regenerated tables are written under `results/pseudo_analysis/`.
- Per-fold evidence: `results/<tag>/<dataset>/fold_XX/{metrics.json,config.json,validation_predictions.npz,test_predictions.npz}`.
- Exact split/GIP evidence: `data/<dataset>/folds/fold_XX/{manifest.json,train_pairs.csv,val_pairs.csv,test_pairs.csv,DrugGIP.npy,DiseaseGIP.npy}` and `refit/`.
- Archived formal outputs supplied with this release: `evidence/main_results/refit_none_fixed_eval_v1/` and `evidence/main_results/main_results_aggregate.csv`.
- Response-only pseudo-supervision summaries: `evidence/pseudo/pseudo_mode_summary.csv` and `evidence/pseudo/pseudo_pairwise_exact.csv`.

## Seeds

Split seed: 1234. Training seed: 1234. Fold offsets: `fold*100`. Validation/test negative seeds: `seed+fold*100+1/+2`. Refit offset: 1,000,000. Epoch/view seeds and sampler strategy are written to each `metrics.json`.

## Integrity

`SHA256SUMS.txt` covers every distributed repository file except itself and the zip container. The formal 30-fold main outputs are bundled under `evidence/main_results/`; newly generated runtime outputs and server logs are not bundled and should be archived separately after a rerun. Regenerate the repository inventory with `python scripts/make_checksums.py --root .`. Do not edit generated folds without regenerating manifests and checksums.
