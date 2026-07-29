import time
import traceback
from datetime import datetime
import numpy as np
import torch
import os
import json
import hashlib
from pathlib import Path

from config import parser
from models.base_models import HyperCL
from optim import RiemannianSGD
from utils.data import Data
from utils.util import set_seed, sp_mat_to_sp_tensor
from utils.log import Logger
from utils.sampler import WarpSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef
)

EFFECTIVE_ENV_KEYS = (
    "PROJ_WEIGHT", "PROJ_ALPHA_METHOD", "PROJ_ALPHA_Q90",
    "PROJ_ALPHA_Q50", "PROJ_ALPHA_Q10", "PROJ_NORM_METHOD",
    "PROJ_GATE_LAMBDA", "PROJ_SUP_METHOD", "PROJ_MIN_SUP",
    "PROJ_MAX_SUP", "CONSIST_GATE", "CONSIST_NEI", "CONSIST_THR",
    "PROJ_LGCN_K", "PROJ_LGCN_TOPK", "PROJ_LGCN_BETA",
    "SIM_AUG_ENABLE", "SIM_AUG_TOPK", "SIM_AUG_WEIGHT",
    "SIM_AUG_THRESHOLD", "SSL_USE_MSG", "SSL_P_OBS",
    "SSL_DEG_SAFE", "SIM_TOPK", "SIM_DROP_SELF", "EMA_DECAY",
)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

def _to_affinity(scores, higher_is_better=False):
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    return s if higher_is_better else -s

def _pick_best_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx]), float(prec[idx]), float(rec[idx])

def _dict_pairs(mapping):
    return {(int(drug), int(disease)) for drug, diseases in mapping.items() for disease in diseases}


def _build_evaluation_set(run_data, split_dict, seed, excluded_negative_pairs=None):
    positive_pairs = sorted(_dict_pairs(split_dict))
    if not positive_pairs:
        raise ValueError("Evaluation split contains no positive pairs")

    observed = (
        _dict_pairs(run_data.train_dict) | _dict_pairs(run_data.val_dict) |
        _dict_pairs(run_data.test_dict)
    )
    projected_block = run_data.adj_train_msg[:run_data.num_drugs, run_data.num_drugs:].tocsr()
    projected = set(zip(*projected_block.nonzero()))
    banned = observed | projected | set(excluded_negative_pairs or ())
    candidates = np.asarray(
        [
            (drug, disease)
            for drug in range(run_data.num_drugs)
            for disease in range(run_data.num_diseases)
            if (drug, disease) not in banned
        ],
        dtype=np.int64,
    )
    if len(candidates) < len(positive_pairs):
        raise ValueError(
            f"Not enough unknown pairs for evaluation: need {len(positive_pairs)}, "
            f"found {len(candidates)}"
        )
    rng = np.random.default_rng(seed)
    selected = candidates[rng.choice(len(candidates), size=len(positive_pairs), replace=False)]
    positives = np.asarray(positive_pairs, dtype=np.int64)
    pairs = np.concatenate([positives, selected], axis=0)
    pairs[:, 1] += run_data.num_drugs
    labels = np.concatenate(
        [np.ones(len(positives), dtype=np.int32), np.zeros(len(selected), dtype=np.int32)]
    )
    return pairs, labels, {tuple(map(int, pair)) for pair in selected}

def _load_fixed_validation_set(run_data):
    if args.fold < 0:
        raise ValueError("Fixed validation pairs require --fold 0..9")
    fixed_dir = (
        Path("data") / args.dataset / "folds"
        / f"fold_{args.fold:02d}" / "fixed_eval"
    )
    pairs_path = fixed_dir / "validation_pairs.npz"
    manifest_path = fixed_dir / "manifest.json"
    if not pairs_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing versioned fixed validation artifacts in {fixed_dir}"
        )
    with np.load(pairs_path, allow_pickle=False) as archive:
        pairs = np.asarray(archive["pairs"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int32).reshape(-1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(labels):
        raise ValueError(f"Malformed fixed validation pairs: {pairs_path}")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError(f"Duplicate fixed validation pairs: {pairs_path}")
    local_diseases = pairs[:, 1] - run_data.num_drugs
    if (
        np.any(pairs[:, 0] < 0)
        or np.any(pairs[:, 0] >= run_data.num_drugs)
        or np.any(local_diseases < 0)
        or np.any(local_diseases >= run_data.num_diseases)
    ):
        raise ValueError(f"Out-of-range fixed validation pairs: {pairs_path}")
    actual_positives = {
        (int(drug), int(disease))
        for drug, disease in zip(pairs[labels == 1, 0], local_diseases[labels == 1])
    }
    if actual_positives != _dict_pairs(run_data.val_dict):
        raise ValueError("Fixed validation positives do not match fold assignments")
    negatives = pairs[labels == 0].copy()
    negatives[:, 1] -= run_data.num_drugs
    negative_set = {tuple(map(int, pair)) for pair in negatives}
    return pairs, labels, negative_set, manifest_path



def _pairs_sha256(pairs):
    canonical = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return hashlib.sha256(canonical.tobytes()).hexdigest()
def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _classification_metrics(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int32)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "threshold": float(threshold),
    }


def _score_pairs(model, embeddings, pairs):
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long, device=args.device)
    return model.score(embeddings, pair_tensor).detach().cpu().numpy().reshape(-1)


def _build_sampler(run_data, seed):
    banned = run_data.adj_train_msg[
        :run_data.num_drugs, run_data.num_drugs:
    ].astype(bool).tocsr()
    banned = (
        banned + run_data.val_csr.astype(bool) + run_data.test_csr.astype(bool)
    ).astype(bool).tocsr()
    return WarpSampler(
        (run_data.num_drugs, run_data.num_diseases),
        run_data.adj_train,
        args.batch_size,
        args.num_neg,
        n_workers=0,
        ban_csr=banned,
        seed=seed,
    )


def _build_model(run_data):
    return HyperCL(
        (run_data.num_drugs, run_data.num_diseases),
        args,
        drug_gip=run_data.drug_gip_sim,
        disease_gip=run_data.disease_gip_sim,
        drug_fp=run_data.drug_fp_sim,
        disease_ps=run_data.disease_ps_sim,
    ).to(args.device)


def _train_one_epoch(
    model,
    run_data,
    run_sampler,
    run_ema,
    optimizer,
    epoch,
    num_batches,
    phase_seed,
    phase_name,
):
    model.train()
    avg_loss = 0.0
    train_start = time.time()
    epoch_seed = int(phase_seed + epoch * 100_000)
    print(f"Creating deterministic subgraphs for {phase_name}...")
    sub_graph1 = sp_mat_to_sp_tensor(
        run_data.create_adj_mat(
            True, 'ed', rng=np.random.default_rng(epoch_seed + 1)
        )
    ).to(args.device)
    sub_graph2 = sp_mat_to_sp_tensor(
        run_data.create_adj_mat(
            True, 'ed', rng=np.random.default_rng(epoch_seed + 2)
        )
    ).to(args.device)
    graph_time = time.time() - train_start

    for batch in range(num_batches):
        triples = run_sampler.next_batch()
        drugs = triples[:, 0].astype(np.int64)
        negatives = triples[:, 2:]
        num_negative = negatives.shape[1]
        rng = np.random.default_rng(epoch_seed + 1_000 + batch)
        for negative_index in range(num_negative):
            chosen = run_data.adaptive_negative_sampling(
                drugs.copy(),
                rng,
                run_data.hard_neg_pool,
                use_similarity_shrink=True,
                shrink_factor=0.7,
            )
            negatives[:, negative_index] = chosen + run_data.num_drugs
            violations = sum(
                int(chosen[row] in run_data.ban_observed[int(drug)])
                for row, drug in enumerate(drugs)
            )
            if violations:
                raise RuntimeError(
                    f"Negative samples collide with observed positives: {violations}"
                )

        embeddings1, embeddings2, embeddings3 = model.encode(
            run_data.adj_train_norm, sub_graph1, sub_graph2
        )
        pseudo_mode = (args.pseudo_mode or "none").lower()
        if pseudo_mode not in {"none", "hard", "weighted"}:
            raise ValueError(
                f"pseudo_mode must be none, hard, or weighted; got {args.pseudo_mode}"
            )
        observed_positive_count = triples.shape[0]
        positive_weights = np.ones(observed_positive_count, dtype=np.float32)
        if pseudo_mode != "none" and len(run_data.pseudo_pos_pairs) > 0:
            eligible = np.flatnonzero(
                run_data.pseudo_pos_confidence
                >= args.pseudo_confidence_threshold
            )
            pseudo_count = min(
                int(observed_positive_count * args.pseudo_pos_fraction),
                len(eligible),
            )
            if pseudo_count > 0:
                indices = rng.choice(
                    eligible, size=pseudo_count, replace=False
                )
                additional_pairs = run_data.pseudo_pos_pairs[indices]
                additional = np.zeros(
                    (pseudo_count, triples.shape[1]), dtype=triples.dtype
                )
                additional[:, 0] = additional_pairs[:, 0]
                additional[:, 1] = (
                    additional_pairs[:, 1] + run_data.num_drugs
                )
                additional[:, 2:] = triples[:pseudo_count, 2:]
                triples = np.vstack([triples, additional])
                if pseudo_mode == "hard":
                    pseudo_weights = np.ones(
                        pseudo_count, dtype=np.float32
                    )
                else:
                    pseudo_weights = run_data.pseudo_pos_confidence[
                        indices
                    ].astype(np.float32)
                positive_weights = np.concatenate(
                    [positive_weights, pseudo_weights], axis=0
                )

        optimizer.zero_grad(set_to_none=True)
        train_loss = model.compute_loss(
            embeddings1,
            embeddings2,
            embeddings3,
            triples,
            positive_weights=positive_weights,
            observed_positive_count=observed_positive_count,
        )
        if torch.isnan(train_loss):
            raise FloatingPointError("Training loss is NaN")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        run_ema.update(model)
        avg_loss += float(train_loss.detach().cpu().item()) / num_batches

    train_time = time.time() - train_start
    log.write(
        f"{phase_name}:{epoch:3d} {avg_loss:.5f} "
        f"{graph_time:.5f} {train_time:.5f}\n"
    )
    return avg_loss


def _run_refit(best_epoch):
    if args.fold < 0:
        raise ValueError("refit_after_selection requires --fold 0..9")
    refit_seed = (
        int(args.seed) + int(args.refit_seed_offset)
        + max(args.fold, 0) * 100
    )
    set_seed(refit_seed)
    refit_data = Data(
        args.dataset,
        args.norm_adj,
        refit_seed,
        args.ssl_ratio,
        fold=args.fold,
        split_variant='refit',
    )
    refit_sampler = _build_sampler(refit_data, refit_seed)
    refit_model = _build_model(refit_data)
    refit_ema = EMA(
        refit_model, decay=float(os.environ.get("EMA_DECAY", "0.999"))
    )
    optimizer = RiemannianSGD(
        params=refit_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    num_pairs = refit_data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(
        f"[REFIT] training on train+validation for {best_epoch} epochs; "
        f"positive_pairs={num_pairs}, seed={refit_seed}"
    )
    try:
        for epoch in range(1, best_epoch + 1):
            print(f"\n========== Refit epoch {epoch} ==========")
            _train_one_epoch(
                refit_model,
                refit_data,
                refit_sampler,
                refit_ema,
                optimizer,
                epoch,
                num_batches,
                refit_seed,
                "RefitTrain",
            )
    finally:
        refit_sampler.close()

    refit_model.eval()
    refit_ema.copy_to(refit_model)
    final_state = {
        key: value.detach().cpu().clone()
        for key, value in refit_model.state_dict().items()
    }
    return refit_data, refit_model, final_state, refit_seed

def _save_evaluation_artifacts(
    best_epoch, threshold, val_data, test_data, val_metrics, test_metrics,
    selection_state, final_state, selection_data, final_data, refit_seed,
):
    fold_name = f"fold_{args.fold:02d}"
    output_dir = Path(args.output_root) / args.run_tag / args.dataset / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    val_pairs, val_labels, val_scores = val_data
    test_pairs, test_labels, test_scores = test_data
    torch.save(final_state, output_dir / 'best_model.pt')
    torch.save(selection_state, output_dir / 'selection_best_model.pt')
    val_negative_pairs = val_pairs[val_labels == 0].copy()
    test_negative_pairs = test_pairs[test_labels == 0].copy()
    val_negative_pairs[:, 1] -= selection_data.num_drugs
    test_negative_pairs[:, 1] -= final_data.num_drugs
    np.savez_compressed(
        output_dir / 'pseudo_candidates.npz',
        pairs=final_data.pseudo_pos_pairs,
        confidence=final_data.pseudo_pos_confidence,
    )
    np.savez_compressed(
        output_dir / 'selection_pseudo_candidates.npz',
        pairs=selection_data.pseudo_pos_pairs,
        confidence=selection_data.pseudo_pos_confidence,
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        pairs=val_pairs,
        labels=val_labels,
        scores=val_scores,
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        pairs=test_pairs,
        labels=test_labels,
        scores=test_scores,
    )
    report = {
        "dataset": args.dataset,
        "fold": None if args.fold < 0 else int(args.fold),
        "selection_metric": "validation_aupr",
        "threshold_source": "validation_f1",
        "best_epoch": int(best_epoch),
        "locked_threshold": float(threshold),
        "training_protocol": {
            "name": (
                "inner_validation_then_outer_refit"
                if int(args.refit_after_selection) else "selection_model_only"
            ),
            "selection_split_path": os.path.abspath(selection_data.split_path),
            "final_split_path": os.path.abspath(final_data.split_path),
            "selection_train_positive_pairs": int(selection_data.train_csr.nnz),
            "selection_validation_positive_pairs": int(selection_data.val_csr.nnz),
            "final_train_positive_pairs": int(final_data.train_csr.nnz),
            "final_training_epochs": int(best_epoch),
            "gip_source": "final_training_associations_only",
            "selection_manifest_sha256": _file_sha256(
                Path(selection_data.split_path) / "manifest.json"
            ),
            "final_manifest_sha256": _file_sha256(
                Path(final_data.split_path) / "manifest.json"
            ),
            "selection_gip_source": "selection_training_associations_only",
            "final_gip_source": "final_training_associations_only",
        },
        "pseudo_supervision": {
            "mode": args.pseudo_mode or "none",
            "fraction": float(args.pseudo_pos_fraction),
            "confidence_threshold": float(args.pseudo_confidence_threshold),
            "selection_available_candidates": int(len(selection_data.pseudo_pos_pairs)),
            "final_available_candidates": int(len(final_data.pseudo_pos_pairs)),
            "candidate_source": "training-only projection with held-out masking",
        },
        "protein_projection_sources": {
            "dataset": args.dataset,
            "drug_protein_path": os.path.abspath(final_data.dp_path),
            "protein_disease_path": os.path.abspath(final_data.pd_path),
            "drug_protein_env_override": bool(final_data.dp_path_overridden),
            "protein_disease_env_override": bool(final_data.pd_path_overridden),
        },
        "validation": val_metrics,
        "effective_environment": {key: os.environ.get(key) for key in EFFECTIVE_ENV_KEYS},
        "negative_sampling": {
            "training_negatives_per_positive": int(args.num_neg),
            "training_strategy": "dynamic hard-negative sampling",
            "evaluation_negatives_per_positive": 1,
            "evaluation_strategy": "fixed seeded sampling without replacement",
            "validation_test_negative_overlap": 0,
            "validation_negative_count": int(len(val_negative_pairs)),
            "test_negative_count": int(len(test_negative_pairs)),
            "validation_negative_sha256": _pairs_sha256(val_negative_pairs),
            "test_negative_sha256": _pairs_sha256(test_negative_pairs),
        },
        "test": test_metrics,
        "seeds": {
            "training": int(args.seed),
            "refit_training": None if refit_seed is None else int(refit_seed),
            "subgraph_strategy": "seed + phase_offset + epoch*100000 + view",
            "sampler_strategy": "explicit numpy Generator seed per phase",
            "validation_negative_sampling": int(args.seed + max(args.fold, 0) * 100 + 1),
            "test_negative_sampling": int(args.seed + max(args.fold, 0) * 100 + 2),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    serializable_config = {
        key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        for key, value in vars(args).items()
    }
    serializable_config["pseudo_mode"] = args.pseudo_mode or "none"
    serializable_config["effective_environment"] = report["effective_environment"]
    (output_dir / "config.json").write_text(
        json.dumps(serializable_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir

def _save_selection_only_artifacts(
    best_epoch, threshold, validation_data, validation_metrics,
    selection_state, selection_data, fixed_manifest_path,
):
    fold_name = f"fold_{args.fold:02d}"
    output_dir = Path(args.output_root) / args.run_tag / args.dataset / fold_name
    output_dir.mkdir(parents=True, exist_ok=True)
    val_pairs, val_labels, val_scores = validation_data
    val_negative_pairs = val_pairs[val_labels == 0].copy()
    val_negative_pairs[:, 1] -= selection_data.num_drugs

    torch.save(selection_state, output_dir / "selection_best_model.pt")
    np.savez_compressed(
        output_dir / "selection_pseudo_candidates.npz",
        pairs=selection_data.pseudo_pos_pairs,
        confidence=selection_data.pseudo_pos_confidence,
    )
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        pairs=val_pairs,
        labels=val_labels,
        scores=val_scores,
    )
    report = {
        "dataset": args.dataset,
        "fold": None if args.fold < 0 else int(args.fold),
        "selection_metric": "validation_aupr",
        "threshold_source": "fixed_validation_f1",
        "best_epoch": int(best_epoch),
        "locked_threshold": float(threshold),
        "training_protocol": {
            "name": "inner_validation_selection_only",
            "selection_split_path": os.path.abspath(selection_data.split_path),
            "selection_train_positive_pairs": int(selection_data.train_csr.nnz),
            "selection_validation_positive_pairs": int(selection_data.val_csr.nnz),
            "selection_manifest_sha256": _file_sha256(
                Path(selection_data.split_path) / "manifest.json"
            ),
            "selection_gip_source": "selection_training_associations_only",
            "fixed_validation_manifest_sha256": (
                _file_sha256(fixed_manifest_path)
                if fixed_manifest_path is not None else None
            ),
        },
        "pseudo_supervision": {
            "mode": args.pseudo_mode or "none",
            "fraction": float(args.pseudo_pos_fraction),
            "confidence_threshold": float(args.pseudo_confidence_threshold),
            "selection_available_candidates": int(
                len(selection_data.pseudo_pos_pairs)
            ),
            "candidate_source": (
                "training-only projection with held-out masking"
            ),
        },
        "validation": validation_metrics,
        "effective_environment": {
            key: os.environ.get(key) for key in EFFECTIVE_ENV_KEYS
        },
        "negative_sampling": {
            "training_negatives_per_positive": int(args.num_neg),
            "training_strategy": "dynamic hard-negative sampling",
            "evaluation_negatives_per_positive": 1,
            "evaluation_strategy": (
                "versioned fixed validation pairs"
                if fixed_manifest_path is not None
                else "fixed seeded validation sampling"
            ),
            "validation_negative_count": int(len(val_negative_pairs)),
            "validation_negative_sha256": _pairs_sha256(val_negative_pairs),
        },
        "test_boundary": {
            "outer_test_labels_used_for_selection": False,
            "outer_test_predictions_generated": False,
            "outer_test_metrics_computed": False,
        },
        "seeds": {
            "training": int(args.seed),
            "subgraph_strategy": "seed + phase_offset + epoch*100000 + view",
            "sampler_strategy": "explicit numpy Generator seed per phase",
            "validation_negative_sampling": (
                "versioned fixed pairs"
                if fixed_manifest_path is not None
                else int(args.seed + max(args.fold, 0) * 100 + 1)
            ),
        },
    }
    (output_dir / "selection_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    serializable_config = {
        key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        for key, value in vars(args).items()
    }
    serializable_config["pseudo_mode"] = args.pseudo_mode or "none"
    serializable_config["effective_environment"] = report["effective_environment"]
    (output_dir / "config.json").write_text(
        json.dumps(serializable_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir



def train(model, data, sampler, ema):
    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    total_parameters = sum(
        np.prod(parameter.size()) for parameter in model.parameters()
    )
    print(f"Total number of parameters: {total_parameters}")

    num_pairs = data.adj_train.count_nonzero() // 2
    num_batches = int(num_pairs / args.batch_size) + 1
    print(f"selection num_batches: {num_batches}")

    fold_seed_offset = max(args.fold, 0) * 100
    validation_seed = args.seed + fold_seed_offset + 1
    test_seed = args.seed + fold_seed_offset + 2
    fixed_validation_manifest = None
    if int(args.use_fixed_validation_pairs):
        (
            validation_pairs,
            validation_labels,
            validation_negative_pairs,
            fixed_validation_manifest,
        ) = _load_fixed_validation_set(data)
    else:
        (
            validation_pairs, validation_labels, validation_negative_pairs,
        ) = _build_evaluation_set(data, data.val_dict, validation_seed)

    best_validation_aupr = -np.inf
    best_epoch = None
    best_state = None
    locked_threshold = None
    best_validation_scores = None
    best_validation_metrics = None

    try:
        for epoch in range(1, args.epochs + 1):
            print(f"\n========== Selection epoch {epoch} ==========")
            _train_one_epoch(
                model,
                data,
                sampler,
                ema,
                optimizer,
                epoch,
                num_batches,
                args.seed + fold_seed_offset * 10_000,
                "SelectionTrain",
            )

            if epoch % args.eval_freq == 0 or epoch == args.epochs:
                model.eval()
                raw_state = {
                    name: parameter.data.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                ema.copy_to(model)
                with torch.no_grad():
                    embeddings = model.encode4eval(data.adj_train_norm)
                    validation_scores = _score_pairs(
                        model, embeddings, validation_pairs
                    )
                    threshold, _, _, _ = _pick_best_threshold(
                        validation_labels, validation_scores
                    )
                    validation_metrics = _classification_metrics(
                        validation_labels, validation_scores, threshold
                    )
                log.write(
                    f"Validation:{epoch:3d}\t"
                    f"AUC={validation_metrics['auc']:.5f}\t"
                    f"AUPR={validation_metrics['aupr']:.5f}\t"
                    f"F1={validation_metrics['f1']:.5f}\t"
                    f"thr={threshold:.6f}\n"
                )
                if validation_metrics["aupr"] > best_validation_aupr:
                    best_validation_aupr = validation_metrics["aupr"]
                    best_epoch = epoch
                    locked_threshold = threshold
                    best_validation_scores = validation_scores.copy()
                    best_validation_metrics = dict(validation_metrics)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        parameter.data.copy_(raw_state[name])
                model.train()
    finally:
        sampler.close()

    if best_state is None or locked_threshold is None:
        raise RuntimeError("No validation checkpoint was selected")

    model.load_state_dict(best_state)
    model.eval()
    if int(args.selection_only):
        output_dir = _save_selection_only_artifacts(
            best_epoch,
            locked_threshold,
            (
                validation_pairs,
                validation_labels,
                best_validation_scores,
            ),
            best_validation_metrics,
            best_state,
            data,
            fixed_validation_manifest,
        )
        log.write(
            f"SelectionOnly:{best_epoch:3d}\t"
            f"AUC={best_validation_metrics['auc']:.5f}\t"
            f"AUPR={best_validation_metrics['aupr']:.5f}\t"
            f"F1={best_validation_metrics['f1']:.5f}\t"
            f"thr={locked_threshold:.6f}\n"
        )
        print(f"Selection-only artifacts saved to {output_dir}")
        return {"validation": best_validation_metrics, "test": None}

    final_data = data
    final_model = model
    final_state = best_state
    refit_seed = None
    if int(args.refit_after_selection):
        final_data, final_model, final_state, refit_seed = _run_refit(
            best_epoch
        )

    test_pairs, test_labels, test_negative_pairs = _build_evaluation_set(
        final_data,
        final_data.test_dict,
        test_seed,
        excluded_negative_pairs=validation_negative_pairs,
    )
    if validation_negative_pairs & test_negative_pairs:
        raise RuntimeError("Validation and test negative samples overlap")

    final_model.eval()
    with torch.no_grad():
        embeddings = final_model.encode4eval(final_data.adj_train_norm)
        test_scores = _score_pairs(final_model, embeddings, test_pairs)
    test_metrics = _classification_metrics(
        test_labels, test_scores, locked_threshold
    )
    output_dir = _save_evaluation_artifacts(
        best_epoch,
        locked_threshold,
        (validation_pairs, validation_labels, best_validation_scores),
        (test_pairs, test_labels, test_scores),
        best_validation_metrics,
        test_metrics,
        best_state,
        final_state,
        data,
        final_data,
        refit_seed,
    )
    phase = "RefitFinalTest" if int(args.refit_after_selection) else "FinalTest"
    log.write(
        f"{phase}:{best_epoch:3d}\tAUC={test_metrics['auc']:.5f}\t"
        f"AUPR={test_metrics['aupr']:.5f}\t"
        f"ACC={test_metrics['accuracy']:.5f}\t"
        f"Precision={test_metrics['precision']:.5f}\t"
        f"Recall={test_metrics['recall']:.5f}\t"
        f"F1={test_metrics['f1']:.5f}\t"
        f"MCC={test_metrics['mcc']:.5f}\t"
        f"locked_val_thr={locked_threshold:.6f}\n"
    )
    print(f"Final test artifacts saved to {output_dir}")
    return {"validation": best_validation_metrics, "test": test_metrics}


if __name__ == '__main__':
    args = parser.parse_args()

    # ---- 默认环境参数（可被外部同名环境变量覆盖）----
    default_env = {
        "PROJ_WEIGHT": "0.9",
        # 自适应 α 配置
        "PROJ_ALPHA_METHOD": "adaptive",
        "PROJ_ALPHA_Q90": "0.8",
        "PROJ_ALPHA_Q50": "0.5",
        "PROJ_ALPHA_Q10": "0.2",
        # 归一化 & 残差门控
        "PROJ_NORM_METHOD": "bidirectional",
        "PROJ_GATE_LAMBDA": "0.5",
        # K-support 策略
        "PROJ_SUP_METHOD": "k_support",
        "PROJ_MIN_SUP": "2",
        "PROJ_MAX_SUP": "40",
        # 一致性门
        "CONSIST_GATE": "1",
        "CONSIST_NEI": "10",
        "CONSIST_THR": "0.6",
        # LGCN 扩散参数
        "PROJ_LGCN_K": "3",
        "PROJ_LGCN_TOPK": "40",
        "PROJ_LGCN_BETA": "0.75",
        # 数据层相似度增强（默认开启）
        "SIM_AUG_ENABLE": "1",
        "SIM_AUG_TOPK": "40",
        "SIM_AUG_WEIGHT": "0.3",
        "SIM_AUG_THRESHOLD": "0.0",
        "SSL_USE_MSG": "1",
        "SSL_P_OBS": "0.6",
        "SSL_DEG_SAFE": "1",
        "SIM_TOPK": "40",
        "SIM_DROP_SELF": "1",
        "EMA_DECAY": "0.999",
    }
    for k, v in default_env.items():
        os.environ.setdefault(k, v)

    if args.fold is None or not 0 <= int(args.fold) <= 9:
        raise ValueError('This archived release requires --fold 0..9')
    args.fold = int(args.fold)
    if int(args.selection_only) and int(args.refit_after_selection):
        raise ValueError(
            "selection_only cannot be combined with refit_after_selection"
        )
    if int(args.use_fixed_validation_pairs) and args.fold < 0:
        raise ValueError(
            "use_fixed_validation_pairs requires --fold 0..9"
        )

    if args.log:
        now = datetime.now().strftime('%m-%d_%H-%M-%S')
        log = Logger(args.log, now)
        for arg in vars(args):
            log.write(arg + '=' + str(getattr(args, arg)) + '\n')
        for key in EFFECTIVE_ENV_KEYS:
            log.write(f"env.{key}={os.environ.get(key)}\n")
    else:
        print(args)

    set_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data = Data(
        args.dataset,
        args.norm_adj,
        args.seed,
        args.ssl_ratio,
        fold=args.fold,
        split_variant='selection',
    )
    total_edges = data.adj_train.count_nonzero()
    args.n_nodes = data.num_drugs + data.num_diseases
    args.feat_dim = args.embedding_dim

    # === 禁止集（D×Di）：观测 ∪ 投影TopK
    R_ban = data.adj_train_msg[:data.num_drugs, data.num_drugs:].astype(bool).tocsr()
    R_ban = (
        R_ban + data.val_csr.astype(bool) + data.test_csr.astype(bool)
    ).astype(bool).tocsr()

    # 注意：你的 WarpSampler 需要已经支持 ban_csr 参数；如果还没改 sampler.py，请先去掉 ban_csr 这一行参数
    sampler = WarpSampler(
        (data.num_drugs, data.num_diseases),
        data.adj_train,          # 正样本仍仅来自观测图
        args.batch_size,
        args.num_neg,
        n_workers=0,
        ban_csr=R_ban,
        seed=args.seed,
    )

    model = HyperCL(
        (data.num_drugs, data.num_diseases),
        args,
        drug_gip=data.drug_gip_sim,
        disease_gip=data.disease_gip_sim,
        drug_fp=data.drug_fp_sim,
        disease_ps=data.disease_ps_sim
    ).to(args.device)

    ema = EMA(model, decay=float(os.environ.get("EMA_DECAY", "0.999")))

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print('model is running on', next(model.parameters()).device)

    try:
        train(model, data, sampler, ema)
    except Exception:
        sampler.close()
        traceback.print_exc()
        raise
