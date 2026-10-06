import hashlib
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from .exp import scTRENDExperiment
from .utils import (
    beta_z_results,
    bulk_deconvolution_results,
    fit_hazard_joint_fixed_epochs,
    fit_hazard_two_stage_fixed_epochs,
    make_inputs,
    make_sample_one_hot_mat,
    optimize_deepcolor,
    optimize_deepcolor_onlyload,
    optimize_scTREND,
    optimize_scTREND_joint,
    optimize_vae,
    optimize_vae_onlyload,
    safe_toarray,
    spatial_results,
    vae_results,
)


# default 10x10 candidate grids
# default_l2_beta_candidates = (
#     0.0, 1e-4, 1e-3, 5e-3, 1e-2,
#     5e-2, 1e-1, 3e-1, 5e-1, 1.0
# )
# default_l2_gamma_candidates = (
#     0.0, 1e-5, 1e-4, 1e-3,
#     1e-2, 3e-2, 1e-1, 1.0, 3.0, 10.0
# )



def _resolve_l2_candidates(candidate_values=None, *, default_values, name: str):
    raw_values = tuple(default_values) if candidate_values is None else tuple(candidate_values)

    cleaned = []
    for v in raw_values:
        v = float(v)
        if v < 0:
            raise ValueError(f"All values in {name} must be >= 0. got {v}")
        cleaned.append(v)

    if len(cleaned) == 0:
        raise ValueError(f"{name} must not be empty.")

    has_zero = any(v == 0.0 for v in cleaned)
    positive = sorted(set(v for v in cleaned if v > 0.0))

    if has_zero:
        return tuple([0.0] + positive)
    return tuple(positive)



def _set_global_seeds(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def _atomic_write_json(path, payload):
    path = os.fspath(path)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _atomic_write_dataframe_csv(path, dataframe):
    path = os.fspath(path)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    dataframe.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)



def _build_snv_flag(driver_genes, driver_bulk_adata, n_obs):
    if not driver_genes or driver_bulk_adata is None:
        return None

    snv_mat = driver_bulk_adata.layers["SNV"]
    if hasattr(snv_mat, "toarray"):
        snv_mat = snv_mat.toarray()
    snv_mat = np.asarray(snv_mat)

    snv_flag = np.zeros(n_obs, dtype=int)
    for bit, gene in enumerate(driver_genes):
        try:
            col = np.where(driver_bulk_adata.var_names == gene)[0][0]
        except IndexError as exc:
            raise ValueError(f"{gene} not found in driver_bulk_adata.var_names") from exc
        mut_vec = snv_mat[:, col].astype(int).ravel()
        snv_flag += mut_vec * (1 << bit)
    return snv_flag

def _build_classical_stratified_folds(
    survival_time_np,
    censor_np,
    snv_flag,
    edges_with_inf,
    *,
    n_splits: int = 3,
    seed: int = 0,
):
    survival_time_np = np.asarray(survival_time_np, dtype=float)
    censor_np = np.asarray(censor_np, dtype=int)
    edges = np.asarray(edges_with_inf, dtype=float)

    bin_idx = np.digitize(survival_time_np, edges, right=True) - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)

    if snv_flag is None:
        strata = bin_idx * 2 + censor_np
        mut_mask = np.ones_like(strata, dtype=bool)
    else:
        snv_flag = np.asarray(snv_flag, dtype=int)
        max_flag = int(np.max(snv_flag)) if snv_flag.size > 0 else 0
        n_bits = max(1, int(np.ceil(np.log2(max_flag + 1)))) if max_flag > 0 else 1
        upper = bin_idx * 2 + censor_np
        strata = (upper.astype(int) << n_bits) + snv_flag.astype(int)
        mut_mask = snv_flag > 0

    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(int(n_splits))]

    for k in np.unique(strata):
        idx_k = np.where(strata == k)[0]
        idx_k = rng.permutation(idx_k)
        parts = np.array_split(idx_k, int(n_splits))
        for fold_id, part in enumerate(parts):
            folds[fold_id].extend(part.tolist())

    fold_tensors = []
    for fold_id, fold in enumerate(folds):
        fold = np.asarray(sorted(fold), dtype=int)
        if fold.size == 0:
            raise ValueError(f"Fold {fold_id} is empty. Reduce n_splits or check strata.")
        fold_tensors.append(torch.as_tensor(fold, dtype=torch.long))

    mut_counts = [int(mut_mask[f.cpu().numpy()].sum()) for f in fold_tensors]
    if np.min(mut_counts) == 0:
        raise ValueError(
            f"At least one validation fold has zero mutated samples: {mut_counts}. "
            "Reduce n_splits or fall back to repeated hold-out."
        )

    return fold_tensors, mut_counts




def _select_candidate_with_gamma_relative_threshold(
    records,
    *,
    gamma_min_rel_improvement: float = 0.01,
    eps: float = 1e-12,
):
    """
    Select beta+gamma only when it improves CV validation NLL over beta-only
    by at least gamma_min_rel_improvement.

    relative improvement =
        (beta_only_cv_val_nll - best_gamma_cv_val_nll)
        / (abs(beta_only_cv_val_nll) + eps)

    The function also annotates every record so that *_l2_search.csv can be
    re-used later to sweep arbitrary delta thresholds without retraining.
    """
    if records is None or len(records) == 0:
        raise ValueError("records is empty; cannot select L2/gamma candidate.")

    def _as_bool(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

    def _nll_key(rec):
        return (
            float(rec.get("cv_val_nll_mean", float("inf"))),
            -float(rec.get("cv_val_c_index_mean", float("-inf"))),
        )

    beta_records = [r for r in records if not _as_bool(r.get("use_gamma", False))]
    gamma_records = [r for r in records if _as_bool(r.get("use_gamma", False))]

    # If beta-only is missing, fall back to pure CV-NLL selection.
    if len(beta_records) == 0:
        selected = min(records, key=_nll_key)
        for r in records:
            r["beta_baseline_cv_val_nll_mean"] = float("nan")
            r["best_gamma_cv_val_nll_mean"] = float("nan")
            r["best_gamma_l2_gamma"] = float("nan")
            r["gamma_rel_improvement_vs_beta"] = float("nan")
            r["gamma_rel_improvement_percent_vs_beta"] = float("nan")
            r["gamma_min_rel_improvement"] = float(gamma_min_rel_improvement)
            r["gamma_min_rel_improvement_percent"] = float(100.0 * gamma_min_rel_improvement)
            r["gamma_passes_rel_improvement_threshold"] = bool(r is selected)
            r["selected_by_gamma_relative_threshold"] = bool(r is selected)
        return selected, records

    beta_best = min(beta_records, key=_nll_key)
    beta_nll = float(beta_best["cv_val_nll_mean"])
    beta_best_l2 = float(beta_best.get("l2_beta", 0.0))

    if len(gamma_records) == 0:
        selected = beta_best
        for r in records:
            r["beta_baseline_cv_val_nll_mean"] = beta_nll
            r["best_gamma_cv_val_nll_mean"] = float("nan")
            r["best_gamma_l2_gamma"] = float("nan")
            r["gamma_rel_improvement_vs_beta"] = float("nan")
            r["gamma_rel_improvement_percent_vs_beta"] = float("nan")
            r["gamma_min_rel_improvement"] = float(gamma_min_rel_improvement)
            r["gamma_min_rel_improvement_percent"] = float(100.0 * gamma_min_rel_improvement)
            r["gamma_passes_rel_improvement_threshold"] = False
            r["selected_by_gamma_relative_threshold"] = bool(r is selected)
        return selected, records

    gamma_best = min(gamma_records, key=_nll_key)
    gamma_nll = float(gamma_best["cv_val_nll_mean"])
    best_gamma_l2 = float(gamma_best.get("l2_gamma", 0.0))
    best_gamma_l2_beta = float(gamma_best.get("l2_beta", 0.0))
    rel_improve = (beta_nll - gamma_nll) / (abs(beta_nll) + eps)

    gamma_passes = bool((gamma_nll < beta_nll) and (rel_improve >= float(gamma_min_rel_improvement)))
    selected = gamma_best if gamma_passes else beta_best
    if gamma_nll >= beta_nll:
        selection_reason = "best_joint_model_did_not_improve_validation_nll"
    elif not gamma_passes:
        selection_reason = "best_joint_model_improved_but_below_relative_threshold"
    else:
        selection_reason = "best_joint_model_passed_relative_improvement_threshold"

    for r in records:
        r["beta_baseline_cv_val_nll_mean"] = beta_nll
        r["beta_baseline_best_l2_beta"] = beta_best_l2
        r["best_gamma_cv_val_nll_mean"] = gamma_nll
        r["best_gamma_l2_beta"] = best_gamma_l2_beta
        r["best_gamma_l2_gamma"] = best_gamma_l2
        r["gamma_rel_improvement_vs_beta"] = float(rel_improve)
        r["gamma_rel_improvement_percent_vs_beta"] = float(100.0 * rel_improve)
        r["gamma_min_rel_improvement"] = float(gamma_min_rel_improvement)
        r["gamma_min_rel_improvement_percent"] = float(100.0 * gamma_min_rel_improvement)
        r["gamma_passes_rel_improvement_threshold"] = gamma_passes
        r["gamma_selection_reason"] = selection_reason
        r["selected_by_gamma_relative_threshold"] = bool(r is selected)

    return selected, records


def tune_l2_beta_classical_3fold(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    x_batch_size_scTREND,
    epoch,
    patience,
    param_save_path,
    warm_path,
    l2_beta_candidates,
    l2_gamma_fixed: float = 0.0,
    tune_epoch: int = None,
    tune_patience: int = None,
    metric: str = "val_nll",
    seed: int = 0,
    n_splits: int = 3,
):
    if metric not in {"val_nll", "val_c_index"}:
        raise ValueError("metric must be either 'val_nll' or 'val_c_index'.")

    if tune_epoch is None:
        tune_epoch = int(epoch)
    if tune_patience is None:
        tune_patience = int(patience)

    beta_candidates = tuple(float(v) for v in l2_beta_candidates)
    if len(beta_candidates) == 0:
        raise ValueError("l2_beta_candidates must not be empty.")

    censor_np = scTREND_exp.cutting_off_0_1.detach().cpu().numpy()
    survival_time_np = scTREND_exp.survival_time.detach().cpu().numpy()
    snv_flag = _build_snv_flag(
        scTREND_exp.driver_genes,
        scTREND_exp.driver_bulk_adata,
        scTREND_exp.bulk_count.shape[0],
    )
    edges_with_inf = tuple(scTREND_exp.edges.detach().cpu().numpy().tolist()) + (np.inf,)

    folds, mut_val_counts = _build_classical_stratified_folds(
        survival_time_np,
        censor_np,
        snv_flag,
        edges_with_inf,
        n_splits=n_splits,
        seed=seed,
    )

    print("[L2 beta 3-fold] beta candidates: " + ", ".join(f"{v:.3e}" for v in beta_candidates))
    print(f"[L2 beta 3-fold] n_splits={n_splits}")
    print(f"[L2 beta 3-fold] mutated val counts per fold={mut_val_counts}")
    print("[L2 beta 3-fold] evaluating beta-only candidates")

    all_idx_np = np.arange(scTREND_exp.bulk_count.shape[0], dtype=int)
    empty_idx = torch.empty(0, dtype=torch.long)

    trial_records = []
    best_rec = None
    best_key = None

    def _score(rec):
        if metric == "val_nll":
            return (-rec["cv_val_nll_mean"], rec["cv_val_c_index_mean"])
        return (rec["cv_val_c_index_mean"], -rec["cv_val_nll_mean"])

    for i, l2b in enumerate(beta_candidates):
        fold_metrics = []
        for fold_id, val_idx in enumerate(folds):
            _set_global_seeds(seed)

            val_idx_np = val_idx.cpu().numpy()
            train_idx_np = np.setdiff1d(all_idx_np, val_idx_np, assume_unique=True)
            train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)

            trial_label = f"l2beta{i:02d}"
            trial_path = param_save_path.replace(".pt", f"_{trial_label}_fold{fold_id:02d}.pt")
            if os.path.exists(trial_path):
                os.remove(trial_path)

            scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
            scTREND_exp.clear_hazard_cache()

            scTREND_exp.bulk_data_manager.train_idx = train_idx.clone()
            scTREND_exp.bulk_data_manager.validation_idx = val_idx.clone()
            scTREND_exp.bulk_data_manager.test_idx = empty_idx.clone()
            scTREND_exp.set_active_bulk_indices(None, None, None)

            scTREND_exp.set_gamma_usage(False)
            scTREND_exp.set_l2_penalty(l2_beta=float(l2b), l2_gamma=float(l2_gamma_fixed))

            scTREND_exp_tmp, metrics = optimize_scTREND(
                scTREND_exp,
                third_lr=third_lr,
                x_batch_size=x_batch_size_scTREND,
                epoch=tune_epoch,
                patience=tune_patience,
                param_save_path=trial_path,
                warm_path=warm_path,
                return_metrics=True,
            )
            fold_metrics.append(metrics)

        rec = {
            "trial": len(trial_records),
            "trial_label": f"l2beta{i:02d}",
            "use_gamma": False,
            "l2_beta": float(l2b),
            "l2_gamma": float(l2_gamma_fixed),
            "cv_val_nll_mean": float(np.mean([m["val_nll"] for m in fold_metrics])),
            "cv_val_c_index_mean": float(np.mean([m["val_c_index"] for m in fold_metrics])),
            "cv_val_nll_scope": str(fold_metrics[0].get("model_selection_val_nll_scope", "unknown")),
            "mutated_val_counts": tuple(int(v) for v in mut_val_counts),
        }
        trial_records.append(rec)

        key = _score(rec)
        if (best_key is None) or (key > best_key):
            best_key = key
            best_rec = rec

    return best_rec, trial_records

def tune_l2_gamma_classical_3fold(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    x_batch_size_scTREND,
    epoch,
    patience,
    param_save_path,
    warm_path,
    l2_beta_fixed,
    l2_gamma_candidates,
    tune_epoch: int = None,
    tune_patience: int = None,
    metric: str = "val_nll",
    seed: int = 0,
    n_splits: int = 3,
):
    if metric not in {"val_nll", "val_c_index"}:
        raise ValueError("metric must be either 'val_nll' or 'val_c_index'.")

    if tune_epoch is None:
        tune_epoch = int(epoch)
    if tune_patience is None:
        tune_patience = int(patience)

    gamma_candidates = tuple(float(v) for v in l2_gamma_candidates)
    if len(gamma_candidates) == 0:
        raise ValueError("l2_gamma_candidates must not be empty.")

    censor_np = scTREND_exp.cutting_off_0_1.detach().cpu().numpy()
    survival_time_np = scTREND_exp.survival_time.detach().cpu().numpy()
    snv_flag = _build_snv_flag(
        scTREND_exp.driver_genes,
        scTREND_exp.driver_bulk_adata,
        scTREND_exp.bulk_count.shape[0],
    )
    edges_with_inf = tuple(scTREND_exp.edges.detach().cpu().numpy().tolist()) + (np.inf,)

    folds, mut_val_counts = _build_classical_stratified_folds(
        survival_time_np,
        censor_np,
        snv_flag,
        edges_with_inf,
        n_splits=n_splits,
        seed=seed,
    )

    print("[L2 gamma 3-fold] gamma candidates: " + ", ".join(f"{v:.3e}" for v in gamma_candidates))
    print(f"[L2 gamma 3-fold] n_splits={n_splits}")
    print(f"[L2 gamma 3-fold] mutated val counts per fold={mut_val_counts}")
    print("[L2 gamma 3-fold] evaluating beta-only baseline and beta+gamma candidates")

    all_idx_np = np.arange(scTREND_exp.bulk_count.shape[0], dtype=int)
    empty_idx = torch.empty(0, dtype=torch.long)

    trial_records = []
    best_rec = None
    best_key = None

    def _score(rec):
        if metric == "val_nll":
            return (-rec["cv_val_nll_mean"], rec["cv_val_c_index_mean"])
        return (rec["cv_val_c_index_mean"], -rec["cv_val_nll_mean"])

    def _run_candidate(*, trial_label: str, use_gamma: bool, l2g: float):
        fold_metrics = []

        for fold_id, val_idx in enumerate(folds):
            _set_global_seeds(seed)

            val_idx_np = val_idx.cpu().numpy()
            train_idx_np = np.setdiff1d(all_idx_np, val_idx_np, assume_unique=True)
            train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)

            trial_path = param_save_path.replace(".pt", f"_{trial_label}_fold{fold_id:02d}.pt")
            if os.path.exists(trial_path):
                os.remove(trial_path)

            scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
            scTREND_exp.clear_hazard_cache()

            scTREND_exp.bulk_data_manager.train_idx = train_idx.clone()
            scTREND_exp.bulk_data_manager.validation_idx = val_idx.clone()
            scTREND_exp.bulk_data_manager.test_idx = empty_idx.clone()
            scTREND_exp.set_active_bulk_indices(None, None, None)

            scTREND_exp.set_gamma_usage(use_gamma)
            scTREND_exp.set_l2_penalty(
                l2_beta=l2_beta_fixed,
                l2_gamma=(float(l2g) if use_gamma else 0.0),
            )


            scTREND_exp_tmp, metrics = optimize_scTREND(
                scTREND_exp,
                third_lr=third_lr,
                x_batch_size=x_batch_size_scTREND,
                epoch=tune_epoch,
                patience=tune_patience,
                param_save_path=trial_path,
                warm_path=warm_path,
                return_metrics=True,
            )
            fold_metrics.append(metrics)

        return {
            "trial_label": str(trial_label),
            "use_gamma": bool(use_gamma),
            "l2_beta": float(l2_beta_fixed),
            "l2_gamma": float(l2g if use_gamma else 0.0),
            "cv_val_nll_mean": float(np.mean([m["val_nll"] for m in fold_metrics])),
            "cv_val_c_index_mean": float(np.mean([m["val_c_index"] for m in fold_metrics])),
            "cv_val_nll_scope": str(fold_metrics[0].get("model_selection_val_nll_scope", "unknown")),
            "mutated_val_counts": tuple(int(v) for v in mut_val_counts),
        }
        
    beta_only_rec = _run_candidate(trial_label="betaonly", use_gamma=False, l2g=0.0)
    beta_only_rec["trial"] = len(trial_records)
    trial_records.append(beta_only_rec)
    best_rec = beta_only_rec
    best_key = _score(beta_only_rec)

    for i, l2g in enumerate(gamma_candidates):
        rec = _run_candidate(trial_label=f"l2gamma{i:02d}", use_gamma=True, l2g=l2g)
        rec["trial"] = len(trial_records)
        trial_records.append(rec)

        key = _score(rec)
        if key > best_key:
            best_key = key
            best_rec = rec

    return best_rec, trial_records


def tune_l2_beta_gamma_joint_classical_3fold(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    x_batch_size_scTREND,
    epoch,
    patience,
    param_save_path,
    warm_path,
    l2_beta_candidates,
    l2_gamma_candidates,
    tune_epoch=None,
    tune_patience=None,
    metric="val_nll",
    seed=0,
    n_splits=3,
    resume=True,
    keep_cv_checkpoints=False,
    eligible_indices=None,
):
    """Compare beta-only and joint beta+gamma grids with fold-level resume markers.

    A fold is considered complete only after its metrics JSON has been written
    atomically. On restart, matching completion records are loaded without
    refitting. A fold interrupted before that marker is the only unit rerun.
    """
    if metric not in {"val_nll", "val_c_index"}:
        raise ValueError("metric must be either 'val_nll' or 'val_c_index'.")
    tune_epoch = int(epoch if tune_epoch is None else tune_epoch)
    tune_patience = int(patience if tune_patience is None else tune_patience)
    beta_candidates = tuple(float(v) for v in l2_beta_candidates)
    gamma_candidates = tuple(float(v) for v in l2_gamma_candidates)
    if not beta_candidates or not gamma_candidates:
        raise ValueError("Both beta and gamma candidate grids must be non-empty.")

    censor_np = scTREND_exp.cutting_off_0_1.detach().cpu().numpy()
    survival_time_np = scTREND_exp.survival_time.detach().cpu().numpy()
    snv_flag = _build_snv_flag(
        scTREND_exp.driver_genes,
        scTREND_exp.driver_bulk_adata,
        scTREND_exp.bulk_count.shape[0],
    )
    edges_np = scTREND_exp.edges.detach().cpu().numpy()
    edges_with_inf = tuple(edges_np.tolist()) + (np.inf,)

    n_bulk = int(scTREND_exp.bulk_count.shape[0])
    if eligible_indices is None:
        eligible_idx_np = np.arange(n_bulk, dtype=int)
        folds, mut_val_counts = _build_classical_stratified_folds(
            survival_time_np,
            censor_np,
            snv_flag,
            edges_with_inf,
            n_splits=n_splits,
            seed=seed,
        )
        cv_scope = "all_data_classical_3fold"
    else:
        if isinstance(eligible_indices, torch.Tensor):
            eligible_idx_np = eligible_indices.detach().cpu().numpy().astype(int, copy=False)
        else:
            eligible_idx_np = np.asarray(eligible_indices, dtype=int)
        eligible_idx_np = np.unique(eligible_idx_np.ravel())
        if eligible_idx_np.size < int(n_splits):
            raise ValueError(
                f"eligible_indices has {eligible_idx_np.size} samples, fewer than n_splits={n_splits}."
            )
        if eligible_idx_np.min() < 0 or eligible_idx_np.max() >= n_bulk:
            raise IndexError("eligible_indices contains an out-of-range bulk index.")
        snv_subset = None if snv_flag is None else snv_flag[eligible_idx_np]
        local_folds, mut_val_counts = _build_classical_stratified_folds(
            survival_time_np[eligible_idx_np],
            censor_np[eligible_idx_np],
            snv_subset,
            edges_with_inf,
            n_splits=n_splits,
            seed=seed,
        )
        folds = [
            torch.as_tensor(eligible_idx_np[local_fold.cpu().numpy()], dtype=torch.long)
            for local_fold in local_folds
        ]
        cv_scope = "eligible_subset_inner_3fold"

    candidates = [("beta_only", l2b, 0.0, False) for l2b in beta_candidates]
    candidates.extend(
        ("joint_beta_gamma", l2b, l2g, True)
        for l2b in beta_candidates
        for l2g in gamma_candidates
    )
    print("[Joint model CV] beta-only candidates:", len(beta_candidates))
    print("[Joint model CV] beta+gamma candidates:", len(beta_candidates) * len(gamma_candidates))
    print("[Joint model CV] mutated validation counts:", mut_val_counts)
    print(
        f"[Joint model CV] scope={cv_scope} | eligible_n={eligible_idx_np.size} "
        f"| resume={bool(resume)} | keep_cv_checkpoints={bool(keep_cv_checkpoints)}"
    )

    dataset_hasher = hashlib.sha256()
    for array in (survival_time_np, censor_np, edges_np):
        dataset_hasher.update(np.ascontiguousarray(array).tobytes())
    if snv_flag is not None:
        dataset_hasher.update(np.ascontiguousarray(snv_flag).tobytes())
    if eligible_indices is not None:
        dataset_hasher.update(np.ascontiguousarray(eligible_idx_np).tobytes())
    dataset_signature = dataset_hasher.hexdigest()
    if warm_path is not None and os.path.exists(warm_path):
        warm_stat = os.stat(warm_path)
        warm_signature = {
            "path": os.path.abspath(warm_path),
            "size": int(warm_stat.st_size),
            "mtime_ns": int(warm_stat.st_mtime_ns),
        }
    else:
        warm_signature = {"path": None if warm_path is None else os.path.abspath(warm_path)}

    all_idx_np = eligible_idx_np
    empty_idx = torch.empty(0, dtype=torch.long)
    trial_records = []
    fold_records = []
    resumed_fold_count = 0
    folds_progress_path = param_save_path.replace(".pt", "_l2_search_folds_progress.csv")
    trials_progress_path = param_save_path.replace(".pt", "_l2_search_candidates_progress.csv")

    def _cleanup_trial_artifacts(trial_path, use_gamma):
        if keep_cv_checkpoints:
            return
        suffix = "_3rd_joint_end.pt" if use_gamma else "_3rd_betaonly_end.pt"
        tag = "" if warm_path is None else "_" + os.path.splitext(os.path.basename(warm_path))[0]
        artifact_paths = [
            trial_path,
            trial_path.replace(".pt", suffix),
            trial_path.replace(".pt", "") + tag + "_metrics.pt",
        ]
        for artifact_path in artifact_paths:
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)

    for trial_id, (family, l2b, l2g, use_gamma) in enumerate(candidates):
        candidate_fold_metrics = []
        candidate_resumed = 0
        for fold_id, val_idx in enumerate(folds):
            _set_global_seeds(seed)
            val_idx_np = val_idx.cpu().numpy()
            train_idx_np = np.setdiff1d(all_idx_np, val_idx_np, assume_unique=True)
            train_idx = torch.as_tensor(train_idx_np, dtype=torch.long)
            trial_path = param_save_path.replace(
                ".pt", f"_jointcv_trial{trial_id:03d}_fold{fold_id:02d}.pt"
            )
            resume_path = trial_path.replace(".pt", "_complete.json")
            signature = {
                "resume_schema": "joint_cv_fold_v1",
                "trial": int(trial_id),
                "model_family": family,
                "use_gamma": bool(use_gamma),
                "l2_beta": float(l2b),
                "l2_gamma": float(l2g if use_gamma else 0.0),
                "fold": int(fold_id),
                "cv_seed": int(seed),
                "n_splits": int(n_splits),
                "validation_indices": [int(v) for v in val_idx_np.tolist()],
                "dataset_signature": dataset_signature,
                "warm_checkpoint": warm_signature,
                "tune_epoch": int(tune_epoch),
                "tune_patience": int(tune_patience),
                "third_lr": float(third_lr),
                "x_batch_size_scTREND": int(x_batch_size_scTREND),
                "hazard_penalty": scTREND_exp.hazard_penalty,
                "posterior_is_epsilon": scTREND_exp.posterior_is_epsilon,
                "posterior_is_samples_per_cell": 1,
                "hazard_full_batch": bool(getattr(scTREND_exp, "hazard_full_batch", False)),
            }
            if eligible_indices is not None:
                signature.update({
                    "resume_schema": "joint_cv_fold_v2_eligible_subset",
                    "cv_scope": cv_scope,
                    "eligible_n": int(eligible_idx_np.size),
                })
            cv_early_stopping_metric = str(
                getattr(scTREND_exp, "hazard_early_stopping_metric", "val_nll")
            )
            cv_early_stopping_min_delta = float(
                getattr(scTREND_exp, "hazard_early_stopping_min_delta", 0.0)
            )
            # Preserve byte-for-byte-compatible resume signatures for existing
            # canonical NLL CV records. Only non-default stopping policies need
            # additional identity fields.
            if cv_early_stopping_metric != "val_nll" or cv_early_stopping_min_delta != 0.0:
                signature.update({
                    "hazard_early_stopping_metric": cv_early_stopping_metric,
                    "hazard_early_stopping_min_delta": cv_early_stopping_min_delta,
                })

            metrics = None
            resumed_from_cache = False
            if bool(resume) and os.path.isfile(resume_path):
                try:
                    with open(resume_path, "r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    cached_metrics = payload.get("metrics")
                    required_metrics = {
                        "train_nll", "val_nll", "train_c_index", "val_c_index", "joint_best_epoch"
                    }
                    if payload.get("signature") == signature and required_metrics.issubset(cached_metrics or {}):
                        metrics = cached_metrics
                        resumed_from_cache = True
                        candidate_resumed += 1
                        resumed_fold_count += 1
                        print(
                            f"[Joint model CV] reuse trial={trial_id:03d} fold={fold_id:02d} "
                            f"family={family} l2_beta={l2b:.3e} l2_gamma={l2g:.3e}"
                        )
                except Exception as exc:
                    print(f"[Joint model CV] invalid resume record; rerun {resume_path}: {exc}")

            if not resumed_from_cache:
                scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
                scTREND_exp.clear_hazard_cache()
                scTREND_exp.bulk_data_manager.train_idx = train_idx.clone()
                scTREND_exp.bulk_data_manager.validation_idx = val_idx.clone()
                scTREND_exp.bulk_data_manager.test_idx = empty_idx.clone()
                scTREND_exp.set_active_bulk_indices(None, None, None)
                scTREND_exp.set_gamma_usage(use_gamma)
                scTREND_exp.set_l2_penalty(
                    l2_beta=l2b,
                    l2_gamma=(l2g if use_gamma else 0.0),
                )
                _, metrics = optimize_scTREND_joint(
                    scTREND_exp,
                    third_lr=third_lr,
                    x_batch_size=x_batch_size_scTREND,
                    epoch=tune_epoch,
                    patience=tune_patience,
                    param_save_path=trial_path,
                    warm_path=warm_path,
                    return_metrics=True,
                )
                _atomic_write_json(
                    resume_path,
                    {"signature": signature, "metrics": metrics},
                )
                _cleanup_trial_artifacts(trial_path, use_gamma)

            # Also cleans stale artifacts when a previous process wrote the
            # completion record and then stopped during cleanup.
            _cleanup_trial_artifacts(trial_path, use_gamma)
            candidate_fold_metrics.append(metrics)
            fold_records.append({
                "trial": int(trial_id),
                "model_family": family,
                "use_gamma": bool(use_gamma),
                "l2_beta": float(l2b),
                "l2_gamma": float(l2g if use_gamma else 0.0),
                "fold": int(fold_id),
                "n_train": int(train_idx.numel()),
                "n_validation": int(val_idx.numel()),
                "n_mutated_validation": int(mut_val_counts[fold_id]),
                "train_nll": float(metrics["train_nll"]),
                "val_nll": float(metrics["val_nll"]),
                "train_c_index": float(metrics["train_c_index"]),
                "val_c_index": float(metrics["val_c_index"]),
                "best_epoch": int(metrics["joint_best_epoch"]),
                "resumed_from_complete_record": bool(resumed_from_cache),
                "completion_record": str(resume_path),
            })
            _atomic_write_dataframe_csv(folds_progress_path, pd.DataFrame(fold_records))

        val_nlls = np.asarray([m["val_nll"] for m in candidate_fold_metrics], dtype=float)
        val_cs = np.asarray([m["val_c_index"] for m in candidate_fold_metrics], dtype=float)
        rec = {
            "trial": int(trial_id),
            "model_family": family,
            "use_gamma": bool(use_gamma),
            "l2_beta": float(l2b),
            "l2_gamma": float(l2g if use_gamma else 0.0),
            "cv_val_nll_mean": float(np.nanmean(val_nlls)),
            "cv_val_nll_std": float(np.nanstd(val_nlls, ddof=1)),
            "cv_val_c_index_mean": float(np.nanmean(val_cs)),
            "cv_val_c_index_std": float(np.nanstd(val_cs, ddof=1)),
            "cv_val_nll_scope": (
                "full_validation_WT_plus_Mut"
                if eligible_indices is None
                else "outer_train_inner_validation_WT_plus_Mut"
            ),
            "cv_scope": cv_scope,
            "eligible_n": int(eligible_idx_np.size),
            "n_splits": int(n_splits),
            "n_resumed_folds": int(candidate_resumed),
            "mutated_val_counts": ";".join(str(int(v)) for v in mut_val_counts),
            "fold_val_nll_values": ";".join(f"{v:.12g}" for v in val_nlls),
            "fold_val_c_index_values": ";".join(f"{v:.12g}" for v in val_cs),
        }
        trial_records.append(rec)
        _atomic_write_dataframe_csv(trials_progress_path, pd.DataFrame(trial_records))

    beta_best = min(
        (r for r in trial_records if not r["use_gamma"]),
        key=lambda r: (r["cv_val_nll_mean"], -r["cv_val_c_index_mean"]),
    )
    beta_nll = float(beta_best["cv_val_nll_mean"])
    for rec in trial_records:
        candidate_improvement = (
            beta_nll - float(rec["cv_val_nll_mean"])
        ) / (abs(beta_nll) + 1e-12)
        rec["candidate_rel_improvement_vs_best_beta"] = float(candidate_improvement)
        rec["candidate_rel_improvement_percent_vs_best_beta"] = float(100.0 * candidate_improvement)
        rec["candidate_is_worse_than_best_beta"] = bool(candidate_improvement < 0.0)

    scTREND_exp.joint_cv_fold_records = fold_records
    scTREND_exp.joint_cv_resumed_fold_count = int(resumed_fold_count)
    scTREND_exp.joint_cv_total_fold_count = int(len(fold_records))
    scTREND_exp.joint_cv_scope = cv_scope
    scTREND_exp.joint_cv_eligible_n = int(eligible_idx_np.size)
    if metric == "val_nll":
        best_raw = min(
            trial_records,
            key=lambda r: (r["cv_val_nll_mean"], -r["cv_val_c_index_mean"]),
        )
    else:
        best_raw = max(
            trial_records,
            key=lambda r: (r["cv_val_c_index_mean"], -r["cv_val_nll_mean"]),
        )
    return best_raw, trial_records

def tune_l2_beta_gamma_grid(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    x_batch_size_scTREND,
    epoch,
    patience,
    param_save_path,
    warm_path,
    l2_beta_candidates,
    l2_gamma_candidates,
    tune_epoch: int = None,
    tune_patience: int = None,
    metric: str = "val_nll",
    seed: int = 0,
    cv_bulk_seeds = (0,1,2,3,4),
):
    if metric not in {"val_c_index", "val_nll"}:
        raise ValueError("metric must be either 'val_c_index' or 'val_nll'.")

    if tune_epoch is None:
        tune_epoch = int(epoch)
    if tune_patience is None:
        tune_patience = int(patience)

    beta_candidates = tuple(float(v) for v in l2_beta_candidates)
    gamma_candidates = tuple(float(v) for v in l2_gamma_candidates)

    if len(beta_candidates) == 0:
        raise ValueError("l2_beta_candidates must not be empty.")
    if len(gamma_candidates) == 0:
        raise ValueError("l2_gamma_candidates must not be empty.")

    candidates = []
    for l2b in beta_candidates:
        candidates.append((l2b, 0.0, False))
        candidates.extend((l2b, l2g, True) for l2g in gamma_candidates)


    print("[L2 grid] beta candidates : " + ", ".join(f"{v:.3e}" for v in beta_candidates))
    print("[L2 grid] gamma candidates: " + ", ".join(f"{v:.3e}" for v in gamma_candidates))
    print(f"[L2 grid] total trials (including beta-only baselines): {len(candidates)}")

    trial_records = []
    best_rec = None
    best_key = None
    
    censor_np = scTREND_exp.cutting_off_0_1.detach().cpu().numpy()
    snv_flag = _build_snv_flag(
        scTREND_exp.driver_genes,
        scTREND_exp.driver_bulk_adata,
        scTREND_exp.bulk_count.shape[0],
    )
    edges_with_inf = tuple(scTREND_exp.edges.detach().cpu().numpy().tolist()) + (np.inf,)


    for i, (l2b, l2g, use_gamma) in enumerate(candidates):
        fold_metrics = []
        
        for fold_id, bulk_seed in enumerate(cv_bulk_seeds):
            _set_global_seeds(seed)

            trial_path = param_save_path.replace(".pt", f"_l2trial{i:02d}_fold{fold_id:02d}.pt")
            if os.path.exists(trial_path):
                os.remove(trial_path)

            scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
            scTREND_exp.clear_hazard_cache()
            scTREND_exp.bulk_data_split(
                bulk_seed,
                scTREND_exp.bulk_validation_num_or_ratio,
                scTREND_exp.bulk_test_num_or_ratio,
                censor_np,
                snv_flag,
                edges=edges_with_inf,
            )
            scTREND_exp.set_gamma_usage(use_gamma)
            scTREND_exp.set_l2_penalty(
                l2_beta=l2b,
                l2_gamma=(float(l2g) if use_gamma else 0.0),
            )

            scTREND_exp_tmp, metrics = optimize_scTREND(
                scTREND_exp,
                third_lr=third_lr,
                x_batch_size=x_batch_size_scTREND,
                epoch=tune_epoch,
                patience=tune_patience,
                param_save_path=trial_path,
                warm_path=warm_path,
                return_metrics=True,
            )
            fold_metrics.append(metrics)

        rec = {
            "trial": i,
            "ckpt": trial_path,
            "use_gamma": bool(use_gamma),
            "l2_beta": float(l2b),
            "l2_gamma": float(l2g if use_gamma else 0.0),
            "cv_val_nll_mean": float(np.mean([m["val_nll"] for m in fold_metrics])),
            "cv_val_c_index_mean": float(np.mean([m["val_c_index"] for m in fold_metrics])),
            "cv_val_nll_scope": str(fold_metrics[0].get("model_selection_val_nll_scope", "unknown")),
        }
        trial_records.append(rec)

        if metric == "val_nll":
            key = (-rec["cv_val_nll_mean"], rec["cv_val_c_index_mean"])
        else:
            key = (rec["cv_val_c_index_mean"], -rec["cv_val_nll_mean"])

        if (best_key is None) or (key > best_key):
            best_key = key
            best_rec = rec

    return best_rec, trial_records



def run_scTREND(
    sc_adata,
    bulk_adata,
    param_save_path,
    warm_path,
    epoch,
    batch_key,
    driver_genes,
    driver_bulk_adata,
    bulk_seed=0,
    spatial_adata=None,
    survival_time_label="survival_times",
    survival_time_censor="vital_status",
    edges=None,
    l2_beta=None,
    l2_gamma=None,
    l2_search: bool = False,
    l2_search_seed: int = 0,
    l2_beta_candidates=None,
    l2_gamma_candidates=None,
    l2_tune_epoch=None,
    l2_tune_patience=None,
    l2_search_metric: str = "val_nll",
    gamma_min_rel_improvement: float = 0.01,
    force_use_gamma=None,
    hazard_fit_mode: str = "joint",
    resume_l2_search: bool = True,
    keep_cv_checkpoints: bool = False,
    beta_fit_group: str = "all",
    gamma_fit_group: str = "mutated_only",
    use_deterministic_z_in_hazard: bool = True,
    refit_hazard_on_full: bool = False,
    hazard_penalty: str = "posterior_is_clipped",
    posterior_is_epsilon: float = 0.01,
    posterior_is_samples_per_cell: int = 1,
    optimizer_weight_decay: float = 0.0,
    hazard_full_batch: bool = False,
    postprocess_minimal: bool = False,
    patience: int = 30,
    x_batch_size_VAE: int = 1000,
    x_batch_size_DeepCOLOR: int = 1000,
    x_batch_size_scTREND: int = 1000,
    sc_num_workers: int = 0,
    sc_pin_memory: bool = True,
    sc_persistent_workers: bool = False,
    sc_prefetch_factor: int = 2,
    enable_tf32: bool = True,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
    non_blocking_transfer: bool = True,
    hazard_early_stopping_metric: str = "val_nll",
    hazard_early_stopping_min_delta: float = 0.0,
    l2_search_scope: str = "all_data",
):
    survival_time_np = np.asarray(bulk_adata.obs[survival_time_label].values, dtype=float)
    survival_time = torch.tensor(survival_time_np)
    hazard_fit_mode = str(hazard_fit_mode).strip().lower()
    if hazard_fit_mode not in {"sequential", "joint"}:
        raise ValueError("hazard_fit_mode must be either 'sequential' or 'joint'.")
    hazard_early_stopping_metric = str(hazard_early_stopping_metric).strip().lower()
    if hazard_early_stopping_metric not in {"val_nll", "val_c_index"}:
        raise ValueError(
            "hazard_early_stopping_metric must be either 'val_nll' or 'val_c_index'."
        )
    hazard_early_stopping_min_delta = float(hazard_early_stopping_min_delta)
    if hazard_early_stopping_min_delta < 0.0:
        raise ValueError("hazard_early_stopping_min_delta must be >= 0.")
    optimizer_weight_decay = float(optimizer_weight_decay)
    if optimizer_weight_decay < 0.0:
        raise ValueError("optimizer_weight_decay must be >= 0.")
    l2_search_scope = str(l2_search_scope).strip().lower()
    if l2_search_scope not in {"all_data", "outer_train"}:
        raise ValueError("l2_search_scope must be either 'all_data' or 'outer_train'.")

    if edges is None:
        t0 = 0.0
        t1 = float(np.nanmax(survival_time_np))
        if (not np.isfinite(t1)) or (t1 <= t0):
            t1 = t0 + 1.0
        edges = np.array([t0, t1], dtype=float)
    else:
        edges = np.asarray(edges, dtype=float).ravel()
        if edges.size == 1:
            t0 = float(edges[0])
            t1 = float(np.nanmax(survival_time_np))
            if (not np.isfinite(t1)) or (t1 <= t0):
                t1 = t0 + 1.0
            edges = np.array([t0, t1], dtype=float)
        if edges.size < 2:
            raise ValueError("edges must have length >= 2, or None for single-bin mode.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError(f"edges must be strictly increasing. got {edges}")

    EDGES = tuple(edges.tolist())
    EDGES_WITH_INF = EDGES + (np.inf,)
    K = len(EDGES) - 1

    x_batch_size_VAE = int(x_batch_size_VAE)
    x_batch_size_DeepCOLOR = int(x_batch_size_DeepCOLOR)
    x_batch_size_scTREND = int(x_batch_size_scTREND)

    first_lr = 0.01
    second_lr = 0.01
    third_lr = 0.0001
    bulk_validation_num_or_ratio = 0.2
    bulk_test_num_or_ratio = 0.2

    use_val_loss_mean = True
    model_params = {
        "z_dim": 20,
        "h_dim": 100,
        "num_enc_z_layers": 1,
        "num_dec_z_layers": 1,
        "num_dec_p_layers": 1,
        "num_dec_b_layers": 1,
    }
    model_params["l2_beta"] = 0.0
    model_params["l2_gamma"] = 0.0
    model_params["gamma_min_rel_improvement"] = float(gamma_min_rel_improvement)
    model_params["num_time_bins"] = K
    model_params["edges"] = EDGES
    model_params["beta_fit_group"] = beta_fit_group
    model_params["gamma_fit_group"] = gamma_fit_group
    initial_use_gamma = bool(len(driver_genes) > 0) if force_use_gamma is None else bool(force_use_gamma)
    model_params["use_gamma"] = initial_use_gamma
    model_params["hazard_fit_mode"] = hazard_fit_mode
    model_params["resume_l2_search"] = bool(resume_l2_search)
    model_params["keep_cv_checkpoints"] = bool(keep_cv_checkpoints)
    model_params["use_deterministic_z_in_hazard"] = bool(use_deterministic_z_in_hazard)
    model_params["hazard_penalty"] = str(hazard_penalty)
    model_params["posterior_is_epsilon"] = float(posterior_is_epsilon)
    model_params["posterior_is_samples_per_cell"] = posterior_is_samples_per_cell
    model_params["optimizer_weight_decay"] = optimizer_weight_decay
    model_params["hazard_full_batch"] = bool(hazard_full_batch)
    model_params["sc_num_workers"] = int(sc_num_workers)
    model_params["sc_pin_memory"] = bool(sc_pin_memory)
    model_params["sc_persistent_workers"] = bool(sc_persistent_workers)
    model_params["sc_prefetch_factor"] = int(sc_prefetch_factor)
    model_params["enable_tf32"] = bool(enable_tf32)
    model_params["matmul_precision"] = str(matmul_precision)
    model_params["cudnn_benchmark"] = bool(cudnn_benchmark)
    model_params["non_blocking_transfer"] = bool(non_blocking_transfer)
    model_params["hazard_early_stopping_metric"] = hazard_early_stopping_metric
    model_params["hazard_early_stopping_min_delta"] = hazard_early_stopping_min_delta

    patience = int(patience)
    usePoisson_sc = True

    if l2_tune_epoch is None:
        l2_tune_epoch = int(epoch)
    if l2_tune_patience is None:
        l2_tune_patience = int(patience)

    default_l2_beta_candidates = (0.0, 1e0, 1e1, 1e2, 1e3)
    default_l2_gamma_candidates = (0.0, 1e0, 1e1, 1e2, 1e3)

    resolved_l2_beta_candidates = _resolve_l2_candidates(
        l2_beta_candidates,
        default_values=default_l2_beta_candidates,
        name="l2_beta_candidates",
    )
    resolved_l2_gamma_candidates = _resolve_l2_candidates(
        l2_gamma_candidates,
        default_values=default_l2_gamma_candidates,
        name="l2_gamma_candidates",
    )

    x_count, bulk_count = make_inputs(sc_adata, bulk_adata)
    batch_onehot = make_sample_one_hot_mat(sc_adata, batch_key)

    if survival_time_censor == "vital_status":
        valid_values = ["Dead", "Alive"]
        if not all(value in valid_values for value in bulk_adata.obs[survival_time_censor]):
            raise ValueError(
                "Invalid values found in bulk_adata.obs[survival_time_censor]. Only 'Dead' and 'Alive' are allowed."
            )
        vital_status_values = np.where(bulk_adata.obs[survival_time_censor] == "Dead", 1, 0)
        cutting_off_0_1 = torch.tensor(vital_status_values)
    else:
        cutting_off_0_1 = torch.tensor(bulk_adata.obs[survival_time_censor].values)

    model_params["x_dim"] = x_count.size()[1]
    model_params_dict = {
        "1st_lr": first_lr,
        "2nd_lr": second_lr,
        "3rd_lr": third_lr,
        "patience": patience,
        "bulk_seed": bulk_seed,
        "x_batch_size_VAE": x_batch_size_VAE,
        "x_batch_size_DeepCOLOR": x_batch_size_DeepCOLOR,
        "x_batch_size_scTREND": x_batch_size_scTREND,
        "n_var": sc_adata.n_vars,
        "usePoisson_sc": usePoisson_sc,
        "batch_key": batch_key,
        "n_obs_sc": sc_adata.n_obs,
        "n_obs_bulk": bulk_adata.n_obs,
        "use_val_loss_mean": use_val_loss_mean,
        "bulk_validation_num_or_ratio": bulk_validation_num_or_ratio,
        "bulk_test_num_or_ratio": bulk_test_num_or_ratio,
        "l2_search_mode": "beta_gamma_grid_10x10",
        "l2_beta_candidates": tuple(float(v) for v in resolved_l2_beta_candidates),
        "l2_gamma_candidates": tuple(float(v) for v in resolved_l2_gamma_candidates),
        "l2_search_total_trials": len(resolved_l2_beta_candidates) * (1 + len(resolved_l2_gamma_candidates)),
        "l2_search_seed": l2_search_seed,
        "l2_tune_epoch": l2_tune_epoch,
        "l2_tune_patience": l2_tune_patience,
        "l2_search_metric": l2_search_metric,
        "l2_search_scope": l2_search_scope,
        "gamma_min_rel_improvement": float(gamma_min_rel_improvement),
        "gamma_min_rel_improvement_percent": float(100.0 * gamma_min_rel_improvement),
        "force_use_gamma": (None if force_use_gamma is None else bool(force_use_gamma)),
        "hazard_fit_mode": hazard_fit_mode,
        "resume_l2_search": bool(resume_l2_search),
        "keep_cv_checkpoints": bool(keep_cv_checkpoints),
        "beta_fit_group": beta_fit_group,
        "gamma_fit_group": gamma_fit_group,
        "use_gamma": initial_use_gamma,
        "use_deterministic_z_in_hazard": bool(use_deterministic_z_in_hazard),
        "refit_hazard_on_full": bool(refit_hazard_on_full),
        "hazard_penalty": str(hazard_penalty),
        "posterior_is_epsilon": float(posterior_is_epsilon),
        "posterior_is_samples_per_cell": 1,
        "hazard_full_batch": bool(hazard_full_batch),
        "postprocess_minimal": bool(postprocess_minimal),
        "sc_num_workers": int(sc_num_workers),
        "sc_pin_memory": bool(sc_pin_memory),
        "sc_persistent_workers": bool(sc_persistent_workers),
        "sc_prefetch_factor": int(sc_prefetch_factor),
        "enable_tf32": bool(enable_tf32),
        "matmul_precision": str(matmul_precision),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "non_blocking_transfer": bool(non_blocking_transfer),
        "hazard_early_stopping_metric": hazard_early_stopping_metric,
        "hazard_early_stopping_min_delta": hazard_early_stopping_min_delta,
    }

    if spatial_adata is not None:
        spatial_count = torch.tensor(safe_toarray(spatial_adata.X))
        model_params_dict["n_obs_spatial"] = spatial_adata.n_obs
    else:
        spatial_count = None

    model_params_dict.update(model_params)
    print(model_params_dict)

    scTREND_exp = scTRENDExperiment(
        model_params=model_params,
        x_count=x_count,
        bulk_count=bulk_count,
        survival_time=survival_time,
        cutting_off_0_1=cutting_off_0_1,
        x_batch_size=x_batch_size_VAE,
        checkpoint=param_save_path,
        usePoisson_sc=usePoisson_sc,
        batch_onehot=batch_onehot,
        spatial_count=spatial_count,
        use_val_loss_mean=use_val_loss_mean,
        driver_genes=driver_genes,
        driver_bulk_adata=driver_bulk_adata,
    )
    scTREND_exp.third_lr = float(third_lr)
    sc_adata.uns['third_lr'] = float(third_lr)

    if warm_path is not None:
        scTREND_exp.scTREND.load_state_dict(torch.load(warm_path), strict=False)
        scTREND_exp = optimize_vae_onlyload(
            scTREND_exp=scTREND_exp,
            first_lr=first_lr,
            x_batch_size=x_batch_size_VAE,
            epoch=epoch,
            patience=patience,
            param_save_path=warm_path,
        )
        scTREND_exp = optimize_deepcolor_onlyload(
            scTREND_exp=scTREND_exp,
            second_lr=second_lr,
            x_batch_size=x_batch_size_DeepCOLOR,
            epoch=epoch,
            patience=patience,
            param_save_path=warm_path,
        )
    else:
        scTREND_exp = optimize_vae(
            scTREND_exp=scTREND_exp,
            first_lr=first_lr,
            x_batch_size=x_batch_size_VAE,
            epoch=epoch,
            patience=patience,
            param_save_path=param_save_path,
        )
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path.replace(".pt", "") + "_1st_end.pt")
        scTREND_exp = optimize_deepcolor(
            scTREND_exp=scTREND_exp,
            second_lr=second_lr,
            x_batch_size=x_batch_size_DeepCOLOR,
            epoch=epoch,
            patience=patience,
            param_save_path=param_save_path,
            spatial_adata=spatial_adata,
        )
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path.replace(".pt", "") + "_2nd_end.pt")

    snv_flag = _build_snv_flag(driver_genes, driver_bulk_adata, bulk_adata.n_obs)

    scTREND_exp.bulk_data_split(
        bulk_seed,
        bulk_validation_num_or_ratio,
        bulk_test_num_or_ratio,
        cutting_off_0_1,
        snv_flag,
        edges=EDGES_WITH_INF,
    )

    base_state_dict = {k: v.detach().cpu().clone() for k, v in scTREND_exp.scTREND.state_dict().items()}
    scTREND_exp.pre_hazard_state_dict = {k: v.detach().cpu().clone() for k, v in base_state_dict.items()}

    if l2_search and (l2_beta is None or l2_gamma is None):
        beta_candidates = (float(l2_beta),) if l2_beta is not None else resolved_l2_beta_candidates
        gamma_candidates = (float(l2_gamma),) if l2_gamma is not None else resolved_l2_gamma_candidates

        if l2_search_scope == "outer_train" and hazard_fit_mode != "joint":
            raise ValueError(
                "l2_search_scope='outer_train' currently requires hazard_fit_mode='joint'."
            )
        cv_eligible_indices = (
            scTREND_exp.bulk_data_manager.train_idx.detach().cpu().clone()
            if l2_search_scope == "outer_train"
            else None
        )
        if hazard_fit_mode == "joint":
            best_rec, trials = tune_l2_beta_gamma_joint_classical_3fold(
                scTREND_exp,
                base_state_dict=base_state_dict,
                third_lr=third_lr,
                x_batch_size_scTREND=x_batch_size_scTREND,
                epoch=epoch,
                patience=patience,
                param_save_path=param_save_path,
                warm_path=warm_path,
                l2_beta_candidates=beta_candidates,
                l2_gamma_candidates=gamma_candidates,
                tune_epoch=l2_tune_epoch,
                tune_patience=l2_tune_patience,
                metric=l2_search_metric,
                seed=l2_search_seed,
                n_splits=3,
                resume=resume_l2_search,
                keep_cv_checkpoints=keep_cv_checkpoints,
                eligible_indices=cv_eligible_indices,
            )
            model_params_dict["l2_search_mode"] = (
                "joint_beta_gamma_vs_beta_only_outer_train_inner_3fold"
                if l2_search_scope == "outer_train"
                else "joint_beta_gamma_vs_beta_only_classical_3fold"
            )
            model_params_dict["l2_search_scope"] = l2_search_scope
            model_params_dict["l2_search_eligible_n"] = int(
                getattr(scTREND_exp, "joint_cv_eligible_n", bulk_adata.n_obs)
            )
            model_params_dict["l2_search_resumed_folds"] = int(
                getattr(scTREND_exp, "joint_cv_resumed_fold_count", 0)
            )
            model_params_dict["l2_search_total_folds"] = int(
                getattr(scTREND_exp, "joint_cv_total_fold_count", 0)
            )
            model_params_dict["l2_search_total_trials"] = len(beta_candidates) * (1 + len(gamma_candidates))
        elif (l2_beta is None) and (l2_gamma is not None):
            best_rec, trials = tune_l2_beta_classical_3fold(
                scTREND_exp,
                base_state_dict=base_state_dict,
                third_lr=third_lr,
                x_batch_size_scTREND=x_batch_size_scTREND,
                epoch=epoch,
                patience=patience,
                param_save_path=param_save_path,
                warm_path=warm_path,
                l2_beta_candidates=beta_candidates,
                l2_gamma_fixed=float(l2_gamma),
                tune_epoch=l2_tune_epoch,
                tune_patience=l2_tune_patience,
                metric=l2_search_metric,
                seed=l2_search_seed,
                n_splits=3,
            )
            model_params_dict["l2_search_mode"] = "beta_model_classical_3fold"
            model_params_dict["l2_search_total_trials"] = len(beta_candidates)
        elif (l2_beta is not None) and (l2_gamma is None):
            best_rec, trials = tune_l2_gamma_classical_3fold(
                scTREND_exp,
                base_state_dict=base_state_dict,
                third_lr=third_lr,
                x_batch_size_scTREND=x_batch_size_scTREND,
                epoch=epoch,
                patience=patience,
                param_save_path=param_save_path,
                warm_path=warm_path,
                l2_beta_fixed=float(l2_beta),
                l2_gamma_candidates=gamma_candidates,
                tune_epoch=l2_tune_epoch,
                tune_patience=l2_tune_patience,
                metric=l2_search_metric,
                seed=l2_search_seed,
                n_splits=3,
            )
            model_params_dict["l2_search_mode"] = "gamma_model_classical_3fold"
            model_params_dict["l2_search_total_trials"] = 1 + len(gamma_candidates)
        else:
            best_rec, trials = tune_l2_beta_gamma_grid(
                scTREND_exp,
                base_state_dict=base_state_dict,
                third_lr=third_lr,
                x_batch_size_scTREND=x_batch_size_scTREND,
                epoch=epoch,
                patience=patience,
                param_save_path=param_save_path,
                warm_path=warm_path,
                l2_beta_candidates=beta_candidates,
                l2_gamma_candidates=gamma_candidates,
                tune_epoch=l2_tune_epoch,
                tune_patience=l2_tune_patience,
                metric=l2_search_metric,
                seed=l2_search_seed,
            )
            model_params_dict["l2_search_mode"] = "beta_gamma_model_repeated_holdout"
            model_params_dict["l2_search_total_trials"] = len(beta_candidates) * (1 + len(gamma_candidates))

        if l2_search_metric == "val_nll":
            best_rec, trials = _select_candidate_with_gamma_relative_threshold(
                trials,
                gamma_min_rel_improvement=float(gamma_min_rel_improvement),
            )
            print(
                f"[L2 grid] gamma threshold | delta={float(gamma_min_rel_improvement):.4%} "
                f"| beta_cv_nll={best_rec.get('beta_baseline_cv_val_nll_mean', float('nan')):.6g} "
                f"| best_gamma_cv_nll={best_rec.get('best_gamma_cv_val_nll_mean', float('nan')):.6g} "
                f"| rel_improve={best_rec.get('gamma_rel_improvement_percent_vs_beta', float('nan')):.3f}% "
                f"| pass={best_rec.get('gamma_passes_rel_improvement_threshold', False)}"
            )

        try:
            df = pd.DataFrame(trials)
            csv_path = param_save_path.replace(".pt", "") + "_l2_search.csv"
            df.to_csv(csv_path, index=False)
            print(f"[L2 grid] saved: {csv_path}")
            fold_records = getattr(scTREND_exp, "joint_cv_fold_records", None)
            if fold_records:
                folds_csv_path = param_save_path.replace(".pt", "") + "_l2_search_folds.csv"
                pd.DataFrame(fold_records).to_csv(folds_csv_path, index=False)
                print(f"[L2 grid] saved fold records: {folds_csv_path}")
        except Exception as e:
            print("[L2 grid] WARNING: could not save csv:", e)
            
        selected_use_gamma = bool(best_rec.get("use_gamma", len(driver_genes) > 0))
        scTREND_exp.set_gamma_usage(selected_use_gamma)
        model_params["use_gamma"] = selected_use_gamma
        model_params_dict["use_gamma"] = selected_use_gamma
        model_params_dict["gamma_model_selected"] = "beta_plus_gamma" if selected_use_gamma else "beta_only"

        if l2_beta is None:
            l2_beta = float(best_rec["l2_beta"])
        if l2_gamma is None:
            l2_gamma = float(best_rec["l2_gamma"])

        for _k in [
            "trial",
            "trial_label",
            "cv_val_nll_mean",
            "cv_val_c_index_mean",
            "cv_val_nll_scope",
            "cv_scope",
            "eligible_n",
            "beta_baseline_cv_val_nll_mean",
            "beta_baseline_best_l2_beta",
            "best_gamma_cv_val_nll_mean",
            "best_gamma_l2_beta",
            "best_gamma_l2_gamma",
            "gamma_rel_improvement_vs_beta",
            "gamma_rel_improvement_percent_vs_beta",
            "gamma_min_rel_improvement",
            "gamma_min_rel_improvement_percent",
            "gamma_passes_rel_improvement_threshold",
            "gamma_selection_reason",
            "selected_by_gamma_relative_threshold",
        ]:
            if _k in best_rec:
                model_params_dict[_k] = best_rec[_k]
        model_params_dict["l2_search_cv_val_nll_scope"] = str(best_rec.get("cv_val_nll_scope", "unknown"))

        print(
            f"[L2 grid] BEST | use_gamma={selected_use_gamma} | l2_beta={l2_beta:.3e} | l2_gamma={l2_gamma:.3e} "
            f"| cv_val_nll={best_rec.get('cv_val_nll_mean', float('nan')):.4f} "
            f"| cv_val_c={best_rec.get('cv_val_c_index_mean', float('nan')):.4f}"
        )

    if l2_beta is None:
        l2_beta = 0.0
    if l2_gamma is None:
        l2_gamma = 0.0
        
    selected_use_gamma = bool(force_use_gamma) if force_use_gamma is not None else bool(getattr(scTREND_exp, "use_gamma", len(driver_genes) > 0))
    if not selected_use_gamma:
        l2_gamma = 0.0
    scTREND_exp.set_gamma_usage(selected_use_gamma)

    scTREND_exp.set_l2_penalty(l2_beta=l2_beta, l2_gamma=l2_gamma)
    scTREND_exp.bulk_data_split(
        bulk_seed,
        bulk_validation_num_or_ratio,
        bulk_test_num_or_ratio,
        cutting_off_0_1,
        snv_flag,
        edges=EDGES_WITH_INF,
    )
    scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
    scTREND_exp.clear_hazard_cache()
    model_params["l2_beta"] = float(l2_beta)
    model_params["l2_gamma"] = float(l2_gamma)
    model_params["use_gamma"] = bool(selected_use_gamma)
    model_params_dict["l2_beta"] = float(l2_beta)
    model_params_dict["l2_gamma"] = float(l2_gamma)
    model_params_dict["use_gamma"] = bool(selected_use_gamma)
    model_params_dict["gamma_min_rel_improvement"] = float(gamma_min_rel_improvement)
    model_params_dict["gamma_min_rel_improvement_percent"] = float(100.0 * gamma_min_rel_improvement)

    scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
    scTREND_exp.clear_hazard_cache()

    optimize_hazard = optimize_scTREND_joint if hazard_fit_mode == "joint" else optimize_scTREND
    scTREND_exp = optimize_hazard(
        scTREND_exp,
        third_lr=third_lr,
        x_batch_size=x_batch_size_scTREND,
        epoch=epoch,
        patience=patience,
        param_save_path=param_save_path,
        warm_path=warm_path,
    )

    model_params_dict["beta_best_epoch"] = int(getattr(scTREND_exp, "beta_best_epoch", -1))
    model_params_dict["gamma_best_epoch"] = int(getattr(scTREND_exp, "gamma_best_epoch", -1))
    model_params_dict["joint_best_epoch"] = int(getattr(scTREND_exp, "joint_best_epoch", -1))
    model_params_dict["hazard_early_stopping_metric"] = str(
        getattr(scTREND_exp, "hazard_early_stopping_metric", hazard_early_stopping_metric)
    )
    model_params_dict["hazard_early_stopping_min_delta"] = float(
        getattr(scTREND_exp, "hazard_early_stopping_min_delta", hazard_early_stopping_min_delta)
    )
    model_params_dict["hazard_early_stopping_best_metric"] = float(
        getattr(scTREND_exp, "hazard_early_stopping_best_metric", float("nan"))
    )
    model_params_dict["hazard_early_stopping_best_epoch"] = int(
        getattr(scTREND_exp, "hazard_early_stopping_best_epoch", -1)
    )
    model_params_dict["hazard_early_stopping_stop_epoch"] = int(
        getattr(scTREND_exp, "hazard_early_stopping_stop_epoch", -1)
    )
    model_params_dict["hazard_early_stopping_best_val_nll"] = float(
        getattr(scTREND_exp, "hazard_early_stopping_best_val_nll", float("nan"))
    )
    model_params_dict["hazard_early_stopping_best_val_c_index"] = float(
        getattr(scTREND_exp, "hazard_early_stopping_best_val_c_index", float("nan"))
    )
    model_params_dict["hazard_metrics_split"] = dict(getattr(scTREND_exp, "hazard_metrics", {}))

    if refit_hazard_on_full and hazard_fit_mode == "joint":
        joint_refit_epochs = max(1, int(getattr(scTREND_exp, "joint_best_epoch", epoch - 1)) + 1)
        print(f"[Full refit] joint/beta-only epochs={joint_refit_epochs}")
        fit_idx = torch.arange(bulk_adata.n_obs, dtype=torch.long)
        scTREND_exp = fit_hazard_joint_fixed_epochs(
            scTREND_exp,
            base_state_dict=scTREND_exp.pre_hazard_state_dict,
            third_lr=third_lr,
            num_epochs=joint_refit_epochs,
            fit_indices=fit_idx,
            param_save_path=param_save_path,
            x_batch_size=x_batch_size_scTREND,
        )
        model_params_dict["hazard_refit_joint_epochs"] = joint_refit_epochs
    elif refit_hazard_on_full:
        beta_refit_epochs = max(1, int(getattr(scTREND_exp, "beta_best_epoch", epoch - 1)) + 1)
        if bool(getattr(scTREND_exp, "use_gamma", False)) and len(driver_genes) > 0:
            gamma_refit_epochs = max(1, int(getattr(scTREND_exp, "gamma_best_epoch", epoch - 1)) + 1)
        else:
            gamma_refit_epochs = 0

        print(
            f"[Full refit] beta epochs={beta_refit_epochs} | gamma epochs={gamma_refit_epochs} | "
            f"beta_fit_group={beta_fit_group} | gamma_fit_group={gamma_fit_group}"
        )
        fit_idx = torch.arange(bulk_adata.n_obs, dtype=torch.long)
        scTREND_exp = fit_hazard_two_stage_fixed_epochs(
            scTREND_exp,
            base_state_dict=scTREND_exp.pre_hazard_state_dict,
            third_lr=third_lr,
            beta_num_epochs=beta_refit_epochs,
            gamma_num_epochs=gamma_refit_epochs,
            fit_indices=fit_idx,
            param_save_path=param_save_path,
            beta_fit_group=beta_fit_group,
            gamma_fit_group=gamma_fit_group,
        )
        model_params_dict["hazard_refit_beta_epochs"] = beta_refit_epochs
        model_params_dict["hazard_refit_gamma_epochs"] = gamma_refit_epochs
        model_params_dict["hazard_refit_gamma_fit_group"] = gamma_fit_group
    else:
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path)

    sc_adata.uns["param_save_path"] = param_save_path
    if postprocess_minimal:
        sc_adata, bulk_adata = beta_z_results(scTREND_exp, sc_adata, bulk_adata, driver_genes)
        print("Done minimal post process")
    else:
        sc_adata, bulk_adata = vae_results(scTREND_exp, sc_adata, bulk_adata, param_save_path)
        sc_adata, bulk_adata = bulk_deconvolution_results(scTREND_exp, sc_adata, bulk_adata)
        sc_adata, bulk_adata = beta_z_results(scTREND_exp, sc_adata, bulk_adata, driver_genes)
        if spatial_adata is not None:
            sc_adata, spatial_adata = spatial_results(scTREND_exp, sc_adata, spatial_adata)
        print("Done post process")
    return sc_adata, bulk_adata, model_params_dict, spatial_adata, scTREND_exp

def scTREND_preprocess(sc_adata, bulk_adata, per=0.01, n_top_genes=5000, highly_variable="bulk", driver_genes=None):
    common_genes_before = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    sc_adata = sc_adata[:, common_genes_before].copy()
    bulk_adata = bulk_adata[:, common_genes_before].copy()
    print("common_genes_before", len(common_genes_before))

    sc_min_cells = int(sc_adata.n_obs * per)
    bulk_min_cells = int(bulk_adata.n_obs * per)
    sc.pp.filter_genes(sc_adata, min_cells=sc_min_cells)
    sc.pp.filter_genes(bulk_adata, min_cells=bulk_min_cells)

    common_genes_filtered = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    if driver_genes is not None:
        common_driver_genes = np.intersect1d(common_genes_filtered, driver_genes)
    sc_adata = sc_adata[:, common_genes_filtered].copy()
    bulk_adata = bulk_adata[:, common_genes_filtered].copy()
    print("common_genes_filtered", len(common_genes_filtered))

    raw_sc_adata = sc_adata.copy()
    raw_bulk_adata = bulk_adata.copy()

    if highly_variable == "bulk":
        sc.pp.normalize_total(bulk_adata)
        sc.pp.log1p(bulk_adata)
        sc.pp.highly_variable_genes(bulk_adata, n_top_genes=n_top_genes)
        if driver_genes is not None:
            bulk_adata.var.loc[common_driver_genes, "highly_variable"] = True
        bulk_adata = bulk_adata[:, bulk_adata.var["highly_variable"]].copy()
    elif highly_variable == "sc":
        sc.pp.normalize_total(sc_adata)
        sc.pp.log1p(sc_adata)
        sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes)
        if driver_genes is not None:
            sc_adata.var.loc[common_driver_genes, "highly_variable"] = True
        sc_adata = sc_adata[:, sc_adata.var["highly_variable"]].copy()

    common_genes_highly_variable = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    if len(common_genes_highly_variable) == 0:
        raise ValueError("No common genes found between sc_adata and bulk_adata.")
    sc_adata = sc_adata[:, common_genes_highly_variable].copy()
    bulk_adata = bulk_adata[:, common_genes_highly_variable].copy()
    print("common_genes_highly_variable", len(common_genes_highly_variable))

    sc_adata.X = raw_sc_adata[:, sc_adata.var_names].X
    bulk_adata.X = raw_bulk_adata[:, bulk_adata.var_names].X
    return sc_adata, bulk_adata



def scTREND_preprocess_spatial(sc_adata, bulk_adata, spatial_adata, per=0.01, n_top_genes=5000, highly_variable="bulk"):
    common_genes_before = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    common_genes_before = np.intersect1d(common_genes_before, spatial_adata.var_names)
    sc_adata = sc_adata[:, common_genes_before].copy()
    bulk_adata = bulk_adata[:, common_genes_before].copy()
    spatial_adata = spatial_adata[:, common_genes_before].copy()
    print("common_genes_before", len(common_genes_before))

    sc_min_cells = int(sc_adata.n_obs * per)
    bulk_min_cells = int(bulk_adata.n_obs * per)
    sp_min_cells = int(spatial_adata.n_obs * per)
    sc.pp.filter_genes(sc_adata, min_cells=sc_min_cells)
    sc.pp.filter_genes(bulk_adata, min_cells=bulk_min_cells)
    sc.pp.filter_genes(spatial_adata, min_cells=sp_min_cells)
    common_genes_filtered = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    common_genes_filtered = np.intersect1d(common_genes_filtered, spatial_adata.var_names)
    sc_adata = sc_adata[:, common_genes_filtered].copy()
    bulk_adata = bulk_adata[:, common_genes_filtered].copy()
    spatial_adata = spatial_adata[:, common_genes_filtered].copy()
    print("common_genes_filtered", len(common_genes_filtered))

    raw_sc_adata = sc_adata.copy()
    raw_bulk_adata = bulk_adata.copy()
    raw_spatial_adata = spatial_adata.copy()
    if highly_variable == "bulk":
        sc.pp.normalize_total(bulk_adata)
        sc.pp.log1p(bulk_adata)
        sc.pp.highly_variable_genes(bulk_adata, n_top_genes=n_top_genes)
        bulk_adata = bulk_adata[:, bulk_adata.var["highly_variable"]].copy()
    elif highly_variable == "spatial":
        sc.pp.normalize_total(spatial_adata)
        sc.pp.log1p(spatial_adata)
        sc.pp.highly_variable_genes(spatial_adata, n_top_genes=n_top_genes)
        spatial_adata = spatial_adata[:, spatial_adata.var["highly_variable"]].copy()
    elif highly_variable == "sc":
        sc.pp.normalize_total(sc_adata)
        sc.pp.log1p(sc_adata)
        sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes)
        sc_adata = sc_adata[:, sc_adata.var["highly_variable"]].copy()

    common_genes_highly_variable = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    common_genes_highly_variable = np.intersect1d(common_genes_highly_variable, spatial_adata.var_names)
    if len(common_genes_highly_variable) == 0:
        raise ValueError("No common genes found between sc_adata, bulk_adata and spatial_adata.")
    sc_adata = sc_adata[:, common_genes_highly_variable].copy()
    bulk_adata = bulk_adata[:, common_genes_highly_variable].copy()
    spatial_adata = spatial_adata[:, common_genes_highly_variable].copy()
    print("common_genes_highly_variable", len(common_genes_highly_variable))

    sc_adata.X = raw_sc_adata[:, sc_adata.var_names].X
    bulk_adata.X = raw_bulk_adata[:, bulk_adata.var_names].X
    spatial_adata.X = raw_spatial_adata[:, spatial_adata.var_names].X
    return sc_adata, bulk_adata, spatial_adata
