import os

import numpy as np
import pandas as pd
import torch

from .concordance import concordance_index_unweighted_dynamic


def safe_toarray(x):
    if type(x) != np.ndarray:
        x = x.toarray()
        if not np.all(x == np.floor(x)):
            raise ValueError("target layer of adata should be raw count")
        return x
    if not np.all(x == np.floor(x)):
        raise ValueError("target layer of adata should be raw count")
    return x


def make_sample_one_hot_mat(adata, sample_key):
    print("make_sample_one_hot_mat")
    if sample_key is not None:
        sidxs = np.sort(adata.obs[sample_key].unique())
        b = np.array([(sidxs == sidx).astype(int) for sidx in adata.obs[sample_key]]).astype(float)
        b = torch.tensor(b).float()
    else:
        b = np.zeros((len(adata.obs_names), 1))
        b = torch.tensor(b).float()
    return b


def input_checks(adata, layer_name):
    if layer_name == "X":
        if np.sum((adata.X - adata.X.astype(int))) ** 2 != 0:
            raise ValueError("`X` includes non integer number, while count data is required for `X`.")
    else:
        if np.sum((adata.layers[layer_name] - adata.layers[layer_name].astype(int))) ** 2 != 0:
            raise ValueError(
                f"layers `{layer_name}` includes non integer number, while count data is required for `{layer_name}`."
            )



def make_inputs(sc_adata, bulk_adata, layer_name="X"):
    input_checks(sc_adata, layer_name)
    if layer_name == "X":
        x = torch.tensor(safe_toarray(sc_adata.X))
        s = torch.tensor(safe_toarray(bulk_adata.X))
    else:
        x = torch.tensor(safe_toarray(sc_adata.layers[layer_name]))
        s = torch.tensor(safe_toarray(bulk_adata.layers[layer_name]))
    return x, s



def optimize_vae(scTREND_exp, first_lr, x_batch_size, epoch, patience, param_save_path):
    print("Start first opt", "lr=", first_lr)
    scTREND_exp.scTREND.sc_mode()
    scTREND_exp.initialize_optimizer(first_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    scTREND_exp.train_total(epoch, patience)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    val_loss_vae = scTREND_exp.evaluate(mode="val")
    test_loss_vae = scTREND_exp.evaluate(mode="test")
    print(f"Done {scTREND_exp.scTREND.mode} mode,", f"Val Loss: {val_loss_vae}", f"Test Loss: {test_loss_vae}")
    return scTREND_exp



def optimize_vae_onlyload(scTREND_exp, first_lr, x_batch_size, epoch, patience, param_save_path):
    print("Start first opt", "lr=", first_lr)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    return scTREND_exp



def optimize_deepcolor(scTREND_exp, second_lr, x_batch_size, epoch, patience, param_save_path, spatial_adata):
    scTREND_exp.scTREND.bulk_mode()
    scTREND_exp.initialize_optimizer(second_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    print(f"{scTREND_exp.scTREND.mode} mode", "lr=", second_lr)
    scTREND_exp.train_total(epoch, patience)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    val_loss_bulk = scTREND_exp.evaluate(mode="val")
    test_loss_bulk = scTREND_exp.evaluate(mode="test")
    print(f"Done {scTREND_exp.scTREND.mode} mode,", f"Val Loss: {val_loss_bulk}", f"Test Loss: {test_loss_bulk}")

    if spatial_adata is not None:
        scTREND_exp.scTREND.spatial_mode()
        scTREND_exp.initialize_optimizer(second_lr)
        scTREND_exp.initialize_loader(x_batch_size)
        print(f"{scTREND_exp.scTREND.mode} mode", "lr=", second_lr)
        scTREND_exp.train_total(epoch, patience)
        scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
        val_loss_spatial = scTREND_exp.evaluate(mode="val")
        test_loss_spatial = scTREND_exp.evaluate(mode="test")
        print(
            f"Done {scTREND_exp.scTREND.mode} mode,",
            f"Val Loss: {val_loss_spatial}",
            f"Test Loss: {test_loss_spatial}",
        )
    return scTREND_exp



def optimize_deepcolor_onlyload(scTREND_exp, second_lr, x_batch_size, epoch, patience, param_save_path, spatial_adata=None):
    print(f"{scTREND_exp.scTREND.mode} mode", "lr=", second_lr)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    return scTREND_exp




def _evaluate_hazard_on_original_bulk_split(scTREND_exp):
    """
    Temporarily clear active train/validation/test indices and evaluate hazard NLL
    on the original train / validation / test split.

    This is critical for fair model selection:
      - beta is trained/evaluated on WT + Mut.
      - gamma may be trained on mutated-only samples.
      - however, beta-only vs beta+gamma candidates must be compared on the
        same full validation split (WT + Mut).
    """
    old_active_train = getattr(scTREND_exp, "active_train_idx", None)
    old_active_val = getattr(scTREND_exp, "active_validation_idx", None)
    old_active_test = getattr(scTREND_exp, "active_test_idx", None)

    try:
        scTREND_exp.set_active_bulk_indices(None, None, None)
        scTREND_exp.clear_hazard_cache()

        train_nll = scTREND_exp.evaluate_train()
        val_nll = scTREND_exp.evaluate("validation")
        test_nll = scTREND_exp.evaluate("test")
        return float(train_nll), float(val_nll), float(test_nll)

    finally:
        scTREND_exp.set_active_bulk_indices(
            old_active_train,
            old_active_val,
            old_active_test,
        )
        scTREND_exp.clear_hazard_cache()


def optimize_scTREND(scTREND_exp, third_lr, x_batch_size, epoch, patience, param_save_path, warm_path, return_metrics: bool = False):
    scTREND_exp.checkpoint = param_save_path
    scTREND_exp.clear_hazard_cache()
    scTREND_exp.hazard_forward_deterministic_override = None
    scTREND_exp.set_active_bulk_indices(None, None, None)

    gamma_enabled = bool(
        getattr(
            scTREND_exp,
            "use_gamma",
            len(getattr(scTREND_exp, "driver_genes", [])) > 0,
        )
    )
    gamma_enabled = gamma_enabled and (len(getattr(scTREND_exp, "driver_genes", [])) > 0)

    if hasattr(scTREND_exp, "set_gamma_usage"):
        scTREND_exp.set_gamma_usage(gamma_enabled)

    # ============================================================
    # Stage 1: beta only
    # ============================================================
    scTREND_exp.activate_beta_stage_indices(
        beta_fit_group=getattr(scTREND_exp, "beta_fit_group", "all")
    )
    scTREND_exp.scTREND.hazard_beta_z_mode()
    scTREND_exp.initialize_optimizer(third_lr)
    scTREND_exp.initialize_loader(x_batch_size)

    print(
        f"{scTREND_exp.scTREND.mode} stage (beta-only) lr={third_lr} "
        f"| beta_fit_group={getattr(scTREND_exp, 'beta_fit_group', 'all')} "
        f"| l2_beta={getattr(scTREND_exp, 'l2_beta', 0.0):.3e} "
        f"| use_gamma={gamma_enabled}"
    )

    beta_best_epoch = scTREND_exp.train_total(epoch, patience)
    scTREND_exp.beta_best_epoch = int(beta_best_epoch)
    scTREND_exp.beta_stop_epoch = int(beta_best_epoch)

    scTREND_exp.scTREND.load_state_dict(
        torch.load(param_save_path, map_location=scTREND_exp.device)
    )
    torch.save(
        scTREND_exp.scTREND.state_dict(),
        param_save_path.replace(".pt", "_3rd_beta_end.pt"),
    )
    scTREND_exp.clear_hazard_cache()

    # Active metrics are useful diagnostics. With beta_fit_group='all', these
    # should be identical to the full split metrics, but we keep both for clarity.
    beta_train_nll_active = float(scTREND_exp.evaluate_train())
    beta_val_nll_active = float(scTREND_exp.evaluate("validation"))
    beta_test_nll_active = float(scTREND_exp.evaluate("test"))

    beta_train_nll, beta_val_nll, beta_test_nll = _evaluate_hazard_on_original_bulk_split(
        scTREND_exp
    )

    scTREND_exp.beta_stage_metrics = {
        "beta_train_nll": float(beta_train_nll),
        "beta_val_nll": float(beta_val_nll),
        "beta_test_nll": float(beta_test_nll),
        "beta_train_nll_active": float(beta_train_nll_active),
        "beta_val_nll_active": float(beta_val_nll_active),
        "beta_test_nll_active": float(beta_test_nll_active),
        "beta_best_epoch": int(beta_best_epoch),
        "l2_beta": float(getattr(scTREND_exp, "l2_beta", 0.0)),
        "beta_fit_group": getattr(scTREND_exp, "beta_fit_group", "all"),
    }

    print(
        f"Done beta-only stage | "
        f"Full Train NLL: {beta_train_nll:.4f} | "
        f"Full Val NLL: {beta_val_nll:.4f} | "
        f"Full Test NLL: {beta_test_nll:.4f}"
    )

    # ============================================================
    # Stage 2: gamma only, beta frozen
    # ============================================================
    gamma_train_nll = float("nan")
    gamma_val_nll = float("nan")
    gamma_test_nll = float("nan")

    gamma_train_nll_active = float("nan")
    gamma_val_nll_active = float("nan")
    gamma_test_nll_active = float("nan")

    gamma_best_epoch = -1

    if gamma_enabled:
        scTREND_exp.activate_gamma_stage_indices(
            gamma_fit_group=getattr(scTREND_exp, "gamma_fit_group", "mutated_only")
        )
        scTREND_exp.scTREND.hazard_gamma_z_mode()
        scTREND_exp.initialize_optimizer(third_lr)
        scTREND_exp.initialize_loader(x_batch_size)

        print(
            f"{scTREND_exp.scTREND.mode} stage (gamma-only, beta frozen) lr={third_lr} "
            f"| gamma_fit_group={getattr(scTREND_exp, 'gamma_fit_group', 'mutated_only')} "
            f"| l2_gamma={getattr(scTREND_exp, 'l2_gamma', 0.0):.3e}"
        )

        gamma_best_epoch = scTREND_exp.train_total(epoch, patience)
        scTREND_exp.gamma_best_epoch = int(gamma_best_epoch)
        scTREND_exp.gamma_stop_epoch = int(gamma_best_epoch)

        scTREND_exp.scTREND.load_state_dict(
            torch.load(param_save_path, map_location=scTREND_exp.device)
        )
        torch.save(
            scTREND_exp.scTREND.state_dict(),
            param_save_path.replace(".pt", "_3rd_gamma_end.pt"),
        )
        scTREND_exp.clear_hazard_cache()

        # Active-set diagnostics. If gamma_fit_group='mutated_only', these are
        # mutated-only NLL values and should NOT be used for model selection.
        gamma_train_nll_active = float(scTREND_exp.evaluate_train())
        gamma_val_nll_active = float(scTREND_exp.evaluate("validation"))
        gamma_test_nll_active = float(scTREND_exp.evaluate("test"))

        # Model-selection metrics. These are full split NLL values using WT + Mut.
        gamma_train_nll, gamma_val_nll, gamma_test_nll = _evaluate_hazard_on_original_bulk_split(
            scTREND_exp
        )

        print(
            f"Done gamma-only stage | "
            f"Active Val NLL: {gamma_val_nll_active:.4f} | "
            f"Full Val NLL: {gamma_val_nll:.4f}"
        )

    else:
        print("Skip gamma stage: use_gamma=False or no driver genes.")

    scTREND_exp.gamma_stage_metrics = {
        # Full split NLL values used for model selection and final metrics.
        "gamma_train_nll": float(gamma_train_nll),
        "gamma_val_nll": float(gamma_val_nll),
        "gamma_test_nll": float(gamma_test_nll),

        # Active split diagnostics. With gamma_fit_group='mutated_only', these
        # are mutated-only values.
        "gamma_train_nll_active": float(gamma_train_nll_active),
        "gamma_val_nll_active": float(gamma_val_nll_active),
        "gamma_test_nll_active": float(gamma_test_nll_active),

        "gamma_best_epoch": int(gamma_best_epoch),
        "l2_gamma": float(getattr(scTREND_exp, "l2_gamma", 0.0)),
        "gamma_fit_group": getattr(scTREND_exp, "gamma_fit_group", "mutated_only"),
    }

    # ============================================================
    # Final predictive metrics on original split
    # ============================================================
    scTREND_exp.set_active_bulk_indices(None, None, None)
    scTREND_exp.clear_hazard_cache()

    edges_np = scTREND_exp.edges.cpu().numpy()

    def _c(bidx):
        if bidx is None or len(bidx) == 0:
            return float("nan")
        try:
            lam = scTREND_exp.predict_lambda_table(bidx)
            surv = scTREND_exp.survival_time[bidx.to(scTREND_exp.device)].cpu().numpy()
            event = scTREND_exp.cutting_off_0_1[bidx.to(scTREND_exp.device)].cpu().numpy()
            c, _ = concordance_index_unweighted_dynamic(surv, event, lam, edges_np)
            return float(c)
        except Exception:
            return float("nan")

    c_train = _c(scTREND_exp.bulk_data_manager.train_idx)
    c_val = _c(scTREND_exp.bulk_data_manager.validation_idx)
    c_test = _c(scTREND_exp.bulk_data_manager.test_idx)

    print(
        f"Final C-index on full split | "
        f"Train: {c_train:.3f} | Val: {c_val:.3f} | Test: {c_test:.3f}"
    )

    final_train_nll = (
        gamma_train_nll
        if gamma_enabled and not np.isnan(gamma_train_nll)
        else beta_train_nll
    )
    final_val_nll = (
        gamma_val_nll
        if gamma_enabled and not np.isnan(gamma_val_nll)
        else beta_val_nll
    )
    final_test_nll = (
        gamma_test_nll
        if gamma_enabled and not np.isnan(gamma_test_nll)
        else beta_test_nll
    )

    metrics_dict = {
        **scTREND_exp.beta_stage_metrics,
        **scTREND_exp.gamma_stage_metrics,

        # workflow.py model selection reads these keys.
        # They must always mean full original split NLL.
        "train_nll": float(final_train_nll),
        "val_nll": float(final_val_nll),
        "test_nll": float(final_test_nll),

        "train_c_index": float(c_train),
        "val_c_index": float(c_val),
        "test_c_index": float(c_test),

        "l2_beta": float(getattr(scTREND_exp, "l2_beta", 0.0)),
        "l2_gamma": float(getattr(scTREND_exp, "l2_gamma", 0.0)),
        "beta_fit_group": getattr(scTREND_exp, "beta_fit_group", "all"),
        "gamma_fit_group": getattr(scTREND_exp, "gamma_fit_group", "mutated_only"),
        "use_gamma": bool(gamma_enabled),

        "model_selection_val_nll_scope": "full_original_validation_WT_plus_Mut",
        "gamma_training_val_nll_scope": getattr(
            scTREND_exp, "gamma_fit_group", "mutated_only"
        ),
    }
    scTREND_exp.hazard_metrics = metrics_dict

    tag = "" if warm_path is None else ("_" + os.path.splitext(os.path.basename(warm_path))[0])
    combined_path = param_save_path.replace(".pt", "") + tag + "_metrics.pt"
    torch.save(metrics_dict, combined_path)

    scTREND_exp.set_active_bulk_indices(None, None, None)
    if return_metrics:
        return scTREND_exp, metrics_dict
    return scTREND_exp


def optimize_scTREND_joint(
    scTREND_exp,
    third_lr,
    x_batch_size,
    epoch,
    patience,
    param_save_path,
    warm_path,
    return_metrics: bool = False,
):
    """Fit beta and gamma together, or the matched beta-only comparator."""
    scTREND_exp.checkpoint = param_save_path
    scTREND_exp.clear_hazard_cache()
    scTREND_exp.hazard_forward_deterministic_override = None
    scTREND_exp.set_active_bulk_indices(None, None, None)

    gamma_enabled = bool(
        getattr(scTREND_exp, "use_gamma", False)
        and len(getattr(scTREND_exp, "driver_genes", [])) > 0
    )
    scTREND_exp.set_gamma_usage(gamma_enabled)
    if gamma_enabled:
        scTREND_exp.scTREND.hazard_joint_z_mode()
        fit_label = "beta+gamma joint"
    else:
        scTREND_exp.scTREND.hazard_beta_z_mode()
        fit_label = "beta-only matched comparator"

    scTREND_exp.initialize_optimizer(third_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    print(
        f"{scTREND_exp.scTREND.mode} stage ({fit_label}) lr={third_lr} "
        f"| l2_beta={getattr(scTREND_exp, 'l2_beta', 0.0):.3e} "
        f"| l2_gamma={getattr(scTREND_exp, 'l2_gamma', 0.0):.3e}"
    )

    best_epoch = scTREND_exp.train_total(epoch, patience)
    scTREND_exp.joint_best_epoch = int(best_epoch)
    scTREND_exp.beta_best_epoch = int(best_epoch) if not gamma_enabled else -1
    scTREND_exp.gamma_best_epoch = int(best_epoch) if gamma_enabled else -1
    scTREND_exp.scTREND.load_state_dict(
        torch.load(param_save_path, map_location=scTREND_exp.device)
    )
    suffix = "_3rd_joint_end.pt" if gamma_enabled else "_3rd_betaonly_end.pt"
    torch.save(scTREND_exp.scTREND.state_dict(), param_save_path.replace(".pt", suffix))
    scTREND_exp.clear_hazard_cache()

    train_nll, val_nll, test_nll = _evaluate_hazard_on_original_bulk_split(scTREND_exp)
    edges_np = scTREND_exp.edges.cpu().numpy()

    def _c_index(bidx):
        if bidx is None or len(bidx) == 0:
            return float("nan")
        try:
            lam = scTREND_exp.predict_lambda_table(bidx)
            bidx_dev = bidx.to(scTREND_exp.device)
            surv = scTREND_exp.survival_time[bidx_dev].cpu().numpy()
            event = scTREND_exp.cutting_off_0_1[bidx_dev].cpu().numpy()
            c_index, _ = concordance_index_unweighted_dynamic(
                surv, event, lam, edges_np
            )
            return float(c_index)
        except Exception:
            return float("nan")

    c_train = _c_index(scTREND_exp.bulk_data_manager.train_idx)
    c_val = _c_index(scTREND_exp.bulk_data_manager.validation_idx)
    c_test = _c_index(scTREND_exp.bulk_data_manager.test_idx)
    metrics_dict = {
        "train_nll": float(train_nll),
        "val_nll": float(val_nll),
        "test_nll": float(test_nll),
        "train_c_index": float(c_train),
        "val_c_index": float(c_val),
        "test_c_index": float(c_test),
        "l2_beta": float(getattr(scTREND_exp, "l2_beta", 0.0)),
        "l2_gamma": float(getattr(scTREND_exp, "l2_gamma", 0.0)),
        "use_gamma": bool(gamma_enabled),
        "hazard_fit_mode": "joint_beta_gamma" if gamma_enabled else "beta_only",
        "joint_best_epoch": int(best_epoch),
        "hazard_early_stopping_metric": str(
            getattr(scTREND_exp, "hazard_early_stopping_metric", "val_nll")
        ),
        "hazard_early_stopping_min_delta": float(
            getattr(scTREND_exp, "hazard_early_stopping_min_delta", 0.0)
        ),
        "hazard_early_stopping_best_metric": float(
            getattr(scTREND_exp, "hazard_early_stopping_best_metric", float("nan"))
        ),
        "hazard_early_stopping_best_epoch": int(
            getattr(scTREND_exp, "hazard_early_stopping_best_epoch", best_epoch)
        ),
        "hazard_early_stopping_stop_epoch": int(
            getattr(scTREND_exp, "hazard_early_stopping_stop_epoch", best_epoch)
        ),
        "hazard_early_stopping_best_val_nll": float(
            getattr(scTREND_exp, "hazard_early_stopping_best_val_nll", float("nan"))
        ),
        "hazard_early_stopping_best_val_c_index": float(
            getattr(scTREND_exp, "hazard_early_stopping_best_val_c_index", float("nan"))
        ),
        "model_selection_val_nll_scope": "full_original_validation_WT_plus_Mut",
        "gamma_training_val_nll_scope": "full_original_validation_WT_plus_Mut",
    }
    scTREND_exp.hazard_metrics = metrics_dict
    tag = "" if warm_path is None else "_" + os.path.splitext(os.path.basename(warm_path))[0]
    torch.save(metrics_dict, param_save_path.replace(".pt", "") + tag + "_metrics.pt")
    print(
        f"Done {fit_label} | Val NLL: {val_nll:.6f} | "
        f"Val unweighted dynamic C-index: {c_val:.6f}"
    )
    if return_metrics:
        return scTREND_exp, metrics_dict
    return scTREND_exp


def fit_hazard_joint_fixed_epochs(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    num_epochs,
    fit_indices=None,
    param_save_path=None,
    x_batch_size=1000,
):
    """Full-data refit for the selected beta-only or joint beta+gamma model."""
    if fit_indices is None:
        fit_indices = torch.arange(scTREND_exp.bulk_count.shape[0], dtype=torch.long)
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long)
    if fit_indices.numel() == 0:
        raise ValueError("fit_indices is empty in fit_hazard_joint_fixed_epochs().")

    scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
    scTREND_exp.clear_hazard_cache()
    scTREND_exp.set_active_bulk_indices(fit_indices, fit_indices, fit_indices)
    gamma_enabled = bool(
        getattr(scTREND_exp, "use_gamma", False)
        and len(getattr(scTREND_exp, "driver_genes", [])) > 0
    )
    if gamma_enabled:
        scTREND_exp.scTREND.hazard_joint_z_mode()
    else:
        scTREND_exp.scTREND.hazard_beta_z_mode()
    # Explicit hazard batch size; never inherit the VAE-stage loader setting.
    scTREND_exp.initialize_loader(int(x_batch_size))
    scTREND_exp.initialize_optimizer(third_lr)
    for _ in range(int(num_epochs)):
        scTREND_exp.train_epoch()

    scTREND_exp.set_active_bulk_indices(None, None, None)
    scTREND_exp.clear_hazard_cache()
    scTREND_exp.prepare_hazard_cache(force=True, sample_z=False)
    if param_save_path is not None:
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path)
    return scTREND_exp

def fit_hazard_two_stage_fixed_epochs(
    scTREND_exp,
    *,
    base_state_dict,
    third_lr,
    beta_num_epochs,
    gamma_num_epochs,
    fit_indices=None,
    param_save_path=None,
    beta_fit_group=None,
    gamma_fit_group=None,
    sample_z_for_hazard_cache=False,
    hazard_cache_seed=None,
):
    gamma_enabled = bool(
        getattr(scTREND_exp, "use_gamma",
                len(getattr(scTREND_exp, "driver_genes", [])) > 0)
    )
    if fit_indices is None:
        fit_indices = torch.arange(scTREND_exp.bulk_count.shape[0], dtype=torch.long)
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long)

    if beta_fit_group is None:
        beta_fit_group = getattr(scTREND_exp, "beta_fit_group", "all")
    if gamma_fit_group is None:
        gamma_fit_group = getattr(scTREND_exp, "gamma_fit_group", "mutated_only")

    scTREND_exp.scTREND.load_state_dict(base_state_dict, strict=False)
    scTREND_exp.clear_hazard_cache()

    current_x_batch_size = getattr(getattr(scTREND_exp.x_data_manager, "train_loader", None), "batch_size", None)
    if current_x_batch_size is None:
        current_x_batch_size = 500
    scTREND_exp.initialize_loader(int(current_x_batch_size))

    prev_override = getattr(scTREND_exp, "hazard_forward_deterministic_override", None)
    if bool(sample_z_for_hazard_cache):
        # In bootstrap mode, use sampled z during hazard fitting and also cache a sampled
        # latent realization afterward for coefficient export / CI construction.
        scTREND_exp.hazard_forward_deterministic_override = False
    else:
        scTREND_exp.hazard_forward_deterministic_override = None

    try:
        # beta stage
        if beta_fit_group == "reference_only":
            beta_fit_indices = scTREND_exp._restrict_indices_to_reference(fit_indices)
        elif beta_fit_group == "all":
            beta_fit_indices = fit_indices.clone()
        else:
            raise ValueError("beta_fit_group must be either 'reference_only' or 'all'.")

        if beta_fit_indices.numel() == 0:
            raise ValueError("No samples are available for beta-stage fitting in fit_hazard_two_stage_fixed_epochs().")

        scTREND_exp.set_active_bulk_indices(beta_fit_indices, beta_fit_indices, beta_fit_indices)
        scTREND_exp.scTREND.hazard_beta_z_mode()
        scTREND_exp.initialize_optimizer(third_lr)
        for _ in range(int(beta_num_epochs)):
            scTREND_exp.train_epoch()

        # gamma stage
        if gamma_enabled and len(getattr(scTREND_exp, "driver_genes", [])) > 0 and int(gamma_num_epochs) > 0:
            if gamma_fit_group in {"mutated_only", "any_mutated_only"}:
                gamma_fit_indices = scTREND_exp._restrict_indices_to_mutated(fit_indices)
            elif gamma_fit_group == "all":
                gamma_fit_indices = fit_indices.clone()
            else:
                raise ValueError("gamma_fit_group must be either 'mutated_only' or 'all'.")

            if gamma_fit_indices.numel() == 0:
                print("[fit_hazard_two_stage_fixed_epochs] WARNING: no samples matched gamma_fit_group; gamma stage was skipped.")
            else:
                scTREND_exp.set_active_bulk_indices(gamma_fit_indices, gamma_fit_indices, gamma_fit_indices)
                scTREND_exp.scTREND.hazard_gamma_z_mode()
                scTREND_exp.initialize_optimizer(third_lr)
                for _ in range(int(gamma_num_epochs)):
                    scTREND_exp.train_epoch()

        scTREND_exp.set_active_bulk_indices(None, None, None)
        scTREND_exp.clear_hazard_cache()
        scTREND_exp.prepare_hazard_cache(
            force=True,
            sample_z=bool(sample_z_for_hazard_cache),
            seed=hazard_cache_seed,
        )
        if param_save_path is not None:
            torch.save(scTREND_exp.scTREND.state_dict(), param_save_path)
        return scTREND_exp
    finally:
        scTREND_exp.hazard_forward_deterministic_override = prev_override


def vae_results(scTREND_exp, sc_adata, bulk_adata, param_save_path):
    print("vae_results")
    with torch.no_grad():
        torch.cuda.empty_cache()
        batch_onehot = scTREND_exp.x_data_manager.batch_onehot.to(scTREND_exp.device)
        x = scTREND_exp.x_data_manager.x_count.to(scTREND_exp.device)
        x_np = x.detach().cpu().numpy()
        xb = torch.cat([x, batch_onehot], dim=-1)
        z, qz = scTREND_exp.scTREND.enc_z(xb)
        zl = qz.loc
        xxx_list = []
        for _ in range(100):
            zzz = qz.sample()
            zb = torch.cat([zzz, batch_onehot], dim=-1)
            xxx_np = scTREND_exp.scTREND.dec_z2x(zb).detach().cpu().numpy()
            xxx_list.append(xxx_np)
        xld_np = np.mean(xxx_list, axis=0)
        sc_adata.obsm["zl"] = zl.detach().cpu().numpy()
        sc_adata.layers["xld"] = xld_np
        xnorm_mat = scTREND_exp.x_data_manager.xnorm_mat
        xnorm_mat_np = xnorm_mat.cpu().detach().numpy()
        x_df = pd.DataFrame(x_np, columns=list(sc_adata.var_names))
        xld_df = pd.DataFrame(xld_np, columns=list(sc_adata.var_names))
        train_idx = scTREND_exp.x_data_manager.train_idx
        val_idx = scTREND_exp.x_data_manager.validation_idx
        test_idx = scTREND_exp.x_data_manager.test_idx
        x_correlation_gene = xld_df.corrwith((x_df / xnorm_mat_np)).mean()
        train_x_correlation_gene = (xld_df.T[train_idx].T).corrwith((x_df / xnorm_mat_np).T[train_idx].T).mean()
        val_x_correlation_gene = (xld_df.T[val_idx].T).corrwith((x_df / xnorm_mat_np).T[val_idx].T).mean()
        test_x_correlation_gene = (xld_df.T[test_idx].T).corrwith((x_df / xnorm_mat_np).T[test_idx].T).mean()
        metrics_dict = {
            "all_x_correlation_gene": x_correlation_gene,
            "train_x_correlation_gene": train_x_correlation_gene,
            "val_x_correlation_gene": val_x_correlation_gene,
            "test_x_correlation_gene": test_x_correlation_gene,
        }
        combined_path = param_save_path.replace(".pt", "") + "_correlation.pt"
        torch.save(metrics_dict, combined_path)
        print(
            "all_x_correlation_gene",
            f"{x_correlation_gene:.3f}",
            "train_x_correlation_gene",
            f"{train_x_correlation_gene:.3f}",
            "val_x_correlation_gene",
            f"{val_x_correlation_gene:.3f}",
            "test_x_correlation_gene",
            f"{test_x_correlation_gene:.3f}",
        )
        return sc_adata, bulk_adata



def bulk_deconvolution_results(scTREND_exp, sc_adata, bulk_adata):
    print("deconvolution_results")
    with torch.no_grad():
        torch.cuda.empty_cache()
        batch_onehot = scTREND_exp.x_data_manager.batch_onehot.to(scTREND_exp.device)
        x = scTREND_exp.x_data_manager.x_count.to(scTREND_exp.device)
        xb = torch.cat([x, batch_onehot], dim=-1)
        z, qz = scTREND_exp.scTREND.enc_z(xb)
        ppp_list = []
        for _ in range(100):
            zzz = qz.sample()
            ppp = scTREND_exp.scTREND.dec_z2p_bulk(zzz).detach().cpu().numpy()
            ppp_list.append(ppp)
        bulk_pl_np = np.mean(ppp_list, axis=0)
        bulk_scoeff_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_bulk_coeff).cpu().detach().numpy()
        bulk_scoeff_add_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_bulk_coeff_add).cpu().detach().numpy()
        xld_np = sc_adata.layers["xld"]
        bulk_hat_np = np.matmul(bulk_pl_np, xld_np * bulk_scoeff_np) + bulk_scoeff_add_np
        bulk_p_df = pd.DataFrame(bulk_pl_np.transpose(), index=sc_adata.obs_names, columns=bulk_adata.obs_names)
        sc_adata.obsm["map2bulk"] = bulk_p_df.values
        bulk_norm_mat = scTREND_exp.bulk_data_manager.bulk_norm_mat
        bulk_norm_mat_np = bulk_norm_mat.cpu().detach().numpy()
        bulk_count = scTREND_exp.bulk_data_manager.bulk_count
        bulk_count_df = pd.DataFrame(bulk_count.cpu().detach().numpy(), columns=list(bulk_adata.var_names))
        bulk_hat_df = pd.DataFrame(bulk_hat_np, columns=list(bulk_adata.var_names))
        bulk_adata.layers["bulk_hat"] = pd.DataFrame(
            bulk_hat_np,
            index=list(bulk_adata.obs_names),
            columns=list(bulk_adata.var_names),
        )
        bulk_target_df = bulk_count_df / bulk_norm_mat_np
        bulk_corr_by_gene = bulk_hat_df.replace([np.inf, -np.inf], np.nan).corrwith(
            bulk_target_df.replace([np.inf, -np.inf], np.nan)
        )
        bulk_corr_valid = bulk_corr_by_gene[np.isfinite(bulk_corr_by_gene)]
        bulk_correlation_gene = bulk_corr_valid.mean() if len(bulk_corr_valid) else np.nan
        bulk_adata.uns["bulk_correlation_gene"] = float(bulk_correlation_gene)
        bulk_adata.uns["bulk_correlation_gene_valid_n"] = int(len(bulk_corr_valid))
        bulk_adata.uns["bulk_correlation_gene_total_n"] = int(len(bulk_corr_by_gene))
        bulk_adata.uns["bulk_correlation_gene_nan_n"] = int(bulk_corr_by_gene.isna().sum())
        print("bulk_correlation_gene", bulk_correlation_gene)
        print(
            "bulk_correlation_gene_valid",
            len(bulk_corr_valid),
            "of",
            len(bulk_corr_by_gene),
        )
        return sc_adata, bulk_adata



def spatial_results(scTREND_exp, sc_adata, spatial_adata):
    print("spatial_results")
    with torch.no_grad():
        torch.cuda.empty_cache()
        batch_onehot = scTREND_exp.x_data_manager.batch_onehot.to(scTREND_exp.device)
        x = scTREND_exp.x_data_manager.x_count.to(scTREND_exp.device)
        xb = torch.cat([x, batch_onehot], dim=-1)
        z, qz = scTREND_exp.scTREND.enc_z(xb)
        ppp_list = []
        for _ in range(100):
            zzz = qz.sample()
            ppp = scTREND_exp.scTREND.dec_z2p_spatial(zzz).detach().cpu().numpy()
            ppp_list.append(ppp)
        spatial_pl_np = np.mean(ppp_list, axis=0)
        spatial_coeff_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_spatial_coeff).cpu().detach().numpy()
        spatial_coeff_add_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_spatial_coeff_add).cpu().detach().numpy()
        xld_np = sc_adata.layers["xld"]
        spatial_hat_np = np.matmul(spatial_pl_np, xld_np * spatial_coeff_np) + spatial_coeff_add_np
        spatial_p_df = pd.DataFrame(spatial_pl_np.transpose(), index=sc_adata.obs_names, columns=spatial_adata.obs_names)
        sc_adata.obsm["map2spatial"] = spatial_p_df.values
        if "raw_beta_z" not in sc_adata.obsm:
            raise KeyError("raw_beta_z not found in sc_adata.obsm. Call beta_z_results() before spatial_results().")
        beta_ck = sc_adata.obsm["raw_beta_z"]
        P_cs = sc_adata.obsm["map2spatial"]
        eta_spot_k = P_cs.T @ beta_ck
        spatial_adata.obsm["eta_spot_timebins"] = eta_spot_k
        lambda_spot_k = np.log1p(np.exp(eta_spot_k))
        spatial_adata.obsm["lambda_spot_timebins"] = lambda_spot_k
        K = lambda_spot_k.shape[1]
        for k in range(K):
            spatial_adata.obs[f"lambda_timebin_{k + 1}"] = lambda_spot_k[:, k]
            h = lambda_spot_k[:, k]
            spatial_adata.obs[f"lambda_rel_timebin_{k + 1}"] = h / h.mean()
        total_h = lambda_spot_k.sum(axis=1)
        spatial_adata.obs["Hazard_rates"] = total_h / total_h.mean()
        spatial_norm_mat = scTREND_exp.spatial_data_manager.spatial_norm_mat
        spatial_norm_mat_np = spatial_norm_mat.cpu().detach().numpy()
        spatial_count = scTREND_exp.spatial_data_manager.spatial_count
        spatial_count_df = pd.DataFrame(spatial_count.cpu().detach().numpy(), columns=list(spatial_adata.var_names))
        spatial_hat_df = pd.DataFrame(spatial_hat_np, columns=list(spatial_adata.var_names))
        spatial_adata.layers["spatial_hat"] = pd.DataFrame(
            spatial_hat_np,
            index=list(spatial_adata.obs_names),
            columns=list(spatial_adata.var_names),
        )
        spatial_correlation_gene = spatial_hat_df.corrwith(spatial_count_df / spatial_norm_mat_np).mean()
        print("spatial_correlation_gene", spatial_correlation_gene)
        return sc_adata, spatial_adata



def beta_z_results(scTREND_exp, sc_adata, bulk_adata, driver_genes):
    with torch.no_grad():
        torch.cuda.empty_cache()
        beta_z_np, gamma_z_np_dict = scTREND_exp.export_hazard_coefficients()

        batch_onehot = scTREND_exp.x_data_manager.batch_onehot.to(scTREND_exp.device)
        x = scTREND_exp.x_data_manager.x_count.to(scTREND_exp.device)
        xb = torch.cat([x, batch_onehot], dim=-1)
        _, qz = scTREND_exp.scTREND.enc_z(xb)
        zl = qz.loc.detach().cpu().numpy()

        if beta_z_np.ndim == 1:
            beta_z_np = beta_z_np[:, None]
        sc_adata.obsm["raw_beta_z"] = beta_z_np
        sc_adata.obsm["raw_beta_zl"] = beta_z_np.copy()
        sc_adata.obs["beta_z_mean"] = beta_z_np.mean(1)
        if beta_z_np.shape[1] == 1:
            sc_adata.obs["raw_beta_z_1d"] = beta_z_np[:, 0]
            sc_adata.obs["raw_beta_zl_1d"] = beta_z_np[:, 0]

        if driver_genes is not None:
            for gene in driver_genes:
                if gene not in gamma_z_np_dict:
                    continue
                gamma_np = gamma_z_np_dict[gene]
                if gamma_np.ndim == 1:
                    gamma_np = gamma_np[:, None]
                sc_adata.obsm[f"raw_gamma_{gene}_z"] = gamma_np
                sc_adata.obsm[f"raw_gamma_{gene}_zl"] = gamma_np.copy()
                sc_adata.obs[f"gamma_{gene}_mean"] = gamma_np.mean(1)
                if gamma_np.shape[1] == 1:
                    sc_adata.obs[f"raw_gamma_{gene}_z_1d"] = gamma_np[:, 0]
                    sc_adata.obs[f"raw_gamma_{gene}_zl_1d"] = gamma_np[:, 0]

        sc_adata.obsm["z_sample_avg"] = zl
        return sc_adata, bulk_adata
