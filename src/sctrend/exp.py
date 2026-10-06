import math
from collections import deque
from statistics import mean

import numpy as np
import torch
import torch.distributions as dist

from .dataset import BulkDataManager, ScDataManager, SpatialDataManager
from .concordance import concordance_index_unweighted_dynamic
from .modules_SNV import scTREND


def _to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class EarlyStopping:
    def __init__(self, patience, path, min_delta=0.0, strict_improvement=False):
        self.patience = patience
        self.min_delta = max(0.0, float(min_delta))
        self.strict_improvement = bool(strict_improvement)
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model, epoch=None):
        checkpoint_updated = False
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = -1 if epoch is None else int(epoch)
            self.checkpoint(model)
            checkpoint_updated = True
        elif self.strict_improvement:
            improved = val_loss < (self.best_score - self.min_delta)
            if improved:
                self.best_score = val_loss
                self.best_epoch = -1 if epoch is None else int(epoch)
                self.checkpoint(model)
                self.counter = 0
                checkpoint_updated = True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        elif val_loss > (self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_epoch = -1 if epoch is None else int(epoch)
            self.checkpoint(model)
            self.counter = 0
            checkpoint_updated = True
        return checkpoint_updated

    def checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


class scTRENDExperiment:
    def __init__(
        self,
        model_params,
        x_count,
        bulk_count,
        survival_time,
        cutting_off_0_1,
        x_batch_size,
        checkpoint,
        usePoisson_sc,
        batch_onehot,
        spatial_count,
        use_val_loss_mean,
        driver_genes,
        driver_bulk_adata,
    ):
        print("torch.cuda.is_available()", torch.cuda.is_available())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_onehot = batch_onehot
        self.device = torch.device(device)

        if self.device.type == "cuda":
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = bool(model_params.get("enable_tf32", True))
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = bool(model_params.get("enable_tf32", True))
            if bool(model_params.get("cudnn_benchmark", True)):
                torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                try:
                    torch.set_float32_matmul_precision(str(model_params.get("matmul_precision", "high")))
                except Exception:
                    pass

        self.non_blocking_transfer = bool(model_params.get("non_blocking_transfer", True))
        self.x_data_manager = ScDataManager(
            x_count,
            batch_size=x_batch_size,
            batch_onehot=batch_onehot,
            num_workers=int(model_params.get("sc_num_workers", 0)),
            pin_memory=bool(model_params.get("sc_pin_memory", True)),
            persistent_workers=bool(model_params.get("sc_persistent_workers", False)),
            prefetch_factor=model_params.get("sc_prefetch_factor", 2),
        )
        self.bulk_data_manager = BulkDataManager(
            bulk_count,
            survival_time=survival_time,
            cutting_off_0_1=cutting_off_0_1,
        )
        self.bulk_count = self.bulk_data_manager.bulk_count.to(self.device)
        self.bulk_norm_mat = self.bulk_data_manager.bulk_norm_mat.to(self.device)
        self.cutting_off_0_1 = self.bulk_data_manager.cutting_off_0_1.to(self.device)
        self.survival_time = self.bulk_data_manager.survival_time.to(self.device)
        self.spatial_count = spatial_count
        self.driver_genes = list(driver_genes) if driver_genes is not None else []
        self.driver_bulk_adata = driver_bulk_adata
        self.latest_c_index = {}
        self.hazard_early_stopping_metric = str(
            model_params.get("hazard_early_stopping_metric", "val_nll")
        ).strip().lower()
        if self.hazard_early_stopping_metric not in {"val_nll", "val_c_index"}:
            raise ValueError(
                "hazard_early_stopping_metric must be either 'val_nll' or 'val_c_index'."
            )
        self.hazard_early_stopping_min_delta = max(
            0.0,
            _to_float(model_params.get("hazard_early_stopping_min_delta", 0.0), 0.0),
        )
        self.hazard_early_stopping_history = []
        self.hazard_early_stopping_best_metric = float("nan")
        self.hazard_early_stopping_best_epoch = -1
        self.hazard_early_stopping_stop_epoch = -1
        self.hazard_early_stopping_best_val_nll = float("nan")
        self.hazard_early_stopping_best_val_c_index = float("nan")
        self.edges = torch.tensor(model_params["edges"], dtype=torch.float32, device=self.device)
        self.delta = torch.diff(self.edges)

        if driver_bulk_adata is not None:
            temp_SNV = driver_bulk_adata.layers["SNV"]
            if hasattr(temp_SNV, "toarray"):
                temp_SNV = temp_SNV.toarray()
            temp_SNV = np.array(temp_SNV).astype(float)
            self.driver_bulk_SNV = torch.tensor(temp_SNV, device=self.device)
            self.driver_bulk_SNV_dict = {
                gene: self.driver_bulk_SNV[:, i]
                for i, gene in enumerate(driver_bulk_adata.var_names)
            }
        else:
            self.driver_bulk_SNV = None
            self.driver_bulk_SNV_dict = {}

        self.model_params = model_params
        self.l2_beta = max(0.0, _to_float(model_params.get("l2_beta", 0.0), 0.0))
        self.l2_gamma = max(0.0, _to_float(model_params.get("l2_gamma", 0.0), 0.0))
        self.optimizer_weight_decay = max(
            0.0,
            _to_float(model_params.get("optimizer_weight_decay", 0.0), 0.0),
        )
        self.beta_fit_group = str(model_params.get("beta_fit_group", "all"))
        self.gamma_fit_group = str(model_params.get("gamma_fit_group", "mutated_only"))
        self.use_deterministic_z_in_hazard = bool(model_params.get("use_deterministic_z_in_hazard", True))
        self.hazard_full_batch = bool(model_params.get("hazard_full_batch", False))
        self.hazard_penalty = str(model_params.get("hazard_penalty", "posterior_is_clipped"))
        if self.hazard_penalty != "posterior_is_clipped":
            raise ValueError("This fork supports only hazard_penalty='posterior_is_clipped'.")
        self.posterior_is_epsilon = float(model_params.get("posterior_is_epsilon", 0.01))
        if not math.isfinite(self.posterior_is_epsilon) or self.posterior_is_epsilon <= 0:
            raise ValueError("posterior_is_epsilon must be finite and positive.")
        self.posterior_is_samples_per_cell = 1
        if model_params.get("posterior_is_samples_per_cell", 1) != 1:
            raise ValueError("Exactly one posterior sample per cell is supported.")
        obsolete = {"penalty_type_beta", "penalty_type_gamma", "functional_penalty_samples"}
        if obsolete.intersection(model_params):
            raise ValueError("Obsolete penalty settings are not accepted in this fork.")
        default_use_gamma = len(self.driver_genes) > 0
        self.use_gamma = bool(model_params.get("use_gamma", default_use_gamma)) and default_use_gamma

        scTREND_kwargs = {
            k: v
            for k, v in model_params.items()
            if k
            in {
                "x_dim",
                "z_dim",
                "h_dim",
                "num_enc_z_layers",
                "num_dec_z_layers",
                "num_dec_p_layers",
                "num_dec_b_layers",
                "num_time_bins",
            }
        }

        if spatial_count is not None:
            self.spatial_data_manager = SpatialDataManager(spatial_count)
            self.spatial_count = self.spatial_data_manager.spatial_count.to(self.device)
            self.spatial_norm_mat = self.spatial_data_manager.spatial_norm_mat.to(self.device)
            spatial_num = self.spatial_data_manager.spatial_count.shape[0]
        else:
            self.spatial_data_manager = None
            self.spatial_norm_mat = None
            spatial_num = 0

        self.scTREND = scTREND(
            bulk_num=self.bulk_data_manager.bulk_count.shape[0],
            spatial_num=spatial_num,
            batch_onehot_dim=batch_onehot.shape[1],
            driver_genes=self.driver_genes,
            **scTREND_kwargs,
        )
        self.scTREND.to(self.device)

        self.checkpoint = checkpoint
        self.usePoisson_sc = usePoisson_sc
        self.epoch = 0
        self.use_val_loss_mean = use_val_loss_mean
        self.bulk_test_num_or_ratio = None
        self.bulk_validation_num_or_ratio = None

        self.active_train_idx = None
        self.active_validation_idx = None
        self.active_test_idx = None
        self.hazard_cache = None
        self.pre_hazard_state_dict = None
        self.beta_stage_metrics = {}
        self.gamma_stage_metrics = {}
        self.hazard_metrics = {}
        self.beta_best_epoch = -1
        self.gamma_best_epoch = -1
        self.beta_stop_epoch = -1
        self.gamma_stop_epoch = -1
        self.hazard_forward_deterministic_override = None

    def _to_device(self, x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(self.device, non_blocking=self.non_blocking_transfer)
        return x

    def bulk_data_split(self, n_bulk_split, validation_num, test_num, censor_np, snv_flag, *, edges=None):
        self.bulk_test_num_or_ratio = test_num
        self.bulk_validation_num_or_ratio = validation_num
        if edges is None:
            edges = self.edges
        self.bulk_data_manager.bulk_split(
            n_bulk_split,
            validation_num,
            test_num,
            censor_np,
            snv_flag,
            edges=edges,
        )
        self.set_active_bulk_indices(None, None, None)

    def set_l2_penalty(self, l2_beta=None, l2_gamma=None):
        if l2_beta is not None:
            self.l2_beta = max(0.0, float(l2_beta))
            self.model_params["l2_beta"] = self.l2_beta
        if l2_gamma is not None:
            self.l2_gamma = max(0.0, float(l2_gamma))
            self.model_params["l2_gamma"] = self.l2_gamma
            
    def set_gamma_usage(self, use_gamma=None):
        if use_gamma is None:
            return
        default_use_gamma = len(self.driver_genes) > 0
        self.use_gamma = bool(use_gamma) and default_use_gamma
        self.model_params["use_gamma"] = self.use_gamma
        self.clear_hazard_cache()

    def clear_hazard_cache(self):
        self.hazard_cache = None

    def set_active_bulk_indices(self, train_idx=None, validation_idx=None, test_idx=None):
        self.active_train_idx = self._coerce_long_cpu_tensor(train_idx)
        self.active_validation_idx = self._coerce_long_cpu_tensor(validation_idx)
        self.active_test_idx = self._coerce_long_cpu_tensor(test_idx)

    def _coerce_long_cpu_tensor(self, idx):
        if idx is None:
            return None
        if isinstance(idx, torch.Tensor):
            return idx.detach().cpu().long().clone()
        return torch.as_tensor(idx, dtype=torch.long)

    def _resolve_active_indices(self, mode):
        mode = str(mode)
        if mode in {"train", "training"}:
            idx = self.active_train_idx
            if idx is None:
                idx = self.bulk_data_manager.train_idx
            return idx
        if mode in {"validation", "val"}:
            idx = self.active_validation_idx
            if idx is None:
                idx = self.bulk_data_manager.validation_idx
            return idx
        if mode == "test":
            idx = self.active_test_idx
            if idx is None:
                idx = self.bulk_data_manager.test_idx
            return idx
        raise ValueError(f"Unknown mode: {mode}")

    def _reference_bulk_mask(self):
        n_bulk = int(self.bulk_count.shape[0])
        if len(self.driver_genes) == 0 or self.driver_bulk_SNV is None:
            return torch.ones(n_bulk, dtype=torch.bool)

        mask = torch.ones(n_bulk, dtype=torch.bool, device=self.device)
        for gene in self.driver_genes:
            if gene not in self.driver_bulk_SNV_dict:
                raise KeyError(f"{gene} not found in driver_bulk_SNV_dict")
            mask = mask & (self.driver_bulk_SNV_dict[gene].float() <= 0.0)
        return mask.detach().cpu()

    def _mutated_bulk_mask(self):
        n_bulk = int(self.bulk_count.shape[0])
        if len(self.driver_genes) == 0 or self.driver_bulk_SNV is None:
            return torch.zeros(n_bulk, dtype=torch.bool)

        mask = torch.zeros(n_bulk, dtype=torch.bool, device=self.device)
        for gene in self.driver_genes:
            if gene not in self.driver_bulk_SNV_dict:
                raise KeyError(f"{gene} not found in driver_bulk_SNV_dict")
            mask = mask | (self.driver_bulk_SNV_dict[gene].float() > 0.0)
        return mask.detach().cpu()

    def _restrict_indices_to_reference(self, idx):
        idx = self._coerce_long_cpu_tensor(idx)
        if idx is None:
            return None
        if idx.numel() == 0:
            return idx
        ref_mask = self._reference_bulk_mask()
        keep = ref_mask[idx]
        return idx[keep]

    def _restrict_indices_to_mutated(self, idx):
        idx = self._coerce_long_cpu_tensor(idx)
        if idx is None:
            return None
        if idx.numel() == 0:
            return idx
        mut_mask = self._mutated_bulk_mask()
        keep = mut_mask[idx]
        return idx[keep]

    def activate_beta_stage_indices(self, beta_fit_group=None):
        if beta_fit_group is None:
            beta_fit_group = self.beta_fit_group

        train_idx = self.bulk_data_manager.train_idx
        val_idx = self.bulk_data_manager.validation_idx
        test_idx = self.bulk_data_manager.test_idx

        if beta_fit_group == "reference_only":
            train_idx = self._restrict_indices_to_reference(train_idx)
            val_idx = self._restrict_indices_to_reference(val_idx)
            test_idx = self._restrict_indices_to_reference(test_idx)
        elif beta_fit_group == "all":
            pass
        else:
            raise ValueError("beta_fit_group must be either 'reference_only' or 'all'.")

        if train_idx is None or train_idx.numel() == 0:
            raise ValueError(
                "No training samples are available for beta-stage fitting. "
                "If your reference/control group is too small, either change the split seed or use beta_fit_group='all'."
            )
        if val_idx is None or val_idx.numel() == 0:
            raise ValueError(
                "No validation samples are available for beta-stage fitting. "
                "If your reference/control group is too small, either change the split seed or use beta_fit_group='all'."
            )

        self.set_active_bulk_indices(train_idx, val_idx, test_idx)

    def activate_gamma_stage_indices(self, gamma_fit_group=None):
        if gamma_fit_group is None:
            gamma_fit_group = self.gamma_fit_group

        train_idx = self.bulk_data_manager.train_idx
        val_idx = self.bulk_data_manager.validation_idx
        test_idx = self.bulk_data_manager.test_idx

        if gamma_fit_group in {"mutated_only", "any_mutated_only"}:
            train_idx = self._restrict_indices_to_mutated(train_idx)
            val_idx = self._restrict_indices_to_mutated(val_idx)
            test_idx = self._restrict_indices_to_mutated(test_idx)
        elif gamma_fit_group == "all":
            pass
        else:
            raise ValueError("gamma_fit_group must be either 'mutated_only' or 'all'.")

        if train_idx is None or train_idx.numel() == 0:
            raise ValueError(
                "No mutated training samples are available for gamma-stage fitting. "
                "Use a different split seed or gamma_fit_group='all' if you intentionally want that behavior."
            )
        if val_idx is None or val_idx.numel() == 0:
            print("[gamma stage] WARNING: no validation samples matched gamma_fit_group; using the gamma-stage training indices for validation.")
            val_idx = train_idx.clone()

        self.set_active_bulk_indices(train_idx, val_idx, test_idx)

    def _sample_latent_from_q(self, qz, seed=None):
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            eps = torch.randn(qz.loc.shape, device=self.device, generator=generator)
        else:
            eps = torch.randn(qz.loc.shape, device=self.device)
        return qz.loc + qz.scale * eps

    def _hazard_forward_deterministic(self):
        if self.hazard_forward_deterministic_override is not None:
            return bool(self.hazard_forward_deterministic_override)
        return bool(self.use_deterministic_z_in_hazard)

    def prepare_hazard_cache(self, force=False, sample_z=None, seed=None):
        if (self.hazard_cache is not None) and (not force) and (sample_z is None):
            return self.hazard_cache

        if sample_z is None:
            sample_z = not self.use_deterministic_z_in_hazard

        was_training = self.scTREND.training
        self.scTREND.eval()
        with torch.no_grad():
            x = self.x_data_manager.x_count.to(self.device)
            batch_onehot = self.x_data_manager.batch_onehot.to(self.device)
            xb = torch.cat([x, batch_onehot], dim=-1)
            _, qz = self.scTREND.enc_z(xb)
            if sample_z:
                z_fixed = self._sample_latent_from_q(qz, seed=seed)
            else:
                z_fixed = qz.loc
            p_bulk_fixed = self.scTREND.dec_z2p_bulk(z_fixed)

            cache = {
                "z_fixed": z_fixed.detach(),
                "p_bulk_fixed": p_bulk_fixed.detach(),
                "sample_z": bool(sample_z),
                "seed": None if seed is None else int(seed),
            }
            if self.scTREND.spatial_num > 0:
                cache["p_spatial_fixed"] = self.scTREND.dec_z2p_spatial(z_fixed).detach()
            self.hazard_cache = cache

        if was_training:
            self.scTREND.train()
        return self.hazard_cache

    def _get_hazard_decoder_outputs(self):
        if not self.scTREND.training and (
            self.hazard_cache is None or self.hazard_cache.get("sample_z", False)
        ):
            cache = self.prepare_hazard_cache(force=True, sample_z=False)
        else:
            cache = self.prepare_hazard_cache(force=False)
        z_fixed = cache["z_fixed"]
        p_bulk_fixed = cache["p_bulk_fixed"]

        beta_z_all = self.scTREND.dec_beta_z(z_fixed)
        if beta_z_all.dim() == 1:
            beta_z_all = beta_z_all.unsqueeze(1)

        if (not getattr(self, "use_gamma", True)) or len(self.scTREND.dec_gammas) == 0:
            gamma_z_dict = None
        else:
            gamma_z_dict = {}
            for key, decoder in self.scTREND.dec_gammas.items():
                gene = key.replace("dec_gamma_", "").replace("_z", "")
                gamma_val = decoder(z_fixed)
                if gamma_val.dim() == 1:
                    gamma_val = gamma_val.unsqueeze(1)
                gamma_z_dict[gene] = gamma_val

        return z_fixed, p_bulk_fixed, beta_z_all, gamma_z_dict

    def _posterior_is_penalty(self, qz):
        """Cell-mean f(z)/max(q_cell(z), epsilon), one fresh draw per cell.

        f is the time-bin mean of squared decoder outputs. The diagonal-normal
        joint density is that of the SAME cell that generated z; no mixture,
        prior-density numerator, self-normalization, or epoch divisor is used.
        """
        mode = self.scTREND.mode
        beta_active = mode in {"beta_z", "joint_z"} and self.l2_beta > 0.0
        gamma_active = (
            mode in {"gamma_z", "joint_z"}
            and self.use_gamma
            and self.l2_gamma > 0.0
            and len(self.scTREND.dec_gammas) > 0
        )
        zero = torch.zeros((), dtype=torch.float64, device=self.device)
        if not (beta_active or gamma_active):
            return zero
        if qz is None:
            raise ValueError("Posterior IS penalty requires the current cell batch's qz.")
        if qz.loc.ndim != 2 or qz.loc.shape[0] == 0:
            raise ValueError("qz must be a nonempty cell-by-latent diagonal Normal.")
        with torch.no_grad():
            z_sample = qz.sample().detach()
            q64 = dist.Normal(qz.loc.detach().double(), qz.scale.detach().double())
            log_q = q64.log_prob(z_sample.double()).sum(dim=-1)
            log_denominator = log_q.clamp_min(math.log(self.posterior_is_epsilon))
            weights = torch.exp(-log_denominator)
            if not torch.isfinite(weights).all():
                raise FloatingPointError("Non-finite posterior IS weights.")

        def weighted_square(decoder):
            parameter = next(decoder.parameters())
            output = decoder(z_sample.to(device=parameter.device, dtype=parameter.dtype))
            if output.ndim == 1:
                output = output.unsqueeze(-1)
            # Cast BEFORE squaring: tiny high-dimensional inverse densities must
            # not be prematurely rounded to zero by float32 arithmetic.
            square_per_cell = output.double().square().mean(dim=-1)
            return (weights * square_per_cell).mean()

        penalty = zero
        if beta_active:
            penalty = penalty + self.l2_beta * weighted_square(self.scTREND.dec_beta_z)
        if gamma_active:
            gamma_terms = [weighted_square(decoder) for decoder in self.scTREND.dec_gammas.values()]
            penalty = penalty + self.l2_gamma * torch.stack(gamma_terms).mean()
        return penalty

    def elbo_loss(self, x, xnorm_mat, bulk_count, bulk_norm_mat, spatial_count, spatial_norm_mat, batch_onehot, bulk_idx):
        if self.scTREND.mode in {"sc", "bulk", "spatial"}:
            (
                z,
                qz,
                x_hat,
                p_bulk,
                p_spatial,
                bulk_hat,
                spatial_hat,
                theta_x,
                theta_bulk,
                theta_spatial,
                beta_z,
                gamma_z_dict,
            ) = self.scTREND(x, batch_onehot, gene_name=None)

            if self.scTREND.mode == "sc":
                elbo_loss = self.calc_kld(qz).sum()
                if self.usePoisson_sc:
                    elbo_loss += self.calc_poisson_loss(ld=x_hat, norm_mat=xnorm_mat, obs=x).sum()
                else:
                    elbo_loss += self.calc_nb_loss(x_hat, xnorm_mat, theta_x, x).sum()
                return elbo_loss

            if self.scTREND.mode == "bulk":
                return self.calc_nb_loss(bulk_hat, bulk_norm_mat, theta_bulk, bulk_count).sum()

            return self.calc_nb_loss(spatial_hat, spatial_norm_mat, theta_spatial, spatial_count).sum()

        if self.scTREND.mode not in {"beta_z", "gamma_z", "joint_z"}:
            raise ValueError(f"Unsupported mode: {self.scTREND.mode}")

        bulk_idx = self._coerce_long_cpu_tensor(bulk_idx)
        if bulk_idx is None or bulk_idx.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        bulk_idx_dev = bulk_idx.to(self.device)
        qz = None
        if (x is not None) and (batch_onehot is not None):
            (
                _,
                qz,
                _,
                p_bulk_all,
                _,
                _,
                _,
                _,
                _,
                _,
                beta_z_all,
                gamma_all_dict,
            ) = self.scTREND(
                x,
                batch_onehot,
                gene_name=None,
                deterministic_z=self._hazard_forward_deterministic(),
            )
            p_bulk_sel = p_bulk_all[bulk_idx_dev]
        else:
            _, p_bulk_all, beta_z_all, gamma_all_dict = self._get_hazard_decoder_outputs()
            if not getattr(self, "use_gamma", True):
                gamma_all_dict = None
            p_bulk_sel = p_bulk_all[bulk_idx_dev]

        event_obs = self.cutting_off_0_1[bulk_idx_dev].float()

        gamma_use_dict = None
        T_driver_sel = None
        if self.scTREND.mode in {"gamma_z", "joint_z"} and gamma_all_dict is not None:
            gamma_use_dict = gamma_all_dict
            T_driver_sel = {
                gene: self.driver_bulk_SNV_dict[gene][bulk_idx_dev].float()
                for gene in gamma_use_dict.keys()
            }

        neg_log_like = self._piecewise_const_loss(
            beta_z_all,
            p_bulk_sel,
            self.survival_time[bulk_idx_dev],
            event_obs,
            gamma_z_all_dict=gamma_use_dict,
            T_driver_dict=T_driver_sel,
        )
        elbo_loss = neg_log_like

        if self.scTREND.training:
            elbo_loss = elbo_loss + self._posterior_is_penalty(qz)

        if not self.scTREND.training:
            lam_tbl = self.predict_lambda_table(bulk_idx)
            surv_np = self.survival_time[bulk_idx_dev].cpu().numpy()
            event_np = event_obs.cpu().numpy()
            edges_np = self.edges.cpu().numpy()
            if lam_tbl is not None and len(lam_tbl) > 0:
                try:
                    c_index, _ = concordance_index_unweighted_dynamic(
                        event_times=surv_np,
                        event_observed=event_np,
                        lam=lam_tbl,
                        times=edges_np,
                    )
                    self.latest_c_index[self.scTREND.mode] = c_index
                except Exception:
                    pass
        return elbo_loss

    def train_epoch(self):
        self.scTREND.train()

        if self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}:
            bulk_idx = self._resolve_active_indices("train")
            if bulk_idx is None or bulk_idx.numel() == 0:
                raise ValueError("Active training bulk indices are empty.")

            if self.hazard_full_batch:
                self.scTREND_optimizer.zero_grad()
                loss = self.elbo_loss(
                    None,
                    None,
                    self.bulk_count,
                    self.bulk_norm_mat,
                    self.spatial_count,
                    self.spatial_norm_mat,
                    None,
                    bulk_idx,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.scTREND.parameters(), max_norm=5.0)
                self.scTREND_optimizer.step()
                return float(loss.item() if torch.is_tensor(loss) else loss)

            if not hasattr(self.x_data_manager, "hazard_loader"):
                self.x_data_manager.initialize_hazard_loader(1000)
            total_loss = 0.0
            update_num = 0
            for x, xnorm_mat, batch_onehot in self.x_data_manager.hazard_loader:
                x = self._to_device(x)
                xnorm_mat = self._to_device(xnorm_mat)
                batch_onehot = self._to_device(batch_onehot)
                self.scTREND_optimizer.zero_grad()
                loss = self.elbo_loss(
                    x,
                    xnorm_mat,
                    self.bulk_count,
                    self.bulk_norm_mat,
                    self.spatial_count,
                    self.spatial_norm_mat,
                    batch_onehot,
                    bulk_idx,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.scTREND.parameters(), max_norm=5.0)
                self.scTREND_optimizer.step()
                update_num += 1
                total_loss += loss.item() if torch.is_tensor(loss) else loss

            # Each loss is already a bulk-patient mean plus a cell-mean penalty.
            # Report the mean optimizer-step objective, not objective / cells.
            loss_val = total_loss / max(update_num, 1)
            return float(loss_val)

        total_loss = 0.0
        entry_num = 0
        for x, xnorm_mat, batch_onehot in self.x_data_manager.train_loader:
            x = self._to_device(x)
            xnorm_mat = self._to_device(xnorm_mat)
            batch_onehot = self._to_device(batch_onehot)
            self.scTREND_optimizer.zero_grad()
            bulk_idx = self.bulk_data_manager.train_idx.to(self.device)
            loss = self.elbo_loss(
                x,
                xnorm_mat,
                self.bulk_count,
                self.bulk_norm_mat,
                self.spatial_count,
                self.spatial_norm_mat,
                batch_onehot,
                bulk_idx,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scTREND.parameters(), max_norm=5.0)
            self.scTREND_optimizer.step()
            entry_num += x.shape[0]
            total_loss += loss.item() if torch.is_tensor(loss) else loss

        loss_val = total_loss / max(entry_num, 1)
        return float(loss_val)

    def evaluate(self, mode="test"):
        with torch.no_grad():
            self.scTREND.eval()

            if self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}:
                bulk_idx = self._resolve_active_indices(mode)
                if bulk_idx is None or bulk_idx.numel() == 0:
                    return float("nan")

                # Evaluate every patient split against the SAME full reference.
                loss = self.elbo_loss(
                    None,
                    None,
                    self.bulk_count,
                    self.bulk_norm_mat,
                    self.spatial_count,
                    self.spatial_norm_mat,
                    None,
                    bulk_idx,
                )
                return loss.item() if torch.is_tensor(loss) else float(loss)

            if mode == "test":
                x = self._to_device(self.x_data_manager.test_x)
                xnorm_mat = self._to_device(self.x_data_manager.test_xnorm_mat)
                batch_onehot = self._to_device(self.x_data_manager.test_batch_onehot)
                bulk_idx = self.bulk_data_manager.test_idx.to(self.device)
            else:
                x = self._to_device(self.x_data_manager.validation_x)
                xnorm_mat = self._to_device(self.x_data_manager.validation_xnorm_mat)
                batch_onehot = self._to_device(self.x_data_manager.validation_batch_onehot)
                bulk_idx = self.bulk_data_manager.validation_idx.to(self.device)

            loss = self.elbo_loss(
                x,
                xnorm_mat,
                self.bulk_count,
                self.bulk_norm_mat,
                self.spatial_count,
                self.spatial_norm_mat,
                batch_onehot,
                bulk_idx,
            )
            entry_num = max(int(x.shape[0]), 1)
            loss_val = loss / entry_num
            return loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)

    def evaluate_train(self):
        with torch.no_grad():
            self.scTREND.eval()

            if self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}:
                bulk_idx = self._resolve_active_indices("train")
                if bulk_idx is None or bulk_idx.numel() == 0:
                    return float("nan")

                loss = self.elbo_loss(
                    None,
                    None,
                    self.bulk_count,
                    self.bulk_norm_mat,
                    self.spatial_count,
                    self.spatial_norm_mat,
                    None,
                    bulk_idx,
                )
                return loss.item() if torch.is_tensor(loss) else float(loss)

            x = self._to_device(self.x_data_manager.train_x)
            xnorm_mat = self._to_device(self.x_data_manager.train_xnorm_mat)
            batch_onehot = self._to_device(self.x_data_manager.train_batch_onehot)
            bulk_idx = self.bulk_data_manager.train_idx.to(self.device)
            loss = self.elbo_loss(
                x,
                xnorm_mat,
                self.bulk_count,
                self.bulk_norm_mat,
                self.spatial_count,
                self.spatial_norm_mat,
                batch_onehot,
                bulk_idx,
            )
            entry_num = max(int(x.shape[0]), 1)
            loss_val = loss / entry_num
            return loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)

    def train_total(self, epoch_num, patience):
        hazard_mode = self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}
        early_stopping_metric = (
            self.hazard_early_stopping_metric if hazard_mode else "val_nll"
        )
        use_c_index_for_early_stopping = (
            hazard_mode and early_stopping_metric == "val_c_index"
        )
        earlystopping = EarlyStopping(
            patience=patience,
            path=self.checkpoint,
            min_delta=(
                self.hazard_early_stopping_min_delta
                if use_c_index_for_early_stopping
                else 0.0
            ),
            # The unweighted dynamic C-index is stepwise and often tied across
            # epochs. A tie must not reset patience, otherwise a one-bin fit can
            # run indefinitely.
            strict_improvement=use_c_index_for_early_stopping,
        )
        val_loss_list = deque(maxlen=patience)
        history = []
        last_epoch = -1

        for epoch in range(epoch_num):
            loss = self.train_epoch()
            if hazard_mode:
                # evaluate() stores the C-index for the active validation split
                # under the current hazard-stage key. Remove stale values first
                # so a C-index failure cannot silently reuse the previous epoch.
                self.latest_c_index.pop(self.scTREND.mode, None)
            val_loss = self.evaluate(mode="validation")
            val_c_index = (
                float(self.latest_c_index.get(self.scTREND.mode, float("nan")))
                if hazard_mode
                else float("nan")
            )

            if use_c_index_for_early_stopping:
                if not np.isfinite(val_c_index):
                    raise RuntimeError(
                        "Validation unweighted dynamic C-index is not finite; "
                        "refusing to fall back silently to NLL early stopping. "
                        "Check validation events/times."
                    )
                # EarlyStopping minimizes its score, so negate C-index here.
                monitor_value = -val_c_index
                monitor_display_value = val_c_index
                monitor_name = "val_unweighted_dynamic_c_index"
            else:
                val_loss_list.append(val_loss)
                val_loss_mean = mean(val_loss_list)
                monitor_value = val_loss_mean if self.use_val_loss_mean else val_loss
                monitor_display_value = monitor_value
                monitor_name = (
                    "val_nll_rolling_mean" if self.use_val_loss_mean else "val_nll"
                )

            checkpoint_updated = earlystopping(
                monitor_value,
                self.scTREND,
                epoch=epoch,
            )
            history.append({
                "epoch": int(epoch),
                "train_objective": float(loss),
                "val_nll": float(val_loss),
                "val_unweighted_dynamic_c_index": float(val_c_index),
                "early_stopping_metric": monitor_name,
                "early_stopping_value": float(monitor_display_value),
                "checkpoint_updated": bool(checkpoint_updated),
                "patience_counter": int(earlystopping.counter),
            })

            last_epoch = epoch
            if epoch % 10 == 0 or epoch == epoch_num - 1:
                message = (
                    f"[{self.scTREND.mode}] epoch {epoch}: "
                    f"train loss {loss:.6f} validation NLL {val_loss:.6f}"
                )
                if hazard_mode:
                    message += (
                        " validation unweighted dynamic C-index "
                        f"{val_c_index:.6f}"
                    )
                message += f" | early stop={monitor_name}"
                print(message)

            if earlystopping.early_stop:
                best_display_value = (
                    -float(earlystopping.best_score)
                    if use_c_index_for_early_stopping
                    else float(earlystopping.best_score)
                )
                print(
                    f"Early Stopping! at {epoch} epoch, "
                    f"metric={monitor_name}, best value={best_display_value}, "
                    f"best_epoch={earlystopping.best_epoch}"
                )
                break
            if math.isnan(loss):
                print("loss is nan")
                break

        best_epoch = earlystopping.best_epoch if earlystopping.best_epoch >= 0 else last_epoch
        if hazard_mode:
            best_row = next(
                (row for row in history if row["epoch"] == int(best_epoch)),
                None,
            )
            self.hazard_early_stopping_history = history
            self.hazard_early_stopping_best_epoch = int(best_epoch)
            self.hazard_early_stopping_stop_epoch = int(last_epoch)
            self.hazard_early_stopping_best_metric = (
                -float(earlystopping.best_score)
                if use_c_index_for_early_stopping
                else float(earlystopping.best_score)
            )
            if best_row is not None:
                self.hazard_early_stopping_best_val_nll = float(best_row["val_nll"])
                self.hazard_early_stopping_best_val_c_index = float(
                    best_row["val_unweighted_dynamic_c_index"]
                )
        return best_epoch

    def initialize_optimizer(self, lr):
        hazard_stage = self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}
        self.scTREND_optimizer = torch.optim.AdamW(
            self.scTREND.parameters(),
            lr=lr,
            weight_decay=0.0 if hazard_stage else self.optimizer_weight_decay,
        )

    def initialize_loader(self, x_batch_size):
        if self.scTREND.mode in {"beta_z", "gamma_z", "joint_z"}:
            self.x_data_manager.initialize_hazard_loader(x_batch_size)
        else:
            self.x_data_manager.initialize_loader(x_batch_size)

    def calc_kld(self, qz):
        kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        return kld

    def calc_nb_loss(self, ld, norm_mat, theta, obs):
        ld = norm_mat * ld
        ld = ld + 1.0e-10
        theta = theta + 1.0e-10
        lp = ld.log() - theta.log()
        p_z = dist.NegativeBinomial(theta, logits=lp)
        l = -p_z.log_prob(obs)
        return l

    def calc_poisson_loss(self, ld, norm_mat, obs):
        p_z = dist.Poisson(ld * norm_mat + 1.0e-10)
        l = -p_z.log_prob(obs)
        return l

    def _piecewise_const_loss(
        self,
        beta_z_all,
        p_bulk,
        T_b,
        delta_b,
        *,
        gamma_z_all_dict=None,
        T_driver_dict=None,
    ):
        device = beta_z_all.device
        B, K = p_bulk.size(0), beta_z_all.size(1)
        edges = self.edges.to(device)
        delta_k = self.delta.to(device)

        eta_bk = torch.matmul(p_bulk, beta_z_all)
        if gamma_z_all_dict is not None and T_driver_dict is not None:
            for gene, gamma_c_k in gamma_z_all_dict.items():
                indic_b = T_driver_dict[gene].to(device)
                eta_bk = eta_bk + indic_b.unsqueeze(1) * torch.matmul(p_bulk, gamma_c_k)

        lam_bk = torch.nn.functional.softplus(eta_bk).clamp(max=1e4)
        cum_H = torch.cumsum(lam_bk * delta_k, dim=1)

        k_star = torch.bucketize(T_b, edges[1:-1], right=True)
        k_star = torch.clamp(k_star, max=K - 1)

        row = torch.arange(B, device=device)
        lam_k = lam_bk[row, k_star]
        H_prev = torch.zeros_like(T_b)
        mask = k_star > 0
        H_prev[mask] = cum_H[row[mask], k_star[mask] - 1]

        t_prev = edges[k_star]
        diff = (T_b - t_prev).clamp(min=0)
        H_T = H_prev + lam_k * diff

        eps = 1e-8
        nll = torch.mean(H_T - delta_b * torch.log(lam_k + eps))
        return nll

    def get_beta_per_bin(self, x, batch_onehot):
        with torch.no_grad():
            x = x.to(self.device)
            boh = batch_onehot.to(self.device)
            z, _ = self.scTREND.enc_z(torch.cat([x, boh], dim=-1))
            beta_z_all = self.scTREND.dec_beta_z(z)
        return beta_z_all.cpu()

    def predict_lambda_table(self, bulk_idx):
        bulk_idx = self._coerce_long_cpu_tensor(bulk_idx)
        if bulk_idx is None or bulk_idx.numel() == 0:
            return None

        self.scTREND.eval()
        with torch.no_grad():
            _, p_bulk_all, beta_z_all, gamma_z_dict = self._get_hazard_decoder_outputs()
            bulk_idx_dev = bulk_idx.to(self.device)
            p_bulk = p_bulk_all[bulk_idx_dev]
            eta_bk = p_bulk @ beta_z_all

            if gamma_z_dict is not None and len(gamma_z_dict) > 0:
                for g, gamma_ck in gamma_z_dict.items():
                    indic = self.driver_bulk_SNV_dict[g][bulk_idx_dev].float()
                    eta_bk = eta_bk + indic.unsqueeze(1) * (p_bulk @ gamma_ck)

            lam_bk = torch.nn.functional.softplus(eta_bk)
            return lam_bk.cpu().numpy()

    def export_hazard_coefficients(self):
        self.scTREND.eval()
        with torch.no_grad():
            _, _, beta_z_all, gamma_z_dict = self._get_hazard_decoder_outputs()
            beta_np = beta_z_all.detach().cpu().numpy()
            if gamma_z_dict is None:
                gamma_np_dict = {}
            else:
                gamma_np_dict = {
                    gene: gamma_val.detach().cpu().numpy()
                    for gene, gamma_val in gamma_z_dict.items()
                }
        return beta_np, gamma_np_dict
