import torch
import numpy as np
from .modules_SNV import scTREND
from .dataset import ScDataManager, BulkDataManager, SpatialDataManager
from statistics import mean
import torch.distributions as dist
import math
from collections import deque
from .ipcw import concordance_index_ipcw

class EarlyStopping:
    def __init__(self, patience, path):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.checkpoint(model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.checkpoint(model)
            self.counter = 0

    def checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class scTRENDExperiment:
    def __init__(self, model_params, x_count, bulk_count, survival_time, cutting_off_0_1, x_batch_size, checkpoint, usePoisson_sc, 
                batch_onehot, spatial_count, use_val_loss_mean, driver_genes, driver_bulk_adata):
        print('torch.cuda.is_available()', torch.cuda.is_available())
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_onehot = batch_onehot
        self.device = torch.device(device)
        self.x_data_manager = ScDataManager(x_count, batch_size=x_batch_size, batch_onehot=batch_onehot)
        self.bulk_data_manager = BulkDataManager(bulk_count, survival_time = survival_time, cutting_off_0_1=cutting_off_0_1)
        self.bulk_count = self.bulk_data_manager.bulk_count.to(self.device)
        self.bulk_norm_mat = self.bulk_data_manager.bulk_norm_mat.to(self.device)
        self.cutting_off_0_1 = self.bulk_data_manager.cutting_off_0_1.to(self.device)
        self.survival_time = self.bulk_data_manager.survival_time.to(self.device)
        self.spatial_count = spatial_count
        self.driver_genes = driver_genes
        self.driver_bulk_adata = driver_bulk_adata
        self.latest_c_index = {}
        self.edges  = torch.tensor(model_params["edges"], dtype=torch.float32, device=self.device)
        self.delta  = torch.diff(self.edges)
        if driver_bulk_adata is not None:
            temp_SNV = driver_bulk_adata.layers["SNV"]
            if hasattr(temp_SNV, "toarray"):
                temp_SNV = temp_SNV.toarray()
            temp_SNV = np.array(temp_SNV).astype(float)
            self.driver_bulk_SNV = torch.tensor(temp_SNV, device=self.device)
            self.driver_bulk_SNV_dict = { gene: self.driver_bulk_SNV[:, i] 
                                        for i, gene in enumerate(driver_bulk_adata.var_names) }
        self.model_params = model_params
        
        scTREND_kwargs = {k: v for k, v in model_params.items() if k in {
            "x_dim", "z_dim", "h_dim",
            "num_enc_z_layers", "num_dec_z_layers",
            "num_dec_p_layers", "num_dec_b_layers",
            "num_time_bins"
        }}
        
        if spatial_count is not None:
            self.spatial_data_manager = SpatialDataManager(spatial_count)
            self.spatial_count = self.spatial_data_manager.spatial_count.to(self.device)
            self.spatial_norm_mat = self.spatial_data_manager.spatial_norm_mat.to(self.device)
            spatial_num = self.spatial_data_manager.spatial_count.shape[0]    
        else:
            self.spatial_norm_mat = None
            spatial_num = 0
            
        self.scTREND = scTREND(
            bulk_num = self.bulk_data_manager.bulk_count.shape[0],
            spatial_num = spatial_num,
            batch_onehot_dim = batch_onehot.shape[1],
            driver_genes = self.driver_genes,
            **scTREND_kwargs
        )
        self.scTREND.to(self.device)
        self.checkpoint=checkpoint
        self.usePoisson_sc = usePoisson_sc
        self.epoch = 0
        self.use_val_loss_mean = use_val_loss_mean
        self.bulk_test_num_or_ratio = None
        self.bulk_validation_num_or_ratio = None

    def bulk_data_split(self, n_bulk_split, validation_num, test_num, censor_np, snv_flag, *, edges=None):
        self.bulk_test_num_or_ratio = test_num
        self.bulk_validation_num_or_ratio = validation_num
        if edges is None:
            edges = self.edges              # 引数が無いときだけ self.edges を使う
        self.bulk_data_manager.bulk_split(n_bulk_split, validation_num, test_num, censor_np, snv_flag, edges = edges)

    def elbo_loss(self, x, xnorm_mat, bulk_count, bulk_norm_mat, spatial_count, spatial_norm_mat, batch_onehot, bulk_idx):
        z, qz, x_hat, p_bulk, p_spatial, bulk_hat, spatial_hat, theta_x, theta_bulk, theta_spatial, beta_z, gamma_z_dict = self.scTREND(x, batch_onehot, gene_name = None)
        if self.scTREND.mode == 'sc':
            elbo_loss = self.calc_kld(qz).sum()
            if self.usePoisson_sc:
                elbo_loss += self.calc_poisson_loss(ld=x_hat, norm_mat=xnorm_mat, obs=x).sum()
            else:
                elbo_loss += self.calc_nb_loss(x_hat, xnorm_mat, theta_x, x).sum()
        elif self.scTREND.mode == 'bulk':
            elbo_loss = self.calc_nb_loss(bulk_hat, bulk_norm_mat, theta_bulk, bulk_count).sum()
        elif self.scTREND.mode == 'spatial':
            elbo_loss = self.calc_nb_loss(spatial_hat, spatial_norm_mat, theta_spatial, spatial_count).sum()
        elif self.scTREND.mode in ('beta_z', 'gamma_z'):
            all_cens = self.cutting_off_0_1.to(self.device)
            event_obs    = all_cens[bulk_idx]
            beta_z_all = beta_z
            p_bulk_sel = p_bulk[bulk_idx]
            gamma_all_dict = gamma_z_dict
            if gamma_all_dict is not None:
                T_driver_sel = {
                    gene: self.driver_bulk_SNV_dict[gene][bulk_idx].float()
                    for gene in gamma_all_dict.keys()
                }
            else:
                T_driver_sel = None
            neg_log_like = self._piecewise_const_loss(
                beta_z_all, p_bulk_sel,
                self.survival_time[bulk_idx],
                event_obs.float(),
                gamma_z_all_dict = gamma_all_dict,
                T_driver_dict    = T_driver_sel
            )
            elbo_loss = neg_log_like
            if not self.scTREND.training:
                lam_tbl = self.predict_lambda_table(bulk_idx)
                surv_np  = self.survival_time[bulk_idx].cpu().numpy()
                event_np = event_obs.cpu().numpy()
                edges_np = self.edges.cpu().numpy()
                c_uno, _ = concordance_index_ipcw(
                    event_times    = surv_np,
                    event_observed = event_np,
                    lam      = lam_tbl,
                    times          = edges_np
                )
                self.latest_c_index[self.scTREND.mode] = c_uno
        return elbo_loss
        
    def train_epoch(self):
        self.scTREND.train()
        total_loss = 0
        entry_num = 0
        for x, xnorm_mat, batch_onehot in self.x_data_manager.train_loader:
            x = x.to(self.device)
            xnorm_mat = xnorm_mat.to(self.device)
            batch_onehot = batch_onehot.to(self.device)
            self.scTREND_optimizer.zero_grad()
            bulk_idx = self.bulk_data_manager.train_idx.to(self.device)
            loss = self.elbo_loss(
                x, xnorm_mat, self.bulk_count, self.bulk_norm_mat,
                self.spatial_count, self.spatial_norm_mat,
                batch_onehot, bulk_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.scTREND.parameters(), max_norm=5.0)
            self.scTREND_optimizer.step()
            entry_num += x.shape[0]
            total_loss += loss.item() if torch.is_tensor(loss) else loss
        loss_val = total_loss / entry_num
        return loss_val
        
    def evaluate(self, mode='test'):
        with torch.no_grad():
            self.scTREND.eval()
            if mode == 'test':
                x = self.x_data_manager.test_x.to(self.device)
                xnorm_mat = self.x_data_manager.test_xnorm_mat.to(self.device)
                batch_onehot = self.x_data_manager.test_batch_onehot.to(self.device)
                bulk_idx = self.bulk_data_manager.test_idx.to(self.device)
            else:
                x = self.x_data_manager.validation_x.to(self.device)
                xnorm_mat = self.x_data_manager.validation_xnorm_mat.to(self.device)
                batch_onehot = self.x_data_manager.validation_batch_onehot.to(self.device)
                bulk_idx = self.bulk_data_manager.validation_idx.to(self.device)
            loss = self.elbo_loss(x, xnorm_mat, self.bulk_count, self.bulk_norm_mat, self.spatial_count, self.spatial_norm_mat, batch_onehot, bulk_idx)
            entry_num = x.shape[0]
            loss_val = loss / entry_num
            if isinstance(loss_val, torch.Tensor):
                return loss_val.item()
            else:
                return loss_val
    
    def evaluate_train(self):
        with torch.no_grad():
            self.scTREND.eval()
            x = self.x_data_manager.train_x.to(self.device)
            xnorm_mat = self.x_data_manager.train_xnorm_mat.to(self.device)
            batch_onehot = self.x_data_manager.train_batch_onehot.to(self.device)
            idx = self.bulk_data_manager.train_idx.to(self.device)
            loss = self.elbo_loss(x, xnorm_mat, self.bulk_count, self.bulk_norm_mat, self.spatial_count, self.spatial_norm_mat, batch_onehot, idx)
            entry_num = x.shape[0]
            loss_val = loss / entry_num
            return loss_val.item() if torch.is_tensor(loss_val) else loss_val

    def train_total(self, epoch_num, patience):
        earlystopping = EarlyStopping(patience=patience, path=self.checkpoint)
        val_loss_list = deque(maxlen=patience)
        for epoch in range(epoch_num):
            loss = self.train_epoch()
            val_loss = self.evaluate(mode='validation')
            val_loss_list.append(val_loss)
            val_loss_mean = mean(val_loss_list)
            if self.use_val_loss_mean == True:
                earlystopping(val_loss_mean, self.scTREND)
            else:
                earlystopping(val_loss, self.scTREND)
            if earlystopping.early_stop:
                print(f"Early Stopping! at {epoch} epoch, best score={earlystopping.best_score}")
                break
            if self.scTREND.mode == 'beta_z':
                if epoch % 10 == 0:
                    lam_train = self.predict_lambda_table(self.bulk_data_manager.train_idx)
                    lam_val = self.predict_lambda_table(self.bulk_data_manager.validation_idx)
                    surv_train = self.survival_time[self.bulk_data_manager.train_idx].cpu().numpy()
                    surv_val   = self.survival_time[self.bulk_data_manager.validation_idx].cpu().numpy()
                    event_train = self.cutting_off_0_1[self.bulk_data_manager.train_idx].cpu().numpy()
                    event_val   = self.cutting_off_0_1[self.bulk_data_manager.validation_idx].cpu().numpy()
                    edges_np    = self.edges.cpu().numpy()
                    c_train, _ = concordance_index_ipcw(surv_train, event_train, lam_train, edges_np)
                    c_val, _   = concordance_index_ipcw(surv_val,   event_val,   lam_val,   edges_np)
                    print(f"epoch {epoch}: Uno-c train {c_train:.4f} | Uno-c val {c_val:.4f}")
            elif epoch % 50 == 0:
                print(f'epoch {epoch}: train loss {loss} validation loss {val_loss}')
            if math.isnan(loss):
                print('loss is nan')
                break
        return epoch

    def initialize_optimizer(self, lr):
        self.scTREND_optimizer = torch.optim.AdamW(self.scTREND.parameters(), lr=lr)

    def initialize_loader(self, x_batch_size):
        self.x_data_manager.initialize_loader(x_batch_size)
  
    def calc_kld(self, qz):
        kld = -0.5 * (1 + qz.scale.pow(2).log() - qz.loc.pow(2) - qz.scale.pow(2))
        return kld
    
    def calc_nb_loss(self, ld, norm_mat, theta, obs):
        ld = norm_mat * ld
        ld = ld + 1.0e-10
        theta = theta + 1.0e-10
        lp =  ld.log() - (theta).log()
        p_z = dist.NegativeBinomial(theta, logits=lp)
        l = - p_z.log_prob(obs)
        return l
    
    def calc_poisson_loss(self, ld, norm_mat, obs):
        p_z = dist.Poisson(ld * norm_mat + 1.0e-10)
        l = - p_z.log_prob(obs)
        return l
    
    def _piecewise_const_loss(
            self,
            beta_z_all,
            p_bulk,
            T_b,
            delta_b,
            *,
            gamma_z_all_dict=None,
            T_driver_dict=None):

        device   = beta_z_all.device
        B, K     = p_bulk.size(0), beta_z_all.size(1)
        edges    = self.edges.to(device)
        delta_k  = self.delta.to(device)

        eta_bk = torch.matmul(p_bulk, beta_z_all)
        if gamma_z_all_dict is not None and T_driver_dict is not None:
            for gene, gamma_c_k in gamma_z_all_dict.items():
                indic_b = T_driver_dict[gene].to(device)
                eta_bk = eta_bk + indic_b.unsqueeze(1) * torch.matmul(p_bulk, gamma_c_k)
        lam_bk = torch.nn.functional.softplus(eta_bk).clamp(max=1e4)
        cum_H = torch.cumsum(lam_bk * delta_k, dim=1)

        k_star = torch.bucketize(T_b, edges[1:-1], right=True)
        k_star = torch.clamp(k_star, max=K-1)

        row    = torch.arange(B, device=device)
        lam_k  = lam_bk[row, k_star]
        H_prev = torch.zeros_like(T_b)
        mask   = k_star > 0
        H_prev[mask] = cum_H[row[mask], k_star[mask]-1]

        t_prev = edges[k_star]
        diff   = (T_b - t_prev).clamp(min=0)
        H_T    = H_prev + lam_k * diff

        eps = 1e-8
        nll = torch.sum(H_T - delta_b * torch.log(lam_k + eps))
        return nll

    def get_beta_per_bin(self, x, batch_onehot):
        with torch.no_grad():
            x   = x.to(self.device)
            boh = batch_onehot.to(self.device)
            z,_ = self.scTREND.enc_z(torch.cat([x, boh], dim=-1))
            beta_z_all = self.scTREND.dec_beta_z(z)
        return beta_z_all.cpu()
    
    def predict_lambda_table(self, bulk_idx):
        self.scTREND.eval()
        with torch.no_grad():
            x_all  = torch.cat([self.x_data_manager.train_x,
                                self.x_data_manager.validation_x,
                                self.x_data_manager.test_x], dim=0).to(self.device)
            boh_all = torch.cat([self.x_data_manager.train_batch_onehot,
                                self.x_data_manager.validation_batch_onehot,
                                self.x_data_manager.test_batch_onehot], dim=0).to(self.device)

            (_, _, _,
            p_bulk_all, _, _, _,
            _, _, _,
            beta_z_all, gamma_z_dict) = self.scTREND(x_all, boh_all)

            p_bulk = p_bulk_all[bulk_idx]
            eta_bk = p_bulk @ beta_z_all

            if gamma_z_dict is not None:
                for g, gamma_ck in gamma_z_dict.items():
                    indic = self.driver_bulk_SNV_dict[g][bulk_idx].float()
                    eta_bk += indic.unsqueeze(1) * (p_bulk @ gamma_ck)

            lam_bk = torch.nn.functional.softplus(eta_bk)
            return lam_bk.cpu().numpy()
