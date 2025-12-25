import torch
import pandas as pd
import numpy as np
from .ipcw import concordance_index_ipcw

def safe_toarray(x):
    if type(x) != np.ndarray:
        x = x.toarray()
        if not np.all(x == np.floor(x)):
            raise ValueError('target layer of adata should be raw count')
        return x
    else:
        if not np.all(x == np.floor(x)):
            raise ValueError('target layer of adata should be raw count')
        return x

def make_sample_one_hot_mat(adata, sample_key):
    print('make_sample_one_hot_mat')
    if sample_key is not None:
        sidxs = np.sort(adata.obs[sample_key].unique())
        b = np.array([
            (sidxs == sidx).astype(int)
            for sidx in adata.obs[sample_key]]).astype(float)
        b = torch.tensor(b).float()
    else:
        b = np.zeros((len(adata.obs_names), 1))
        b = torch.tensor(b).float()
    return b

def input_checks(adata, layer_name):
    if layer_name == 'X':
        if np.sum((adata.X - adata.X.astype(int)))**2 != 0:
            raise ValueError('`X` includes non integer number, while count data is required for `X`.')
    else:
        if np.sum((adata.layers[layer_name] - adata.layers[layer_name].astype(int)))**2 != 0:
            raise ValueError(f'layers `{layer_name}` includes non integer number, while count data is required for `{layer_name}`.')

def make_inputs(sc_adata, bulk_adata, layer_name='X'):
    input_checks(sc_adata, layer_name)
    if layer_name == 'X':
        x = torch.tensor(safe_toarray(sc_adata.X))
        s = torch.tensor(safe_toarray(bulk_adata.X))
    else:
        x = torch.tensor(safe_toarray(sc_adata.layers[layer_name]))
        s = torch.tensor(safe_toarray(bulk_adata.layers[layer_name]))
    return x, s

def optimize_vae(scTREND_exp, first_lr, x_batch_size, epoch, patience, param_save_path):
    print('Start first opt', 'lr=', first_lr)
    scTREND_exp.scTREND.sc_mode()
    scTREND_exp.initialize_optimizer(first_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    stop_epoch_vae = scTREND_exp.train_total(epoch, patience)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    val_loss_vae = scTREND_exp.evaluate(mode='val')
    test_loss_vae = scTREND_exp.evaluate(mode='test')
    print(f'Done {scTREND_exp.scTREND.mode} mode,', f'Val Loss: {val_loss_vae}', f'Test Loss: {test_loss_vae}')
    return scTREND_exp

def optimize_vae_onlyload(scTREND_exp, first_lr, x_batch_size, epoch, patience, param_save_path):
    print('Start first opt', 'lr=', first_lr)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    return scTREND_exp

def optimize_deepcolor(scTREND_exp, second_lr, x_batch_size, epoch, patience, param_save_path, spatial_adata):
    scTREND_exp.scTREND.bulk_mode()
    scTREND_exp.initialize_optimizer(second_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    print(f'{scTREND_exp.scTREND.mode} mode', 'lr=', second_lr)
    stop_epoch_bulk = scTREND_exp.train_total(epoch, patience)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    val_loss_bulk = scTREND_exp.evaluate(mode='val')
    test_loss_bulk = scTREND_exp.evaluate(mode='test')
    print(f'Done {scTREND_exp.scTREND.mode} mode,', f'Val Loss: {val_loss_bulk}', f'Test Loss: {test_loss_bulk}')
    if spatial_adata is not None:
        scTREND_exp.scTREND.spatial_mode()
        scTREND_exp.initialize_optimizer(second_lr)
        scTREND_exp.initialize_loader(x_batch_size)
        print(f'{scTREND_exp.scTREND.mode} mode', 'lr=', second_lr)
        stop_epoch_spatial = scTREND_exp.train_total(epoch, patience)
        scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path))
        val_loss_spatial = scTREND_exp.evaluate(mode='val')
        test_loss_spatial = scTREND_exp.evaluate(mode='test')
        print(f'Done {scTREND_exp.scTREND.mode} mode,', f'Val Loss: {val_loss_spatial}', f'Test Loss: {test_loss_spatial}')
    return scTREND_exp

def optimize_deepcolor_onlyload(scTREND_exp, second_lr, x_batch_size, epoch, patience, param_save_path, spatial_adata=None):
    print(f'{scTREND_exp.scTREND.mode} mode', 'lr=', second_lr)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path), strict=False)
    return scTREND_exp

def optimize_scTREND(scTREND_exp, third_lr, x_batch_size, epoch, patience, param_save_path, warm_path):
    scTREND_exp.scTREND.hazard_beta_z_mode()
    scTREND_exp.initialize_optimizer(third_lr)
    scTREND_exp.initialize_loader(x_batch_size)
    print(f'{scTREND_exp.scTREND.mode} mode', 'lr=', third_lr)
    stop_epoch_beta_z = scTREND_exp.train_total(epoch, patience)
    scTREND_exp.scTREND.load_state_dict(torch.load(param_save_path))
    train_nll = scTREND_exp.evaluate_train()
    val_nll   = scTREND_exp.evaluate('validation')
    test_nll  = scTREND_exp.evaluate('test')
    print(f"Done beta_z mode | "
          f"Train NLL: {train_nll:.4f} | "
          f"Val NLL: {val_nll:.4f} | "
          f"Test NLL: {test_nll:.4f}")
    edges_np = scTREND_exp.edges.cpu().numpy()
    def _c(bidx):
        lam = scTREND_exp.predict_lambda_table(bidx)
        surv  = scTREND_exp.survival_time[bidx].cpu().numpy()
        event = scTREND_exp.cutting_off_0_1[bidx].cpu().numpy()
        c, _  = concordance_index_ipcw(surv, event, lam, edges_np)
        return c
    c_train = _c(scTREND_exp.bulk_data_manager.train_idx)
    c_val   = _c(scTREND_exp.bulk_data_manager.validation_idx)
    c_test  = _c(scTREND_exp.bulk_data_manager.test_idx)
    print(f"Final C-index | Train: {c_train:.3f} | "
          f"Val: {c_val:.3f} | Test: {c_test:.3f}")
    metrics_dict = {
        "train_nll":  train_nll,
        "val_nll":    val_nll,
        "test_nll":   test_nll,
        "train_c_index": c_train,
        "val_c_index":   c_val,
        "test_c_index":  c_test
    }
    tag = "" if warm_path is None else warm_path.replace(".pt", "")
    combined_path = param_save_path.replace(".pt", "") + tag + "_metrics.pt"
    torch.save(metrics_dict, combined_path)
    return scTREND_exp

def vae_results(scTREND_exp, sc_adata, bulk_adata, param_save_path):
    print('vae_results')
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
        sc_adata.obsm['zl'] = zl.detach().cpu().numpy()
        sc_adata.layers['xld'] = xld_np
        xnorm_mat=scTREND_exp.x_data_manager.xnorm_mat
        xnorm_mat_np = xnorm_mat.cpu().detach().numpy()
        x_df = pd.DataFrame(x_np, columns=list(sc_adata.var_names))
        xld_df = pd.DataFrame(xld_np,columns=list(sc_adata.var_names))
        train_idx = scTREND_exp.x_data_manager.train_idx
        val_idx = scTREND_exp.x_data_manager.validation_idx
        test_idx = scTREND_exp.x_data_manager.test_idx
        x_correlation_gene=(xld_df).corrwith(x_df / xnorm_mat_np).mean()
        train_x_correlation_gene = (xld_df.T[train_idx].T).corrwith((x_df / xnorm_mat_np).T[train_idx].T).mean()
        val_x_correlation_gene = (xld_df.T[val_idx].T).corrwith((x_df / xnorm_mat_np).T[val_idx].T).mean()
        test_x_correlation_gene = (xld_df.T[test_idx].T).corrwith((x_df / xnorm_mat_np).T[test_idx].T).mean()
        metrics_dict = {
            "all_x_correlation_gene": x_correlation_gene,
            "train_x_correlation_gene": train_x_correlation_gene,
            "val_x_correlation_gene": val_x_correlation_gene,
            "test_x_correlation_gene": test_x_correlation_gene
        }
        combined_path = param_save_path.replace('.pt', '') + "_correlation.pt"
        torch.save(metrics_dict, combined_path)
        print('all_x_correlation_gene', f"{x_correlation_gene:.3f}", 'train_x_correlation_gene', f"{train_x_correlation_gene:.3f}", 'val_x_correlation_gene', f"{val_x_correlation_gene:.3f}", 'test_x_correlation_gene', f"{test_x_correlation_gene:.3f}")
        return sc_adata, bulk_adata

def bulk_deconvolution_results(scTREND_exp, sc_adata, bulk_adata):
    print('deconvolution_results')
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
        del zzz, ppp
        bulk_scoeff_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_bulk_coeff).cpu().detach().numpy()
        bulk_scoeff_add_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_bulk_coeff_add).cpu().detach().numpy()
        xld_np = sc_adata.layers['xld']
        bulk_hat_np = np.matmul(bulk_pl_np, xld_np * bulk_scoeff_np) + bulk_scoeff_add_np
        bulk_p_df = pd.DataFrame(bulk_pl_np.transpose(), index=sc_adata.obs_names, columns=bulk_adata.obs_names)
        sc_adata.obsm['map2bulk'] = bulk_p_df.values
        bulk_norm_mat=scTREND_exp.bulk_data_manager.bulk_norm_mat
        bulk_norm_mat_np = bulk_norm_mat.cpu().detach().numpy()
        bulk_count = scTREND_exp.bulk_data_manager.bulk_count
        bulk_count_df = pd.DataFrame(bulk_count.cpu().detach().numpy(), columns=list(bulk_adata.var_names))
        bulk_hat_df = pd.DataFrame(bulk_hat_np, columns=list(bulk_adata.var_names))
        bulk_adata.layers['bulk_hat'] = pd.DataFrame(bulk_hat_np, index = list(bulk_adata.obs_names), columns=list(bulk_adata.var_names))
        bulk_correlation_gene=(bulk_hat_df).corrwith(bulk_count_df / bulk_norm_mat_np).mean()
        print('bulk_correlation_gene', bulk_correlation_gene)
        return sc_adata, bulk_adata
    
def spatial_results(scTREND_exp, sc_adata, spatial_adata):
    print('spatial_results')
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
        del zzz, ppp
        spatial_coeff_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_spatial_coeff).cpu().detach().numpy()
        spatial_coeff_add_np = scTREND_exp.scTREND.softplus(scTREND_exp.scTREND.log_spatial_coeff_add).cpu().detach().numpy()
        xld_np = sc_adata.layers['xld']
        spatial_hat_np = np.matmul(spatial_pl_np, xld_np * spatial_coeff_np) + spatial_coeff_add_np
        spatial_p_df = pd.DataFrame(spatial_pl_np.transpose(), index=sc_adata.obs_names, columns=spatial_adata.obs_names)
        sc_adata.obsm['map2spatial'] = spatial_p_df.values
        if 'raw_beta_z' not in sc_adata.obsm:
            raise KeyError(
                'raw_beta_z not found in sc_adata.obsm. '
                'Call beta_z_results() before spatial_results().'
            )
        beta_ck = sc_adata.obsm['raw_beta_z']       
        P_cs = sc_adata.obsm['map2spatial']      
        eta_spot_k = P_cs.T @ beta_ck             
        spatial_adata.obsm['eta_spot_timebins'] = eta_spot_k
        lambda_spot_k = np.log1p(np.exp(eta_spot_k)) # softplus
        spatial_adata.obsm['lambda_spot_timebins'] = lambda_spot_k
        K = lambda_spot_k.shape[1]
        for k in range(K):
            spatial_adata.obs[f'lambda_timebin_{k+1}'] = lambda_spot_k[:, k]

            h = lambda_spot_k[:, k]
            spatial_adata.obs[f'lambda_rel_timebin_{k+1}'] = h / h.mean()

        total_h = lambda_spot_k.sum(axis=1)
        spatial_adata.obs['Hazard_rates'] = total_h / total_h.mean()
        
        spatial_norm_mat=scTREND_exp.spatial_data_manager.spatial_norm_mat
        spatial_norm_mat_np = spatial_norm_mat.cpu().detach().numpy()
        spatial_count = scTREND_exp.spatial_data_manager.spatial_count
        spatial_count_df = pd.DataFrame(spatial_count.cpu().detach().numpy(), columns=list(spatial_adata.var_names))
        spatial_hat_df = pd.DataFrame(spatial_hat_np,columns=list(spatial_adata.var_names))
        spatial_adata.layers['spatial_hat'] = pd.DataFrame(spatial_hat_np, index = list(spatial_adata.obs_names), columns=list(spatial_adata.var_names))
        spatial_correlation_gene=(spatial_hat_df).corrwith(spatial_count_df / spatial_norm_mat_np).mean()
        print('spatial_correlation_gene', spatial_correlation_gene)
        return sc_adata, spatial_adata

def beta_z_results(scTREND_exp, sc_adata, bulk_adata, driver_genes):
    with torch.no_grad():
        torch.cuda.empty_cache()
        batch_onehot = scTREND_exp.x_data_manager.batch_onehot.to(scTREND_exp.device)
        x = scTREND_exp.x_data_manager.x_count.to(scTREND_exp.device)
        xb = torch.cat([x, batch_onehot], dim=-1)
        z, qz = scTREND_exp.scTREND.enc_z(xb)
        beta_list = []
        if driver_genes is not None:
            gamma_list_dict = { gene: [] for gene in driver_genes }
        zl = qz.loc
        z_list = []
        for _ in range(100):
            z_sample = qz.sample()
            z_for_beta = z_sample
            z_list.append(z_sample.detach().cpu().numpy())
            b_sample= scTREND_exp.scTREND.dec_beta_z(z_for_beta).detach().cpu().numpy()
            beta_list.append(b_sample)
            if driver_genes is not None:
                for key, decoder in scTREND_exp.scTREND.dec_gammas.items():
                    gene = key.replace("dec_gamma_", "").replace("_z", "")
                    gamma_sample = decoder(z_for_beta).detach().cpu().numpy()
                    gamma_list_dict[gene].append(gamma_sample)
        z_sample_avg = np.mean(z_list, axis=0)
        zl_for_beta = zl
        beta_z_np = np.mean(beta_list, axis=0)
        beta_zl_np = scTREND_exp.scTREND.dec_beta_z(zl_for_beta).detach().cpu().numpy()
        if driver_genes is not None:
            gamma_z_np_dict = { gene: np.mean(gamma_list, axis=0)
                               for gene, gamma_list in gamma_list_dict.items() }
            gamma_zl_np_dict = {
                key.replace("dec_gamma_", "").replace("_z", ""): decoder(zl_for_beta).detach().cpu().numpy()
                for key, decoder in scTREND_exp.scTREND.dec_gammas.items()
                }
            for gene, gamma_np in gamma_z_np_dict.items():
                if gamma_np.ndim == 2 and gamma_np.shape[1] > 1:
                    sc_adata.obsm[f"raw_gamma_{gene}_z"]  = gamma_np
                    sc_adata.obsm[f"raw_gamma_{gene}_zl"] = gamma_zl_np_dict[gene]
                    sc_adata.obs[f"gamma_{gene}_mean"] = gamma_np.mean(1)
                else:
                    sc_adata.obs[f"raw_gamma_{gene}_z"]  = gamma_np.ravel()
                    sc_adata.obs[f"raw_gamma_{gene}_zl"] = gamma_zl_np_dict[gene].ravel()
        if beta_z_np.ndim == 2 and beta_z_np.shape[1] > 1:
            sc_adata.obsm["raw_beta_z"]  = beta_z_np
            sc_adata.obsm["raw_beta_zl"] = beta_zl_np
            sc_adata.obs["beta_z_mean"] = beta_z_np.mean(1)
        else:
            sc_adata.obs["raw_beta_z"]  = beta_z_np.ravel()
            sc_adata.obs["raw_beta_zl"] = beta_zl_np.ravel()
        sc_adata.obsm['z_sample_avg'] = z_sample_avg
        return sc_adata, bulk_adata
