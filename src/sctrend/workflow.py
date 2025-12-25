import torch
import scanpy as sc
import numpy as np
from .exp import scTRENDExperiment
from .utils import make_inputs, safe_toarray, make_sample_one_hot_mat, vae_results, bulk_deconvolution_results, beta_z_results, spatial_results, optimize_vae, optimize_vae_onlyload, optimize_deepcolor, optimize_deepcolor_onlyload, optimize_scTREND

def run_scTREND(
        sc_adata, bulk_adata, param_save_path, warm_path, epoch, batch_key, driver_genes, driver_bulk_adata, bulk_seed = 0,
        spatial_adata = None, survival_time_label = 'survival_times', survival_time_censor = 'vital_status', edges = None):
    
    assert edges is not None, "Error: 'edges' must be provided (e.g. edges=[0, 0.5, 1.0, 2.0, ...])"
    EDGES = tuple(edges)
    EDGES_WITH_INF = EDGES + (np.inf,)
    K = len(EDGES) - 1
    
    x_batch_size_VAE = 1000
    x_batch_size_DeepCOLOR = 1000
    x_batch_size_scTREND = 500
    
    first_lr = 0.01
    second_lr = 0.01
    third_lr = 0.0001
    bulk_validation_num_or_ratio = 0.2
    bulk_test_num_or_ratio = 0.2
    
    use_val_loss_mean = True
    model_params = {"z_dim": 20, "h_dim": 100, "num_enc_z_layers": 1, "num_dec_z_layers": 1, "num_dec_p_layers": 1, "num_dec_b_layers": 1}
    model_params["num_time_bins"] = K
    model_params["edges"] = EDGES
    
    patience = 30
    usePoisson_sc = True
    
    survival_time_np = (bulk_adata.obs[survival_time_label]).values
    survival_time = torch.tensor(survival_time_np)

    x_count, bulk_count = make_inputs(sc_adata, bulk_adata)
    batch_onehot = make_sample_one_hot_mat(sc_adata, batch_key)
    
    if survival_time_censor == 'vital_status':
        valid_values = ['Dead', 'Alive']
        if not all(value in valid_values for value in bulk_adata.obs[survival_time_censor]):
            raise ValueError("Invalid values found in bulk_adata.obs[survival_time_censor]. Only 'Dead' and 'Alive' are allowed.")
        vital_status_values = np.where(bulk_adata.obs[survival_time_censor] == 'Dead', 1, 0)
        cutting_off_0_1 = torch.tensor(vital_status_values)
    else:
        cutting_off_0_1 = torch.tensor(bulk_adata.obs[survival_time_censor].values)

    model_params['x_dim'] = x_count.size()[1]
    model_params_dict = {
        '1st_lr': first_lr, '2nd_lr': second_lr, '3rd_lr': third_lr, 'patience': patience, 'bulk_seed': bulk_seed,
        'x_batch_size_VAE': x_batch_size_VAE, 'x_batch_size_DeepCOLOR': x_batch_size_DeepCOLOR, 'x_batch_size_scTREND': x_batch_size_scTREND,
        'n_var': sc_adata.n_vars, 'usePoisson_sc': usePoisson_sc, 'batch_key': batch_key, 
        'n_obs_sc': sc_adata.n_obs, 'n_obs_bulk': bulk_adata.n_obs, 'use_val_loss_mean' : use_val_loss_mean,
        'bulk_validation_num_or_ratio': bulk_validation_num_or_ratio, 'bulk_test_num_or_ratio': bulk_test_num_or_ratio,
    }
    
    if spatial_adata is not None:
        spatial_count = torch.tensor(safe_toarray(spatial_adata.X))
        model_params_dict['n_obs_spatial'] = spatial_adata.n_obs
    else:
        spatial_count = None
    model_params_dict.update(model_params)
    print(model_params_dict)

    scTREND_exp = scTRENDExperiment(
        model_params=model_params, x_count=x_count, bulk_count=bulk_count, survival_time=survival_time,
        cutting_off_0_1=cutting_off_0_1, x_batch_size=x_batch_size_VAE, checkpoint=param_save_path,
        usePoisson_sc=usePoisson_sc, batch_onehot=batch_onehot, spatial_count=spatial_count, use_val_loss_mean=use_val_loss_mean, driver_genes=driver_genes, driver_bulk_adata=driver_bulk_adata
    )

    if warm_path is not None:
        scTREND_exp.scTREND.load_state_dict(torch.load(warm_path), strict=False)
        scTREND_exp = optimize_vae_onlyload(scTREND_exp=scTREND_exp, first_lr=first_lr, x_batch_size=x_batch_size_VAE, epoch=epoch, patience=patience, param_save_path=warm_path)
        scTREND_exp = optimize_deepcolor_onlyload(scTREND_exp=scTREND_exp, second_lr=second_lr, x_batch_size=x_batch_size_DeepCOLOR, epoch=epoch, patience=patience, param_save_path=warm_path)
    else:
        scTREND_exp = optimize_vae(scTREND_exp=scTREND_exp, first_lr=first_lr, x_batch_size=x_batch_size_VAE, epoch=epoch, patience=patience, param_save_path=param_save_path)
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path.replace('.pt', '') + '_1st_end.pt')
        scTREND_exp = optimize_deepcolor(scTREND_exp=scTREND_exp, second_lr=second_lr, x_batch_size=x_batch_size_DeepCOLOR, epoch=epoch, patience=patience, param_save_path=param_save_path, spatial_adata=spatial_adata)
        torch.save(scTREND_exp.scTREND.state_dict(), param_save_path.replace('.pt', '') + '_2nd_end.pt')

    snv_flag = None
    if driver_genes and driver_bulk_adata is not None:
        snv_mat = driver_bulk_adata.layers["SNV"]
        if hasattr(snv_mat, "toarray"):
            snv_mat = snv_mat.toarray()
        snv_flag = np.zeros(bulk_adata.n_obs, dtype=int)
        for bit, gene in enumerate(driver_genes):
            try:
                col = np.where(driver_bulk_adata.var_names == gene)[0][0]
            except IndexError:
                raise ValueError(f"{gene} not found in driver_bulk_adata.var_names")
            mut_vec = snv_mat[:, col].astype(int).ravel()
            snv_flag += mut_vec * (1 << bit)

    scTREND_exp.bulk_data_split(
        bulk_seed,
        bulk_validation_num_or_ratio,
        bulk_test_num_or_ratio,
        cutting_off_0_1,
        snv_flag,
        edges=EDGES_WITH_INF
    )

    scTREND_exp = optimize_scTREND(scTREND_exp, third_lr=third_lr, x_batch_size=x_batch_size_scTREND, epoch=epoch, patience=patience, param_save_path=param_save_path, warm_path=warm_path)
    torch.save(scTREND_exp.scTREND.state_dict(), param_save_path)
    sc_adata.uns['param_save_path'] = param_save_path
    sc_adata, bulk_adata = vae_results(scTREND_exp, sc_adata, bulk_adata, param_save_path)
    sc_adata, bulk_adata = bulk_deconvolution_results(scTREND_exp, sc_adata, bulk_adata)
    sc_adata, bulk_adata = beta_z_results(scTREND_exp, sc_adata, bulk_adata, driver_genes)
    if spatial_adata is not None:
        sc_adata, spatial_adata = spatial_results(scTREND_exp, sc_adata, spatial_adata)
    print('Done post process')
    return sc_adata, bulk_adata, model_params_dict, spatial_adata, scTREND_exp

def scTREND_preprocess(sc_adata, bulk_adata, per=0.01, n_top_genes=5000, highly_variable='bulk', driver_genes=None):
    common_genes_before = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    sc_adata = sc_adata[:, common_genes_before].copy()
    bulk_adata = bulk_adata[:, common_genes_before].copy()
    print('common_genes_before', len(common_genes_before))

    sc_min_cells = int(sc_adata.n_obs * per)
    bulk_min_cells = int(bulk_adata.n_obs * per)
    sc.pp.filter_genes(sc_adata, min_cells=sc_min_cells)
    sc.pp.filter_genes(bulk_adata, min_cells=bulk_min_cells)

    common_genes_filtered = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    if driver_genes is not None:
        common_driver_genes = np.intersect1d(common_genes_filtered, driver_genes)
    sc_adata = sc_adata[:, common_genes_filtered].copy()
    bulk_adata = bulk_adata[:, common_genes_filtered].copy()
    print('common_genes_filtered', len(common_genes_filtered))

    raw_sc_adata = sc_adata.copy()
    raw_bulk_adata = bulk_adata.copy()

    if highly_variable == 'bulk':
        sc.pp.normalize_total(bulk_adata)
        sc.pp.log1p(bulk_adata)
        sc.pp.highly_variable_genes(bulk_adata, n_top_genes=n_top_genes)
        if driver_genes is not None:
            bulk_adata.var.loc[common_driver_genes, "highly_variable"] = True
        bulk_adata = bulk_adata[:, bulk_adata.var['highly_variable']].copy()
    elif highly_variable == 'sc':
        sc.pp.normalize_total(sc_adata)
        sc.pp.log1p(sc_adata)
        sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes)
        if driver_genes is not None:
            sc_adata.var.loc[common_driver_genes, "highly_variable"] = True
        sc_adata = sc_adata[:, sc_adata.var['highly_variable']].copy()

    common_genes_highly_variable = np.intersect1d(sc_adata.var_names, bulk_adata.var_names)
    if len(common_genes_highly_variable) == 0:
        raise ValueError("No common genes found between sc_adata and bulk_adata.")
    sc_adata = sc_adata[:, common_genes_highly_variable].copy()
    bulk_adata = bulk_adata[:, common_genes_highly_variable].copy()
    print('common_genes_highly_variable', len(common_genes_highly_variable))
    
    sc_adata.X = raw_sc_adata[:, sc_adata.var_names].X
    bulk_adata.X = raw_bulk_adata[:, bulk_adata.var_names].X
    return sc_adata, bulk_adata

def scTREND_preprocess_spatial(sc_adata, bulk_adata, spatial_adata, per =  0.01, n_top_genes = 5000, highly_variable='bulk'):
        common_genes_before = np.intersect1d(sc_adata.var_names, bulk_adata.var_names) #scとbulkの共通
        common_genes_before = np.intersect1d(common_genes_before, spatial_adata.var_names) #上での共通とSTとの共通
        sc_adata = sc_adata[:, common_genes_before].copy()
        bulk_adata = bulk_adata[:, common_genes_before].copy()
        spatial_adata = spatial_adata[:, common_genes_before].copy()
        print('common_genes_before', len(common_genes_before))

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
        print('common_genes_filtered', len(common_genes_filtered))

        raw_sc_adata = sc_adata.copy()
        raw_bulk_adata = bulk_adata.copy()
        raw_spatial_adata = spatial_adata.copy()
        if highly_variable=='bulk':
            sc.pp.normalize_total(bulk_adata)
            sc.pp.log1p(bulk_adata)
            sc.pp.highly_variable_genes(bulk_adata, n_top_genes=n_top_genes)
            bulk_adata = bulk_adata[:, bulk_adata.var['highly_variable']].copy()
        elif highly_variable=='spatial':
            sc.pp.normalize_total(spatial_adata)
            sc.pp.log1p(spatial_adata)
            sc.pp.highly_variable_genes(spatial_adata, n_top_genes=n_top_genes)
            spatial_adata = spatial_adata[:, spatial_adata.var['highly_variable']].copy()
        elif highly_variable=='sc':
            sc.pp.normalize_total(sc_adata)
            sc.pp.log1p(sc_adata)
            sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes)
            sc_adata = sc_adata[:, sc_adata.var['highly_variable']].copy()

        common_genes_highly_variable = np.intersect1d(sc_adata.var_names, bulk_adata.var_names) #HVGsを記録したbulkとscとの共通
        common_genes_highly_variable = np.intersect1d(common_genes_highly_variable, spatial_adata.var_names) #上とSTとの共通
        if len(common_genes_highly_variable) == 0:
            raise ValueError("No common genes found between sc_adata, bulk_adata and spatial_adata.")
        sc_adata = sc_adata[:, common_genes_highly_variable].copy()
        bulk_adata = bulk_adata[:, common_genes_highly_variable].copy()
        spatial_adata = spatial_adata[:, common_genes_highly_variable].copy()
        print('common_genes_highly_variable', len(common_genes_highly_variable))

        sc_adata.X = raw_sc_adata[:, sc_adata.var_names].X
        bulk_adata.X = raw_bulk_adata[:, bulk_adata.var_names].X
        spatial_adata.X = raw_spatial_adata[:, spatial_adata.var_names].X
        return sc_adata, bulk_adata, spatial_adata
