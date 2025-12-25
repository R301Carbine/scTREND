# scTREND

## Description
scTREND (single-cell time-resolved and condition-dependent hazard model) is a novel deep generative framework that integrates single-cell latent representations from a VAE with bulk-level cell-type proportions and hazard coefficients.
This enables the computation of patient-level risk scores and the identification of cell populations whose prognostic impact dynamically changes across time and clinical conditions.

Teppei Shimamura's lab, Institute of Science Tokyo, Tokyo, Japan

<p align="center">
  <img src="Overview_git.png" alt="Overview of the scTCHM framework" width="1100">
</p>

## Model architecture
The model comprises three main components: VAE for latent representation of single cells, bulk deconvolution based on DeepCOLOR, and a conditional piecewise constant hazard model for time- and condition-dependent risk estimation.

## Requirements
Python >= 3.8.16

torch >= 1.13.1

lifelines >= 0.27.8

scanpy >= 1.9.5

pandas >= 1.5.3

numpy >= 1.23.5

matplotlib >= 3.7.2

scipy >= 1.10.1

## Application example
In this tutorial, we present an application of scTREND using a melanoma single-cell RNA-seq dataset (GSE115978) together with a bulk RNA-seq dataset from TCGA-SKCM.
BRAF mutation status is incorporated as a driver condition, and the survival time axis is discretized into four time intervals.
Under this setting, both the coefficients shared across all patients and the coefficients specific to BRAF-mutant patients can be visualized as shown below.

```python
driver_genes = ["BRAF"]
edges = [...]  # time bin edges used in training
sc_adata, bulk_adata = workflow.scTREND_preprocess(sc_adata, bulk_adata,
     per=0.01, n_top_genes=5000, highly_variable="bulk", driver_genes=driver_genes)
driver_bulk_adata = bulk_adata[:, bulk_adata.var_names.isin(driver_genes)]
driver_bulk_adata.layers["SNV"] = ...  # samples × driver_genes (0:wild-type 1:mutated)
sc_adata, bulk_adata, model_params_dict, spatial_adata, exp = workflow.run_scTREND(
     sc_adata, bulk_adata,
     param_save_path="scTREND.pt",
     epoch=10000,
     batch_key="samples",
     driver_genes=driver_genes,
     driver_bulk_adata=driver_bulk_adata,
     edges=edges
)


```

<p align="center">
  <img src="SKCM_umap_celltype.png" alt="Cell type" width="1100">
</p>

<p align="center">
  <img src="umap_beta_per_bin.png" alt="Baseline contribution (beta)" width="1100">
</p>

<p align="center">
  <img src="umap_gamma_BRAF_per_bin.png" alt="BRAF-specific contribution (gamma)" width="1100">
</p>


