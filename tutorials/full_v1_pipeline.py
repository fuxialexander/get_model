#%% [markdown]
# NOTE: tabix has to be >= 1.17
! tabix --version
#%% [markdown]
# Before you start, make sure you have the conda environment installed or use the docker image at https://hub.docker.com/r/fuxialexander/get_model
# The required data is a peak bed file with log10(counts/counts.sum()*1e5+1) values for each peak and a CSV file with gene expression in log10(counts/counts.sum()*1e6+1) values. 
# checkout the `astrocyte.atac.bed` and `astrocyte.rna.csv` for the file format. Or follow the `prepare_data_from_snapatac2.py` to prepare the data.
#%%
import os
from pathlib import Path
from gcell.cell.celltype import GETHydraCellType
from gcell.cell.mutincell import GETHydraCellMutCollection
from gcell.rna.gencode import Gencode
from preprocess_utils import (add_atpm, add_exp, create_peak_motif,
                              download_motif, get_motif, join_peaks,
                              query_motif, unzip_zarr, zip_zarr)

from get_model.config.config import load_config, pretty_print_config
from get_model.dataset.zarr_dataset import (InferenceRegionMotifDataset,
                                            RegionMotifDataset)
from get_model.run_region import run_zarr as run
from get_model.utils import print_shape
from gcell._settings import get_setting
annotation_dir = Path(get_setting('annotation_dir'))
print("gcell currently using annotation directory:", annotation_dir)
# %% [markdown]
# # Preprocessing
# %% [markdown]
# ## Download motif bed file
# %%
motif_bed_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz"
motif_bed_index_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi"


if (
    motif_bed_url
    and motif_bed_index_url
    and not (
        (annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz").exists()
        or (annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz.tbi").exists()
    )
):
    download_motif(motif_bed_url, motif_bed_index_url, motif_dir=annotation_dir)
    motif_bed = str(annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz")
else:
    motif_bed = str(annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz")

# %% [markdown]
# ## Query motif and get motifs in the peaks
# Since the mutliome data is processed with snapatac2, we can directly use the peak bed file from snapatac2. Noted that the consensus peak set has been called by snapatac2, so here we will just query the motif in the consensus peak set (i.e. any of the peak bed we saved).
# %%
peak_bed = "cd4_naive.atac.bed"
peaks_motif = query_motif(peak_bed, motif_bed)
get_motif_output = get_motif(peak_bed, peaks_motif)

# %% [markdown]
# ## Create peak motif zarr file
# %% [markdown]
# Create a peak x motif matrix stored in a zarr file. If you are working on multiple cell types with the same peak set, you can use the same peak bed and zarr file for all cell types.
# %%
create_peak_motif(get_motif_output, "pbmc10k_multiome.zarr", peak_bed) # all cell types will be added to the same zarr file as we use the same peak set.

# %% [markdown]
# ## Add aCPM data to region x motif matrix
# %%
celltype_for_modeling = ['cd4_naive', 'cd8_naive', 'cd4_tcm', 'cd14_mono']
#%%
for cell_type in celltype_for_modeling:
    add_atpm(
        "pbmc10k_multiome.zarr",
        f"{cell_type}.atac.bed",
        cell_type,
    )
# %%
# ## Add expression and TSS data to region x motif matrix
# %%
# add expression and TSS data for multiple cell types
for cell_type in celltype_for_modeling:
    add_exp(
        "pbmc10k_multiome.zarr",
        f"{cell_type}.rna.csv",
        f"{cell_type}.atac.bed",
        cell_type,
        assembly="hg38",
        version=44,
        extend_bp=300, # extend TSS region to 300bp upstream and downstream when overlapping with peaks
    id_or_name="gene_name", # use gene_name or gene_id to match the gene expression data, checkout your rna.csv file column names, should be either [gene_name, TPM] or [gene_id, TPM]
)
# %% [markdown]
# optionally zip the zarr file for download or storage
# ```python
# zip_zarr("pbmc10k_multiome.zarr")
# ```
# %% [markdown]
# ## Clean up intermediate files
# %%
for file in [peaks_motif, get_motif_output]:
    os.remove(file)
# %% [markdown]
# ## Unzip if necessary (e.g. if you get the zarr file from other people)
#%%
# ```python
unzip_zarr("pbmc10k_multiome.zarr")
# ```
# %% [markdown]
# ## Load the zarr file as a dataset to validate the data preprocessing works fine
# %%
# load the zarr file as a dataset. 
region_motif_dataset = RegionMotifDataset(
    "pbmc10k_multiome.zarr",
    celltypes=','.join(celltype_for_modeling),
    quantitative_atac=True,
    num_region_per_sample=200,
    leave_out_celltypes='cd4_tcm', # training on just this cd4_naive
    leave_out_chromosomes="chr11",
    is_train=True,
)
# %%
print_shape(region_motif_dataset[0]) # hic is not used in this tutorial, it returns a placeholder
# %%
# load the zarr file as a inference dataset, where you only focus on the genes of interest. 
gencode = Gencode(assembly="hg38", version=44)
inference_region_motif_dataset = InferenceRegionMotifDataset(
    zarr_path="pbmc10k_multiome.zarr",
    gencode_obj={'hg38':gencode},
    assembly='hg38',
    gene_list=None,
    celltypes=','.join(celltype_for_modeling),
    quantitative_atac=True,
    num_region_per_sample=200,
    leave_out_celltypes=None,
    leave_out_chromosomes=None,
    is_train=True,
)
#%%
print_shape(inference_region_motif_dataset[3])
#%%
len(inference_region_motif_dataset) 

# %% [markdown]
# This is basically all genes with TSS overlapping with peaks. Since the snapatac2 tutorial data contains only highly variable genes, we only have 2316 genes here. You should use a full gene list for your own data.
#%% [markdown]
# # Finetune the model (note that with incomplete gene expression data, the model performance will be poor due to missing values in the expression labels)
#%%
# Download checkpoint from s3 
if not Path('./checkpoint-best.pth').exists():
    s3_checkpoint_url = "s3://2023-get-xf2217/get_demo/checkpoints/regulatory_inference_checkpoint_fetal_adult/finetune_fetal_adult_leaveout_astrocyte/checkpoint-best.pth"
    ! aws s3 cp $s3_checkpoint_url ./checkpoint-best.pth --no-sign-request
#%% [markdown]
# Load the finetuning config. have a look at the config file to see what options are available.
# %%
cfg = load_config('finetune_tutorial')
pretty_print_config(cfg)
#%% [markdown]
# Setup config. Change state to 'fit'
cfg.stage = 'fit'
cfg.run.run_name='training_from_scratch_no_chr_split'
cfg.run.project_name='finetune_pbmc10k_multiome'
cfg.dataset.zarr_path = "./pbmc10k_multiome.zarr"
cfg.dataset.celltypes = ','.join(celltype_for_modeling) # the celltypes you want to finetune
cfg.finetune.checkpoint = None # "./checkpoint-799.pth"
cfg.finetune.use_lora = False # True
cfg.finetune.strict = False
cfg.finetune.layers_with_lora = ['encoder', 'region_embed']
cfg.dataset.leave_out_celltypes = 'cd4_tcm'
cfg.dataset.quantitative_atac = True
cfg.dataset.leave_out_chromosomes = None #'chr10,chr11'
cfg.run.use_wandb=True # enabled wandb logging
cfg.machine.num_devices=1 # use 0 for cpu training; >=1 for gpu training
cfg.training.epochs=30 # use as demo there
#%% [markdown]
# # Model training...
#%%
trainer = run(cfg)
#%%
# ## Check the training metrics.
import pandas as pd
csv_logger_version = trainer.loggers[1].version
csv_logger_path = Path(trainer.log_dir) / f'csv_logs/lightning_logs/version_{csv_logger_version}/metrics.csv'
metrics = pd.read_csv(csv_logger_path)
metrics.query('~exp_pearson.isna()').plot(x='epoch', y='exp_spearman')
#%%
# ## Run inference to get the jacobian matrix for genes
# Change state to 'predict'
cfg.stage = 'predict'
cfg.machine.batch_size=1
# resume from the best checkpoint we just traineds
cfg.finetune.resume_ckpt = '/home/xf2217/output/finetune_pbmc10k_multiome/training_from_pretrained_lora/checkpoints/best-v1.ckpt' # trainer.checkpoint_callback.best_model_path 
cfg.run.run_name='interpret'
print('Saved checkpoint is at:', cfg.finetune.resume_ckpt, 'resuming...')
#%% [markdown]
# ## Prediction and interpretation over all genes
cfg.dataset.leave_out_celltypes = 'cd4_naive'
cfg.task.gene_list = None # set to None to predict all genes
  #%%
for cell_type in celltype_for_modeling:
    cfg.run.run_name = f'{cell_type}'
    cfg.dataset.leave_out_celltypes = cell_type
    run(cfg)
#%%
import zarr
zp = '/home/xf2217/output/finetune_pbmc10k_multiome/cd4_naive/cd4_naive.zarr'
z = zarr.open(zp)
# %% [markdown]
# ## Load the inference result as a celltype object
# %%
hydra_celltype = GETHydraCellType.from_config(cfg)
#%%
hydra_celltype.gene_annot
# %% [markdown]
# ## Get the jacobian matrix for MYC, summarize by region
# %%
hydra_celltype.get_gene_jacobian_summary('MYC', 'region')
# %% [markdown]
# ## Get the jacobian matrix for MYC, summarize by motif
# %%
hydra_celltype.get_gene_jacobian_summary('MYC', 'motif').sort_values().tail(20)*200
#%% [markdown]
# ## Run Mutation analysis for one mutation. Multiple mutations can be passed as a comma separated string like 'rs55705857,rs55705858'
# Note that the mutation analysis is still under development and the results might not be reliable, especially with the current peak-based model. The main purpose for this analysis would be to pinpoint important altered motifs, rather than predicting the corresponding expression changes, as the model is not
# trained on mutation or nucleotide resolution data. For more accurate mutation analysis, try ChromBPNet (https://www.biorxiv.org/content/10.1101/2024.12.25.630221v1).
#%%
fasta_path = Path(get_setting('genome_dir')) / 'hg38.fa'
cfg.machine.fasta_path = str(fasta_path)
cfg.task.mutations = 'rs55705857'
cfg.dataset.leave_out_celltypes = 'astrocyte'
cell_mut_col = GETHydraCellMutCollection(cfg)
# %%
cell_mut_col.variant_to_genes
# %%
scores = cell_mut_col.get_all_variant_scores()
# %%
scores
