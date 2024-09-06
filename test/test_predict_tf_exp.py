#%%
from get_model.dataset.zarr_dataset import RegionMotifDataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from caesar.io.gencode import Gencode
# load the zarr file as a dataset. 
region_motif_dataset = RegionMotifDataset(
    "/home/xf2217/Projects/4dn_h1esc/peak_motif/original/H1ESC.4dn_h1esc.4dn_h1esc.peak_motif.zarr/",
    celltypes="H1ESC.4dn_h1esc.4dn_h1esc",
    quantitative_atac=True,
    num_region_per_sample=400,
    leave_out_celltypes=None,
    leave_out_chromosomes="chr11",
    is_train=False,
    hic_path="/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic",
    hic_resolution=5000,
    hic_method="observed",
    hic_normalization="KR"
)

gencode = Gencode(gtf_dir="/home/xf2217/Projects/get_data/")
# %%
id_to_name = gencode.gtf[['gene_name', 'gene_id']].drop_duplicates().set_index('gene_id').to_dict()['gene_name']
# %%
import pandas as pd
rna = pd.read_csv('/home/xf2217/Projects/4dn_h1esc/raw/4DNFIXN6KBV6.tsv', sep='\t')
# %%
rna['gene_name'] = rna['gene_id'].str.split('.').str[0].map(id_to_name)
# %%
rna.dropna()
# %%
tf_list = pd.read_csv('/home/xf2217/Projects/get_model/modules/caesar/data/tf_list.txt', header=None)[0].tolist()

# %%
rna[rna['gene_name'].isin(tf_list)]

# %%
tf_motif_map = pd.read_csv('/home/xf2217/Projects/atac_rna_data_processing/human/tf_list.motif_cluster.csv')
# %%
tf_motif_map.query('in_cluster=="True"')


# %%
region_motif_dataset.region_motifs['H1ESC.4dn_h1esc.4dn_h1esc'].data.shape
# %%
