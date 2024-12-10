#%%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from get_model.dataset.hic import HiCDataProcessor
k562 = HiCDataProcessor('/home/xf2217/Projects/encode_hg38atac/raw/ENCFF621AIY.hic')
k562
#%%
h1esc = HiCDataProcessor('/home/xf2217/Projects/get_data/H1_ESC.hic')
h1esc
#%%
h1_jz = HiCDataProcessor('/home/xf2217/Projects/get_data/resources/4DNFI9GMP2J8.rebinned.mcool')
h1_jz
#%%
CHROM = 'chr5'
START = 10000000+2000000
END = 10000000+6000000
RESOLUTION = 5000
#%%
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
k562.plot_matrix(CHROM, START, END, resolution=RESOLUTION, count_cutoff=1, normalization='NONE', method='observed', ax=axs[0])
h1esc.plot_matrix(CHROM, START, END, resolution=RESOLUTION, count_cutoff=1, normalization='NONE', method='observed', ax=axs[1])
h1_jz.plot_matrix(CHROM, START, END, resolution=RESOLUTION, count_cutoff=1, normalization=False, method='observed', ax=axs[2])
# %%
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
h1esc.plot_matrix(CHROM, START, END, resolution=RESOLUTION, coarse_grain=True, count_cutoff=2, normalization='KR', method='oe', ax=axs[0], cmap='Blues', vmin=0, vmax=1)
k562.plot_matrix(CHROM, START, END, resolution=RESOLUTION, coarse_grain=True, count_cutoff=2, normalization='SCALE', method='oe', ax=axs[1], cmap='Blues', vmin=0, vmax=1)
h1_jz.plot_matrix(CHROM, START, END, resolution=RESOLUTION, coarse_grain=True, count_cutoff=2, normalization='KR', method='oe', ax=axs[2], cmap='Blues', vmin=0, vmax=1)
# %%
from tqdm import tqdm
for i in tqdm(range(1000)):
    k562.get_coarse_grain_matrix(CHROM, START, END, resolution=RESOLUTION, count_cutoff=2, normalization='SCALE', method='oe')
# %%
