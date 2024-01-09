# %%
import logging

import numpy as np
import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm

from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr',
                            '/pmglocal/xf2217/get_data/bingren_adult_dense.zarr'],
                           '/pmglocal/xf2217/get_data/hg38.zarr', [
                           '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/manitou/pmg/users/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue', preload_count=2, n_packs=1,
                           max_peak_length=5000, center_expand_target=500, n_peaks_lower_bound=5, n_peaks_upper_bound=200)
pretrain.__len__()
#%%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=2,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn
)
# %%
for i, batch in tqdm(enumerate(data_loader_train)):
    sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, peak_split, mask, n_peaks, max_n_peaks, total_peak_len, sample_total_insertion, conv_sizes, motif_mean_std  = batch
    if min(peak_split)<0:
        continue
    break
#%%
import matplotlib.pyplot as plt
y = sample_track[3].float().cpu().numpy()[1024*1:1024*2]
# plt.plot(np.convolve(y, np.ones(int(np.floor(1024/y.sum()))*2), mode='same'))
# plt.plot(sample_track[0].float().cpu().numpy()[0:2000])
#%%
plt.plot(y)
#%%
# fft to check the squareness of the signal
import matplotlib.pyplot as plt
conv_y = np.convolve(y, np.ones(400)/400, mode='same')
# y = sample_track[0].float().cpu().numpy()[0:2000]
plt.plot(np.abs(np.fft.fft(conv_y/conv_y.max()))[1:conv_y.shape[0]//2])
# quantify the squareness
#%%

# kurtosis
from scipy.stats import kurtosis
conv_y = np.convolve(y, np.ones(200), mode='same')
kurtosis(np.abs(np.fft.fft(conv_y/conv_y.max())), fisher=False)
#%%
np.abs(np.fft.fft(conv_y/conv_y.max())).var()
#%%
vars = []
y = sample_track[1].float().cpu().numpy()[12000:13000]
for conv_size in range(1, 300):
    conv_y = np.convolve(y, np.ones(conv_size)/conv_size, mode='same')
    vars += [kurtosis(np.abs(np.fft.fft(conv_y/conv_y.max(),)))]

x = np.arange(1, 300)
plt.plot(x, vars)
#%%
from get_model.model.model import GETPretrain
#%%
model = GETPretrain(
        num_regions=200,
        num_res_block=0,
        motif_prior=True,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=True,
        nhead=12,
        dropout=0.1,
        output_dim=1274,
        pos_emb_components=[],
    )
model.eval()
model.cuda()
#%%
bool_mask_pos = mask.clone()
bool_mask_pos[bool_mask_pos == -10000] = 0

peak_seq = peak_seq.bfloat16().cuda()
sample_track = sample_track.bfloat16().cuda()
# #%%
# del model
# del sample_track
# del peak_seq
# torch.cuda.empty_cache()
#%%
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    a = model.forward(peak_seq, sample_track, bool_mask_pos.bool().cuda(), peak_split, n_peaks.cuda(), max_n_peaks,
                      sample_total_insertion.cuda(), motif_mean_std.cuda())
#%%
import seaborn as sns
d = (a[2]).detach().cpu().numpy().reshape(-1,1290)
#%%
d_std = (d-d.mean(0))/d.std(0)
# row_color based on samples, each sample has 200 row
row_color = np.repeat(np.arange(2), 200)
# d = (d-d.mean(0))/d.std(0)

# assign color to each row
row_color =  [sns.color_palette("Set2", 2)[i] for i in row_color]
row_color = np.array(row_color)[d.mean(1)!=0]

d = d[d.mean(1)!=0]
d_std = d_std[d_std.mean(1)!=0]
#%%

d_log = np.log10(d+1)
#%%
# kmeans and average the cluster
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=256, random_state=0).fit(d.T)
#%%
d_centroid = np.zeros((d.shape[0], 256))
for i in range(256):
    d_centroid[:,i] = d[:,kmeans.labels_==i].mean(1)


#%%
sns.clustermap(d_std, cmap='RdBu_r',row_cluster=True, col_cluster=True, method='ward')
# %%
sns.clustermap((d_log-d_log.mean(0))/d_log.std(), cmap='Blues', row_cluster=False, col_cluster=True, method='complete', row_colors=row_color)
# %%
a[2].device
# %%
import pandas as pd 
pd.read_csv('../log.txt', sep='loss', skiprows=442).iloc[:,1].str.split('(').str[1].str.split(')').str[0].astype(float).plot(xlim=(-200,1562*10), ylim=(0.3, 0.6))
# %%
