#%%
from get_model.config.config import load_config, pretty_print_config
from get_model.run_motif_adaptor import run
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
motif_clusters = pd.read_csv("../../geneformer_esc/data/motif_cluster.txt", sep='\t', names=['cluster'])['cluster'].values

# inline
%matplotlib inline
#%%
# load config
cfg = load_config('nucleotide_motif_adaptor')
pretty_print_config(cfg)
#%%
cfg.stage='validate'
cfg.finetune.resume_ckpt = '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
cfg.finetune.checkpoint = '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
cfg.run.use_wandb=False
cfg.dataset.leave_out_chromosomes = 'chr11'
trainer = run(cfg)
#%%
trainer.model.model.to('cuda')
trainer.model.model.half()
trainer.model.model.eval()
# %%
# for i, batch in enumerate(trainer.val_dataloaders):
#     if i == 2:
#         print(batch['sequence'].shape)
#         print(batch['motif'].shape)
#         input_data = trainer.model.model.get_input(batch)
#         input_data['sequence'] = input_data['sequence'].to('cuda').half()
#         output = trainer.model.model(**input_data)
#         a = batch['motif'][:,50:-50,:].cpu().numpy().flatten()
#         b = output[:,50:-50,:].detach().cpu().numpy().flatten()
#         print(np.corrcoef(a, b))
#         break
# # %%
# sns.scatterplot(x=a, y=b,s=3, palette='tab20')
# plt.xlabel('Observed')
# plt.ylabel('Predicted')
# plt.show()
# #%%
# motif_idx = 142
# plt.plot(a.reshape(-1,282)[:,motif_idx])
# plt.plot(b.reshape(-1,282)[:,motif_idx])
# plt.show()
# # %%
# np.unique((a.reshape(-1,282)[:,motif_idx][a.reshape(-1,282)[:,motif_idx]>=5])).sum()

# #%%
# # find length of consecutive true in b.reshape(-1,282)[:,16]>5
# detector = np.diff(np.where(b.reshape(-1,282)[:,motif_idx]<5)[0])
# (b.reshape(-1,282)[:,motif_idx][b.reshape(-1,282)[:,motif_idx]>=5]).sum()/detector[detector>=5].mean()
# # %%
# # heatmap side by side
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# sns.heatmap(a.reshape(-1,282), ax=axs[0], label='Observed', vmax=25)
# sns.heatmap(b.reshape(-1,282), ax=axs[1], label='Predicted', vmax=25)
# axs[0].set_title('Observed')
# axs[1].set_title('Predicted')
# plt.show()
# #%%
# DNA_BASES = ['A', 'C', 'G', 'T']
# def one_hot_to_seq(one_hot):
#     return ''.join([DNA_BASES[np.argmax(one_hot)] for one_hot in one_hot])
# # %%
# one_hot_to_seq(batch['sequence'][0, 220:350])
# # %%
# # use deep lift shap to examine the kernel learned by the model
# from tangermeme.deep_lift_shap import deep_lift_shap
# from tangermeme.utils import one_hot_encode
# X = one_hot_encode(one_hot_to_seq(batch['sequence'][1])).unsqueeze(0).float().to('cuda')
# # %%
# import torch
# # wrap model to be a function that permute x (0,2,1) to conform the input requirement of deep lift shap
# class ModelPermute(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, x):
#         return self.model(x.permute(0,2,1)).permute(0,2,1)
# # %%
# model_permute = ModelPermute(trainer.model.model).to('cuda').half().eval()
# X_attr_permute = deep_lift_shap(model_permute, X.half(), batch_size=1, target=35, random_state=0, n_shuffles=100)
# # %%
# from matplotlib import pyplot as plt
# import seaborn; seaborn.set_style('whitegrid')
# from tangermeme.plot import plot_logo
# plt.figure(figsize=(5, 1))
# ax = plt.subplot(111)
# plot_logo(X_attr_permute[0], ax=ax)

# plt.xlabel("Genomic Coordinate")
# plt.xlim(220, 350)
# plt.show()
# # %%
# # Validate on k562 regionxmotif data to see if the model can predict the motif vector
# from scipy.sparse import load_npz
# from atac_rna_data_processing.io.region import GenomicRegionCollection, Genome
# from tqdm import tqdm
# k562_peak = pd.read_csv('/home/xf2217/Projects/pretrain_human_bingren_shendure_apr2023/k562/k562_bulk_peak_400.csv',index_col=0)
# k562_npz = load_npz('/home/xf2217/Projects/pretrain_human_bingren_shendure_apr2023/k562/k562_bulk_peak_400.natac.npz')
# assert k562_npz.shape[0] == k562_peak.shape[0]
# hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
# k562_regions = GenomicRegionCollection(genome=hg38, df = k562_peak.head(5000))
# k562_regions_sequences = k562_regions.collect_sequence(upstream=56, downstream=56)
# # %%
# def get_motif_vector(X):
#     """A function to calculate the motif vector for a given sequence"""
#     scores = []
#     for motif_idx in range(X.shape[1]):
#         detector = np.diff(np.where(X[:,motif_idx]<5)[0])
#         if len(detector[detector>=5])>0:
#             scores.append((X[:,motif_idx][X[:,motif_idx]>=5]).sum()/detector[detector>=5].mean())
#         else:
#             scores.append(0)
#     return np.array(scores)
# # %%
# output_scores = []
# for i in tqdm(range(5000)):
#     X = one_hot_encode(k562_regions_sequences.sequences[i].seq).T
#     output = trainer.model.model(X.unsqueeze(0).float().to('cuda').half()).detach().cpu().numpy()[0]
#     output_score = get_motif_vector(output[56:-56])
#     output_scores.append(output_score)
# output_scores = np.array(output_scores)
# # %%
# # plot correlation for a single motif, CTCF
# sns.scatterplot(x=output_scores[:, 16], y=k562_npz[0:5000][:, 16].toarray().flatten(), s=1)
# # %%
# # calculate correlation for each motif
# corrs = []
# n_instances = []
# for i in range(282):
#     n_instances.append(np.where(k562_npz[0:5000][:, i].toarray().flatten()!=0)[0].shape[0])
#     corrs.append(np.corrcoef(output_scores[:, i], k562_npz[0:5000][:, i].toarray().flatten())[0,1])

# corrs = np.array(corrs)

# # %%
# # plot correlation vs number of instances to see if the correlation is biased by the number of instances
# sns.jointplot(x=n_instances, y=corrs)
# plt.xlabel('Number of Instances')
# plt.ylabel('Correlation')
# plt.show()
# %%
# curate hic training data input
from get_model.dataset.zarr_dataset import CuratedMotifHiCDataset, RegionMotif, RegionMotifConfig, get_hic_from_idx 

h1 = RegionMotif(RegionMotifConfig(root='/home/xf2217/Projects/get_data/', data='H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max.peak_motif.zarr', celltype='H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max', normalize=False, motif_scaler=1.0, leave_out_motifs=None))
from atac_rna_data_processing.io.region import GenomicRegionCollection, Genome
from tqdm import tqdm
#%%
h1.peaks['summit'] = (h1.peaks['Start'] + h1.peaks['End'])//2
h1.peaks['Start'] = h1.peaks['summit'] - 1024
h1.peaks['End'] = h1.peaks['summit'] + 1024
h1.peaks['atpm'] = h1.atpm
#%%
from caesar.io.zarr_io import CelltypeDenseZarrIO
# for each peak, get the atac signal from dense zarr
dense_zarr = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/')
celltype = 'H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max'
atac_signals = []
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
def get_atac_signal(args):
    dense_zarr, celltype, row = args
    atac = dense_zarr.get_track(celltype, row['Chromosome'], row['Start'], row['End']).reshape(64, 32).mean(axis=1)
    return atac
#%%
# Create arguments for each task
args_list = [(dense_zarr, celltype, row) for _, row in h1.peaks.iterrows()]

# Process in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=16) as executor:
    atac_signals = list(tqdm(
        executor.map(get_atac_signal, args_list), 
        total=len(args_list),
        desc="Processing ATAC signals"
    ))
#%%
import numpy as np
atac_signals = np.stack(atac_signals)
np.save('h1_esc_atac_signals.npy', atac_signals)
#%%
import numpy as np
atac_signals = np.load('h1_esc_atac_signals.npy')
# %%

hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
h1_regions = GenomicRegionCollection(genome=hg38, df = h1.peaks)
#%%
h1_regions_sequences = h1_regions.collect_sequence(upstream=50, downstream=50)

# %%

import torch
batch_size = 32
output_scores = []
for i in tqdm(range(len(h1_regions_sequences.sequences)//batch_size)):
    # get a batch of sequences
    X = torch.from_numpy(np.stack([h1_regions_sequences.sequences[j].one_hot for j in range(i*batch_size, (i+1)*batch_size)])).float().to('cuda').half()
    output = trainer.model.model(X).detach().cpu().numpy()
    output = output[:, 50:-50, :] 
    # bin with 32 bp
    output = output.reshape(-1, 64, 32, 282).mean(axis=2)
    output_scores.append(output)
output_scores = np.array(output_scores)
#%%
output_scores = output_scores.reshape(-1, 64, 282)
#%%
# get output for h1_regions_sequences.sequences[1].seq
import torch
plt.plot(trainer.model.model(torch.from_numpy(np.stack([h1_regions_sequences.sequences[4].one_hot for j in range(1)])).float()).detach().cpu().numpy()[0, 50:-50, 16]/132)
chrom, start, end = h1.peaks.iloc[4]['Chromosome'], h1.peaks.iloc[4]['Start'], h1.peaks.iloc[4]['End']
atac_track = dense_zarr.get_track_obj(chrom, start, end, ids=[celltype])
plt.plot(atac_track.convoluted_tracks_agg)
#%%
plt.plot(atac_signals[4, :])

#%%
# save to npy
np.save('h1_esc_nucleotide_motif_adaptor_output.npy', output_scores)
#%%
output_scores = np.load('h1_esc_nucleotide_motif_adaptor_output.npy')
#%%
# Initialize zarr file and create groups
import zarr
from numcodecs import Blosc

z = zarr.open('h1_esc_nucleotide_motif_adaptor_output.zarr', mode='w')
z.create_group('motifs', overwrite=True)
z.create_group('hic', overwrite=True)
z.create_group('hic_10k', overwrite=True)
z.create_group('peak_coords', overwrite=True)
z.create_group('atpm', overwrite=True)
z.create_group('atac', overwrite=True)

#%%
# Process and save data chromosome by chromosome
saved_peaks = h1.peaks.iloc[0:len(output_scores)]
#%%
import hicstraw
import numpy as np
from caesar.io.zarr_io import CelltypeDenseZarrIO
hic = hicstraw.HiCFile('/home/xf2217/Projects/get_data/H1_ESC.hic')

#%%
for chr in tqdm(saved_peaks['Chromosome'].unique()):
    # Get indices for current chromosome
    chr_indices = np.where(saved_peaks['Chromosome']==chr)[0]
    
    # Skip if we don't have enough peaks for at least one sample
    if len(chr_indices) < 400:
        print(f"Skipping {chr}: not enough peaks ({len(chr_indices)})")
        continue
    
    # Truncate to multiple of 400
    saved_peaks_chr = saved_peaks.query('Chromosome == @chr')
    print(chr_indices[0:10])
    output_scores_chr = output_scores[chr_indices]
    atac_signals_chr = atac_signals[chr_indices]
    chr_size = len(saved_peaks_chr)
    # construct indices into samples of 400 peaks each, stride 100
    sample_indices = np.arange(0, chr_size-400, 100)
    print(sample_indices[0:10])
    # Lists to store valid samples
    valid_motif_samples = []
    valid_hic_samples = []
    valid_peak_coords = []
    valid_atpm = []
    valid_atac = []
    valid_hic_samples_10k = []
    # Process each sample of 400 peaks
    for sample_idx in tqdm(sample_indices, desc=f'Processing {chr}'):
        # Get HiC matrix for current sample
        hic_matrix = get_hic_from_idx(hic, saved_peaks_chr.iloc[sample_idx:sample_idx+400], method='oe', normalization='KR', resolution=5000, count_cutoff=0)
        hic_matrix_10k = get_hic_from_idx(hic, saved_peaks_chr.iloc[sample_idx:sample_idx+400], method='oe', normalization='KR', resolution=10000, count_cutoff=0)
        
        # Skip if HiC matrix is not valid
        if not isinstance(hic_matrix, np.ndarray):
            continue

        if not isinstance(hic_matrix_10k, np.ndarray):
            continue
            
        # Get corresponding motif data
        motif_data = output_scores_chr[sample_idx:sample_idx+400]
        peak_coords = saved_peaks_chr.iloc[sample_idx:sample_idx+400][['Start', 'End']].values
        
        # Store valid samples
        valid_motif_samples.append(motif_data)
        valid_hic_samples.append(hic_matrix)
        valid_hic_samples_10k.append(hic_matrix_10k)
        valid_peak_coords.append(peak_coords)
        valid_atpm.append(saved_peaks_chr.iloc[sample_idx:sample_idx+400].atpm.values)
        valid_atac.append(atac_signals_chr[sample_idx:sample_idx+400])
    # Skip chromosome if no valid samples
    if not valid_motif_samples:
        print(f"Skipping {chr}: no valid samples")
        continue
    
    # Stack valid samples
    valid_motif_samples = np.stack(valid_motif_samples)
    valid_hic_samples = np.stack(valid_hic_samples)
    valid_peak_coords = np.stack(valid_peak_coords)
    valid_atpm = np.stack(valid_atpm)
    valid_atac = np.stack(valid_atac)
    valid_hic_samples_10k = np.stack(valid_hic_samples_10k)
    print(f"{chr}: {len(valid_motif_samples)} valid samples out of {len(sample_indices)} total")
    
    # Save motif data
    z['motifs'].create_dataset(chr, 
                           data=valid_motif_samples,
                           chunks=(1, 400, 64, 282),
                           compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                           overwrite=True,
                           dtype='float16')
    
    # Save HiC matrices
    z['hic'].create_dataset(chr, 
                           data=valid_hic_samples,
                           chunks=(1, 400, 400),
                           compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                           overwrite=True,
                           dtype='float16')
    z['hic_10k'].create_dataset(chr, 
                           data=valid_hic_samples_10k,
                           chunks=(1, 400, 400),
                           compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                           overwrite=True,
                           dtype='float16')
    
    # Save peak coordinates
    z['peak_coords'].create_dataset(chr,
                                  data=valid_peak_coords,
                                  chunks=(1, 400, 2),
                                  compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                                  overwrite=True)

    # Save atpm
    z['atpm'].create_dataset(chr,
                                  data=valid_atpm,
                                  chunks=(1, 400),
                                  compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                                  overwrite=True)
    
    # Save atac
    z['atac'].create_dataset(chr,
                                  data=valid_atac,
                                  chunks=(1, 400, 64),
                                  dtype='float16',
                                  compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
                                  overwrite=True)
#%%
# Print summary statistics
print("\nDataset Summary:")
print("-" * 50)
for chr in z['motifs'].keys():
    n_samples = len(z['motifs'][chr])
    print(f"{chr}: {n_samples} samples")
print("-" * 50)

#%%
from get_model.dataset.zarr_dataset import CuratedMotifHiCDataset
dataset = CuratedMotifHiCDataset(
        curated_zarr='h1_esc_nucleotide_motif_adaptor_output.zarr',
        is_train=False,
        leave_out_chromosomes='chr2'
    )
# %%

# dst_count = np.log10(numpy_matrix_count + 1)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(dst, ax=axs, cbar=False)
plt.show()
#%%
dst[dst>1.2] = 1
# %%
plt.plot(dst.max(0))
# %%
from caesar.io.zarr_io import CelltypeDenseZarrIO
# for each peak, get the atac signal from dense zarr
dense_zarr = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/')
celltype = 'H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max'
atac = dense_zarr.get_track(celltype, 'chr1', 5000000, 7000000).reshape(-1, 5000).mean(1)
#%%
peak_track = np.zeros(2000000) 
for i, row in h1.peaks.query('Chromosome == "chr1"').iterrows():
    try:
        peak_track[row['Start']-5000000:row['End']-5000000] = row['atpm']
    except:
        continue
# %%
peak_track = peak_track.reshape(-1, 5000).max(1)

#%%
# plot atac and dst max in one plot, heatmap on top
fig, axs = plt.subplots(4, 1, figsize=(4, 8), sharex=True, gridspec_kw={'height_ratios': [5, 1, 1, 1]})
sns.heatmap(dst, ax=axs[0], cbar=False)
axs[0].set_title('')
axs[1].plot(atac/atac.max())
axs[1].set_title('')
axs[2].plot((peak_track>0.01) * (dst.max(0)/dst.max())[0:400])
axs[2].set_title('')
axs[2].set_xticks([])
axs[3].plot(dst.max(0)/dst.max())
axs[3].set_title('')
plt.show()
# %%
sns.heatmap(numpy_matrix_count<3)
# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(get_hic_from_idx(hic, h1.peaks.query('Chromosome == "chr1" & Start > 5000000 & End < 7000000'), method='oe', normalization='KR', resolution=5000, count_cutoff=0), ax=axs, cbar=False)
plt.show()
#%%
import numpy as np
mean_count = np.load('mean_hic_cotact.npy')
# %%
mzd_count = hic.getMatrixZoomData(
        '9', '9', 'observed', 'KR', "BP", 5000
)
numpy_matrix_count = mzd_count.getRecordsAsMatrix(  
    109892000, 111892000, 109892000, 111892000
)[0:400,0:400]
mean_count = mean_count[0:400, 0:400]
numpy_matrix_count = numpy_matrix_count + mean_count.min()
dst = np.nan_to_num(np.log(numpy_matrix_count/(mean_count+mean_count.min())))
import matplotlib.pyplot as plt
import seaborn as sns
#%%
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(dst, ax=axs, cbar=False, cmap='RdBu_r')
plt.show()
# %%
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(get_hic_from_idx(hic, h1.peaks.query('Chromosome=="chr9" & Start>109892000 & End < 111892000'), method='oe', normalization='KR', resolution=5000, count_cutoff=0), ax=axs, cbar=False, cmap='RdBu_r')
plt.show()
#%%
# for row/col in dst that does not overlap with a h1.peaks, set them to 0
# Assuming `h1.peaks` contains columns 'Start' and 'End' in terms of the chromosome coordinates,
# we first need to convert these to matrix indices in `dst`.
# %%

# This example assumes that each bin is 5000 bp, so we can calculate the start and end indices accordingly.

start_coord = 109892000  # Start coordinate of the matrix window
end_coord = 111892000  # End coordinate of the matrix window
bin_size = 5000  # Resolution/bin size
mzd_count = hic.getMatrixZoomData(
        '8', '8', 'observed', 'KR', "BP", bin_size
)
numpy_matrix_count = mzd_count.getRecordsAsMatrix(  
    start_coord, end_coord, start_coord, end_coord
)[0:400,0:400]
mean_count = mean_count[0:400, 0:400]
numpy_matrix_count = numpy_matrix_count + mean_count.min()
dst = np.nan_to_num(np.log(numpy_matrix_count/(mean_count+mean_count.min())))
dst_nonmasked = dst.copy()
# Initialize a mask of the same shape as `dst` with all zeros
mask = np.zeros(dst.shape, dtype=bool)

peak_start_idxs = []
# Iterate over 1each peak in `h1.peaks` and mark the overlapping regions in the mask
for _, peak in h1.peaks.query('Chromosome=="chr8" & Start > @start_coord & End < @end_coord').iterrows():
    peak_start_idx = (peak['Start'] - start_coord) // bin_size
    peak_end_idx = (peak['End'] - start_coord) // bin_size
    peak_start_idxs.append(peak_start_idx)
# idx not in peak_start_idxs
non_peak_idxs = np.setdiff1d(np.arange(dst.shape[0]), peak_start_idxs)
mask[non_peak_idxs, :] = True
mask[:, non_peak_idxs] = True
# Apply the mask to set non-overlapping regions in `dst` to zero
dst[mask] = np.nan
#%%
# use low resolution for jupyter notebook plot
# Visualize the result
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
sns.heatmap(dst, ax=axs[0], cbar=False, cmap='RdBu_r')
sns.heatmap(dst_nonmasked, ax=axs[1], cbar=False, cmap='RdBu_r')
plt.tight_layout(dpi=100)
plt.show()
#%%
sns.heatmap(get_hic_from_idx(hic, h1.peaks.query('Chromosome=="chr8" & Start > 109892000 & End < 111892000'), method='oe', normalization='KR', resolution=5000, count_cutoff=0))
# %%
dense_zarr = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/')
celltype = 'H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max'
# chr8:109,509,411-109,572,685
atac = dense_zarr.get_track_obj('chr8', 109509411, 109572685, [celltype])
plt.plot(atac.convoluted_tracks_agg)
# %%
h1.peaks[['Chromosome', 'Start', 'End']].to_csv('h1_esc_peaks.bed', sep='\t', index=False)
# %%
