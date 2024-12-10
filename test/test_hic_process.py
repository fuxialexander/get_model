#%%
from cooltools.lib.numutils import adaptive_coarsegrain
import cooler
import torch
import numpy as np
import hicstraw
import numpy as np
from caesar.io.zarr_io import CelltypeDenseZarrIO
hic = hicstraw.HiCFile('/home/xf2217/Projects/get_data/H1_ESC.hic')

# %%
from get_model.dataset.hic import get_hic_from_idx
import pyranges as pr
from get_model.dataset.zarr_dataset import RegionMotif, RegionMotifConfig 

from caesar.io.zarr_io import CelltypeDenseZarrIO 

dense_zarr = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/')
celltype = 'H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max'
#%%
peaks = dense_zarr.call_peaks()
#%%
import pandas as pd
ctcf_peak = pd.read_csv('remap_ctcf.bedgraph', sep='\t', header=None, names=['Chromosome', 'Start', 'End', 'Score']).query('Score> 10')
#%%
# union of ctcf peaks and h1 peaks
union_peak = pr.set_union(pr(ctcf_peak), pr(peaks)).merge().as_df()
# %%
from caesar.io.genome import ChromSize 
hg38 = ChromSize('hg38', '../data')
from get_model.config.config import load_config, pretty_print_config
from get_model.run_motif_adaptor import run
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
motif_clusters = pd.read_csv("../../geneformer_esc/data/motif_cluster.txt", sep='\t', names=['cluster'])['cluster'].values

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

#%%
from atac_rna_data_processing.io.region import GenomicRegionCollection, Genome
hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
#%%
import torch
def get_motif_for_sequence(seq):
    return trainer.model.model(torch.from_numpy(np.stack([seq.one_hot])).cuda().half()).detach().cpu().numpy()
def get_motif_for_region(chrom, start, end):
    seq = hg38.get_sequence(chrom, start, end)
    return get_motif_for_sequence(seq)
def get_motif_for_sequences(sequences):
    X = torch.from_numpy(np.stack([sequences[j].one_hot for j in range(len(sequences))])).int().to('cuda')
    return trainer.model.model(X).detach().cpu().numpy()

#%%
PEAK_EXTEND = 512
SCAN_CHUNK_SIZE = 32
SCAN_CHUNKS = PEAK_EXTEND // SCAN_CHUNK_SIZE
union_peak_ = union_peak.copy().query('Chromosome.isin(["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"])')
union_peak_['summit'] = (union_peak_['Start'] + union_peak_['End']) // 2
union_peak_['Start'] = union_peak_['summit'] - PEAK_EXTEND
union_peak_['End'] = union_peak_['summit'] + PEAK_EXTEND
h1_regions = GenomicRegionCollection(genome=hg38, df = union_peak_)
h1_regions_sequences = h1_regions.collect_sequence(upstream=64, downstream=64)

#%%
import torch
from tqdm import tqdm
batch_size = 32
output_scores = []
for i in tqdm(range(len(h1_regions_sequences.sequences)//batch_size)):
    # get a batch of sequences
    output = get_motif_for_sequences(h1_regions_sequences.sequences[i*batch_size:(i+1)*batch_size])
    output = output[:, 64:-64, :] 
    output = output.reshape(-1, SCAN_CHUNKS, SCAN_CHUNK_SIZE, 282).max(axis=2)
    output_scores.append(output)
output_scores = np.array(output_scores)
# save output_scores
np.save('h1_regions_motif_scores.npy', output_scores)
#%%
output_scores = np.load('h1_regions_motif_scores.npy')
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
    atac = dense_zarr.get_track(celltype, row['Chromosome'], row['Start'], row['End']).reshape(SCAN_CHUNKS, SCAN_CHUNK_SIZE).max(axis=1)
    return atac
#%%
# Create arguments for each task
args_list = [(dense_zarr, celltype, row) for _, row in union_peak_.iterrows()]

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
np.save('h1_regions_atac_signals.npy', atac_signals)
#%%
atac_signals = np.load('h1_regions_atac_signals.npy')
#%%
# save union_peaks
union_peak_.to_csv('union_peaks.csv', index=False)
#%%

#%%
# Process and save data chromosome by chromosome
saved_peaks = union_peak_.iloc[0:len(output_scores)]
#%%
import hicstraw
import numpy as np
# Initialize zarr file and create groups
import zarr
from numcodecs import Blosc

z = zarr.open('h1_esc_nucleotide_motif_adaptor_output.zarr', mode='w')
z.create_group('motifs', overwrite=True)
z.create_group('hic', overwrite=True)
z.create_group('peak_coords', overwrite=True)
z.create_group('atpm', overwrite=True)
z.create_group('atac', overwrite=True)

output_scores = np.load('h1_regions_motif_scores.npy')
atac_signals = np.load('h1_regions_atac_signals.npy')
union_peak_ = pd.read_csv('union_peaks.csv')
saved_peaks = union_peak_.iloc[0:len(output_scores)]
from tqdm import tqdm
from caesar.io.zarr_io import CelltypeDenseZarrIO
hic = hicstraw.HiCFile('/home/xf2217/Projects/get_data/H1_ESC.hic')
from numcodecs import Blosc
#%%
# for chr in tqdm(saved_peaks['Chromosome'].unique()):
#     if len(chr) > 5:
#         continue
#     # Get indices for current chromosome
#     chr_indices = np.where(saved_peaks['Chromosome']==chr)[0]
    
#     # Skip if we don't have enough peaks for at least one sample
#     if len(chr_indices) < 400:
#         print(f"Skipping {chr}: not enough peaks ({len(chr_indices)})")
#         continue
    
#     # Truncate to multiple of 400
#     saved_peaks_chr = saved_peaks.query('Chromosome == @chr')
#     print(chr_indices[0:10])
#     output_scores_chr = output_scores[chr_indices]
#     atac_signals_chr = atac_signals[chr_indices]
#     chr_size = len(saved_peaks_chr)
#     # construct indices into samples of 400 peaks each, stride 100
#     sample_indices = np.arange(0, chr_size-400, 100)
#     print(sample_indices[0:10])
#     # Lists to store valid samples
#     valid_motif_samples = []
#     valid_hic_samples = []
#     valid_peak_coords = []
#     valid_atac = []
#     # Process each sample of 400 peaks
#     for sample_idx in tqdm(sample_indices, desc=f'Processing {chr}'):
#         # Get HiC matrix for current sample
#         hic_matrix = get_hic_from_idx(hic, saved_peaks_chr.iloc[sample_idx:sample_idx+400], resolution=5000, method='oe', normalization='KR', return_log=False)
        
#         # Skip if HiC matrix is not valid
#         if not isinstance(hic_matrix, np.ndarray):
#             continue

            
#         # Get corresponding motif data
#         motif_data = output_scores_chr[sample_idx:sample_idx+400]
#         peak_coords = saved_peaks_chr.iloc[sample_idx:sample_idx+400][['Start', 'End']].values
        
#         # Store valid samples
#         valid_motif_samples.append(motif_data.astype(np.float16))
#         valid_hic_samples.append(hic_matrix.astype(np.float16))
#         valid_peak_coords.append(peak_coords)
#         valid_atac.append(atac_signals_chr[sample_idx:sample_idx+400].astype(np.float16))
#     # Skip chromosome if no valid samples
#     if not valid_motif_samples:
#         print(f"Skipping {chr}: no valid samples")
#         continue
    
#     # Stack valid samples
#     valid_motif_samples = np.stack(valid_motif_samples).astype(np.float16)
#     valid_hic_samples = np.stack(valid_hic_samples).astype(np.float16)
#     valid_peak_coords = np.stack(valid_peak_coords)
#     valid_atac = np.stack(valid_atac).astype(np.float16)
#     print(f"{chr}: {len(valid_motif_samples)} valid samples out of {len(sample_indices)} total")
    
#     # Save motif data
#     z['motifs'].create_dataset(chr, 
#                            data=valid_motif_samples,
#                            chunks=(1, 400, 32, 282),
#                            compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
#                            overwrite=True,
#                            dtype='float16')
    
#     # Save HiC matrices
#     z['hic'].create_dataset(chr, 
#                            data=valid_hic_samples,
#                            chunks=(1, 400, 400),
#                            compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
#                            overwrite=True,
#                            dtype='float16')

    
#     # Save peak coordinates
#     z['peak_coords'].create_dataset(chr,
#                                   data=valid_peak_coords,
#                                   chunks=(1, 400, 2),
#                                   compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
#                                   overwrite=True)


    
#     # Save atac
#     z['atac'].create_dataset(chr,
#                                   data=valid_atac,
#                                   chunks=(1, 400, 32),
#                                   dtype='float16',
#                                   compressor=Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE),
#                                   overwrite=True)
#     # remove valid_motif_samples, valid_hic_samples, valid_peak_coords, valid_atac
#     del valid_motif_samples, valid_hic_samples, valid_peak_coords, valid_atac
#     import gc
#     gc.collect()







# %%
import hicstraw
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm
import pandas as pd
#%%
from cooltools.lib.numutils import adaptive_coarsegrain
import cooler
import torch
import numpy as np
import hicstraw
import numpy as np
from caesar.io.zarr_io import CelltypeDenseZarrIO
hic = hicstraw.HiCFile('/home/xf2217/Projects/get_data/H1_ESC.hic')

# Initialize zarr file and create groups
z = zarr.open('h1_esc_nucleotide_motif_adaptor_output.zarr', mode='w')
z.create_group('motifs', overwrite=True)
z.create_group('hic', overwrite=True)
z.create_group('peak_coords', overwrite=True)
z.create_group('atpm', overwrite=True)
z.create_group('atac', overwrite=True)

# Load initial data
output_scores = np.load('h1_regions_motif_scores.npy').reshape(-1, 32, 282)
atac_signals = np.load('h1_regions_atac_signals.npy')
union_peak_ = pd.read_csv('union_peaks.csv')
saved_peaks = union_peak_.iloc[0:len(output_scores)]

# Initialize HiC file
hic = hicstraw.HiCFile('/home/xf2217/Projects/get_data/H1_ESC.hic')

# Common compressor configuration
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

for chr in tqdm(saved_peaks['Chromosome'].unique()):
    if len(chr) > 5:
        continue
        
    # Get indices and data for current chromosome
    saved_peaks_chr = saved_peaks.query('Chromosome == @chr')
    chr_indices = np.where(saved_peaks['Chromosome']==chr)[0]
    
    if len(chr_indices) < 400:
        print(f"Skipping {chr}: not enough peaks ({len(chr_indices)})")
        continue
    
    output_scores_chr = output_scores[chr_indices]
    atac_signals_chr = atac_signals[chr_indices]
    chr_size = len(saved_peaks_chr)
    
    # Calculate number of samples
    sample_indices = np.arange(0, chr_size-400, 100)
    n_samples = len(sample_indices)
    
    if n_samples == 0:
        print(f"Skipping {chr}: no valid samples possible")
        continue
    
    # Pre-create zarr arrays with appropriate shapes
    motifs_array = z['motifs'].create_dataset(chr,
        shape=(n_samples, 400, 32, 282),
        chunks=(1, 400, 32, 282),
        compressor=compressor,
        dtype='float16',
        overwrite=True)
        
    hic_array = z['hic'].create_dataset(chr,
        shape=(n_samples, 400, 400),
        chunks=(1, 400, 400),
        compressor=compressor,
        dtype='float16',
        overwrite=True)
        
    peak_coords_array = z['peak_coords'].create_dataset(chr,
        shape=(n_samples, 400, 2),
        chunks=(1, 400, 2),
        compressor=compressor,
        overwrite=True)
        
    atac_array = z['atac'].create_dataset(chr,
        shape=(n_samples, 400, 32),
        chunks=(1, 400, 32),
        compressor=compressor,
        dtype='float16',
        overwrite=True)
    
    # Process and save each sample directly
    valid_sample_count = 0
    for sample_idx_pos, sample_idx in enumerate(tqdm(sample_indices, desc=f'Processing {chr}')):
        # Get HiC matrix
        try:
            hic_matrix = get_hic_from_idx(hic, 
                saved_peaks_chr.iloc[sample_idx:sample_idx+400], 
                resolution=5000, 
                method='oe', 
                normalization='KR', 
                return_log=False)
        except Exception as e:
            print(e)
            continue
        
        # Skip invalid HiC matrices
        if not isinstance(hic_matrix, np.ndarray):
            continue
            
        # Get corresponding data for this sample
        motif_data = output_scores_chr[sample_idx:sample_idx+400]
        peak_coords = saved_peaks_chr.iloc[sample_idx:sample_idx+400][['Start', 'End']].values
        atac_data = atac_signals_chr[sample_idx:sample_idx+400]
        
        # Save directly to zarr arrays
        motifs_array[valid_sample_count] = motif_data.astype(np.float16)
        hic_array[valid_sample_count] = hic_matrix.astype(np.float16)
        peak_coords_array[valid_sample_count] = peak_coords
        atac_array[valid_sample_count] = atac_data.astype(np.float16)
        
        valid_sample_count += 1
        
    # If we had any invalid samples, resize the arrays
    if valid_sample_count < n_samples:
        print(f"{chr}: {valid_sample_count} valid samples out of {n_samples} total")
        
        motifs_array.resize(valid_sample_count, 400, 32, 282)
        hic_array.resize(valid_sample_count, 400, 400)
        peak_coords_array.resize(valid_sample_count, 400, 2)
        atac_array.resize(valid_sample_count, 400, 32)
    
    # Clean up memory after each chromosome
    import gc
    gc.collect()
# %%
from get_model.dataset.zarr_dataset import CuratedMotifHiCDataset
val_dataset = CuratedMotifHiCDataset(
    curated_zarr='h1_esc_nucleotide_motif_adaptor_output.zarr',
    is_train=False,
    leave_out_chromosomes='chr16,chr17,chr18'
)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True)
fig.set_dpi(100)
idx=2
sns.heatmap(val_dataset[idx]['hic_oe'], ax=axs[0], cbar=False, cmap='RdBu_r')
sns.heatmap(val_dataset[idx]['hic'], ax=axs[1], cbar=False, cmap='RdBu_r')
sns.heatmap(val_dataset[idx]['distance_map'], ax=axs[2], cbar=False, cmap='Blues')
plt.show()
# %%
