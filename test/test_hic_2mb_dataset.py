"""
HiC Data Processing and Analysis Script
This script processes HiC data, generates motif scores, and saves results in Zarr format.
"""

import zarr
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import hicstraw
from cooltools.lib.numutils import adaptive_coarsegrain

# Configuration and model imports
from caesar.io.zarr_io import CelltypeDenseZarrIO
from get_model.config.config import load_config, pretty_print_config
from get_model.run_motif_adaptor import run
from caesar.io.genome import ChromSize
from atac_rna_data_processing.io.region import Genome

class ModelHandler:
    def __init__(self, config_name='nucleotide_motif_adaptor', device='cuda'):
        self.device = device
        self.model = self._initialize_model(config_name)
    
    def _initialize_model(self, config_name):
        cfg = load_config(config_name)
        cfg.stage = 'validate'
        cfg.finetune.resume_ckpt = '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
        cfg.finetune.checkpoint = cfg.finetune.resume_ckpt
        cfg.run.use_wandb = False
        cfg.dataset.leave_out_chromosomes = 'chr11'
        
        trainer = run(cfg)
        model = trainer.model.model.to(self.device)
        model.half()
        model.eval()
        return model
    
    def get_motif_for_sequence(self, seq):
        """Process a single sequence through the model."""
        return self.model(torch.from_numpy(np.stack([seq.one_hot])).cuda().half()).detach().cpu().numpy()
    
    def get_motif_for_sequences(self, sequences):
        """Process multiple sequences through the model."""
        X = torch.from_numpy(np.stack([seq.one_hot for seq in sequences])).int().to(self.device)
        return self.model(X).detach().cpu().numpy()
    
    def get_motif_for_region(self, genome, chrom, start, end):
        """Get motif scores for a genomic region."""
        seq = genome.get_sequence(chrom, start, end)
        return self.get_motif_for_sequence(seq)

class HiCDataProcessor:
    def __init__(self, hic_file):
        self.hic = hicstraw.HiCFile(hic_file)
    
    def get_matrix_from_hic_obj(self, chrom, start, end, resolution=5000, method="oe", 
                               normalization="SCALE", return_log=True):
        """Retrieve HiC matrix for a given genomic region."""
        mzd = self.hic.getMatrixZoomData(chrom, chrom, method, normalization, "BP", resolution)
        numpy_matrix = mzd.getRecordsAsMatrix(start, end, start, end)
        numpy_matrix = np.nan_to_num(numpy_matrix)
        if return_log:
            numpy_matrix = np.log10(numpy_matrix + 1)
        return numpy_matrix
    
    def get_coarse_grain_hic(self, chrom, start, end, resolution=5000):
        """Get coarse-grained HiC matrix."""
        countar = self.get_matrix_from_hic_obj(chrom, start, end, resolution=resolution, 
                                             method="observed", normalization="NONE", return_log=False)[0:-1, 0:-1]
        ar = self.get_matrix_from_hic_obj(chrom, start, end, resolution=resolution, 
                                        method="observed", normalization="KR", return_log=False)[0:-1, 0:-1]
        
        mean_count = np.load('mean_hic_cotact.npy')[0:ar.shape[0], 0:ar.shape[1]]
        c = adaptive_coarsegrain(ar, countar, cutoff=5)
        c = np.nan_to_num(c, posinf=0, neginf=0)[0:ar.shape[0], 0:ar.shape[1]]
        c = np.log((c + mean_count.min()) / (mean_count + mean_count.min()))
        
        # Mask zero-sum rows/columns
        ar_mask = np.where(ar.sum(0) == 0)[0]
        for i in ar_mask:
            c[i, :] = 0
            c[:, i] = 0
        return c

class ZarrWriter:
    def __init__(self, output_path):
        self.zarr_root = zarr.open_group(output_path)
        self.hic = self.zarr_root.create_group('hic', overwrite=True)
        self.motif = self.zarr_root.create_group('motif', overwrite=True)
        self.atac = self.zarr_root.create_group('atac', overwrite=True)
    
    def create_chromosome_groups(self, chromosomes):
        """Initialize chromosome groups in Zarr store."""
        for chrom in chromosomes:
            self.hic.create_group(chrom, overwrite=True)
            self.motif.create_group(chrom, overwrite=True)
            self.atac.create_group(chrom, overwrite=True)
    
    def write_data(self, chrom, start, hic_matrix, motif_score, atac_signal):
        """Write data for a genomic region to Zarr store."""
        self.hic[chrom].create_dataset(f'{start}', data=hic_matrix, 
                                     chunks=hic_matrix.shape, dtype=np.float16, overwrite=True)
        self.hic[chrom][f'{start}'].attrs['seq_id'] = f'{chrom}:{start}-{start+4000000}'
        
        self.motif[chrom].create_dataset(f'{start}', data=motif_score, 
                                       chunks=motif_score.shape, dtype=np.float16, overwrite=True)
        
        self.atac[chrom].create_dataset(f'{start}', data=atac_signal, 
                                      chunks=atac_signal.shape, dtype=np.float16, overwrite=True)

def main():
    # Initialize objects
    genome = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')
    chrom_sizes = ChromSize('hg38', '../data')
    hic_processor = HiCDataProcessor('/home/xf2217/Projects/get_data/H1_ESC.hic')
    model_handler = ModelHandler()
    zarr_writer = ZarrWriter('/home/xf2217/Projects/get_data/hic_matrix_4mb.zarr')
    dense_zarr = CelltypeDenseZarrIO('/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/')
    
    # Process chromosomes
    for chrom in tqdm(chrom_sizes.dict):
        if len(chrom) <= 5 and chrom not in ['chrM', 'chrY']:
            chrom_size = chrom_sizes.dict[chrom]
            zarr_writer.create_chromosome_groups([chrom])
            print(f"Processing {chrom}")
            
            for start in range(0, chrom_size//4000000*4000000-3000000, 1000000):
                try:
                    # Get HiC matrix
                    hic_matrix = hic_processor.get_coarse_grain_hic(chrom[3:], start, start+4000000)
                    
                    # Calculate motif scores
                    motif_score = np.zeros((4000000//50, 282))
                    for i in range(4000000//1000000):
                        motif_score_i = model_handler.get_motif_for_region(genome, chrom, start+i*1000000, start+(i+1)*1000000).reshape(-1, 50, 282).max(axis=1)
                        # Combine motif scores
                        motif_score[i*1000000//50:(i+1)*1000000//50, :] = motif_score_i
                    for i in range(1, 4000000//1000000):
                        motif_score_mid = model_handler.get_motif_for_region(genome, chrom, start+i*1000000-1000, start+i*1000000+1000).reshape(-1, 50, 282).max(axis=1)
                        motif_score[i*1000000//50-1000//50:i*1000000//50+1000//50, :] = (motif_score_mid + motif_score[i*1000000//50-1000//50:i*1000000//50+1000//50, :]) / 2
                    
                    # Get ATAC signal
                    atac_signal = dense_zarr.get_track_obj(
                        chrom, start, start+4000000, 
                        ['H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max']
                    ).convoluted_tracks_agg.reshape(-1, 50).mean(axis=1)
                    
                    # Write to Zarr store
                    zarr_writer.write_data(chrom, start, hic_matrix, motif_score, atac_signal)
                    
                except Exception as e:
                    print(f"Error processing {chrom}:{start}-{start+4000000}: {str(e)}")
                    continue

if __name__ == "__main__":
    main()
# #%%
# from get_model.dataset.zarr_dataset import HiCMatrix2MBDataset
# # Initialize dataset
# dataset = HiCMatrix2MBDataset(
#     zarr_path='/home/xf2217/Projects/get_data/hic_matrix_4mb.zarr',
#     is_train=False,
#     leave_out_chromosomes=['chr2'],  # optional
#     window_size=2000000,  # 2MB windows
#     resolution=5000       # 5kb resolution
# )

# # Get a sample
# sample = dataset[0]
# print(f"Chromosome: {sample['chrom']}")
# print(f"HiC matrix shape: {sample['hic'].shape}")
# print(f"Motif scores shape: {sample['motif'].shape}")
# # %%
# len(dataset)
# # %%
# dataset[0]['motif'].shape
# %%
import zarr
from caesar.io.zarr_io import DenseZarrIO
hg38_motif_zarr = '/home/xf2217/Projects/get_data/hg38_motif.zarr'
mz = DenseZarrIO(hg38_motif_zarr).get_track('chr2', 0, 2000000)[:, 16]
mz[mz<10]=10
mz = mz.reshape(-1, 5000).max(1)
# %%
mzs = zarr.open_group('/home/xf2217/Projects/get_data/hic_matrix_2mb.zarr')['motif/chr2']['0'][:]
mzs = mzs[:, 16]
mzs = mzs.reshape(-1, 100).max(1)
mzs[mzs<10]=10
# %%
sns.scatterplot(x=mzs, y=mz)
# %%
m = zarr.open_group('/home/xf2217/Projects/get_data/hic_matrix_2mb.zarr')['hic/chr2']['0'][:]
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %%
import pandas as pd
remap = pd.read_csv('/home/xf2217/Projects/get_model/test/remap_ctcf.bedgraph', sep='\t', header=None, names=['chrom', 'start', 'end', 'score'])
# %%
remap
# %%
bed_score = np.zeros(2000000)
for _, row in remap.query('chrom == "chr2" and start >= 0 and start < 2000000').iterrows():
    bed_score[row['start']:row['end']] = row['score']
# %%
bed_score = bed_score.reshape(-1, 5000).mean(1)
# %%
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [0.8, 0.1, 0.1, 0.1]})
sns.heatmap(m, ax=axs[0], cbar=False)
sns.lineplot(x=np.arange(bed_score.shape[0]), y=bed_score, ax=axs[1])
sns.lineplot(x=np.arange(mzs.shape[0]), y=mzs, ax=axs[2])
sns.lineplot(x=np.arange(mz.shape[0]), y=mz, ax=axs[3])
# %%
sns.scatterplot(x=mzs, y=bed_score)
plt.xlabel('CTCF Motif score')
plt.ylabel('Num. cell types with binding')
# %%
sns.histplot(x=mzs)
