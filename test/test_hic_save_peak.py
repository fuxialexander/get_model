#%%
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from pyranges import PyRanges as pr
import torch
from atac_rna_data_processing.io.region import Genome, GenomicRegionCollection
from caesar.io.zarr_io import CelltypeDenseZarrIO
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import hicstraw

from get_model.config.config import load_config
from get_model.run_motif_adaptor import run
from get_model.dataset.hic import get_hic_from_idx
#%%
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')

def get_motif_vector(X: np.ndarray, detector_threshold: int = 5) -> np.ndarray:
    """Calculate the motif vector for a given sequence.
    
    Args:
        X: Motif scores array of shape (sequence_length, n_motifs)
        
    Returns:
        Array of motif scores, one score per motif
    """
    scores = []
    for motif_idx in range(X.shape[1]):
        detector = np.diff(np.where(X[:,motif_idx]<detector_threshold)[0])
        if len(detector[detector>=detector_threshold])>0:
            scores.append((X[:,motif_idx][X[:,motif_idx]>=detector_threshold]).sum()/detector[detector>=detector_threshold].mean())
        else:
            scores.append(0)
    return np.array(scores)

class MotifProcessor:
    """Class to handle motif score processing for sequences."""
    
    def __init__(self, model_path: str):
        """Initialize with model path and setup model."""
        cfg = load_config('nucleotide_motif_adaptor')
        cfg.stage = 'validate'
        cfg.finetune.resume_ckpt = model_path
        cfg.finetune.checkpoint = model_path
        cfg.run.use_wandb = False
        
        trainer = run(cfg)
        self.model = trainer.model.model.to('cuda').half().eval()

    def process_sequence(self, sequence) -> np.ndarray:
        """Process a single sequence through the model."""
        with torch.no_grad():
            X = torch.from_numpy(sequence.one_hot).unsqueeze(0).half().to('cuda')
            output = self.model(X).detach().cpu().numpy()[0]
            # Remove padding and get motif vector
            output = output[56:-56]  # Remove padding
            return get_motif_vector(output)

    def process_sequences(self, sequences: List) -> np.ndarray:
        """Process multiple sequences and return their motif vectors."""
        motif_scores = []
        for seq in tqdm(sequences, desc="Processing sequences"):
            motif_scores.append(self.process_sequence(seq))
        return np.array(motif_scores)

class HiCProcessor:
    """Class to handle HiC data processing."""
    
    def __init__(self, hic_file_path: str, output_path: str):
        """Initialize with file paths and setup zarr store."""
        self.hic_file_path = hic_file_path
        self.output_path = output_path
        self.compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.sample_size = 500
        self.stride = 200
        
    def process_chromosome(self, chr_data: Dict) -> Optional[Dict]:
        """Process HiC data for a single chromosome."""
        chr_name = chr_data['chromosome']
        peaks_chr = chr_data['peaks']
        motif_scores = chr_data['motif_scores']
        atac_signals = chr_data['atac_signals']
        
        if len(peaks_chr) < self.sample_size:
            logger.info(f"Skipping {chr_name}: not enough peaks ({len(peaks_chr)})")
            return None
            
        sample_indices = np.arange(0, len(peaks_chr) - self.sample_size, self.stride)
        n_samples = len(sample_indices)
        
        if n_samples == 0:
            logger.info(f"Skipping {chr_name}: no valid samples possible")
            return None
            
        # Initialize arrays for this chromosome
        motifs_data = np.zeros((n_samples, self.sample_size, motif_scores.shape[1]), dtype=np.float16)
        hic_data = np.zeros((n_samples, self.sample_size, self.sample_size), dtype=np.float16)
        hic_oe_data = np.zeros((n_samples, self.sample_size, self.sample_size), dtype=np.float16)
        peak_coords_data = np.zeros((n_samples, self.sample_size, 2), dtype=np.int32)
        atac_data = np.zeros((n_samples, self.sample_size), dtype=np.float16)
        
        try:
            # hic = {self.hic_file_path: hicstraw.HiCFile(self.hic_file_path)}
            hic = HiCDataProcessor('/home/xf2217/Projects/get_data/resources/4DNFI9GMP2J8.rebinned.mcool')
            valid_sample_count = 0
            
            for sample_idx in sample_indices:
                try:
                    sample_slice = slice(sample_idx, sample_idx + self.sample_size)
                    peaks_sample = peaks_chr.iloc[sample_slice]
                    # if peak span > 4M, continue
                    if peaks_sample['End'].max() - peaks_sample['Start'].min() > 4000000:
                        continue
                    # Get HiC matrices
                    m = get_hic_from_idx(hic, peaks_chr, method='oe', normalization=True, count_cutoff=1, resolution=1000, coarse_grain=False)
                    hic_oe = get_hic_from_idx(hic, peaks_sample, method='oe', normalization='KR', count_cutoff=2, resolution=2000)
                    hic_obs = get_hic_from_idx(hic, peaks_sample, method='observed', normalization='KR', count_cutoff=2, resolution=2000)
                    
                    if not isinstance(hic_oe, np.ndarray) or not isinstance(hic_obs, np.ndarray):
                        continue
                        
                    # Set hic_oe to 0 where hic_obs is 0
                    hic_oe[hic_obs == 0] = 0
                    
                    motifs_data[valid_sample_count] = motif_scores[sample_slice]
                    hic_data[valid_sample_count] = hic_obs
                    hic_oe_data[valid_sample_count] = hic_oe
                    peak_coords_data[valid_sample_count] = peaks_sample[['Start', 'End']].values
                    atac_data[valid_sample_count] = atac_signals[sample_slice]
                    
                    valid_sample_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {chr_name} at index {sample_idx}: {str(e)}")
                    continue
            
            # Trim arrays to valid samples
            if valid_sample_count < n_samples:
                motifs_data = motifs_data[:valid_sample_count]
                hic_data = hic_data[:valid_sample_count]
                hic_oe_data = hic_oe_data[:valid_sample_count]
                peak_coords_data = peak_coords_data[:valid_sample_count]
                atac_data = atac_data[:valid_sample_count]
            
            return {
                'chromosome': chr_name,
                'motifs': motifs_data,
                'hic': hic_data,
                'hic_oe': hic_oe_data,
                'peak_coords': peak_coords_data,
                'atac': atac_data
            }
            
        except Exception as e:
            logger.error(f"Error processing {chr_name}: {str(e)}")
            return None

    def save_results(self, results: Dict) -> None:
        """Save processed results to zarr store."""
        store = zarr.open(self.output_path, mode='w')
        
        # Create groups
        for group in ['motifs', 'hic', 'hic_oe', 'peak_coords', 'atac']:
            store.create_group(group)
            
        # Save data for each chromosome
        for chr_name, data in results.items():
            if data is None:
                continue
                
            for key in ['motifs', 'hic', 'hic_oe', 'peak_coords', 'atac']:
                chunks = (1, self.sample_size, self.sample_size) if key in ['hic', 'hic_oe'] else None
                store[key].create_dataset(
                    chr_name,
                    data=data[key],
                    chunks=chunks,
                    compressor=self.compressor
                )

class ATACProcessor:
    """Class to handle ATAC signal processing."""
    
    def __init__(self, dense_zarr_path: str, celltype: str, max_workers: int = 16):
        """Initialize ATAC processor.
        
        Args:
            dense_zarr_path: Path to dense zarr file
            celltype: Cell type identifier
            max_workers: Maximum number of parallel workers
        """
        self.dense_zarr = CelltypeDenseZarrIO(dense_zarr_path)
        self.celltype = celltype
        self.max_workers = max_workers

    def get_atac_signal(self, args: Tuple) -> np.ndarray:
        """Get ATAC signal for a given region."""
        _, celltype, row = args
        return self.dense_zarr.get_track(
            celltype,
            row['Chromosome'],
            row['Start'],
            row['End']
        ).mean()  # Aggregate signal over the peak region

    def process_peaks(self, peaks_df: pd.DataFrame) -> np.ndarray:
        """Process ATAC signals for all peaks in parallel."""
        logger.info("Processing ATAC signals...")
        
        args_list = [
            (self.dense_zarr, self.celltype, row)
            for _, row in peaks_df.iterrows()
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            atac_signals = list(tqdm(
                executor.map(self.get_atac_signal, args_list),
                total=len(args_list),
                desc="Processing ATAC signals"
            ))
        
        return np.array(atac_signals)
#%%



def main():
    """Main function to run the complete processing pipeline."""
    # Initialize processors
    motif_processor = MotifProcessor(
        '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
    )
    
    hic_processor = HiCProcessor(
        '/home/xf2217/Projects/get_data/H1_ESC.hic',
        'h1_esc_nucleotide_motif_adaptor_output_peak_2000.zarr'
    )
    
    atac_processor = ATACProcessor(
        dense_zarr_path='/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/',
        celltype='H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max'
    )
    
    # Load peaks
    peaks = pd.read_csv('remap_ctcf.bedgraph', sep='\t', 
                       names=['Chromosome', 'Start', 'End', 'Score'])
    peaks = peaks.query('Score > 1')
    
    # Process by chromosome
    results = {}
    for chr_name in peaks['Chromosome'].unique():
        # try:
        if len(chr_name) > 5:  # Skip non-standard chromosomes
            continue

                
        logger.info(f"Processing {chr_name}")
        peaks_chr = peaks[peaks['Chromosome'] == chr_name]
        
        # Get sequences and process motifs
        regions = GenomicRegionCollection(genome=hg38, df=peaks_chr)
        sequences = regions.collect_sequence()
        motif_scores = motif_processor.process_sequences(sequences.sequences)
        
        # Get ATAC signals for this chromosome
        atac_signals = atac_processor.process_peaks(peaks_chr)
        
        # Process HiC data
        chr_results = hic_processor.process_chromosome({
            'chromosome': chr_name,
            'peaks': peaks_chr,
            'motif_scores': motif_scores,
            'atac_signals': atac_signals
        })
        
        results[chr_name] = chr_results
        # except Exception as e:
        #     logger.error(f"Error processing {chr_name}: {str(e)}")
            
    
    # Save all results
    hic_processor.save_results(results)

if __name__ == '__main__':
    main() 
# %%
from get_model.dataset.hic import HiCDataProcessor, get_hic_from_idx
#%%
import pandas as pd
peaks = pd.read_csv('remap_ctcf.bedgraph', sep='\t', 
                       names=['Chromosome', 'Start', 'End', 'Score'])
peaks_chr = peaks[peaks['Chromosome'] == 'chr2'].iloc[500:900]
hic = HiCDataProcessor('/home/xf2217/Projects/get_data/4DNFI9GMP2J8.mcool')
# hic.generate_mean_contact_matrix(resolution=1000, matrix_size=2000, eps=0.000001, max_iterations=1000)
#%%
m = hic.get_submatrix_from_regions(peaks_chr, method='observed', normalization=False, return_log=False, coarse_grain=False, resolution=1000, count_cutoff=0)
# %%
m.shape
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(m)
# %%
hic.resolutions

# %%
# %%
# m = hic.get_matrix(resolution=1000, chrom='chr2', start=2000000, end=4000000, method='observed', normalization=True, return_log=False, count_cutoff=0)
m, m_raw = hic.get_coarse_grain_matrix(resolution=1000, chrom='chr2', start=2000000, end=4000000, method='observed', normalization=True, return_log=False, count_cutoff=0)
# %%
plt.imshow(np.log(m))
# %%