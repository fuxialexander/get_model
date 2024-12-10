import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from concurrent.futures import ProcessPoolExecutor
import hicstraw
from tqdm import tqdm
import gc
from typing import Tuple, Dict, Any, List
import logging
from get_model.dataset.hic import get_hic_from_idx
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ZarrDataProcessor:
    def __init__(self, 
                 hic_file_path: str,
                 output_zarr_path: str,
                 output_scores_path: str,
                 atac_signals_path: str,
                 peaks_csv_path: str):
        """
        Initialize the ZarrDataProcessor with file paths and configuration.
        
        Args:
            hic_file_path: Path to the HiC file
            output_zarr_path: Path where the zarr store will be created
            output_scores_path: Path to the numpy file containing output scores
            atac_signals_path: Path to the numpy file containing ATAC signals
            peaks_csv_path: Path to the CSV containing peak information
        """
        self.hic_file_path = hic_file_path
        self.output_zarr_path = output_zarr_path
        self.output_scores_path = output_scores_path
        self.atac_signals_path = atac_signals_path
        self.peaks_csv_path = peaks_csv_path
        
        # Common configuration
        self.compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
        self.sample_size = 500
        self.stride = 200

    def initialize_zarr_store(self) -> zarr.Group:
        """Initialize the zarr store with required groups."""
        store = zarr.open(self.output_zarr_path, mode='w')
        groups = ['motifs', 'hic', 'hic_oe', 'peak_coords', 'atac']
        for group in groups:
            store.create_group(group, overwrite=True)
        return store

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load all required data files."""
        output_scores = np.load(self.output_scores_path).reshape(-1, 32, 282)
        atac_signals = np.load(self.atac_signals_path)
        peaks_df = pd.read_csv(self.peaks_csv_path)
        peaks_df = peaks_df.iloc[0:len(output_scores)]
        return output_scores, atac_signals, peaks_df

    @staticmethod
    def get_hic_from_idx(hic_file: hicstraw.HiCFile, peaks_df: pd.DataFrame, 
                        resolution: int = 5000, method: str = 'oe', 
                        normalization: str = 'KR', count_cutoff: int = 3) -> np.ndarray:
        """Get HiC matrix for given peaks."""
        return get_hic_from_idx(hic_file, peaks_df, resolution=resolution, method=method, normalization=normalization, count_cutoff=count_cutoff)

    def process_chromosome(self, chr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single chromosome's data.
        
        Args:
            chr_data: Dictionary containing chromosome processing information
            
        Returns:
            Dictionary containing processed arrays for the chromosome
        """
        chr_name = chr_data['chromosome']
        indices = chr_data['indices']
        peaks_chr = chr_data['peaks']
        output_scores_chr = chr_data['output_scores']
        atac_signals_chr = chr_data['atac_signals']
        
        if len(indices) < self.sample_size:
            logger.info(f"Skipping {chr_name}: not enough peaks ({len(indices)})")
            return None
            
        sample_indices = np.arange(0, len(peaks_chr) - self.sample_size, self.stride)
        n_samples = len(sample_indices)
        
        if n_samples == 0:
            logger.info(f"Skipping {chr_name}: no valid samples possible")
            return None
            
        # Initialize arrays for this chromosome
        motifs_data = np.zeros((n_samples, self.sample_size, 32, 282), dtype=np.float16)
        hic_data = np.zeros((n_samples, self.sample_size, self.sample_size), dtype=np.float16)
        hic_oe_data = np.zeros((n_samples, self.sample_size, self.sample_size), dtype=np.float16)
        peak_coords_data = np.zeros((n_samples, self.sample_size, 2), dtype=np.int32)
        atac_data = np.zeros((n_samples, self.sample_size, 32), dtype=np.float16)
        
        # Open HiC file for this process
        hic = {self.hic_file_path: hicstraw.HiCFile(self.hic_file_path)}
        valid_sample_count = 0
        
        for sample_idx in sample_indices:
            try:
                hic_oe_matrix = self.get_hic_from_idx(
                    hic,
                    peaks_chr.iloc[sample_idx:sample_idx + self.sample_size],
                    resolution=5000,
                    method='oe',
                    normalization='KR',
                    count_cutoff=1
                )
                hic_matrix = self.get_hic_from_idx(
                    hic,
                    peaks_chr.iloc[sample_idx:sample_idx + self.sample_size],
                    resolution=5000,
                    method='observed',
                    normalization='KR',
                    count_cutoff=1
                )
                
                if not isinstance(hic_oe_matrix, np.ndarray) or not isinstance(hic_matrix, np.ndarray):
                    continue

                # set hic_oe_matrix to 0 if hic_matrix is 0
                hic_oe_matrix[hic_matrix == 0] = 0
                    
                motifs_data[valid_sample_count] = output_scores_chr[sample_idx:sample_idx + self.sample_size]
                hic_data[valid_sample_count] = hic_matrix
                hic_oe_data[valid_sample_count] = hic_oe_matrix
                peak_coords_data[valid_sample_count] = peaks_chr.iloc[
                    sample_idx:sample_idx + self.sample_size
                ][['Start', 'End']].values
                atac_data[valid_sample_count] = atac_signals_chr[sample_idx:sample_idx + self.sample_size]
                
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
            'atac': atac_data,
            'valid_samples': valid_sample_count
        }

    def save_chromosome_data(self, store: zarr.Group, chr_results: Dict[str, Any]) -> None:
        """Save processed chromosome data to zarr store."""
        chr_name = chr_results['chromosome']
        
        # Create datasets for this chromosome
        store['motifs'].create_dataset(
            chr_name,
            data=chr_results['motifs'],
            chunks=(1, self.sample_size, 32, 282),
            compressor=self.compressor,
            dtype='float16'
        )
        
        store['hic'].create_dataset(
            chr_name,
            data=chr_results['hic'],
            chunks=(1, self.sample_size, self.sample_size),
            compressor=self.compressor,
            dtype='float16'
        )
        
        store['hic_oe'].create_dataset(
            chr_name,
            data=chr_results['hic_oe'],
            chunks=(1, self.sample_size, self.sample_size),
            compressor=self.compressor,
            dtype='float16'
        )
        
        store['peak_coords'].create_dataset(
            chr_name,
            data=chr_results['peak_coords'],
            chunks=(1, self.sample_size, 2),
            compressor=self.compressor
        )
        
        store['atac'].create_dataset(
            chr_name,
            data=chr_results['atac'],
            chunks=(1, self.sample_size, 32),
            compressor=self.compressor,
            dtype='float16'
        )

    def process_all_chromosomes(self, max_workers: int = 4) -> None:
        """
        Process all chromosomes in parallel using ProcessPoolExecutor.
        
        Args:
            max_workers: Maximum number of parallel processes to use
        """
        # Load all required data
        output_scores, atac_signals, peaks_df = self.load_data()
        store = self.initialize_zarr_store()
        
        # Prepare chromosome data for parallel processing
        chr_data_list = []
        for chr_name in peaks_df['Chromosome'].unique():
            if len(chr_name) > 5:  # Skip non-standard chromosomes
                continue
                
            chr_indices = np.where(peaks_df['Chromosome'] == chr_name)[0]
            peaks_chr = peaks_df[peaks_df['Chromosome'] == chr_name]
            
            chr_data_list.append({
                'chromosome': chr_name,
                'indices': chr_indices,
                'peaks': peaks_chr,
                'output_scores': output_scores[chr_indices],
                'atac_signals': atac_signals[chr_indices]
            })
        
        # Process chromosomes in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chr_results in tqdm(
                executor.map(self.process_chromosome, chr_data_list),
                total=len(chr_data_list),
                desc="Processing chromosomes"
            ):
                if chr_results is not None:
                    self.save_chromosome_data(store, chr_results)
                    
        logger.info("Completed processing all chromosomes")

def main():
    """Main function to run the parallel processing."""
    processor = ZarrDataProcessor(
        hic_file_path='/home/xf2217/Projects/get_data/H1_ESC.hic',
        output_zarr_path='h1_esc_nucleotide_motif_adaptor_output_5000.zarr',
        output_scores_path='preprocessed_data/h1_regions_motif_scores.npy',
        atac_signals_path='preprocessed_data/h1_regions_atac_signals.npy',
        peaks_csv_path='preprocessed_data/union_peaks.csv'
    )
    
    processor.process_all_chromosomes(max_workers=14)

if __name__ == '__main__':
    main()
