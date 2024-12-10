import zarr
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from .utils import _chromosome_splitter

class CuratedPeakMotifHiCDataset(Dataset):
    """Dataset for peak-level motif and HiC data."""
    
    def __init__(self, curated_zarr: str, is_train=True, leave_out_chromosomes: Optional[str] = None):
        """Initialize dataset.
        
        Args:
            curated_zarr (str): Path to zarr store
            is_train (bool): Whether this is training data
            leave_out_chromosomes (Optional[str]): Comma-separated list of chromosomes to exclude
        """
        self.curated_zarr = curated_zarr
        self.is_train = is_train
        self.leave_out_chromosomes = leave_out_chromosomes
        self.zarr_store = zarr.open(self.curated_zarr, mode='r')
        self.chromosomes = list(self.zarr_store['motifs'].keys())
        
        self.setup()
    
    def setup(self):
        """Load data into memory and set up indices."""
        input_chromosomes = _chromosome_splitter(
            self.chromosomes, self.leave_out_chromosomes, self.is_train
        )
        
        self.chrom_indices = []
        self.sample_size = 0
        self.motifs = {}
        self.hic = {}
        self.hic_oe = {}
        self.peak_coords = {}
        self.atac = {}
        
        for chrom in input_chromosomes:
            num_samples = len(self.zarr_store['motifs'][chrom])
            self.chrom_indices.append((chrom, self.sample_size, self.sample_size + num_samples))
            self.sample_size += num_samples
            
            # Load data to memory
            self.motifs[chrom] = self.zarr_store['motifs'][chrom][:].astype(np.float16)
            self.hic[chrom] = self.zarr_store['hic'][chrom][:].astype(np.float16)
            self.hic_oe[chrom] = self.zarr_store['hic_oe'][chrom][:].astype(np.float16)
            self.peak_coords[chrom] = self.zarr_store['peak_coords'][chrom][:].astype(np.int32)
            self.atac[chrom] = self.zarr_store['atac'][chrom][:].astype(np.float16)
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, index):
        """Get a single sample.
        
        Args:
            index (int): Sample index
            
        Returns:
            dict: Sample data including motif features, HiC matrices, and distances
        """
        chrom, start_idx = self._get_chrom_and_index(index)
        
        # Get data for this sample
        motif = self.motifs[chrom][index - start_idx]
        atac = self.atac[chrom][index - start_idx]
        hic = self.hic[chrom][index - start_idx]
        hic_oe = self.hic_oe[chrom][index - start_idx]
        peak_coords = self.peak_coords[chrom][index - start_idx]

        # Calculate distances
        peak_mid = (peak_coords[:, 0] + peak_coords[:, 1]) / 2
        # Add random jitter to peak positions
        peak_mid += np.random.randint(-50, 50, size=peak_mid.shape)
        
        # Calculate various distance metrics
        distance_1d = np.log10((peak_mid - peak_mid.min())/1000 + 1)
        distance_map = np.log10(np.abs(peak_coords[:, 0][:, None] - peak_coords[:, 1][None, :]) + 1)
        relative_distance = np.diff(peak_mid)
        distance_to_previous = np.log10(np.concatenate([[peak_coords[0, 1]-peak_coords[0, 0]], relative_distance]) + 1)
        distance_to_next = np.log10(np.concatenate([relative_distance, [peak_coords[-1, 1]-peak_coords[-1, 0]]]) + 1)
        
        # Stack distance features
        distance_1d = np.stack([distance_1d, distance_to_next, distance_to_previous], axis=1)
        
        sample = {
            "chrom": chrom,
            "motif": motif.reshape(-1, 1, motif.shape[-1]).astype(np.float16),  # shape: num_peaks, 1, num_motifs
            "atac": atac.reshape(-1, 1, 1).astype(np.float16),  # shape: num_peaks, 1, 1
            "hic": hic.astype(np.float16),  # shape: num_peaks, num_peaks
            "hic_oe": hic_oe.astype(np.float16),  # shape: num_peaks, num_peaks
            "distance_1d": distance_1d.astype(np.float16),  # shape: num_peaks, 3
            "distance_map": distance_map.astype(np.float16)  # shape: num_peaks, num_peaks
        }
        
        return self.transform(sample)
    
    def transform(self, sample):
        """Apply any transformations to the sample."""
        return sample

    def _get_chrom_and_index(self, index):
        """Get chromosome and local index for a global index."""
        for chrom, start, end in self.chrom_indices:
            if start <= index < end:
                return chrom, start
        raise IndexError("Index out of range")
    
    def __repr__(self):
        return f"CuratedPeakMotifHiCDataset(curated_zarr={self.curated_zarr}, is_train={self.is_train}, leave_out_chromosomes={self.leave_out_chromosomes})" 