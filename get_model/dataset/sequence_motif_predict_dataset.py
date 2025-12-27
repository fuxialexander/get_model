"""
Dataset for sequence-based motif prediction.

This module provides a dataset class that loads sequence data from dense zarr files
and provides sequences for motif prediction training. Chromosomes are loaded to
memory for faster data access.
"""

from dataclasses import dataclass
from typing import Optional, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from caesar.io.zarr_io import DenseZarrIO


def _chromosome_splitter(
    all_chromosomes: list, leave_out_chromosomes: str | list | None, is_train=True
):
    """Split chromosomes for train/val split."""
    input_chromosomes = all_chromosomes.copy()
    if leave_out_chromosomes is None:
        leave_out_chromosomes = []
    elif isinstance(leave_out_chromosomes, str):
        if "," in leave_out_chromosomes:
            leave_out_chromosomes = leave_out_chromosomes.split(",")
        else:
            leave_out_chromosomes = [leave_out_chromosomes]

    if is_train or leave_out_chromosomes == [""] or leave_out_chromosomes == []:
        input_chromosomes = [
            chrom for chrom in input_chromosomes if chrom not in leave_out_chromosomes
        ]
    else:
        input_chromosomes = (
            all_chromosomes if leave_out_chromosomes == [] else leave_out_chromosomes
        )

    if isinstance(input_chromosomes, str):
        input_chromosomes = [input_chromosomes]
    print("Leave out chromosomes:", leave_out_chromosomes)
    print("Input chromosomes:", input_chromosomes)
    return input_chromosomes


@dataclass
class SequenceMotifPredictDatasetConfig:
    """Configuration for SequenceMotifPredictDataset."""
    sequence_zarr: str
    sequence_length: int = 512
    leave_out_chromosomes: Optional[str] = None
    is_train: bool = True
    dataset_size: int = 40960
    eval_dataset_size: int = 4096
    peaks_bed: Optional[str] = None  # Path to BED file with peaks for oversampling
    peak_oversample_ratio: float = 3.0  # Ratio of peak samples to non-peak samples


class SequenceMotifPredictDataset(Dataset):
    """
    Dataset for sequence-based motif prediction.
    
    Loads sequence data from dense zarr files and provides sequences for training.
    Chromosomes are loaded to memory for faster data access.
    
    Args:
        sequence_zarr: Path to sequence dense zarr file
        sequence_length: Length of sequences to extract
        leave_out_chromosomes: Comma-separated list of chromosomes to exclude
        is_train: Whether this is training data (affects chromosome split)
        dataset_size: Size of training dataset
        eval_dataset_size: Size of evaluation dataset
    """
    
    def __init__(
        self,
        sequence_zarr: str,
        sequence_length: int = 512,
        leave_out_chromosomes: Optional[str] = None,
        is_train: bool = True,
        dataset_size: int = 40960,
        eval_dataset_size: int = 4096,
        sequence_io: Optional[Any] = None,  # Optional shared DenseZarrIO object
        peaks_bed: Optional[str] = None,  # Path to BED file with peaks for oversampling
        peak_oversample_ratio: float = 3.0,  # Ratio of peak samples to non-peak samples
    ):
        # Use shared sequence_io if provided, otherwise create new one
        if sequence_io is not None:
            self.sequence_dataset = sequence_io
        else:
            self.sequence_dataset = DenseZarrIO(sequence_zarr, dtype="int8", mode="r")
            # Load entire chromosomes to memory for faster IO
            self.sequence_dataset.load_to_memory_dense()
        
        self.sequence_length = sequence_length
        self.leave_out_chromosomes = leave_out_chromosomes
        self.is_train = is_train
        self.dataset_size = dataset_size if is_train else eval_dataset_size
        self.peaks_bed = peaks_bed
        self.peak_oversample_ratio = peak_oversample_ratio
        
        # Peaks will be loaded in setup() after determining input_chromosomes
        # This ensures leave_out_chromosomes are completely excluded
        self.peaks_df = None
        
        self.setup()
    
    def _load_peaks(self, input_chromosomes: list) -> pd.DataFrame:
        """Load peaks from BED file, filtered to only include input chromosomes."""
        try:
            peaks_df = pd.read_csv(
                self.peaks_bed,
                sep='\t',
                header=None,
                names=['Chromosome', 'Start', 'End'],
                usecols=[0, 1, 2]
            )
            # Ensure chromosome names match zarr format (e.g., 'chr1' not '1')
            peaks_df['Chromosome'] = peaks_df['Chromosome'].astype(str)
            
            # Filter peaks to only include chromosomes in input_chromosomes
            # This ensures leave_out_chromosomes are completely excluded
            peaks_df = peaks_df[peaks_df['Chromosome'].isin(input_chromosomes)]
            
            print(f"Loaded {len(peaks_df)} peaks from {self.peaks_bed} (filtered to {len(input_chromosomes)} chromosomes)")
            return peaks_df
        except Exception as e:
            print(f"Warning: Could not load peaks from {self.peaks_bed}: {e}")
            return None
    
    def _is_in_peak(self, chrom: str, start_pos: int, end_pos: int) -> bool:
        """Check if a region overlaps with any peak."""
        if self.peaks_df is None:
            return False
        
        chrom_peaks = self.peaks_df[self.peaks_df['Chromosome'] == chrom]
        if len(chrom_peaks) == 0:
            return False
        
        # Check for overlap: region overlaps peak if start < peak_end and end > peak_start
        overlaps = (
            (start_pos < chrom_peaks['End'].values) & 
            (end_pos > chrom_peaks['Start'].values)
        )
        return overlaps.any()
    
    def setup(self):
        """Set up chromosome indices and sample positions."""
        # Get chromosome names from chrom_sizes keys
        chrom_names = list(self.sequence_dataset.chrom_sizes.keys())
        input_chromosomes = _chromosome_splitter(
            chrom_names,
            self.leave_out_chromosomes,
            self.is_train
        )
        
        # Load peaks filtered to only input chromosomes
        # For training: input_chromosomes excludes leave_out_chromosomes
        # For validation: input_chromosomes = leave_out_chromosomes
        if self.peaks_bed:
            self.peaks_df = self._load_peaks(input_chromosomes)
        
        self.chrom_indices = []
        peak_positions = []
        non_peak_positions = []
        
        for chrom in input_chromosomes:
            chrom_size = self.sequence_dataset.chrom_sizes[chrom]
            # Generate sample positions across the chromosome
            # Sample sequences with stride equal to sequence_length
            num_samples = max(1, chrom_size // self.sequence_length)
            
            for i in range(num_samples):
                start_pos = i * self.sequence_length
                end_pos = min(start_pos + self.sequence_length, chrom_size)
                
                if end_pos - start_pos >= self.sequence_length:
                    pos_tuple = (chrom, start_pos, end_pos)
                    # Check if region overlaps with peaks
                    if self.peaks_df is not None and self._is_in_peak(chrom, start_pos, end_pos):
                        peak_positions.append(pos_tuple)
                    else:
                        non_peak_positions.append(pos_tuple)
        
        # Oversample peak regions
        if len(peak_positions) > 0 and self.peak_oversample_ratio > 0:
            # Calculate target number of peak vs non-peak samples
            # If we want ratio R of peak:non-peak, and total is T:
            # peak_count = T * R / (R + 1)
            # non_peak_count = T / (R + 1)
            total_available = len(peak_positions) + len(non_peak_positions)
            
            if total_available > 0:
                peak_target = int(self.dataset_size * self.peak_oversample_ratio / (self.peak_oversample_ratio + 1))
                non_peak_target = self.dataset_size - peak_target
                
                # Sample peak positions (with replacement if needed)
                if len(peak_positions) >= peak_target:
                    peak_indices = np.random.choice(
                        len(peak_positions),
                        size=peak_target,
                        replace=False
                    )
                    sampled_peaks = [peak_positions[i] for i in peak_indices]
                else:
                    # Repeat if not enough peaks
                    sampled_peaks = peak_positions.copy()
                    while len(sampled_peaks) < peak_target:
                        sampled_peaks.extend(peak_positions)
                    sampled_peaks = sampled_peaks[:peak_target]
                
                # Sample non-peak positions
                if len(non_peak_positions) >= non_peak_target:
                    non_peak_indices = np.random.choice(
                        len(non_peak_positions),
                        size=non_peak_target,
                        replace=False
                    )
                    sampled_non_peaks = [non_peak_positions[i] for i in non_peak_indices]
                else:
                    # Repeat if not enough non-peaks
                    sampled_non_peaks = non_peak_positions.copy()
                    while len(sampled_non_peaks) < non_peak_target:
                        sampled_non_peaks.extend(non_peak_positions)
                    sampled_non_peaks = sampled_non_peaks[:non_peak_target]
                
                self.sample_positions = sampled_peaks + sampled_non_peaks
                print(f"Sampled {len(sampled_peaks)} peak regions and {len(sampled_non_peaks)} non-peak regions")
            else:
                self.sample_positions = peak_positions + non_peak_positions
        else:
            # No peaks or no oversampling: use original logic
            self.sample_positions = peak_positions + non_peak_positions
            
            # Limit dataset size
            if len(self.sample_positions) > self.dataset_size:
                # Randomly sample if we have more positions than needed
                indices = np.random.choice(
                    len(self.sample_positions),
                    size=self.dataset_size,
                    replace=False
                )
                self.sample_positions = [self.sample_positions[i] for i in indices]
            else:
                # Repeat positions if we need more samples
                while len(self.sample_positions) < self.dataset_size:
                    self.sample_positions.extend(self.sample_positions)
                self.sample_positions = self.sample_positions[:self.dataset_size]
        
        # Shuffle for training
        if self.is_train:
            np.random.shuffle(self.sample_positions)
    
    def __len__(self):
        """Return dataset size."""
        return len(self.sample_positions)
    
    def __getitem__(self, index):
        """
        Get a sequence sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with 'sequence' key containing one-hot encoded DNA sequence
            Shape: (sequence_length, 4) as float32
        """
        chrom, start_pos, end_pos = self.sample_positions[index]
        
        # Get sequence from preloaded memory - use raw_array format
        sequence = self.sequence_dataset.get_track(chrom, start_pos, end_pos, output_format='raw_array')
        
        # Convert to numpy array if needed
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
        
        # Ensure correct shape: (length, 4) - transpose if needed
        # DenseZarrIO returns (4, length), we need (length, 4)
        if len(sequence.shape) == 2:
            if sequence.shape[0] == 4:
                sequence = sequence.T  # Transpose from (4, length) to (length, 4)
        
        # Pad if necessary
        if sequence.shape[0] < self.sequence_length:
            padding = np.zeros(
                (self.sequence_length - sequence.shape[0], sequence.shape[1]),
                dtype=sequence.dtype
            )
            sequence = np.concatenate([sequence, padding], axis=0)
        
        # Convert to float32
        sequence = sequence.astype(np.float32)
        
        return {"sequence": sequence}
    
    def __repr__(self):
        """String representation."""
        return (
            f"SequenceMotifPredictDataset("
            f"sequence_zarr={self.sequence_dataset.zarr_path}, "
            f"sequence_length={self.sequence_length}, "
            f"is_train={self.is_train}, "
            f"dataset_size={len(self)})"
        )

