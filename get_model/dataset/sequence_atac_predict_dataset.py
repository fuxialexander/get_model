"""
Dataset for sequence-based ATAC signal prediction.

This module provides dataset classes that load:
- DNA sequences from dense zarr files (input)
- ATAC signals from BPCells store or similar (target)

Supports:
- Training on peaks with ATAC signal targets
- Signal quantile-based upsampling for balanced training
- Chromosome-based train/val/test splits
"""

import os
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from caesar.io.zarr_io import DenseZarrIO

# Try to import BPCells IO
try:
    from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
    BPCELLS_AVAILABLE = True
except ImportError:
    BPCELLS_AVAILABLE = False


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
class SequenceATACPredictDatasetConfig:
    """Configuration for SequenceATACPredictDataset."""
    # Data sources
    sequence_zarr: str  # Path to sequence dense zarr
    peaks_bed: str  # Path to BED file with peaks
    bpcells_path: Optional[str] = None  # Path to BPCells store (for ATAC targets)
    celltype_id: str = "bulk"  # Celltype ID in BPCells store

    # Sequence configuration
    sequence_length: int = 2048  # Total sequence length
    extend_bp: int = 1024  # Base pairs to extend from peak center

    # Split configuration
    leave_out_chromosomes: Optional[str] = None
    is_train: bool = True
    train_column: Optional[str] = None  # Column for filtering training data
    val_column: Optional[str] = None  # Column for filtering validation data

    # Data balancing
    use_upsampling: bool = False  # Whether to use signal-based upsampling
    n_quantiles: int = 20
    upsample_epsilon: float = 0.2

    # ATAC processing
    normalize_factor: float = 1e8  # Library size normalization
    conv_size: int = 20  # Smoothing kernel size for ATAC


class SequenceATACPredictDataset(Dataset):
    """
    Dataset for sequence-based ATAC signal prediction.

    Loads:
    - DNA sequences from genome zarr
    - ATAC signals from BPCells store
    - Peak coordinates from BED file

    Supports chromosome-based splits and signal-based upsampling.
    """

    def __init__(
        self,
        sequence_zarr: str,
        peaks_bed: str,
        bpcells_path: Optional[str] = None,
        celltype_id: str = "bulk",
        sequence_length: int = 2048,
        extend_bp: int = 1024,
        leave_out_chromosomes: Optional[str] = None,
        is_train: bool = True,
        train_column: Optional[str] = None,
        val_column: Optional[str] = None,
        use_upsampling: bool = False,
        n_quantiles: int = 20,
        upsample_epsilon: float = 0.2,
        normalize_factor: float = 1e8,
        conv_size: int = 20,
        sequence_io: Optional[Any] = None,  # Optional shared DenseZarrIO
        atac_io: Optional[Any] = None,  # Optional shared BPCells IO
    ):
        self.sequence_length = sequence_length
        self.extend_bp = extend_bp
        self.celltype_id = celltype_id
        self.normalize_factor = normalize_factor
        self.conv_size = conv_size
        self.is_train = is_train
        self.use_upsampling = use_upsampling

        # Initialize sequence IO
        if sequence_io is not None:
            self.sequence_io = sequence_io
        else:
            self.sequence_io = DenseZarrIO(sequence_zarr, dtype="int8", mode="r")
            self.sequence_io.load_to_memory_dense()

        # Initialize ATAC IO (optional)
        self.atac_io = None
        self.libsize = normalize_factor
        if bpcells_path is not None and BPCELLS_AVAILABLE:
            if atac_io is not None:
                self.atac_io = atac_io
            else:
                self.atac_io = CelltypeDenseBPCellsIO(bpcells_path, mode="r")
                self.atac_io = self.atac_io.subset([celltype_id])
            self.libsize = self.atac_io.libsize.get(celltype_id, normalize_factor)
            print(f"Library size for {celltype_id}: {self.libsize:.2e}")

        # Load peaks
        self.peaks_df = self._load_peaks(peaks_bed)
        print(f"Loaded {len(self.peaks_df)} peaks from {peaks_bed}")

        # Filter by chromosome split
        chrom_names = list(self.sequence_io.chrom_sizes.keys())
        self.input_chromosomes = _chromosome_splitter(
            chrom_names, leave_out_chromosomes, is_train
        )
        self.peaks_df = self.peaks_df[
            self.peaks_df['Chromosome'].isin(self.input_chromosomes)
        ].reset_index(drop=True)
        print(f"After chromosome filter: {len(self.peaks_df)} peaks")

        # Filter by split column (train/val/test)
        split_column = train_column if is_train else val_column
        if split_column is not None and split_column in self.peaks_df.columns:
            self.peaks_df = self.peaks_df[
                self.peaks_df[split_column] == True
            ].reset_index(drop=True)
            print(f"After {split_column} filter: {len(self.peaks_df)} peaks")

        # Apply upsampling for training
        if use_upsampling and is_train and self.atac_io is not None:
            self.peaks_df = self._apply_upsampling(n_quantiles, upsample_epsilon)

        print(f"Final dataset size: {len(self.peaks_df)} peaks")

    def _load_peaks(self, peaks_bed: str) -> pd.DataFrame:
        """Load peaks from BED file."""
        # Try to detect if it's a full annotated BED or simple 3-column BED
        try:
            # First try reading as tab-separated with all columns
            peaks_df = pd.read_csv(peaks_bed, sep='\t')

            # Check if it has standard BED columns
            if 'Chromosome' not in peaks_df.columns:
                # Assume first 3 columns are chrom, start, end
                peaks_df = pd.read_csv(
                    peaks_bed,
                    sep='\t',
                    header=None,
                    names=['Chromosome', 'Start', 'End'],
                    usecols=[0, 1, 2]
                )
        except Exception as e:
            # Fallback to simple 3-column BED
            peaks_df = pd.read_csv(
                peaks_bed,
                sep='\t',
                header=None,
                names=['Chromosome', 'Start', 'End'],
                usecols=[0, 1, 2]
            )

        peaks_df['Chromosome'] = peaks_df['Chromosome'].astype(str)
        return peaks_df

    def _apply_upsampling(
        self,
        n_quantiles: int,
        epsilon: float,
    ) -> pd.DataFrame:
        """Apply signal quantile-based upsampling for balanced training."""
        from tqdm import tqdm

        print(f"\nComputing signal sums for {len(self.peaks_df)} peaks...")
        signal_sums = []

        for idx in tqdm(range(len(self.peaks_df)), desc="Computing signals"):
            row = self.peaks_df.iloc[idx]
            chrom = row["Chromosome"]
            center = (row["Start"] + row["End"]) // 2
            start = center - self.extend_bp
            end = center + self.extend_bp

            try:
                atac = self.atac_io.get_track(
                    chrom, start, end,
                    output_format="raw_array",
                    conv_size=1,
                )
                signal_sums.append(float(atac.sum()))
            except Exception:
                signal_sums.append(0.0)

        self.peaks_df = self.peaks_df.copy()
        self.peaks_df["signal_sum"] = signal_sums
        self.peaks_df["signal_log"] = np.log1p(self.peaks_df["signal_sum"])

        print(f"Signal range: [{min(signal_sums):.2f}, {max(signal_sums):.2f}]")

        # Quantile-based upsampling
        peaks_sorted = self.peaks_df.sort_values("signal_sum").reset_index(drop=True)
        quantile_size = len(peaks_sorted) // n_quantiles

        quantile_groups = []
        for i in range(n_quantiles):
            start_idx = i * quantile_size
            end_idx = (i + 1) * quantile_size if i < n_quantiles - 1 else len(peaks_sorted)
            quantile_groups.append(peaks_sorted.iloc[start_idx:end_idx].copy())

        # Merge similar quantiles based on log signal
        quantile_log_means = [g["signal_log"].mean() for g in quantile_groups]
        merged_groups = []
        current_group = quantile_groups[0].copy()
        current_log_mean = quantile_log_means[0]

        for i in range(1, len(quantile_groups)):
            if abs(quantile_log_means[i] - current_log_mean) < epsilon:
                current_group = pd.concat([current_group, quantile_groups[i]], ignore_index=True)
                current_log_mean = current_group["signal_log"].mean()
            else:
                merged_groups.append(current_group)
                current_group = quantile_groups[i].copy()
                current_log_mean = quantile_log_means[i]
        merged_groups.append(current_group)

        print(f"After merging: {len(merged_groups)} groups")

        # Upsample to largest group size
        max_group_size = max(len(g) for g in merged_groups)
        upsampled_dfs = []

        for i, group in enumerate(merged_groups):
            if len(group) < max_group_size:
                n_repeats = max_group_size // len(group)
                remainder = max_group_size % len(group)
                repeated = pd.concat([group] * n_repeats, ignore_index=True)
                if remainder > 0:
                    additional = group.sample(n=remainder, replace=True, random_state=42)
                    repeated = pd.concat([repeated, additional], ignore_index=True)
                upsampled_dfs.append(repeated)
            else:
                upsampled_dfs.append(group)

        upsampled = pd.concat(upsampled_dfs, ignore_index=True)
        upsampled = upsampled.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Upsampling: {len(self.peaks_df)} -> {len(upsampled)} peaks")
        return upsampled

    def __len__(self) -> int:
        return len(self.peaks_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.peaks_df.iloc[idx]
        chrom = row["Chromosome"]

        # Calculate region around peak center
        center = (row["Start"] + row["End"]) // 2
        start = center - self.extend_bp
        end = center + self.extend_bp

        try:
            # Get DNA sequence (one-hot encoded)
            sequence = self.sequence_io.get_track(
                chrom, start, end, output_format="raw_array"
            )

            # DenseZarrIO returns (4, length), we need (length, 4)
            if sequence.shape[0] == 4:
                sequence = sequence.T

            # Get ATAC signal if available
            if self.atac_io is not None:
                atac = self.atac_io.get_track(
                    chrom, start, end,
                    output_format="raw_array",
                    normalize_factor=self.normalize_factor,
                    conv_size=self.conv_size,
                )
            else:
                atac = np.zeros(self.sequence_length, dtype=np.float32)

            # Validate and pad/truncate
            expected_length = 2 * self.extend_bp
            if sequence.shape[0] < expected_length:
                pad_size = expected_length - sequence.shape[0]
                sequence = np.pad(sequence, ((0, pad_size), (0, 0)), mode='constant')
            elif sequence.shape[0] > expected_length:
                sequence = sequence[:expected_length]

            if len(atac) < expected_length:
                atac = np.pad(atac, (0, expected_length - len(atac)), mode='constant')
            elif len(atac) > expected_length:
                atac = atac[:expected_length]

            # Convert to tensors
            sequence = torch.from_numpy(sequence.astype(np.float32))  # (seq_len, 4)
            atac = torch.from_numpy(atac.astype(np.float32))  # (seq_len,)

            return {
                "sequence": sequence,
                "atac": atac,
                "region_info": {
                    "chromosome": chrom,
                    "start": start,
                    "end": end,
                    "center": center,
                    "idx": idx,
                },
            }

        except Exception as e:
            print(f"Error loading region {idx} ({chrom}:{start}-{end}): {e}")
            expected_length = 2 * self.extend_bp
            return {
                "sequence": torch.zeros(expected_length, 4, dtype=torch.float32),
                "atac": torch.zeros(expected_length, dtype=torch.float32),
                "region_info": {
                    "chromosome": chrom,
                    "start": start,
                    "end": end,
                    "center": center,
                    "idx": idx,
                    "error": str(e),
                },
            }

    def __repr__(self):
        return (
            f"SequenceATACPredictDataset("
            f"num_peaks={len(self.peaks_df)}, "
            f"sequence_length={self.sequence_length}, "
            f"is_train={self.is_train})"
        )


class SequenceATACPredictDatasetFromDF(Dataset):
    """
    Dataset variant that takes a pre-filtered DataFrame directly.

    Useful when peaks have already been filtered and processed.
    """

    def __init__(
        self,
        peaks_df: pd.DataFrame,
        sequence_io: Any,
        atac_io: Optional[Any] = None,
        celltype_id: str = "bulk",
        extend_bp: int = 1024,
        normalize_factor: float = 1e8,
        conv_size: int = 20,
    ):
        self.peaks_df = peaks_df.reset_index(drop=True)
        self.sequence_io = sequence_io
        self.atac_io = atac_io
        self.celltype_id = celltype_id
        self.extend_bp = extend_bp
        self.normalize_factor = normalize_factor
        self.conv_size = conv_size
        self.sequence_length = 2 * extend_bp

        if atac_io is not None:
            self.libsize = atac_io.libsize.get(celltype_id, normalize_factor)
        else:
            self.libsize = normalize_factor

    def __len__(self) -> int:
        return len(self.peaks_df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.peaks_df.iloc[idx]
        chrom = str(row["Chromosome"])
        center = (int(row["Start"]) + int(row["End"])) // 2
        start = center - self.extend_bp
        end = center + self.extend_bp

        try:
            sequence = self.sequence_io.get_track(
                chrom, start, end, output_format="raw_array"
            )
            if sequence.shape[0] == 4:
                sequence = sequence.T

            if self.atac_io is not None:
                atac = self.atac_io.get_track(
                    chrom, start, end,
                    output_format="raw_array",
                    normalize_factor=self.normalize_factor,
                    conv_size=self.conv_size,
                )
            else:
                atac = np.zeros(self.sequence_length, dtype=np.float32)

            # Ensure correct length
            if sequence.shape[0] != self.sequence_length:
                if sequence.shape[0] < self.sequence_length:
                    sequence = np.pad(
                        sequence,
                        ((0, self.sequence_length - sequence.shape[0]), (0, 0)),
                        mode='constant'
                    )
                else:
                    sequence = sequence[:self.sequence_length]

            if len(atac) != self.sequence_length:
                if len(atac) < self.sequence_length:
                    atac = np.pad(atac, (0, self.sequence_length - len(atac)), mode='constant')
                else:
                    atac = atac[:self.sequence_length]

            return {
                "sequence": torch.from_numpy(sequence.astype(np.float32)),
                "atac": torch.from_numpy(atac.astype(np.float32)),
            }

        except Exception as e:
            return {
                "sequence": torch.zeros(self.sequence_length, 4, dtype=torch.float32),
                "atac": torch.zeros(self.sequence_length, dtype=torch.float32),
            }
