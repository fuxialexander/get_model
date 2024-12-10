import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pyranges import PyRanges as pr
import torch
from atac_rna_data_processing.io.region import Genome, GenomicRegionCollection
from caesar.io.zarr_io import CelltypeDenseZarrIO
from tqdm import tqdm

from get_model.config.config import load_config
from get_model.run_motif_adaptor import run

hg38 = Genome('hg38', '/home/xf2217/Projects/common/hg38.fa')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing parameters."""
    PEAK_EXTEND: int = 512
    SCAN_CHUNK_SIZE: int = 32
    BATCH_SIZE: int = 32
    MAX_WORKERS: int = 16
    
    @property
    def SCAN_CHUNKS(self) -> int:
        return self.PEAK_EXTEND*2 // self.SCAN_CHUNK_SIZE

class DataPreprocessor:
    def __init__(self, 
                 config: PreprocessingConfig,
                 dense_zarr_path: str,
                 celltype: str,
                 output_dir: Path):
        """
        Initialize the data preprocessor.
        
        Args:
            config: PreprocessingConfig object containing processing parameters
            dense_zarr_path: Path to the dense zarr file
            celltype: Cell type identifier
            output_dir: Directory to save output files
        """
        self.config = config
        self.dense_zarr_path = dense_zarr_path
        self.celltype = celltype
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dense zarr
        self.dense_zarr = CelltypeDenseZarrIO(dense_zarr_path)
        
    def prepare_peaks(self) -> pd.DataFrame:
        """Prepare and process peak data."""
        logger.info("Preparing peak data...")
        
        # Get peaks from dense zarr
        peaks = self.dense_zarr.call_peaks()
        
        # Load and process CTCF peaks
        ctcf_peak = pd.read_csv(
            'remap_ctcf.bedgraph',
            sep='\t',
            header=None,
            names=['Chromosome', 'Start', 'End', 'Score']
        ).query('Score > 1')
        
        # Create union of peaks
        union_peak = pr.set_union(pr(ctcf_peak), pr(peaks)).merge().as_df()
        
        # Filter chromosomes and process coordinates
        valid_chromosomes = [f"chr{i}" for i in list(range(1, 23)) + ['X']]
        union_peak = union_peak.copy().query('Chromosome.isin(@valid_chromosomes)')
        
        # Calculate summit and adjust coordinates
        union_peak['summit'] = (union_peak['Start'] + union_peak['End']) // 2
        union_peak['Start'] = union_peak['summit'] - self.config.PEAK_EXTEND
        union_peak['End'] = union_peak['summit'] + self.config.PEAK_EXTEND
        
        return union_peak

    def get_motif_for_sequences(self, sequences: List, model: torch.nn.Module) -> np.ndarray:
        """Process sequences through the model to get motif scores."""
        X = torch.from_numpy(
            np.stack([sequences[j].one_hot for j in range(len(sequences))])
        ).half().to('cuda')
        return model(X).detach().cpu().numpy()

    def process_motif_scores(self, 
                           h1_regions_sequences: List,
                           model: torch.nn.Module) -> np.ndarray:
        """Process sequences to generate motif scores."""
        logger.info("Processing motif scores...")
        output_scores = []
        
        for i in tqdm(range(len(h1_regions_sequences.sequences) // self.config.BATCH_SIZE),
                     desc="Processing sequences"):
            batch_start = i * self.config.BATCH_SIZE
            batch_end = (i + 1) * self.config.BATCH_SIZE
            
            # Get motif scores for batch
            output = self.get_motif_for_sequences(
                h1_regions_sequences.sequences[batch_start:batch_end],
                model
            )
            
            # Process output
            output = output[:, 64:-64, :]
            output = output.reshape(
                -1, 
                self.config.SCAN_CHUNKS,
                self.config.SCAN_CHUNK_SIZE,
                282
            ).max(axis=2)
            
            output_scores.append(output)
        
        return np.array(output_scores)

    def get_atac_signal(self, args: Tuple) -> np.ndarray:
        """Get ATAC signal for a given region."""
        dense_zarr, celltype, row = args
        return dense_zarr.get_track(
            celltype,
            row['Chromosome'],
            row['Start'],
            row['End']
        ).reshape(self.config.SCAN_CHUNKS, self.config.SCAN_CHUNK_SIZE).max(axis=1)

    def process_atac_signals(self, peaks_df: pd.DataFrame) -> np.ndarray:
        """Process ATAC signals for all peaks in parallel."""
        logger.info("Processing ATAC signals...")
        
        args_list = [
            (self.dense_zarr, self.celltype, row)
            for _, row in peaks_df.iterrows()
        ]
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            atac_signals = list(tqdm(
                executor.map(self.get_atac_signal, args_list),
                total=len(args_list),
                desc="Processing ATAC signals"
            ))
        
        return np.stack(atac_signals)

    def save_data(self, 
                 output_scores: np.ndarray,
                 atac_signals: np.ndarray,
                 peaks_df: pd.DataFrame) -> None:
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        # Save output scores
        np.save(self.output_dir / 'h1_regions_motif_scores.npy', output_scores)
        
        # Save ATAC signals
        np.save(self.output_dir / 'h1_regions_atac_signals.npy', atac_signals)
        
        # Save peaks data
        peaks_df.to_csv(self.output_dir / 'union_peaks.csv', index=False)
        
        logger.info(f"Data saved to {self.output_dir}")

    def run_preprocessing(self, model: torch.nn.Module) -> None:
        """Run the complete preprocessing pipeline."""
        # Prepare peaks
        peaks_df = self.prepare_peaks()
        
        # Create genomic regions
        h1_regions = GenomicRegionCollection(genome=hg38, df=peaks_df)
        h1_regions_sequences = h1_regions.collect_sequence(upstream=64, downstream=64)
        
        # Process motif scores
        output_scores = self.process_motif_scores(h1_regions_sequences, model)
        
        # Process ATAC signals
        atac_signals = self.process_atac_signals(peaks_df)
        
        # Save all data
        self.save_data(output_scores, atac_signals, peaks_df)

def main():
    """Main function to run the preprocessing pipeline."""
    # Configuration
    config = PreprocessingConfig()
    
    # Initialize trainer and model (assuming these are defined elsewhere)
    cfg = load_config('nucleotide_motif_adaptor')
    cfg.stage = 'validate'
    cfg.finetune.resume_ckpt = '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
    cfg.finetune.checkpoint = '/home/xf2217/output/GETNucleotideMotifAdaptorV3_leave_out_chr10_chr11/scratch_282/checkpoints/best.ckpt'
    cfg.run.use_wandb = False
    cfg.dataset.leave_out_chromosomes = 'chr11'
    
    trainer = run(cfg)
    model = trainer.model.model
    model.to('cuda')
    model.half()
    model.eval()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        config=config,
        dense_zarr_path='/home/xf2217/Projects/get_data/4dn_h1esc_dense.zarr/',
        celltype='H1_ESC_ATAC_Seq_biological_replicates.4dn_h1esc.4DNESLMCRW2C.max',
        output_dir=Path('preprocessed_data')
    )
    
    # Run preprocessing
    preprocessor.run_preprocessing(model)

if __name__ == '__main__':
    main()