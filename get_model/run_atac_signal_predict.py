"""
Run module for ATAC signal prediction model.

This module provides a Lightning DataModule and run function for training
the ATAC signal prediction model that builds upon the motif prediction trunk.

Includes:
- Multiple test dataloaders (interpolation and manifold)
- Mutation effect (caQTL) evaluation during validation
- Comprehensive metrics logging
"""

import os
import logging
from typing import Dict, Any, Optional, List

import lightning as L
import torch
import torch.utils.data
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from scipy import stats

from get_model.dataset.sequence_atac_predict_dataset import (
    SequenceATACPredictDataset,
    SequenceATACPredictDatasetFromDF,
)
from get_model.run import LitModel as BaseLitModel, run_shared


class ATACSignalPredictDataModule(L.LightningDataModule):
    """
    DataModule for ATAC signal prediction dataset.

    Supports:
    - Training and validation datasets
    - Multiple test datasets (interpolation and manifold)
    - Shared IO objects for efficiency
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self._shared_sequence_io = None
        self._shared_atac_io = None

        # Test datasets
        self.dataset_interp_test = None
        self.dataset_manifold_test = None

    def _get_shared_sequence_io(self):
        """Get or create shared DenseZarrIO object."""
        if self._shared_sequence_io is None:
            from caesar.io.zarr_io import DenseZarrIO
            self._shared_sequence_io = DenseZarrIO(
                self.cfg.dataset.sequence_zarr, dtype="int8", mode="r"
            )
            self._shared_sequence_io.load_to_memory_dense()
        return self._shared_sequence_io

    def _get_shared_atac_io(self):
        """Get or create shared BPCells IO object."""
        if self._shared_atac_io is None and hasattr(self.cfg.dataset, 'bpcells_path'):
            bpcells_path = self.cfg.dataset.bpcells_path
            if bpcells_path is not None:
                try:
                    from caesar.io.bpcell_io import CelltypeDenseBPCellsIO
                    self._shared_atac_io = CelltypeDenseBPCellsIO(bpcells_path, mode="r")
                    celltype_id = getattr(self.cfg.dataset, 'celltype_id', 'bulk')
                    self._shared_atac_io = self._shared_atac_io.subset([celltype_id])
                except ImportError:
                    logging.warning("BPCells not available, ATAC IO disabled")
        return self._shared_atac_io

    def _load_peaks(self) -> pd.DataFrame:
        """Load peaks from BED file."""
        peaks_bed = self.cfg.dataset.peaks_bed
        peaks = pd.read_csv(peaks_bed, sep='\t')
        logging.info(f"Loaded {len(peaks)} peaks from {peaks_bed}")
        return peaks

    def build_dataset_from_df(self, peaks_df: pd.DataFrame) -> SequenceATACPredictDatasetFromDF:
        """Build dataset from a pre-filtered DataFrame."""
        return SequenceATACPredictDatasetFromDF(
            peaks_df,
            self._get_shared_sequence_io(),
            atac_io=self._get_shared_atac_io(),
            celltype_id=getattr(self.cfg.dataset, 'celltype_id', 'bulk'),
            extend_bp=getattr(self.cfg.dataset, 'extend_bp', 1024),
            normalize_factor=getattr(self.cfg.dataset, 'normalize_factor', 1e8),
            conv_size=getattr(self.cfg.dataset, 'conv_size', 20),
        )

    def build_training_dataset(self, is_train=True):
        """Build training or validation dataset."""
        dataset_cfg = dict(self.cfg.dataset)
        dataset_cfg['is_train'] = is_train
        dataset_cfg['sequence_io'] = self._get_shared_sequence_io()
        dataset_cfg['atac_io'] = self._get_shared_atac_io()
        return SequenceATACPredictDataset(**dataset_cfg)

    def prepare_data(self):
        """Prepare data (no-op for this dataset)."""
        pass

    def setup(self, stage=None):
        """Set up datasets for different stages."""
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_training_dataset(is_train=True)
            self.dataset_val = self.build_training_dataset(is_train=False)

            # Build test datasets if columns exist
            peaks = self._load_peaks()

            interp_col = getattr(self.cfg.dataset, 'interp_test_column', 'is_test_interpolation')
            manifold_col = getattr(self.cfg.dataset, 'manifold_test_column', 'is_test_manifold')

            if interp_col in peaks.columns:
                interp_peaks = peaks[peaks[interp_col] == True].copy().reset_index(drop=True)
                if len(interp_peaks) > 0:
                    self.dataset_interp_test = self.build_dataset_from_df(interp_peaks)
                    logging.info(f"Created interpolation test dataset: {len(interp_peaks)} peaks")

            if manifold_col in peaks.columns:
                manifold_peaks = peaks[peaks[manifold_col] == True].copy().reset_index(drop=True)
                if len(manifold_peaks) > 0:
                    self.dataset_manifold_test = self.build_dataset_from_df(manifold_peaks)
                    logging.info(f"Created manifold test dataset: {len(manifold_peaks)} peaks")

        if stage == 'predict':
            if self.cfg.task.test_mode == 'predict':
                self.dataset_predict = self.build_training_dataset(is_train=False)
        if stage == 'validate':
            self.dataset_val = self.build_training_dataset(is_train=False)

    def train_dataloader(self):
        """Create training dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader(s)."""
        val_batch_size = getattr(self.cfg.machine, 'val_batch_size', None) or max(1, self.cfg.machine.batch_size // 2)

        # Return multiple dataloaders if test sets exist
        loaders = [
            torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=val_batch_size,
                num_workers=self.cfg.machine.num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
        ]

        if self.dataset_interp_test is not None:
            loaders.append(
                torch.utils.data.DataLoader(
                    self.dataset_interp_test,
                    batch_size=val_batch_size,
                    num_workers=self.cfg.machine.num_workers,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
            )

        if self.dataset_manifold_test is not None:
            loaders.append(
                torch.utils.data.DataLoader(
                    self.dataset_manifold_test,
                    batch_size=val_batch_size,
                    num_workers=self.cfg.machine.num_workers,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                )
            )

        return loaders if len(loaders) > 1 else loaders[0]

    def test_dataloader(self):
        """Create test dataloader."""
        if hasattr(self, 'dataset_test'):
            return torch.utils.data.DataLoader(
                self.dataset_test,
                batch_size=self.cfg.machine.batch_size,
                num_workers=self.cfg.machine.num_workers,
                drop_last=False,
                shuffle=False,
            )
        return None

    def predict_dataloader(self):
        """Create prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )


class ATACSignalLitModel(BaseLitModel):
    """
    Lightning module for ATAC signal prediction with mutation effect evaluation.

    Extends BaseLitModel to add:
    - Multiple validation dataloaders handling
    - caQTL mutation effect evaluation
    - Comprehensive metrics logging
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.mutation_evaluator = None
        self._setup_mutation_evaluator()

    def _setup_mutation_evaluator(self):
        """Set up mutation effect evaluator if configured."""
        if hasattr(self.cfg, 'evaluation') and hasattr(self.cfg.evaluation, 'caqtl_path'):
            caqtl_path = self.cfg.evaluation.caqtl_path
            if caqtl_path is not None and os.path.exists(caqtl_path):
                try:
                    from get_model.model.mutation_effect import MutationEffectEvaluator
                    from caesar.io.zarr_io import DenseZarrIO

                    genome_io = DenseZarrIO(self.cfg.dataset.sequence_zarr, dtype="int8", mode="r")
                    genome_io.load_to_memory_dense()

                    extend_bp = getattr(self.cfg.evaluation, 'extend_bp', 1024)
                    center_crop = getattr(self.cfg.evaluation, 'center_crop', 1024)

                    self.mutation_evaluator = MutationEffectEvaluator(
                        genome_io=genome_io,
                        variants_path=caqtl_path,
                        extend_bp=extend_bp,
                        center_crop=center_crop,
                    )
                    logging.info(f"Initialized mutation effect evaluator with {len(self.mutation_evaluator.variants)} variants")
                except Exception as e:
                    logging.warning(f"Could not initialize mutation evaluator: {e}")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step with support for multiple dataloaders."""
        # Get dataloader name based on index
        dataloader_names = ['val', 'interp_test', 'manifold_test']
        if dataloader_idx < len(dataloader_names):
            dl_name = dataloader_names[dataloader_idx]
        else:
            dl_name = f'dataloader_{dataloader_idx}'

        # Forward pass
        model_input = self.model.get_input(batch)
        output = self.model(**model_input)
        pred, obs = self.model.before_loss(output, batch)

        # Compute loss
        loss_dict = self.loss(pred, obs)

        # Compute metrics
        metrics_dict = self.metrics(pred, obs)

        # Log with dataloader prefix
        total_loss = sum(loss_dict.values())
        self.log(f'{dl_name}_loss', total_loss, on_step=False, on_epoch=True,
                 prog_bar=(dataloader_idx == 0), sync_dist=True, add_dataloader_idx=False)

        for name, value in loss_dict.items():
            self.log(f'{dl_name}_{name}', value, on_step=False, on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)

        for name, value in metrics_dict.items():
            self.log(f'{dl_name}_{name}', value, on_step=False, on_epoch=True,
                     sync_dist=True, add_dataloader_idx=False)

        return {'loss': total_loss, 'pred': pred, 'obs': obs}

    def on_validation_epoch_end(self):
        """Run mutation effect evaluation at end of validation epoch."""
        if self.mutation_evaluator is not None:
            try:
                logging.info("Running mutation effect evaluation...")
                caqtl_metrics = self.mutation_evaluator.evaluate(
                    self.model,
                    device=self.device,
                    desc="caQTL Evaluation",
                    use_amp=self.cfg.machine.precision in ['16', '16-mixed', 'bf16', 'bf16-mixed'],
                )

                # Log caQTL metrics
                self.log('caqtl_n_variants', float(caqtl_metrics['n_variants']),
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log('caqtl_pearson_r', caqtl_metrics['pearson_r'],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log('caqtl_spearman_r', caqtl_metrics['spearman_r'],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log('caqtl_auc_direction', caqtl_metrics['auc_direction'],
                         on_step=False, on_epoch=True, sync_dist=True)
                self.log('caqtl_auc_strong', caqtl_metrics['auc_strong_effects'],
                         on_step=False, on_epoch=True, sync_dist=True)

                logging.info(
                    f"caQTL metrics - n={caqtl_metrics['n_variants']}, "
                    f"pearson_r={caqtl_metrics['pearson_r']:.4f}, "
                    f"spearman_r={caqtl_metrics['spearman_r']:.4f}, "
                    f"auc_direction={caqtl_metrics['auc_direction']:.4f}"
                )

            except Exception as e:
                logging.warning(f"Mutation effect evaluation failed: {e}")


def run(cfg: DictConfig):
    """Run training/validation/prediction for ATAC signal prediction model."""
    torch.set_float32_matmul_precision("medium")

    # Use custom LitModel with mutation evaluation
    model = ATACSignalLitModel(cfg)
    dm = ATACSignalPredictDataModule(cfg)
    model.dm = dm

    return run_shared(cfg, model, dm)
