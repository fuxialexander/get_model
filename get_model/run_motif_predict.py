"""
Run module for motif prediction model.

This module provides a Lightning DataModule and run function for training
the motif prediction model.
"""

import lightning as L
import torch
import torch.utils.data
from hydra.utils import instantiate
from omegaconf import DictConfig

from get_model.dataset.sequence_motif_predict_dataset import SequenceMotifPredictDataset
from get_model.run import LitModel, run_shared


class MotifPredictDataModule(L.LightningDataModule):
    """DataModule for motif prediction dataset."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Share DenseZarrIO object across datasets to avoid reloading chromosomes
        self._shared_sequence_io = None

    def _get_shared_sequence_io(self):
        """Get or create shared DenseZarrIO object."""
        if self._shared_sequence_io is None:
            from caesar.io.zarr_io import DenseZarrIO
            self._shared_sequence_io = DenseZarrIO(
                self.cfg.dataset.sequence_zarr, dtype="int8", mode="r"
            )
            # Load chromosomes to memory once
            self._shared_sequence_io.load_to_memory_dense()
        return self._shared_sequence_io

    def build_training_dataset(self, is_train=True):
        """Build training or validation dataset."""
        dataset_cfg = dict(self.cfg.dataset)
        dataset_cfg['is_train'] = is_train
        # Pass shared sequence_io to avoid reloading
        dataset_cfg['sequence_io'] = self._get_shared_sequence_io()
        return SequenceMotifPredictDataset(**dataset_cfg)

    def prepare_data(self):
        """Prepare data (no-op for this dataset)."""
        pass

    def setup(self, stage=None):
        """Set up datasets for different stages."""
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_training_dataset(is_train=True)
            self.dataset_val = self.build_training_dataset(is_train=False)
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
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        # Use smaller batch size for validation to avoid OOM during metrics computation
        val_batch_size = getattr(self.cfg.machine, 'val_batch_size', None) or max(1, self.cfg.machine.batch_size // 4)
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=val_batch_size,
            num_workers=self.cfg.machine.num_workers,
            shuffle=False,
            drop_last=False,  # Don't drop last batch in validation
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def predict_dataloader(self):
        """Create prediction dataloader."""
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )


def run(cfg: DictConfig):
    """Run training/validation/prediction for motif prediction model."""
    torch.set_float32_matmul_precision("medium")
    model = LitModel(cfg)
    dm = MotifPredictDataModule(cfg)
    model.dm = dm

    return run_shared(cfg, model, dm)

