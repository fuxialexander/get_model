import logging
from functools import partial

import lightning as L
import seaborn as sns
import torch
import torch.utils.data
import wandb
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from minlora import LoRAParametrization
from minlora.model import add_lora_by_name
from omegaconf import DictConfig, OmegaConf

from get_model.config.config import *
from get_model.dataset.zarr_dataset import (SequenceMotifDataset, CuratedSequenceMotifDataset)
from get_model.model.model import *
from get_model.model.modules import *
from get_model.run import LitModel, run_shared
from get_model.utils import (extract_state_dict,
                             load_checkpoint, load_state_dict,
                             rename_state_dict)


class NucleotideMotifDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.accumulated_results = []

    def build_training_dataset(self, is_train=True):
        if 'curated_zarr' in self.cfg.dataset:
            return CuratedSequenceMotifDataset(**self.cfg.dataset, is_train=is_train)
        elif 'sequence_zarr' in self.cfg.dataset and 'motif_zarr' in self.cfg.dataset:
            return SequenceMotifDataset(**self.cfg.dataset, is_train=is_train)
        else:
            raise ValueError("No supported dataset specified")

    def build_inference_dataset(self, is_train=False):
        if 'curated_zarr' in self.cfg.dataset:
            return CuratedSequenceMotifDataset(**self.cfg.dataset, is_train=is_train)
        elif 'sequence_zarr' in self.cfg.dataset and 'motif_zarr' in self.cfg.dataset:
            return SequenceMotifDataset(**self.cfg.dataset, is_train=is_train)
        else:
            raise ValueError("No supported dataset specified")

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset_train = self.build_training_dataset(is_train=True)
            self.dataset_val = self.build_training_dataset(is_train=False)
        if stage == 'predict':
            if self.cfg.task.test_mode == 'predict':
                self.dataset_predict = self.build_training_dataset(
                    is_train=False)
        if stage == 'validate':
            self.dataset_val = self.build_training_dataset(is_train=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=True,
            shuffle=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.cfg.machine.batch_size,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )


class RegionLitModel(LitModel):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def validation_step(self, batch, batch_idx):
        loss, pred, obs = self._shared_step(batch, batch_idx, stage='val')
        pred_motif_mean = pred['original_motif'].reshape(-1, 282).mean(dim=0)
        obs_motif_mean = obs['original_motif'].reshape(-1, 282).mean(dim=0)
        # log the max value of each motif
        if batch_idx == 10 and self.cfg.log_image:
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_max_pred": wandb.Image(sns.scatterplot(y=pred_motif_max.detach().cpu().numpy().flatten(), x=range(282)))
        #     })
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_max_obs": wandb.Image(sns.scatterplot(y=obs_motif_max.detach().cpu().numpy().flatten(), x=range(282)))
        #     })
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_max_obs_pred": wandb.Image(sns.scatterplot(y=obs_motif_max.detach().cpu().numpy().flatten(), x=pred_motif_max.detach().cpu().numpy().flatten()))
        #     })
            plt.clf()
            self.logger.experiment.log({
                "scatter_motif_mean_obs_pred": wandb.Image(sns.scatterplot(y=obs_motif_mean.detach().cpu().numpy().flatten(), x=pred_motif_mean.detach().cpu().numpy().flatten()))
            })
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_mean_obs": wandb.Image(sns.scatterplot(y=obs_motif_mean.detach().cpu().numpy().flatten(), x=range(282)))
        #     })
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_mean_pred": wandb.Image(sns.scatterplot(y=pred_motif_mean.detach().cpu().numpy().flatten(), x=range(282)))
        #     })
        #     plt.clf()
        #     self.logger.experiment.log({
        #         "scatter_motif_mean_obs_max_pred": wandb.Image(sns.scatterplot(y=obs_motif_mean.detach().cpu().numpy().flatten(), x=pred_motif_max.detach().cpu().numpy().flatten()))
        #     })
        # # random draw 1000 points
        total_points = pred['motif'].flatten().shape[0]
        sample_points = min(1000, total_points)
        zero_mask = torch.randint(0, total_points, (sample_points,))
        pred['motif'] = pred['motif'].flatten()[zero_mask]
        obs['motif'] = obs['motif'].flatten()[zero_mask]
        metrics = self.metrics(pred, obs)
        if batch_idx == 10 and self.cfg.log_image:
            # log one example as scatter plot
            for key in ['motif']:
                plt.clf()
                if self.cfg.run.use_wandb:
                    self.logger.experiment.log({
                        f"scatter_{key}": wandb.Image(sns.scatterplot(y=pred[key].detach().cpu().numpy().flatten(), x=obs[key].detach().cpu().numpy().flatten()))})
        distributed = self.cfg.machine.num_devices > 1
        self.log_dict(
            metrics, batch_size=self.cfg.machine.batch_size, sync_dist=distributed)
        self.log("val_loss", loss,
                 batch_size=self.cfg.machine.batch_size, sync_dist=distributed)

    def get_model(self):
        model = instantiate(self.cfg.model)

        # Load main model checkpoint
        if self.cfg.finetune.checkpoint is not None:
            checkpoint_model = load_checkpoint(
                self.cfg.finetune.checkpoint, model_key=self.cfg.finetune.model_key)
            checkpoint_model = extract_state_dict(checkpoint_model)
            checkpoint_model = rename_state_dict(
                checkpoint_model, self.cfg.finetune.rename_config)
            lora_config = {  # specify which layers to add lora to, by default only add to linear layers
                nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=8),
                },
                nn.Conv2d: {
                    "weight": partial(LoRAParametrization.from_conv2d, rank=4),
                },
            }
            if any("lora" in k for k in checkpoint_model.keys()) and self.cfg.finetune.use_lora:
                add_lora_by_name(
                    model, self.cfg.finetune.layers_with_lora, lora_config)
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)
            elif any("lora" in k for k in checkpoint_model.keys()) and not self.cfg.finetune.use_lora:
                raise ValueError(
                    "Model checkpoint contains LoRA parameters but use_lora is set to False")
            elif not any("lora" in k for k in checkpoint_model.keys()) and self.cfg.finetune.use_lora:
                logging.info(
                    "Model checkpoint does not contain LoRA parameters but use_lora is set to True, using the checkpoint as base model")
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)
                add_lora_by_name(
                    model, self.cfg.finetune.layers_with_lora, lora_config)
            else:
                load_state_dict(model, checkpoint_model,
                                strict=self.cfg.finetune.strict)

        # Load additional checkpoints
        if len(self.cfg.finetune.additional_checkpoints) > 0:
            for checkpoint_config in self.cfg.finetune.additional_checkpoints:
                checkpoint_model = load_checkpoint(
                    checkpoint_config.checkpoint, model_key=checkpoint_config.model_key)
                checkpoint_model = extract_state_dict(checkpoint_model)
                checkpoint_model = rename_state_dict(
                    checkpoint_model, checkpoint_config.rename_config)
                load_state_dict(model, checkpoint_model,
                                strict=checkpoint_config.strict)

        if self.cfg.finetune.use_lora:
            # Load LoRA parameters based on the stage
            if self.cfg.stage == 'fit':
                # Load LoRA parameters for training
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(
                        self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config)
                    load_state_dict(model, lora_state_dict, strict=True)
            elif self.cfg.stage in ['validate', 'predict']:
                # Load LoRA parameters for validation and prediction
                if self.cfg.finetune.lora_checkpoint is not None:
                    lora_state_dict = load_checkpoint(
                        self.cfg.finetune.lora_checkpoint)
                    lora_state_dict = extract_state_dict(lora_state_dict)
                    lora_state_dict = rename_state_dict(
                        lora_state_dict, self.cfg.finetune.lora_rename_config)
                    load_state_dict(model, lora_state_dict, strict=True)

        model.freeze_layers(
            patterns_to_freeze=self.cfg.finetune.patterns_to_freeze, invert_match=False)
        print("Model = %s" % str(model))
        return model

    def on_validation_epoch_end(self):
        pass




def run(cfg: DictConfig):
    model = RegionLitModel(cfg)
    print(OmegaConf.to_yaml(cfg))
    dm = NucleotideMotifDataModule(cfg)
    model.dm = dm
    return run_shared(cfg, model, dm)
