#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from get_model.dataset.zarr_dataset import CuratedPeakMotifHiCDataset
from get_model.model.modules import ResBlock, ResBlock1d, symmetrize_bulk, Decoder
from get_model.model.transformer import GETTransformer
import matplotlib.pyplot as plt
#%%

class PeakConvTransHiC(nn.Module):
    """Hi-C interaction prediction model for peak-level data."""
    
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim)
        self.distance_map_encoder = ResBlock(1, hidden_dim, (1, 1))
        self.distance_1d_encoder = ResBlock1d(3, hidden_dim, 3)
        
        self.transformer = GETTransformer(
            num_layers=8,
            num_heads=8,
            embed_dim=hidden_dim,
        )
        
        self.decoder = Decoder(hidden_dim, hidden_dim, num_blocks=10)
        self.distance_decoder = Decoder(hidden_dim, hidden_dim)
        
    def forward(self, motif_features, atac_features, distance_1d, distance_map):
        # Encode features
        x = self.feature_encoder(motif_features)
        dis_map = self.distance_map_encoder(distance_map.unsqueeze(1))
        dis_1d = self.distance_1d_encoder(distance_1d.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        
        # Combine features
        # Hadamard product
        x = x * dis_1d

        
        # Transform
        x, _ = self.transformer(x)
        
        # Create symmetric 2D output
        x = symmetrize_bulk()(x.permute(0, 2, 1))
        # x = torch.cat((x, dis_map), dim=1)
        x = x * dis_map
        
        # Decode
        oe = self.decoder(x).squeeze(1)
        oe = 0.5*(oe + oe.permute(0, 2, 1))
        distance_out = self.distance_decoder(dis_map).squeeze(1)
        hic = oe + distance_out
        
        return hic, oe

    def get_input(self, batch):
        return {
            'motif_features': batch['motif'],
            'atac_features': batch['atac'],
            'distance_1d': batch['distance_1d'],
            'distance_map': batch['distance_map']
        }

class HiCPredictorModule(pl.LightningModule):
    """PyTorch Lightning module for training Hi-C prediction models.
    
    Args:
        model: Model class or instance to train
        learning_rate (float, optional): Learning rate. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay factor. Defaults to 1e-4.
        scheduler_factor (float, optional): Learning rate scheduler reduction factor. Defaults to 0.5.
        scheduler_patience (int, optional): Scheduler patience epochs. Defaults to 5.
        **model_kwargs: Additional arguments passed to model if not instantiated
    """
    def __init__(
        self,
        model,
        learning_rate=1e-4,
        weight_decay=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # MSE Loss for HiC prediction
        self.criterion = nn.MSELoss()
        
    def forward(self, motif_features, atac_features, distance_1d, distance_map):
        return self.model(motif_features, atac_features, distance_1d, distance_map)
    
    def _shared_step(self, batch, batch_idx, stage):
        """Shared step function used in training, validation and testing.
        
        Args:
            batch: Input batch dictionary containing 'motif' and 'hic' tensors
            batch_idx (int): Index of current batch
            stage (str): Current stage ('train', 'val', or 'test')
            
        Returns:
            torch.Tensor: Calculated loss value
        """
        # Extract features and target from batch
        input = self.model.get_input(batch)
        # Forward pass
        hic_pred, oe_pred = self(**input)
        hic_target = batch['hic']
        oe_target = batch['hic_oe']
        # Log example predictions periodically 
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            # self._log_predictions(hic_pred[0].detach(), batch['hic'][0].detach(), stage)
            self._log_predictions(oe_pred[0].detach(), batch['hic_oe'][0].detach(), stage)
        # Calculate loss
        # remove values where hic_target is 0
        mask = torch.ones_like(hic_target).bool()
        mask[hic_target == 0] = 0
        # remove values on the diagonal
        # mask &= (torch.eye(hic_target.size(1)).bool().to(hic_target.device)) == 0
        # remove values on boundary 10 rows and columns
        # mask[:, :50, :50] = 0
        # mask[:, -50:, -50:] = 0
        # mask[:, :50, -50:] = 0
        # mask[:, -50:, :50] = 0
        hic_pred = hic_pred[mask]
        hic_target = hic_target[mask]
        oe_pred = oe_pred[mask]
        oe_target = oe_target[mask]
        loss_hic = self.criterion(hic_pred, hic_target)
        loss_oe = self.criterion(oe_pred, oe_target)
        loss = loss_hic + 5*loss_oe
        
        # Calculate additional metrics
        with torch.no_grad():
            mse_hic = F.mse_loss(hic_pred, hic_target)
            mse_oe = F.mse_loss(oe_pred, oe_target)
            
            # Pearson correlation
            pred_flat = hic_pred.detach().cpu().numpy().flatten()
            oe_flat = oe_pred.detach().cpu().numpy().flatten()
            target_flat = hic_target.detach().cpu().numpy().flatten()
            corr, _ = pearsonr(pred_flat, target_flat)
            corr_oe, _ = pearsonr(oe_flat, oe_target.detach().cpu().numpy().flatten())
        # Log metrics
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_correlation', corr, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{stage}_correlation_oe', corr_oe, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{stage}_mse_hic', mse_hic, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{stage}_mse_oe', mse_oe, on_epoch=True, on_step=False, sync_dist=True)

        return loss
    
    def _log_predictions(self, pred, target, stage):
        """Log prediction visualizations to WandB.
        
        Args:
            pred (torch.Tensor): Predicted Hi-C matrix
            target (torch.Tensor): Target Hi-C matrix
            stage (str): Current stage ('train', 'val', or 'test')
        """
        if self.logger:
            fig = plt.figure(figsize=(15, 5))
            
            plt.subplot(131)
            plt.imshow(pred.cpu().numpy(), vmin=-1, vmax=1)
            plt.title('Predicted HiC')
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(target.cpu().numpy(), vmin=-1, vmax=1)
            plt.title('Target HiC')
            plt.colorbar()
            
            plt.subplot(133)
            # difference between pred and target
            diff = pred - target
            plt.imshow(diff.cpu().numpy(), vmin=-1, vmax=1)
            plt.title('Difference')
            plt.colorbar()
            
            self.logger.experiment.log({
                f'{stage}_predictions': wandb.Image(fig)
            })
            plt.close()
    
    def _log_1d_predictions(self, pred, target, stage):
        """Log example predictions to wandb"""
        if self.logger:
            fig = plt.figure(figsize=(15, 5))
            plt.plot(pred.cpu().numpy(), label='Predicted')
            plt.plot(target.cpu().numpy(), label='Target')
            plt.legend()
            self.logger.experiment.log({
                f'{stage}_1d_predictions': wandb.Image(fig)
            })
            plt.close()
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def train_hic_predictor(
    train_dataset,
    val_dataset,
    test_dataset=None,
    batch_size=32,
    num_workers=4,
    max_epochs=300,
    gpus=1,
    **model_kwargs
):
    """Main training function that sets up and runs the training pipeline.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset (optional): Test dataset
        batch_size (int, optional): Batch size for training. Defaults to 32.
        num_workers (int, optional): Number of data loading workers. Defaults to 4.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
        gpus (int, optional): Number of GPUs to use. Defaults to 1.
        **model_kwargs: Additional arguments passed to HiCPredictor
        
    Returns:
        tuple: Trained model and trainer instances
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project='hic-prediction',
        name='hic-predictor-run',
        log_model=True
    )
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='hic-predictor-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )


    # Initialize model and trainer
    model = HiCPredictorModule(
        model=PeakConvTransHiC(feature_dim=283, hidden_dim=64),
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus else 'cpu',
        devices=gpus,
        precision='16-mixed',
        strategy='auto',
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=10,
        log_every_n_steps=10,
    )
    torch.set_float32_matmul_precision('medium')
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    # Test if test dataset is provided
    if test_loader:
        trainer.test(dataloaders=test_loader)
    
    return model, trainer

if __name__ == "__main__":
    # Create datasets using new peak-level dataset class
    train_dataset = CuratedPeakMotifHiCDataset(
        curated_zarr='h1_esc_nucleotide_motif_adaptor_output_peak.zarr',
        is_train=True,
        leave_out_chromosomes='chr10,chr15',
    )
    
    val_dataset = CuratedPeakMotifHiCDataset(
        curated_zarr='h1_esc_nucleotide_motif_adaptor_output_peak.zarr',
        is_train=False,
        leave_out_chromosomes='chr15'
    )
    
    # Train model with peak-level architecture
    model, trainer = train_hic_predictor(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=0,
        max_epochs=500,
        gpus=1,
        model=PeakConvTransHiC(feature_dim=283, hidden_dim=64)  # Adjust feature_dim based on number of motifs
    )
# %%