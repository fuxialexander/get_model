#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from get_model.model.transformer import GETTransformer
from get_model.model.position_encoding import CorigamiPositionalEncoding as PositionalEncoding


# %%
import pytorch_lightning as pl
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model(**model_kwargs) if model_kwargs else model
        
        # MSE Loss for HiC prediction
        self.criterion = nn.MSELoss()
        
    def forward(self, motif_features):
        return self.model(motif_features)
    
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
        motif_features = batch['motif'].float()  # shape: (batch, 400, 16, 285)
        hic_target = batch['hic'].float()  # shape: (batch, 400, 400)
        
        # Forward pass
        hic_pred = self(motif_features)
        
        # Log example predictions periodically 
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_predictions(hic_pred[0].detach(), hic_target[0].detach(), stage)
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
        loss_hic = self.criterion(hic_pred, hic_target)
        # loss_1d = self.criterion(pred_1d, target_1d)
        loss = loss_hic 
        
        # Calculate additional metrics
        with torch.no_grad():
            mse_hic = F.mse_loss(hic_pred, hic_target)
            
            # Pearson correlation
            pred_flat = hic_pred.detach().cpu().numpy().flatten()
            target_flat = hic_target.detach().cpu().numpy().flatten()
            corr, _ = pearsonr(pred_flat, target_flat)
        # Log metrics
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_correlation', corr, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{stage}_mse_hic', mse_hic, on_epoch=True, on_step=False, sync_dist=True)
        

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
            plt.imshow(pred.cpu().numpy(), vmin=-3, vmax=3)
            plt.title('Predicted HiC')
            plt.colorbar()
            
            plt.subplot(132)
            plt.imshow(target.cpu().numpy(), vmin=-3, vmax=3)
            plt.title('Target HiC')
            plt.colorbar()
            
            plt.subplot(133)
            # difference between pred and target
            diff = pred - target
            plt.imshow(diff.cpu().numpy(), vmin=-3, vmax=3)
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
    max_epochs=100,
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
        model=HiCPredictor,
        **model_kwargs
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
    # Example usage
    from get_model.dataset.zarr_dataset import HiCMatrix2MBDataset
    
    # Create datasets
    train_dataset = HiCMatrix2MBDataset(
        zarr_path='/home/xf2217/Projects/get_data/hic_matrix_2mb.zarr',
        is_train=True,
        leave_out_chromosomes='chr10,chr15',
    )
    
    val_dataset = HiCMatrix2MBDataset(
        zarr_path='/home/xf2217/Projects/get_data/hic_matrix_2mb.zarr',
        is_train=False,
        leave_out_chromosomes='chr15'
    )
    
    # Train model
    model, trainer = train_hic_predictor(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=0,
        max_epochs=500,
        gpus=1,
        feature_dim=2,
        hidden_dim=64
    )
# %%
