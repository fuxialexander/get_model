"""
Model for predicting ATAC signal from DNA sequences.

This module builds upon the motif prediction model by reusing its main trunk
(convolutional layers) and adding an ATAC signal prediction head.

Architecture:
1. Reuse the main trunk from MotifPredictModel:
   - Conv1: 4 → hidden_dim (kernel_size=13), GroupNorm, ReLU
   - Conv2: hidden_dim → hidden_dim (kernel_size=13), GroupNorm, ReLU

2. Add ATAC signal prediction head:
   - Several residual blocks with dilated convolutions
   - Output: (batch, seq_len, 1) for ATAC signal

Optionally, the model can also predict motifs (multi-task learning).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import MISSING

from get_model.model.modules import BaseConfig, BaseModule
from get_model.model.model import BaseGETModel, BaseGETModelConfig


@dataclass
class ATACSignalPredictModelConfig(BaseGETModelConfig):
    """Configuration for ATACSignalPredictModel."""
    # Trunk configuration (from MotifPredictModel)
    motif_kernels_path: str = MISSING
    num_motifs: int = 3222  # Will be inferred from loaded kernels
    kernel_length: int = 13  # Kernel length for trunk CNN layers
    hidden_dim: int = 32  # Hidden dimension for trunk CNN layers
    sequence_length: int = 2048  # Input sequence length

    # ATAC head configuration
    atac_hidden_dim: int = 128  # Hidden dimension for ATAC head
    atac_num_res_blocks: int = 8  # Number of residual blocks in ATAC head
    atac_dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 11, 41, 81, 167])
    atac_kernel_size: int = 3  # Kernel size for ATAC residual blocks
    atac_dropout: float = 0.1  # Dropout rate for ATAC head

    # Multi-task configuration
    predict_motif: bool = False  # Whether to also predict motifs
    share_trunk_features: bool = True  # Whether ATAC head uses trunk features

    # Pretrained model
    pretrained_motif_model_path: Optional[str] = None  # Path to pretrained MotifPredictModel
    freeze_trunk: bool = False  # Whether to freeze the trunk during training


class ResidualBlock1D(nn.Module):
    """Residual block with dilated convolutions and group normalization.

    Similar to SimplifiedDNAATACMotifCNN's residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        num_groups: int = 8,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding="same"
        )
        self.gn1 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=1, padding="same"
        )
        self.gn2 = nn.GroupNorm(
            num_groups=min(num_groups, out_channels), num_channels=out_channels
        )

        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


class ATACSignalHead(nn.Module):
    """ATAC signal prediction head.

    Uses residual blocks with dilated convolutions to predict ATAC signal
    from trunk features.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_res_blocks: int = 8,
        dilations: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_channels: int = 1,
    ):
        super().__init__()

        if dilations is None:
            dilations = [1, 2, 3, 5, 11, 41, 81, 167]

        # Ensure we have enough dilations for all blocks
        if len(dilations) < num_res_blocks:
            dilations = dilations * (num_res_blocks // len(dilations) + 1)
        dilations = dilations[:num_res_blocks]

        # Build residual blocks
        self.res_blocks = nn.ModuleList()
        channels = [in_channels] + [hidden_dim] * num_res_blocks

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResidualBlock1D(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                )
            )

        # Regression head
        self.regression_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, output_channels, kernel_size=1),
            nn.Softplus(),  # Ensure non-negative output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, in_channels, seq_len)

        Returns:
            ATAC predictions of shape (batch, output_channels, seq_len)
        """
        for block in self.res_blocks:
            x = block(x)

        return self.regression_head(x)


class ATACSignalPredictModel(BaseGETModel):
    """
    Model for predicting ATAC signal from DNA sequences.

    Architecture:
    1. Trunk (reused from MotifPredictModel):
       - Frozen motif scanner layer (for target generation if predict_motif=True)
       - Conv1: 4 → hidden_dim, kernel_size=kernel_length, GroupNorm, ReLU
       - Conv2: hidden_dim → hidden_dim, kernel_size=kernel_length, GroupNorm, ReLU

    2. ATAC signal head:
       - Multiple residual blocks with dilated convolutions
       - Regression head with Softplus activation

    3. Optional motif head (if predict_motif=True):
       - Conv3: hidden_dim → num_motifs, kernel_size=1

    The trunk can be initialized from a pretrained MotifPredictModel.
    """

    def __init__(self, cfg: ATACSignalPredictModelConfig):
        super().__init__(cfg)
        self.cfg = cfg

        # Load motif kernels for frozen scanner
        if not os.path.exists(cfg.motif_kernels_path):
            raise FileNotFoundError(
                f"Motif kernels file not found: {cfg.motif_kernels_path}"
            )

        motif_data = torch.load(cfg.motif_kernels_path, map_location="cpu", weights_only=False)
        motif_kernels = motif_data['motif_kernels']

        if isinstance(motif_kernels, torch.Tensor):
            motif_kernels = motif_kernels.numpy()

        num_motifs = motif_kernels.shape[0]
        motif_kernel_length = motif_kernels.shape[1]

        if cfg.num_motifs != 637 and cfg.num_motifs != num_motifs:
            logging.warning(
                f"Config num_motifs ({cfg.num_motifs}) doesn't match loaded kernels ({num_motifs}). "
                f"Using {num_motifs} from loaded kernels."
            )

        self.num_motifs = num_motifs

        # Create frozen motif scanner layer (for target generation)
        if cfg.predict_motif:
            self.frozen_motif_conv = self._create_frozen_motif_layer(
                motif_kernels, motif_kernel_length
            )
        else:
            self.frozen_motif_conv = None

        # ========================
        # TRUNK (from MotifPredictModel)
        # ========================
        self.conv1 = nn.Conv1d(4, cfg.hidden_dim, kernel_size=cfg.kernel_length, padding='same')
        self.gn1 = nn.GroupNorm(min(32, cfg.hidden_dim), cfg.hidden_dim)

        self.conv2 = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=cfg.kernel_length, padding='same')
        self.gn2 = nn.GroupNorm(min(32, cfg.hidden_dim), cfg.hidden_dim)

        # ========================
        # ATAC SIGNAL HEAD
        # ========================
        atac_in_channels = cfg.hidden_dim

        self.atac_head = ATACSignalHead(
            in_channels=atac_in_channels,
            hidden_dim=cfg.atac_hidden_dim,
            num_res_blocks=cfg.atac_num_res_blocks,
            dilations=cfg.atac_dilations,
            kernel_size=cfg.atac_kernel_size,
            dropout=cfg.atac_dropout,
            output_channels=1,
        )

        # ========================
        # OPTIONAL MOTIF HEAD
        # ========================
        if cfg.predict_motif:
            self.conv3 = nn.Conv1d(cfg.hidden_dim, num_motifs, kernel_size=1, padding='same')
        else:
            self.conv3 = None

        # Initialize weights
        self.apply(self._init_weights)

        # Load pretrained trunk if specified
        if cfg.pretrained_motif_model_path:
            self._load_pretrained_trunk(cfg.pretrained_motif_model_path)

        # Freeze trunk if specified
        if cfg.freeze_trunk:
            self._freeze_trunk()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"ATACSignalPredictModel initialized:")
        logging.info(f"  Trunk hidden_dim: {cfg.hidden_dim}")
        logging.info(f"  ATAC head hidden_dim: {cfg.atac_hidden_dim}")
        logging.info(f"  ATAC num_res_blocks: {cfg.atac_num_res_blocks}")
        logging.info(f"  Predict motif: {cfg.predict_motif}")
        logging.info(f"  Freeze trunk: {cfg.freeze_trunk}")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")

    def _create_frozen_motif_layer(self, motif_kernels: np.ndarray, kernel_length: int) -> nn.Conv1d:
        """Create frozen motif convolution layer."""
        conv_layer = nn.Conv1d(
            in_channels=4,
            out_channels=self.num_motifs,
            kernel_size=kernel_length,
            padding='same',
            bias=False
        )

        if motif_kernels.shape[2] == 4:
            motif_weights = torch.tensor(
                motif_kernels.transpose(0, 2, 1),
                dtype=torch.float32
            )
        else:
            raise ValueError(
                f"Unexpected motif_kernels shape: {motif_kernels.shape}. "
                f"Expected (num_motifs, length, 4)"
            )

        conv_layer.weight.data = motif_weights
        conv_layer.weight.requires_grad = False

        return conv_layer

    def _load_pretrained_trunk(self, checkpoint_path: str):
        """Load pretrained trunk weights from a MotifPredictModel checkpoint."""
        if not os.path.exists(checkpoint_path):
            logging.warning(f"Pretrained checkpoint not found: {checkpoint_path}")
            return

        logging.info(f"Loading pretrained trunk from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load only trunk layers (conv1, gn1, conv2, gn2)
        trunk_layers = ['conv1', 'gn1', 'conv2', 'gn2']
        loaded_count = 0

        for name, param in self.named_parameters():
            # Check if this is a trunk layer
            layer_name = name.split('.')[0]
            if layer_name in trunk_layers:
                # Find matching key in checkpoint
                matching_keys = [k for k in state_dict.keys() if name in k or k.endswith(name)]
                if matching_keys:
                    src_key = matching_keys[0]
                    if state_dict[src_key].shape == param.shape:
                        param.data.copy_(state_dict[src_key])
                        loaded_count += 1
                    else:
                        logging.warning(
                            f"Shape mismatch for {name}: "
                            f"checkpoint {state_dict[src_key].shape} vs model {param.shape}"
                        )

        logging.info(f"Loaded {loaded_count} trunk parameters from pretrained model")

    def _freeze_trunk(self):
        """Freeze trunk parameters."""
        trunk_layers = [self.conv1, self.gn1, self.conv2, self.gn2]
        frozen_count = 0

        for layer in trunk_layers:
            for param in layer.parameters():
                param.requires_grad = False
                frozen_count += 1

        logging.info(f"Frozen {frozen_count} trunk parameters")

    def unfreeze_trunk(self):
        """Unfreeze trunk parameters for finetuning."""
        trunk_layers = [self.conv1, self.gn1, self.conv2, self.gn2]
        unfrozen_count = 0

        for layer in trunk_layers:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen_count += 1

        logging.info(f"Unfrozen {unfrozen_count} trunk parameters")

    def get_trunk_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the trunk.

        Args:
            x: Input tensor of shape (batch, 4, seq_len)

        Returns:
            Trunk features of shape (batch, hidden_dim, seq_len)
        """
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)

        return x

    def get_input(self, batch):
        """Extract inputs from batch."""
        inputs = {'sequence': batch['sequence']}
        if 'atac' in batch:
            inputs['atac_target'] = batch['atac']
        return inputs

    def forward(self, sequence: torch.Tensor, atac_target: torch.Tensor = None):
        """
        Forward pass through the model.

        Args:
            sequence: Input sequence tensor of shape (batch, seq_len, 4)
            atac_target: Optional ATAC target for loss computation (batch, seq_len)

        Returns:
            Dictionary with predictions:
            - 'atac_prediction': ATAC signal prediction (batch, seq_len)
            - 'motif_prediction': Motif prediction if predict_motif=True (batch, seq_len, num_motifs)
            - 'motif_target': Motif target from frozen scanner if predict_motif=True
            - 'atac_target': Pass-through ATAC target if provided
        """
        # Convert sequence to (batch, 4, seq_len) for Conv1d
        x = sequence.permute(0, 2, 1)  # (batch, 4, seq_len)

        output = {}

        # Generate motif target from frozen scanner (if doing motif prediction)
        if self.frozen_motif_conv is not None:
            with torch.no_grad():
                motif_target = self.frozen_motif_conv(x)
                motif_target = motif_target.permute(0, 2, 1)  # (batch, seq_len, num_motifs)
                output['motif_target'] = motif_target

        # Pass through trunk
        trunk_features = self.get_trunk_features(x)  # (batch, hidden_dim, seq_len)

        # ATAC signal prediction
        atac_pred = self.atac_head(trunk_features)  # (batch, 1, seq_len)
        atac_pred = atac_pred.squeeze(1)  # (batch, seq_len)
        output['atac_prediction'] = atac_pred

        # Optional motif prediction
        if self.conv3 is not None:
            motif_pred = self.conv3(trunk_features)  # (batch, num_motifs, seq_len)
            motif_pred = motif_pred.permute(0, 2, 1)  # (batch, seq_len, num_motifs)
            output['motif_prediction'] = motif_pred

        # Pass through ATAC target if provided
        if atac_target is not None:
            output['atac_target'] = atac_target

        return output

    def before_loss(self, output, batch):
        """
        Prepare output and target for loss computation.

        Args:
            output: Model output dictionary
            batch: Batch dictionary with targets

        Returns:
            pred: Dictionary with predictions
            obs: Dictionary with observations/targets
        """
        pred = {'atac': output['atac_prediction']}
        obs = {}

        # ATAC target from output (passed through forward) or batch
        if 'atac_target' in output:
            obs['atac'] = output['atac_target']
        elif 'atac' in batch:
            obs['atac'] = batch['atac']

        # Motif predictions/targets if enabled
        if 'motif_prediction' in output:
            pred['motif'] = output['motif_prediction']
        if 'motif_target' in output:
            obs['motif'] = output['motif_target']

        return pred, obs

    def generate_dummy_data(self):
        """Generate dummy input data for testing."""
        B, L = 2, self.cfg.sequence_length
        data = {
            'sequence': torch.randn(B, L, 4).float(),
            'atac': torch.randn(B, L).abs().float(),  # Non-negative ATAC signal
        }
        return data


@dataclass
class ATACSignalPredictFromMotifModelConfig(BaseGETModelConfig):
    """
    Configuration for ATACSignalPredictFromMotifModel.

    This model takes motif scanning results as input and predicts ATAC signal.
    It's simpler than ATACSignalPredictModel as it doesn't need the trunk.
    """
    # Input configuration
    num_motifs: int = 637  # Number of motif channels from scanner
    sequence_length: int = 2048  # Input sequence length

    # ATAC head configuration
    atac_hidden_dim: int = 128  # Hidden dimension for ATAC head
    atac_num_res_blocks: int = 8  # Number of residual blocks in ATAC head
    atac_dilations: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 11, 41, 81, 167])
    atac_kernel_size: int = 3  # Kernel size for ATAC residual blocks
    atac_dropout: float = 0.1  # Dropout rate for ATAC head


class ATACSignalPredictFromMotifModel(BaseGETModel):
    """
    Model for predicting ATAC signal from motif scanning results.

    This is a simpler variant that takes pre-computed motif scanning outputs
    and predicts ATAC signal using residual CNN blocks.

    Input: (batch, seq_len, num_motifs) - Motif scanning results
    Output: (batch, seq_len) - ATAC signal prediction
    """

    def __init__(self, cfg: ATACSignalPredictFromMotifModelConfig):
        super().__init__(cfg)
        self.cfg = cfg

        self.atac_head = ATACSignalHead(
            in_channels=cfg.num_motifs,
            hidden_dim=cfg.atac_hidden_dim,
            num_res_blocks=cfg.atac_num_res_blocks,
            dilations=cfg.atac_dilations,
            kernel_size=cfg.atac_kernel_size,
            dropout=cfg.atac_dropout,
            output_channels=1,
        )

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        logging.info(f"ATACSignalPredictFromMotifModel initialized:")
        logging.info(f"  Input motifs: {cfg.num_motifs}")
        logging.info(f"  ATAC head hidden_dim: {cfg.atac_hidden_dim}")
        logging.info(f"  Total parameters: {total_params:,}")

    def get_input(self, batch):
        """Extract inputs from batch."""
        inputs = {'motif_features': batch['motif_features']}
        if 'atac' in batch:
            inputs['atac_target'] = batch['atac']
        return inputs

    def forward(self, motif_features: torch.Tensor, atac_target: torch.Tensor = None):
        """
        Forward pass.

        Args:
            motif_features: Motif scanning results of shape (batch, seq_len, num_motifs)
            atac_target: Optional ATAC target (batch, seq_len)

        Returns:
            Dictionary with predictions
        """
        # Convert to (batch, num_motifs, seq_len) for Conv1d
        x = motif_features.permute(0, 2, 1)  # (batch, num_motifs, seq_len)

        # ATAC prediction
        atac_pred = self.atac_head(x)  # (batch, 1, seq_len)
        atac_pred = atac_pred.squeeze(1)  # (batch, seq_len)

        output = {'atac_prediction': atac_pred}
        if atac_target is not None:
            output['atac_target'] = atac_target

        return output

    def before_loss(self, output, batch):
        """Prepare output and target for loss computation."""
        pred = {'atac': output['atac_prediction']}
        obs = {}

        if 'atac_target' in output:
            obs['atac'] = output['atac_target']
        elif 'atac' in batch:
            obs['atac'] = batch['atac']

        return pred, obs

    def generate_dummy_data(self):
        """Generate dummy input data for testing."""
        B, L = 2, self.cfg.sequence_length
        return {
            'motif_features': torch.randn(B, L, self.cfg.num_motifs).float(),
            'atac': torch.randn(B, L).abs().float(),
        }
