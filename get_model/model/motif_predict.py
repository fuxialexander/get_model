"""
Model for predicting motif scanning outputs from DNA sequences.

This module provides a simple CNN model that predicts motif scanning outputs
using a frozen first layer initialized from motifs_with_rc_aligned.pt.
"""

import logging
import os
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import MISSING

from get_model.model.modules import BaseConfig, BaseModule
from get_model.model.model import BaseGETModel, BaseGETModelConfig

# Import FIMO functions for p-value calculation
try:
    import sys
    # Try multiple possible paths
    possible_paths = [
        '/home/xf2217/Repos/dlbcl_pipeline_package',
        '/home/xf2217/Repos/motif_chip_model',
    ]
    imported = False
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
        try:
            from mutation_analysis.fimo import _pwm_to_mapping, pvalue_from_logpdf, trim_motif_padding
            imported = True
            break
        except ImportError:
            continue
    
    if not imported:
        raise ImportError("Could not find mutation_analysis.fimo module")
except ImportError as e:
    logging.warning(f"Could not import FIMO functions: {e}. P-value prediction will be disabled.")
    _pwm_to_mapping = None
    pvalue_from_logpdf = None
    trim_motif_padding = None


@dataclass
class MotifPredictModelConfig(BaseGETModelConfig):
    """Configuration for MotifPredictModel."""
    motif_kernels_path: str = MISSING
    num_motifs: int = 3222  # Will be inferred from loaded kernels if not specified
    kernel_length: int = 13  # Kernel length for CNN layers
    hidden_dim: int = 32  # Hidden dimension for CNN layers
    sequence_length: int = 512  # Input sequence length
    predict_pvalue: bool = False  # Whether to predict p-values using FIMO method
    fimo_bin_size: float = 0.1  # Bin size for FIMO p-value calculation


class MotifPredictModel(BaseGETModel):
    """
    Simple CNN model for predicting motif scanning outputs.
    
    Architecture:
    - Frozen motif scanner layer (initialized from motifs_with_rc_aligned.pt)
      - Outputs raw scanning values (no normalization or activation)
    - 3-layer CNN: 4 -> hidden_dim -> hidden_dim -> num_motifs
    - Layer 1: kernel_size=kernel_length, GroupNorm, ReLU
    - Layer 2: kernel_size=kernel_length, GroupNorm, ReLU
    - Layer 3: kernel_size=1 (linear per nucleotide), no ReLU (predicts raw values)
    
    The frozen layer provides raw motif scanning outputs as targets for training.
    """
    
    def __init__(self, cfg: MotifPredictModelConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Load motif kernels
        if not os.path.exists(cfg.motif_kernels_path):
            raise FileNotFoundError(
                f"Motif kernels file not found: {cfg.motif_kernels_path}"
            )
        
        motif_data = torch.load(cfg.motif_kernels_path, map_location="cpu", weights_only=False)
        motif_kernels = motif_data['motif_kernels']
        
        # Handle both numpy and torch tensor formats
        if isinstance(motif_kernels, torch.Tensor):
            motif_kernels = motif_kernels.numpy()
        
        # Infer number of motifs from loaded kernels
        num_motifs = motif_kernels.shape[0]
        motif_kernel_length = motif_kernels.shape[1]
        
        if cfg.num_motifs != 637 and cfg.num_motifs != num_motifs:
            logging.warning(
                f"Config num_motifs ({cfg.num_motifs}) doesn't match loaded kernels ({num_motifs}). "
                f"Using {num_motifs} from loaded kernels."
            )
        
        self.num_motifs = num_motifs
        
        # Create frozen motif scanner layer
        self.frozen_motif_conv = self._create_frozen_motif_layer(
            motif_kernels, motif_kernel_length
        )
        
        # CNN layers: 4 -> hidden_dim -> hidden_dim -> num_motifs
        self.conv1 = nn.Conv1d(4, cfg.hidden_dim, kernel_size=cfg.kernel_length, padding='same')
        self.gn1 = nn.GroupNorm(32, cfg.hidden_dim)
        
        self.conv2 = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=cfg.kernel_length, padding='same')
        self.gn2 = nn.GroupNorm(32, cfg.hidden_dim)
        
        # Final conv layer uses kernel_size=1 (essentially linear per nucleotide)
        self.conv3 = nn.Conv1d(cfg.hidden_dim, num_motifs, kernel_size=1, padding='same')
        
        # Optional p-value prediction head (like layer 3, but predicts p-values)
        self.predict_pvalue = getattr(cfg, 'predict_pvalue', False)
        self.fimo_bin_size = getattr(cfg, 'fimo_bin_size', 0.1)
        
        if self.predict_pvalue:
            # Add nonlinearity for p-value prediction (relationship is logarithmic)
            # Use a small hidden layer with ReLU to capture nonlinear relationship
            self.conv4_pvalue_hidden = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1, padding='same')
            self.gn4_pvalue = nn.GroupNorm(32, cfg.hidden_dim)
            self.conv4_pvalue = nn.Conv1d(cfg.hidden_dim, num_motifs, kernel_size=1, padding='same')
            # Precompute FIMO lookup tables for each motif
            self._fimo_cache = self._precompute_fimo_mappings(motif_kernels)
        else:
            self.conv4_pvalue_hidden = None
            self.gn4_pvalue = None
            self.conv4_pvalue = None
            self._fimo_cache = None
        
        self.apply(self._init_weights)
    
    def _create_frozen_motif_layer(self, motif_kernels: np.ndarray, kernel_length: int) -> nn.Conv1d:
        """
        Create frozen motif convolution layer.
        
        Args:
            motif_kernels: Motif kernels of shape (num_motifs, length, 4)
            kernel_length: Kernel length
            
        Returns:
            Frozen Conv1d layer
        """
        conv_layer = nn.Conv1d(
            in_channels=4,
            out_channels=self.num_motifs,
            kernel_size=kernel_length,
            padding='same',
            bias=False
        )
        
        # Convert kernels from (num_motifs, length, 4) to (num_motifs, 4, length)
        # for Conv1d weight format
        if motif_kernels.shape[2] == 4:
            # Shape is (num_motifs, length, 4), need to transpose to (num_motifs, 4, length)
            motif_weights = torch.tensor(
                motif_kernels.transpose(0, 2, 1),  # (num_motifs, 4, length)
                dtype=torch.float32
            )
        else:
            raise ValueError(
                f"Unexpected motif_kernels shape: {motif_kernels.shape}. "
                f"Expected (num_motifs, length, 4)"
            )
        
        # Set weights and freeze
        conv_layer.weight.data = motif_weights
        conv_layer.weight.requires_grad = False
        
        return conv_layer
    
    def _precompute_fimo_mappings(self, motif_kernels: np.ndarray) -> dict:
        """Precompute FIMO score-to-p-value mappings for each motif."""
        if _pwm_to_mapping is None:
            logging.warning("FIMO functions not available. P-value prediction disabled.")
            return None
        
        fimo_cache = {}
        fimo_cache_gpu = {}  # GPU-optimized cache
        logging.info(f"Precomputing FIMO mappings for {len(motif_kernels)} motifs...")
        
        for motif_idx in range(len(motif_kernels)):
            # motif_kernels shape: (num_motifs, length, 4)
            # Convert to (4, length) for FIMO (log-odds format)
            motif_log_odds = motif_kernels[motif_idx].T  # (4, length)
            
            # Trim padding
            trimmed_motif = trim_motif_padding(motif_log_odds)
            
            if trimmed_motif.shape[1] > 0:
                smallest, logpdf = _pwm_to_mapping(trimmed_motif, self.fimo_bin_size)
                fimo_cache[motif_idx] = (smallest, logpdf)
                # Store GPU tensors for faster lookup
                fimo_cache_gpu[motif_idx] = (
                    smallest,
                    torch.tensor(logpdf, dtype=torch.float32)  # Will move to GPU when needed
                )
            else:
                fimo_cache[motif_idx] = (0, np.array([0.0]))
                fimo_cache_gpu[motif_idx] = (0, torch.tensor([0.0], dtype=torch.float32))
            
            if (motif_idx + 1) % 500 == 0:
                logging.info(f"  Processed {motif_idx + 1}/{len(motif_kernels)} motifs")
        
        logging.info(f"✓ Precomputed FIMO mappings for {len(fimo_cache)} motifs")
        # Store GPU cache for faster lookup
        self._fimo_cache_gpu = fimo_cache_gpu
        return fimo_cache
    
    def _compute_fimo_pvalues(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute FIMO -log10(p-values) from raw scanning scores using GPU-optimized lookup.
        
        Returns -log10(p-value) instead of p-value for better numerical stability.
        This is faster than CPU lookup because:
        1. No CPU-GPU transfers
        2. Vectorized across all motifs
        3. All operations stay on GPU
        """
        if not hasattr(self, '_fimo_cache_gpu') or self._fimo_cache_gpu is None:
            return torch.zeros_like(scores)
        
        # scores shape: (batch, seq_len, num_motifs)
        batch_size, seq_len, num_motifs = scores.shape
        device = scores.device
        neg_log_pvalues = torch.zeros_like(scores)
        
        # Pre-move all logpdfs to device to avoid repeated transfers
        # Process all motifs (vectorized where possible)
        for motif_idx in range(num_motifs):
            if motif_idx not in self._fimo_cache_gpu:
                continue
            
            smallest, logpdf_gpu = self._fimo_cache_gpu[motif_idx]
            
            # Ensure logpdf is on the same device as scores (cache this after first call)
            if logpdf_gpu.device != device:
                logpdf_gpu = logpdf_gpu.to(device)
                self._fimo_cache_gpu[motif_idx] = (smallest, logpdf_gpu)  # Cache on device
            
            # Get scores for this motif: (batch, seq_len)
            motif_scores = scores[:, :, motif_idx]
            
            # Compute indices: (scores / bin_size) - smallest
            # Use floor division for efficiency
            score_indices = ((motif_scores / self.fimo_bin_size) - smallest).long()
            
            # Clip indices to valid range
            score_indices = torch.clamp(score_indices, 0, len(logpdf_gpu) - 1)
            
            # Lookup log p-values using advanced indexing (all on GPU)
            # logpdf is in log2 space, so log_pvalues = log2(p-value)
            log_pvalues = logpdf_gpu[score_indices]
            
            # Convert to -log10(p-value) and handle edge cases efficiently
            # Clamp -inf to -50 (p < 1e-15) before conversion to avoid inf
            log_pvalues_safe = torch.clamp(log_pvalues, -50.0, 0.0)
            motif_neg_log_pvalues = -log_pvalues_safe * 0.30102999566398114  # log10(2)
            
            # Clamp and handle NaN/Inf in one efficient operation
            motif_neg_log_pvalues = torch.clamp(motif_neg_log_pvalues, 0.0, 15.0)
            motif_neg_log_pvalues = torch.nan_to_num(motif_neg_log_pvalues, nan=0.0, posinf=15.0, neginf=0.0)
            
            neg_log_pvalues[:, :, motif_idx] = motif_neg_log_pvalues
        
        return neg_log_pvalues
    
    def get_input(self, batch):
        """Extract sequence from batch."""
        return {'sequence': batch['sequence']}
    
    def forward(self, sequence):
        """
        Forward pass through the model.
        
        Args:
            sequence: Input sequence tensor of shape (batch, seq_len, 4)
                     Called with keyword argument from get_input() output
            
        Returns:
            Dictionary with 'prediction' and 'target' keys
            Both have shape (batch, seq_len, num_motifs)
        """
        # Convert sequence to (batch, 4, seq_len) for Conv1d
        # Input is (batch, seq_len, 4) from dataset
        x = sequence.permute(0, 2, 1)  # (batch, 4, seq_len)
        
        # Get target from frozen motif scanner (raw scanning values, no normalization or activation)
        with torch.no_grad():
            target = self.frozen_motif_conv(x)
            target = target.permute(0, 2, 1)  # (batch, seq_len, num_motifs)
            
            # Compute FIMO p-values from raw scanning scores as targets
            if self.predict_pvalue:
                target_pvalue = self._compute_fimo_pvalues(target)
            else:
                target_pvalue = None
        
        # Forward through CNN layers
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        
        # Branch 1: Predict raw scanning values
        prediction = self.conv3(x)
        prediction = prediction.permute(0, 2, 1)  # (batch, seq_len, num_motifs)
        
        # Branch 2: Predict -log10(p-values) (if enabled)
        if self.predict_pvalue:
            # Apply nonlinear transformation
            prediction_pvalue = self.conv4_pvalue_hidden(x)
            prediction_pvalue = self.gn4_pvalue(prediction_pvalue)
            
            # Final linear layer to output -log10(p-values)
            prediction_pvalue = self.conv4_pvalue(prediction_pvalue)
            # Values typically range from 0 to ~10-15 for p-values from 1 to 1e-15
            prediction_pvalue = prediction_pvalue.permute(0, 2, 1)  # (batch, seq_len, num_motifs)
            
            # Clamp predictions to reasonable range and handle NaN/Inf in one step
            prediction_pvalue = torch.clamp(prediction_pvalue, 0.0, 15.0)
            prediction_pvalue = torch.nan_to_num(prediction_pvalue, nan=0.0, posinf=15.0, neginf=0.0)
        else:
            prediction_pvalue = None
        
        output = {'prediction': prediction, 'target': target}
        if self.predict_pvalue:
            output['prediction_pvalue'] = prediction_pvalue
            output['target_pvalue'] = target_pvalue
        
        return output
    
    def before_loss(self, output, batch):
        """
        Prepare output and target for loss computation.
        
        Args:
            output: Model output dictionary with 'prediction' and 'target'
            batch: Batch dictionary (not used here, targets come from frozen layer)
            
        Returns:
            pred: Dictionary with 'motif' key containing predictions
            obs: Dictionary with 'motif' key containing targets
        """
        pred = {'motif': output['prediction']}
        obs = {'motif': output['target']}
        
        # Add p-value predictions/targets if enabled
        if self.predict_pvalue and 'prediction_pvalue' in output:
            pred['motif_pvalue'] = output['prediction_pvalue']
            obs['motif_pvalue'] = output['target_pvalue']
        
        return pred, obs
    
    def generate_dummy_data(self):
        """Generate dummy input data for testing."""
        B, L = 2, self.cfg.sequence_length
        return {
            'sequence': torch.randn(B, L, 4).float(),
        }

