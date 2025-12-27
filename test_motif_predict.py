#!/usr/bin/env python
"""
Debug script for motif prediction model.
Tests the dataset and model implementation.
"""

import sys
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Test imports
print("Testing imports...")
try:
    from get_model.dataset.sequence_motif_predict_dataset import SequenceMotifPredictDataset
    print("✓ Dataset import successful")
except Exception as e:
    print(f"✗ Dataset import failed: {e}")
    sys.exit(1)

try:
    from get_model.model.motif_predict import MotifPredictModel, MotifPredictModelConfig
    print("✓ Model import successful")
except Exception as e:
    print(f"✗ Model import failed: {e}")
    sys.exit(1)

try:
    from get_model.config.config import SequenceMotifPredictDatasetConfig, NucleotideMotifPredictConfig
    print("✓ Config import successful")
except Exception as e:
    print(f"⚠ Config import failed (may need cooler package): {e}")
    print("  Continuing with model tests...")

# Test model initialization with dummy config
print("\nTesting model initialization...")
try:
    # Create a dummy motif kernels file for testing
    dummy_motif_kernels = torch.randn(637, 21, 4)  # (num_motifs, kernel_length, 4)
    dummy_motif_data = {
        'motif_kernels': dummy_motif_kernels.numpy(),
        'motif_names': [f'motif_{i}' for i in range(637)]
    }
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name
        torch.save(dummy_motif_data, temp_path)
    
    # Create model config - GETLoss will instantiate components from DictConfig
    from get_model.model.model import LossConfig, MetricsConfig
    
    # Create loss config as DictConfig with _target_ fields (GETLoss will instantiate)
    loss_cfg = DictConfig({
        'components': {
            'motif': {
                '_target_': 'torch.nn.MSELoss',
                'reduction': 'mean'
            }
        },
        'weights': {
            'motif': 1.0
        }
    })
    
    metrics_cfg = MetricsConfig(
        components={
            'motif': ['pearson', 'spearman', 'r2']
        }
    )
    
    model_cfg = MotifPredictModelConfig(
        motif_kernels_path=temp_path,
        num_motifs=637,
        kernel_length=13,
        hidden_dim=128,
        sequence_length=512,
        loss=loss_cfg,
        metrics=metrics_cfg
    )
    
    model = MotifPredictModel(model_cfg)
    print("✓ Model initialization successful")
    print(f"  - Number of motifs: {model.num_motifs}")
    print(f"  - Frozen layer frozen: {not model.frozen_motif_conv.weight.requires_grad}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_data = model.generate_dummy_data()
    input_dict = model.get_input(dummy_data)
    output = model(**input_dict)
    
    print(f"✓ Forward pass successful")
    print(f"  - Input shape: {input_dict['sequence'].shape}")
    print(f"  - Prediction shape: {output['prediction'].shape}")
    print(f"  - Target shape: {output['target'].shape}")
    
    # Test before_loss
    print("\nTesting before_loss...")
    pred, obs = model.before_loss(output, dummy_data)
    print(f"✓ before_loss successful")
    print(f"  - pred['motif'] shape: {pred['motif'].shape}")
    print(f"  - obs['motif'] shape: {obs['motif'].shape}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    loss = model.loss(pred, obs)
    print(f"✓ Loss computation successful")
    print(f"  - Loss type: {type(loss)}")
    if isinstance(loss, dict):
        print(f"  - Loss keys: {list(loss.keys())}")
        print(f"  - Loss values: {[v.item() if hasattr(v, 'item') else v for v in loss.values()]}")
    else:
        print(f"  - Loss value: {loss.item() if hasattr(loss, 'item') else loss}")
    
    # Cleanup
    os.unlink(temp_path)
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All model tests passed!")

# Test dataset (requires actual zarr file, so we'll skip if not available)
print("\nTesting dataset (requires zarr file)...")
print("  (Skipping dataset test - requires actual zarr file path)")
print("  To test dataset, provide a valid sequence_zarr path")

print("\n✓ All tests completed successfully!")

