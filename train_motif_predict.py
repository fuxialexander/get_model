#!/usr/bin/env python
"""
Training script for motif prediction model.
"""

import hydra
from omegaconf import DictConfig
from get_model.run_motif_predict import run


@hydra.main(version_base="1.3", config_path="get_model/config", config_name="example_motif_predict")
def main(cfg: DictConfig):
    """Main training function."""
    trainer = run(cfg)
    return trainer


if __name__ == "__main__":
    main()

