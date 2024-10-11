import hydra
import sys
import gc
import torch
from get_model.config.config import Config
from get_model.run_motif_adaptor import run


@hydra.main(config_path="../config", config_name="nucleotide_motif_adaptor", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit()
