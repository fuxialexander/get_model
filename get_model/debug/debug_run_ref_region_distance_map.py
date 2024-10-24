import hydra
import sys
import gc
import torch
from get_model.run_ref_region import run

from get_model.config.config import *


@hydra.main(config_path="../config", config_name="distance_map", version_base="1.3")
def main(cfg: Config):
    run(cfg)


if __name__ == "__main__":
    main()
    gc.collect()
    torch.cuda.empty_cache()
    sys.exit()
