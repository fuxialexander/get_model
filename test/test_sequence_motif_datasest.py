#%%
from get_model.config.config import load_config, pretty_print_config
from get_model.run_motif_adaptor import run
# inline
%matplotlib inline
#%%
# load config
cfg = load_config('nucleotide_motif_adaptor')
pretty_print_config(cfg)

# %%
cfg.training.epochs=5 # use as demo there
cfg.machine.num_workers=1
trainer = run(cfg)
# %%
import zarr
motif_zarr = zarr.open_group('/home/xf2217/Projects/get_data/hg38_motif.zarr')
# %%
data = []
from tqdm import tqdm
for i in tqdm(range(1000000)):
    data.append(motif_zarr['chrs/chr11'][i:i+1000])
# %%
import pandas as pd
cpeaks = pd.read_csv('/home/xf2217/Projects/caesar/data/cPeaks_hg38.bed', sep='\t', header=None)
# %%
cpeaks.head()

# %%
