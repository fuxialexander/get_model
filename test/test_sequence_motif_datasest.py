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
trainer = run(cfg)
# %%
