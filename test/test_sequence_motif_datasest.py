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
cfg.stage='validate'
cfg.finetune.resume_ckpt = '/home/xf2217/output/GETNucleotideMotifAdaptorV3/debug/checkpoints/best-v8.ckpt'
cfg.run.use_wandb=False
cfg.dataset.leave_out_chromosomes = 'chr1'
trainer = run(cfg)
#%%
trainer.model.model.to('cuda')
trainer.model.model.half()
trainer.model.model.eval()
# %%
for i, batch in enumerate(trainer.val_dataloaders):
    if i == 0:
        print(batch['sequence'].shape)
        print(batch['motif'].shape)
        input_data = trainer.model.model.get_input(batch)
        input_data['sequence'] = input_data['sequence'].to('cuda')
        output = trainer.model.model(**input_data)
        break
# %%
obs = batch['motif'][1,:,:].cpu().numpy().flatten()
pred = output[1,:,:].detach().cpu().numpy().flatten()
# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x=obs, y=pred,s=3)
# add axes label
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.show()
# %%
# %%
# heatmap side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(batch['motif'][2,:,:].cpu().numpy(), ax=axs[0], label='Observed')
sns.heatmap(output[2,:,:].detach().cpu().numpy(), ax=axs[1], label='Predicted')
axs[0].set_title('Observed')
axs[1].set_title('Predicted')
plt.show()
# %%
import zarr 
motif_zarr = zarr.open('/home/xf2217/Projects/get_data/hg38_motif.zarr')
# %%
list(motif_zarr.keys())
# %%
import numpy as np
np.quantile(motif_zarr['chrs/chr1'][0:100000], 0.95, axis=0)
# %%
# for each chunk, collect the axis 0 position where the value is larger than 0.999 quantile of the axis 0 
# try with chunk 1. chunk size 100000
data = motif_zarr['chrs/chr1'][0:100000]
quantile_99 = np.quantile(data, 0.99, axis=0)
# find the index where the value is larger than the quantile, treat each axis independently 
quantile_99_indices = {}
for i in range(quantile_99.shape[0]):
    indices = np.argwhere(data[:, i] > quantile_99[i])
    current_idx=0
    previous_idx=0
    if len(indices) > 0:
        quantile_99_indices[i] = np.array([])
        for j in indices:
            previous_idx = current_idx
            current_idx = j
            if current_idx - previous_idx > 236:
                quantile_99_indices[i] = np.append(quantile_99_indices[i], current_idx)
# %%
quantile_99_indices
# %%
