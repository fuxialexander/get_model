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
import pandas as pd
pd.DataFrame(np.stack(np.where(((motif_zarr['chrs/chr1'][0:1000000]>5) & (motif_zarr['chrs/chr1'][0:1000000]<6)))).T, columns=['pos', 'motif_idx']).groupby('motif_idx').sample(2, replace=False)
# %%

# %%
# process executor across chunks 1000000
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
import zarr
import numpy as np
import pandas as pd
motif_zarr = zarr.open('/home/xf2217/Projects/get_data/hg38_motif.zarr')
def sample_pos_by_motif(data, chrom, num_samples=2, lower_threshold=5, upper_threshold=6):
    try:    
        result = pd.DataFrame(np.stack(np.where(((data>=lower_threshold) & (data<upper_threshold)))).T, columns=['pos', 'motif_idx']).groupby('motif_idx').sample(num_samples, replace=False)
        result['chrom'] = chrom
        return result
    except:
        return pd.DataFrame(columns=['pos', 'motif_idx', 'chrom'])



def process_chunk(args):
    chrom, start, end, thresholds = args
    motif_zarr = zarr.open('/home/xf2217/Projects/get_data/hg38_motif.zarr')
    data = motif_zarr[f'chrs/{chrom}'][start:end]
    results = []
    for threshold in tqdm(thresholds):
        result = sample_pos_by_motif(data, chrom, num_samples=2, lower_threshold=threshold, upper_threshold=threshold+1)
        results.append(result)
    return pd.concat(results, ignore_index=True)
#%%
def process_chromosome(motif_zarr, chrom):
    chunk_size = 2000000
    thresholds = np.arange(5, 25, 2)
    
    tasks = []
    for start in range(0, len(motif_zarr[f'chrs/{chrom}']), chunk_size):
        end = start + chunk_size
        tasks.append((chrom, start, end, thresholds))
    
    results = []
    with ProcessPoolExecutor(16) as executor:
        for result in tqdm(executor.map(process_chunk, tasks), total=len(tasks), desc=f"Processing {chrom}"):
            results.append(result)
    
    return pd.concat(results, ignore_index=True)

# Process all chromosomes
all_results = []
for chrom in tqdm(motif_zarr['chrs'].keys(), desc="Chromosomes"):
    chrom_results = process_chromosome(motif_zarr, chrom)
    all_results.append(chrom_results)

# Combine results from all chromosomes
final_results = pd.concat(all_results, ignore_index=True)

# %%
# Display the first few rows of the final results
print(final_results.head())

# %%
# Save the results to a CSV file
final_results.to_csv('motif_positions_results.csv', index=False)
print("Results saved to motif_positions_results.csv")

# %%
final_results_sampled = final_results.groupby('motif_idx').sample(2000, replace=True)[['chrom', 'pos']].reset_index(drop=True).sort_values(['chrom', 'pos'])#.to_csv('motif_positions_results_2000.csv', index=False)

filtered_df = []
for chr, df in final_results_sampled.groupby('chrom'):
    prev_pos = df.iloc[0]['pos']
    for i, row in df.iterrows():
        pos = row['pos']
        if pos-prev_pos > 200:
            filtered_df.append(row.values)
        prev_pos = pos
filtered_df = pd.DataFrame(filtered_df, columns=['chrom', 'pos'])
#%%
# remove chrom name with length>5
filtered_df = filtered_df[filtered_df['chrom'].str.len()<=5]
filtered_df.to_csv('motif_positions_results_2000_filtered.csv', index=False)

# %%
