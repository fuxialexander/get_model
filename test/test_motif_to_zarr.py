# %%
import pandas as pd
import numpy as np
import zarr
import subprocess
import io
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import gc

# %%
def get_motif_df(chrom, start, end):
    """Call tabix to get the motif df for a given chrom, start, end"""
    cmd = f"tabix /home/xf2217/Projects/get_data/hg38.archetype_motifs.v1.0.bed.gz {chrom}:{start}-{end}"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # noqa
    stdout, stderr = process.communicate()
    if stderr:
        raise Exception(stderr)
    motif_df = pd.read_csv(io.StringIO(stdout.decode("utf-8")), sep="\t", header=None, names=["chrom", "start", "end", "cluster", "score", "strand", "seed_motif", "n_motifs"])
    return motif_df

motif_df = get_motif_df('chr1', 1, 500000)

# %%
motif_clusters = sorted(motif_df["cluster"].unique())
# %%
len(motif_clusters)

# %%
z= zarr.open_group("/home/xf2217/Projects/get_data/hg38_motif.zarr", mode='a')


# %%
from caesar.io.genome import ChromSize
chrom_size = ChromSize('hg38', '/home/xf2217/Projects/get_data')
#%%
cluster_to_idx = {cluster: i for i, cluster in enumerate(motif_clusters)}

# %%
from numcodecs import Blosc


# %%
def process_chunk(chrom, chunk_start, chunk_end, cluster_to_idx, zc):
    motif_df = get_motif_df(chrom, chunk_start, chunk_end)
    
    # Calculate the actual chunk size (important for the last chunk)
    actual_chunk_size = min(chunk_end, chrom_size.chrom_sizes[chrom]) - chunk_start
    
    chunk_arr = np.zeros((actual_chunk_size, len(cluster_to_idx)), dtype='float32')
    chunk = motif_df[(motif_df["start"] >= chunk_start) & (motif_df["end"] <= chunk_end)]
    
    for _, row in chunk.iterrows():
        start_offset = max(0, row['start'] - chunk_start)
        end_offset = min(actual_chunk_size, row['end'] - chunk_start)
        chunk_arr[start_offset:end_offset, cluster_to_idx[row['cluster']]] = row['score']
    
    zc[chunk_start:chunk_start + actual_chunk_size, :] = chunk_arr
    
    del motif_df, chunk, chunk_arr
    gc.collect()
    return 1

def process_chromosome(chrom):
    chrom_length = chrom_size.chrom_sizes[chrom]
    zc = z.create_dataset(chrom, shape=(chrom_length, len(motif_clusters)), chunks=(100000, 282), overwrite=True, dtype='float32')
    chunk_starts = list(range(0, chrom_length, 100000))
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chrom, chunk_start, min(chunk_start + 100000, chrom_length), cluster_to_idx, zc) 
                   for chunk_start in chunk_starts]
        
        for chunk_start, future in zip(chunk_starts, tqdm(futures, total=len(chunk_starts), desc=f"Processing {chrom}")):
            future.result()
            gc.collect()
    del zc
    gc.collect()
#%%
import os
for chrom in tqdm(chrom_size.chrom_sizes.keys(), desc="Chromosomes"):
    try:
        process_chromosome(chrom)
    except Exception as e:
        print(f"Error processing {chrom}: {e}")
        os.remove(f"/home/xf2217/Projects/get_data/hg38_motif.zarr/{chrom}")

