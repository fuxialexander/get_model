#%%
import pandas as pd

result = pd.read_csv('/pmglocal/xf2217/output/k562_ref_fit_watac.csv', names=['gene_name', 'strand', 'tss_peak', 'perturb_chrom', 'perturb_start', 'perturb_end', 'pred_wt', 'pred_mut', 'obs'], header=None)
result
# %%
experiment = pd.read_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/EPCrisprBenchmark_Fulco2019_K562_GRCh38.tsv.gz', sep='\t')
# %%
experiment['key'] = experiment['chrom'] + ':' + experiment['chromStart'].astype(str) + '-' + experiment['chromEnd'].astype(str) + ':' + experiment['measuredGeneSymbol']

# %%
result['key'] = result['perturb_chrom'] + ':' + result['perturb_start'].astype(str) + '-' + result['perturb_end'].astype(str) + ':' + result['gene_name']
# %%
import numpy as np
merged = pd.merge(result, experiment, on='key', how='left')
merged['pred_mut'] = merged['pred_mut'].astype(float)
merged['pred_wt'] = merged['pred_wt'].astype(float)
merged['obs'] = merged['obs'].astype(float)
merged['Distance'] = (merged['startTSS'] - merged['perturb_start']).abs()
merged['logfc'] = (merged['pred_mut'] - merged['pred_wt']).abs()/merged['pred_wt']#/merged['Distance']
# merged = merged.query('startTSS-perturb_start>100_000')
# merged = merged.query('pred_wt>0.1').dropna()
# precision recall curve of logfc to significant
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
precision, recall, _ = precision_recall_curve(merged['Significant'], merged['logfc'])
average_precision = average_precision_score(merged['Significant'], merged['logfc'])
plt.plot(recall, precision, label='AP={0:0.3f}'.format(average_precision))
# add random
plt.plot([0, 1], [merged['Significant'].mean(), merged['Significant'].mean()], linestyle='--', label='Random={0:0.3f}'.format(merged['Significant'].mean()))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# %%
import seaborn as sns
sns.scatterplot(x='logfc', y='EffectSize', data=merged, hue='Significant')
# %%
merged[['logfc', 'EffectSize']].corr(method='spearman')
# %%
# group by gene_name and calculate the correlation
correlation = merged.groupby('gene_name')[['logfc', 'EffectSize']].corr(method='spearman').reset_index().query('level_1=="logfc"').dropna()
# %%
correlation
# %%
acc = merged[['gene_name', 'obs','pred_wt']].drop_duplicates()
acc['error'] = (acc['obs'] - acc['pred_wt'])/acc['obs']
# %%
pd.merge(correlation, acc, on='gene_name').plot(x='error', y='EffectSize', kind='scatter')
# %%
merged.rename({'chromStart':'start', 'chromEnd':'end', 'chrom':'chr', 'TargetGene': 'gene_name', 'logfc':'GET'}, axis=1)
# %%
to_save = merged.rename({'chromStart':'start', 'chromEnd':'end', 'chrom':'chr', 'measuredGeneSymbol': 'TargetGene', 'logfc':'GET'}, axis=1)[['chr', 'start', 'end', 'TargetGene', 'GET']]
to_save['CellType'] = 'K562'
to_save.to_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/GETQATAC_K562_Fulco2019Genes_GRCh38.tsv', sep='\t')
# %%
to_save
# %%
# load jacobian-based analysis
import zarr
z = zarr.open_group('/pmglocal/xf2217/output/k562_ref_fit_watac.zarr/', mode='r')
import numpy as np
gene_names = [x.strip(' ') for x in z['gene_name'][:]]
chromosomes = [x.strip(' ') for x in z['chromosome'][:]]

strand = z['strand'][:]
jacobians = np.stack([z['jacobians']['exp'][str(x)]['input'][i] for i, x in enumerate(strand)])
input_data = z['input'][:]
atac = input_data[:,:, 282]
peaks = z['peaks'][:]
input_x_jacob = input_data * jacobians
# %%
dfs = []
import pandas as pd
for i, gene in enumerate(gene_names):
    gene_df = pd.DataFrame({'score': np.absolute(input_x_jacob[i]).sum(1), 'Chromosome': chromosomes[i], 'Start': peaks[i][:, 0], 'End': peaks[i][:, 1], 'gene_name': gene, 'Strand': strand[i], 'atac': atac[i]})
    dfs.append(gene_df)


# %%
dfs = pd.concat(dfs).query('Start>0')
# %%
dfs
# %%
experiment = experiment.rename({'chrom':'Chromosome', 'chromStart':'Start', 'chromEnd':'End', 'measuredGeneSymbol':'gene_name'}, axis=1)
# %%
experiment
# %%
from pyranges import PyRanges as pr

overlap = pr(dfs).join(pr(experiment), suffix='_experiment').df.query('gene_name_experiment==gene_name')
overlap['Distance'] = (overlap['Start'] - overlap['startTSS']).abs()

#%%
hic_gamma = 1.024238616787792
hic_scale = 5.9594510043736655
hic_gamma_reference = 0.87
hic_scale_reference = -4.80 + 11.63 * hic_gamma_reference

def get_powerlaw_at_distance(distances, gamma, scale, min_distance=5000):
    assert gamma > 0
    assert scale > 0

    # The powerlaw is computed for distances > 5kb. We don't know what the contact freq looks like at < 5kb.
    # So just assume that everything at < 5kb is equal to 5kb.
    # TO DO: get more accurate powerlaw at < 5kb
    distances = np.clip(distances, min_distance, np.Inf)
    log_dists = np.log(distances + 1)

    powerlaw_contact = np.exp(scale + -1 * gamma * log_dists)
    return powerlaw_contact

overlap['powerlaw'] = get_powerlaw_at_distance(overlap['Distance'], hic_gamma_reference, hic_scale_reference)
overlap['atac/dis'] = overlap['atac']*overlap['powerlaw']

# %%
import seaborn as sns
sns.scatterplot(x='atac/dis', y='EffectSize', data=overlap, hue='Significant')
# %%
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
precision, recall, _ = precision_recall_curve(overlap['Significant'], overlap['atac/dis'])
average_precision = average_precision_score(overlap['Significant'], overlap['atac/dis'])
plt.plot(recall, precision, label='AP={0:0.3f}'.format(average_precision))
# add random
plt.plot([0, 1], [overlap['Significant'].mean(), overlap['Significant'].mean()], linestyle='--', label='Random={0:0.3f}'.format(overlap['Significant'].mean()))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()
# %%
sns.violinplot(x='Significant', y='atac/dis', data=overlap, cut=0)
# %%
to_save = overlap.rename({'Start_experiment':'start', 'End_experiment':'end', 'Chromosome':'chr', 'gene_name': 'TargetGene', 'atac/dis':'AtacDis'}, axis=1)[['chr', 'start', 'end', 'TargetGene', 'AtacDis']]
to_save['CellType'] = 'K562'
to_save.to_csv('/burg/pmg/users/xf2217/CRISPR_comparison/resources/example/AtacDis_K562_Fulco2019Genes_GRCh38.tsv', sep='\t')
# %%