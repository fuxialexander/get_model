#%%
from matplotlib import legend
import pandas as pd
from pyranges import PyRanges as pr
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

#%%
hyena=pd.read_table('fulco_hyena.tsv').fillna(0)
#%%
experiment = pd.read_csv('EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz', sep='\t')
enformer = pd.read_csv('enformer_background_norm.tsv', sep='\t')
#%%
experiment['hyena_sum_abs'] = hyena.sum_score_elementwise_abs.fillna(0).values
experiment['hyena_sum'] = hyena.sum_score_elementwise_abs.fillna(0).values
experiment['hyena_mean_score'] = hyena.mean_score.fillna(0).values
experiment['hyena_sum_score'] = hyena.sum_score.fillna(0).values
experiment['enformer'] = enformer.enformer_norm_score.fillna(0).values
experiment['key'] = experiment['chrom'] + ':' + experiment['chromStart'].astype(str) + '-' + experiment['chromEnd'].astype(str) + ':' + experiment['measuredGeneSymbol']
experiment = experiment.rename({'chrom':'Chromosome', 'chromStart':'Start', 'chromEnd':'End', 'measuredGeneSymbol':'gene_name'}, axis=1)
# change all boolean to int
experiment['Significant'] = experiment['Significant'].astype(int)
experiment['ValidConnection'] = experiment['ValidConnection'].astype(int)
experiment['Regulated'] = experiment['Significant'].astype(int)
experiment = experiment[~experiment.startTSS.isna()]
gene_to_startTSS_dict = dict(zip(experiment.gene_name, experiment.startTSS))

#%%
# ABC based training
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

from tqdm import tqdm
dfs = []
for f in tqdm(glob('dnase_jacob_abc/*_get.csv')):
    df = pd.read_csv(f, index_col=0)
    df['startTSS'] = df['gene_name'].map(gene_to_startTSS_dict)
    df['Distance'] = (df['Start'] - df['startTSS']).abs()
    df['powerlaw'] = get_powerlaw_at_distance(df['Distance'], hic_gamma_reference, hic_scale_reference)
    df['abc_powerlaw'] = df['atac'] * df['powerlaw']
    df['abc_powerlaw'] = df['abc_powerlaw'] / df['abc_powerlaw'].sum()
    df['get_atac_hic'] = df['atac'] * df['predicted_hic']
    df['get'] = df['jacobian_norm'] * df['atac'] + df['get_atac_hic']
    df['get_atac_hic'] = df['get_atac_hic'] / df['get_atac_hic'].sum() 
    df['get'] = df['get'] / df['get'].sum()
    df['jacob_hic'] = df['jacobian_norm'] * df['predicted_hic']
    df['jacob_hic'] = df['jacob_hic'] / df['jacob_hic'].sum()
    df['jacobian_norm'] = (df['jacobian_norm'] * df['atac']) / (df['jacobian_norm'] * df['atac']).sum()
    dfs.append(df)
dfs = pd.concat(dfs).reset_index(drop=True)
dfs['key'] = dfs['Chromosome'] + ':' + dfs['Start'].astype(str) + '-' + dfs['End'].astype(str) + ':' + dfs['gene_name']
#%%
np.bool = np.bool_
overlap = pr(experiment).join(pr(dfs), suffix='_get').df
overlap = overlap.query('gene_name_get==gene_name')
overlap['Distance'] = (overlap['Start'] - overlap['startTSS']).abs()
#%%
# fit a powerlaw function to predicted hic
from scipy.optimize import curve_fit
# get gamma and scale
def func(x, gamma, scale):
    return np.exp(scale + -1 * gamma * np.log(x + 1))
popt, _ = curve_fit(func, overlap['Distance'], overlap['predicted_hic'])
new_gamma = popt[0]
new_scale = popt[1]
#%%
overlap = overlap[[ 'hyena_mean_score', 'hyena_sum_score', 'hyena_sum_abs', 'hyena_sum',
    'Distance', 'gene_name', 'name', 'atac', 'powerlaw', 'predicted_hic', 'jacobian_norm', 'jacob_hic', 'abc_powerlaw',  'get_atac_hic',  'enformer', 'get', 'Regulated', 'EffectSize',
]].groupby(['name', 'gene_name']).max()
overlap['Regulated'] = overlap['Regulated']>0
overlap_backup = overlap.copy()
#%%
overlap = overlap_backup.copy().reset_index()
#%%
import numpy as np

def plot_aupr(overlap, lower, upper, ax=None, n_resamples=50, resample_size=0.8):
    methods = ['abc_powerlaw', 'enformer', 'Distance', 'get', 'atac', 'get_atac_hic']
    average_precisions = {}
    std_devs = {}

    for method in methods:
        if method == 'Distance':
            scores = 1 / overlap['Distance']
        else:
            scores = overlap[method]

        average_precisions[method] = []

        for _ in range(n_resamples):
            resampled_indices = np.random.choice(len(overlap), size=int(len(overlap) * resample_size), replace=True)
            resampled_overlap = overlap.iloc[resampled_indices]
            resampled_scores = scores.iloc[resampled_indices]

            precision, recall, _ = precision_recall_curve(resampled_overlap['Regulated'], resampled_scores)
            average_precision = average_precision_score(resampled_overlap['Regulated'], resampled_scores)
            average_precisions[method].append(average_precision)

        std_devs[method] = np.std(average_precisions[method])
        average_precisions[method] = np.mean(average_precisions[method])

    if ax is None:
        ax = plt.gca()

    for method in methods:
        if method == 'Distance':
            precision, recall, _ = precision_recall_curve(overlap['Regulated'], 1 / overlap['Distance'])
            label = '1/Distance AP={0:0.3f} ± {1:0.3f}'.format(average_precisions[method], std_devs[method])
        else:
            precision, recall, _ = precision_recall_curve(overlap['Regulated'], overlap[method])
            label = '{} AP={:0.3f} ± {:0.3f}'.format(method, average_precisions[method], std_devs[method])

        ax.plot(recall, precision, label=label, linestyle='--')

    # add random
    ax.plot([0, 1], [overlap['Regulated'].mean(), overlap['Regulated'].mean()], linestyle='--', label='Random={0:0.3f}'.format(overlap['Regulated'].mean()))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve (Distance>{} & Distance<={})'.format(lower, upper))
    ax.legend()

# Rest of the code remains the same
distances = [0, 5000, 20000, 100000]
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.flatten()

# plot the total data in ax[0]
plot_aupr(overlap, lower=0, upper=5e6, ax=axes[0])

for i, distance in enumerate(distances):
    o = overlap.dropna().query('Distance>' + str(distance) + ' & Distance<=' + str(distances[i+1] if i+1<len(distances) else 1e6))
    plot_aupr(o, lower=distance, upper=distances[i+1] if i+1<len(distances) else 5e6, ax=axes[i+1])

plt.tight_layout()
plt.show()
#%%
import numpy as np
import seaborn as sns

def plot_aupr_bar(overlap, lower, upper, ax=None, n_resamples=1000, resample_size=0.8):
    methods = ['get', 'jacobian_norm', 'jacob_hic',  'enformer',  'hyena_sum_abs', 'abc_powerlaw', 'atac', 'Distance',]
    # methods = ['get', 'abc_powerlaw', 'enformer', 'Distance']
    average_precisions = {}
    std_devs = {}
    ci95 = {}

    for method in methods:
        if method == 'Distance':
            scores = 1 / overlap['Distance']
        else:
            scores = overlap[method]

        average_precisions[method] = []

        for _ in range(n_resamples):
            resampled_indices = np.random.choice(len(overlap), size=int(len(overlap) * resample_size), replace=False)
            resampled_overlap = overlap.iloc[resampled_indices]
            resampled_scores = scores.iloc[resampled_indices]

            average_precision = average_precision_score(resampled_overlap['Regulated'], resampled_scores)
            average_precisions[method].append(average_precision)

    for method in methods:
        std_devs[method] = np.std(average_precisions[method])
        # quantile 95% confidence interval
        ci95[method] = np.quantile(average_precisions[method], [0.025, 0.975])
        average_precisions[method] = np.mean(average_precisions[method])
    if ax is None:
        ax = plt.gca()

    labels = ['GET (Jacobian x DNase, Powerlaw)', 'GET (Jacobian x DNase)', 'GET (Jacobian, Powerlaw)', 'Enformer (Input x Attention)', 'HyenaDNA (ISM)',  'ABC Powerlaw', 'DNase', 'Distance']
    values = [average_precisions[method] for method in methods]
    errors = [ci95[method][1] - average_precisions[method] for method in methods]
    # add random
    labels.append('Random')
    values.append(overlap['Regulated'].mean())
    errors.append(0)

    

    sns.barplot(x=labels, y=values, ax=ax, hue=labels, dodge=False, hue_order=labels)
    ax.errorbar(x=range(len(labels)), y=values, yerr=errors, fmt='none', capsize=5, ecolor='black')
    n_positive = overlap['Regulated'].sum()
    n_negative = len(overlap) - n_positive
    ax.set_xlabel('[{},{})bp\nPositive:{}\nNegative:{}'.format(int(lower), int(upper), int(n_positive), int(n_negative)))
    # rotate x labels
    ax.set_xticklabels("")
    
    # use legend instead of ticks
    ax.legend()

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
axes = axes.flatten()
distances = [0, 10000, 100000]

for i, distance in enumerate(distances):
    o = overlap.dropna().query('Distance>' + str(distance) + ' & Distance<=' + str(distances[i+1] if i+1<len(distances) else 1e6))
    plot_aupr_bar(o, lower=distance, upper=distances[i+1] if i+1<len(distances) else 5e6, ax=axes[i])

# remove ax[0,1] legends
axes[0].legend().remove()
axes[1].legend().remove()
axes[2].legend().remove()
# put legend below figure in the middle with out affecting the panels
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=4)

axes[0].set_title('AUPRC vs distance to TSS')
# save to pdf
# plt.tight_layout()
plt.savefig('auprc_vs_distance.pdf')
#%%
# compute pearson correlation by distance cutoff and plot as method * bin heatmap
from scipy.stats import pearsonr
from tqdm import tqdm
correlations = []
for i, distance in enumerate(distances):
    o = overlap.dropna().query('Distance>' + str(distance) + ' & Distance<=' + str(distances[i+1] if i+1<len(distances) else 1e6)
    )
    corrs = []
    for method in ['get', 'jacobian_norm', 'jacob_hic',  'enformer',  'hyena_sum_abs', 'abc_powerlaw', 'atac', 'Distance']:
        if method == 'Distance':
            scores = 1 / o['Distance']
        else:
            scores = o[method]
        corrs.append(pearsonr(scores, o['EffectSize'])[0])
    correlations.append(corrs)
correlations = np.array(correlations)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.heatmap(correlations, annot=True, xticklabels=['GET (Jacobian, DNase, Powerlaw)', 'GET (Jacobian)', 'GET (Jacobian, Powerlaw)',  'Enformer (Input x Attention)', 'HyenaDNA (ISM)', 'ABC Powerlaw', 'DNase', 'Distance',], yticklabels=['0-10kb', '10-100kb', '100kb-5Mb'], ax=ax, cmap='Blues_r')
ax.set_title('Pearson correlation between predicted score and CRISPRi effect size')
plt.savefig('pearson_correlation_vs_distance.pdf')
