#%%
import numpy as np
# import seaborn as sns
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
import wandb
import torch.nn as nn
import matplotlib.pyplot as plt
from get_model.model.model import GETFinetune
from get_model.dataset.collate import get_rev_collate_fn
from get_model.dataset.zarr_dataset import PretrainDataset, ZarrDataPool, PreloadDataPack, CelltypeDenseZarrIO, worker_init_fn_get
# # %%
# cdz = CelltypeDenseZarrIO('/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr')
# # %%
# cdz = cdz.subset_celltypes_with_data_name()
# #%%
# cdz = cdz.leave_out_celltypes_with_pattern('Astrocyte')


# wandb.login()
# run = wandb.init(
#     project="get",
#     name="finetune-gbm",
# )
#%%
pretrain = PretrainDataset(['/pmglocal/xf2217/get_data/shendure_fetal_dense.zarr_v3',
                            ],
                           '/pmglocal/xf2217/get_data/hg38.zarr', 
                           '/pmglocal/xf2217/get_data/hg38_motif_result.zarr', [
                           '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.adjecent.feather', '/pmglocal/xf2217/get_model/data/hg38_4DN_average_insulation.ctcf.longrange.feather'], peak_name='peaks_q0.01_tissue_open_exp', preload_count=200, n_packs=1,
                           max_peak_length=5000, center_expand_target=200, n_peaks_lower_bound=50, n_peaks_upper_bound=100, leave_out_celltypes='Astrocyte', leave_out_chromosomes='chr4', is_train=False, additional_peak_columns=['Expression_positive', 'Expression_negative', 'aTPM', 'TSS'], non_redundant=None, use_insulation=False, dataset_size=4096)
pretrain.__len__()
#%%
df = pretrain.datapool.zarr_dict['shendure_fetal_dense.zarr_v3'].get_peaks(pretrain.datapool.zarr_dict['shendure_fetal_dense.zarr_v3'].ids[2], 'peaks_q0.01_tissue_open_exp').query("TSS>0")
#%%
df['Exp'] =df.Expression_positive + df.Expression_negative

#%%

df[['Exp', 'aTPM']].plot(x='Exp',y='aTPM',kind='scatter',s=1)
#%%
from sklearn.metrics import r2_score
r2_score(df.Exp, df.aTPM)
#%%
def get_rev_collate_fn(batch):
    # zip and convert to list
    sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, motif_mean_std, additional_peak_columns_data  = zip(*batch)
    print('got data')
    celltype_peaks = list(celltype_peaks)
    sample_track = list(sample_track)
    sample_peak_sequence = list(sample_peak_sequence)
    sample_metadata = list(sample_metadata)
    motif_mean_std = list(motif_mean_std)
    additional_peak_columns_data = list(additional_peak_columns_data)
    
    batch_size = len(celltype_peaks)
    mask_ratio = sample_metadata[0]['mask_ratio']

    n_peak_max = max([len(x) for x in celltype_peaks])
    # calculate max length of the sample sequence using peak coordinates, padding is 100 per peak
    sample_len_max = max([(x[:,1]-x[:,0]).sum()+100*x.shape[0] for x in celltype_peaks])
    sample_track_boundary = []
    sample_peak_sequence_boundary = []
    # pad each peaks in the end with 0
    for i in range(len(celltype_peaks)):
        celltype_peaks[i] = np.pad(celltype_peaks[i], ((0, n_peak_max - len(celltype_peaks[i])), (0,0)))
        # pad each track in the end with 0 which is csr_matrix, use sparse operation
        sample_track[i].resize((sample_len_max, sample_track[i].shape[1]))
        sample_peak_sequence[i].resize((sample_len_max, sample_peak_sequence[i].shape[1]))
        sample_track[i] = sample_track[i].todense()
        sample_peak_sequence[i] = sample_peak_sequence[i].todense()
        cov = (celltype_peaks[i][:,1]-celltype_peaks[i][:,0]).sum()
        real_cov = sample_track[i].sum()
        conv = 50#int(min(500, max(100, int(cov/(real_cov+20)))))
        sample_track[i] = np.convolve(np.array(sample_track[i]).reshape(-1), np.ones(50)/50, mode='same')
        # if sample_track[i].max()>0:
        #     sample_track[i] = sample_track[i]/sample_track[i].max()

    celltype_peaks = np.stack(celltype_peaks, axis=0)
    celltype_peaks = torch.from_numpy(celltype_peaks)
    sample_track = np.stack(sample_track, axis=0)
    sample_track = torch.from_numpy(sample_track)
    sample_peak_sequence = np.hstack(sample_peak_sequence)
    sample_peak_sequence = torch.from_numpy(sample_peak_sequence).view(-1, batch_size, 4)
    sample_peak_sequence = sample_peak_sequence.transpose(0,1)
    motif_mean_std = np.stack(motif_mean_std, axis=0)
    motif_mean_std = torch.FloatTensor(motif_mean_std)
    peak_len = celltype_peaks[:,:,1]-celltype_peaks[:,:,0]
    padded_peak_len = peak_len + 100
    total_peak_len = peak_len.sum(1)
    n_peaks = (peak_len>0).sum(1)
    # max_n_peaks = n_peaks.max()
    max_n_peaks = n_peak_max
    peak_peadding_len = n_peaks*100
    tail_len = sample_peak_sequence.shape[1] - peak_peadding_len - peak_len.sum(1)
    # flatten the list
    chunk_size = torch.cat([torch.cat([padded_peak_len[i][0:n],tail_len[i].unsqueeze(0)]) for i, n in enumerate(n_peaks)]).tolist()

    mask = torch.stack([torch.cat([torch.zeros(i), torch.zeros(max_n_peaks-i)-10000]) for i in n_peaks.tolist()])
    maskable_pos = (mask+10000).nonzero()

    for i in range(batch_size):
        maskable_pos_i = maskable_pos[maskable_pos[:,0]==i,1]
        idx = np.random.choice(maskable_pos_i, size=np.ceil(mask_ratio*len(maskable_pos_i)).astype(int), replace=False)
        mask[i,idx] = 1
    
    if additional_peak_columns_data[0] is not None:
        # pad each element to max_n_peaks using zeros
        for i in range(len(additional_peak_columns_data)):
            additional_peak_columns_data[i] = np.pad(additional_peak_columns_data[i], ((0, max_n_peaks - len(additional_peak_columns_data[i])), (0,0)))
        additional_peak_columns_data = np.stack(additional_peak_columns_data, axis=0)
        # if aTPM < 0.1, set the expression to 0
        n_peak_labels = additional_peak_columns_data.shape[-1]
        other_peak_labels = np.zeros(1)
        if n_peak_labels >= 3:
            # assuming the third column is aTPM, use aTPM to thresholding the expression
            additional_peak_columns_data = additional_peak_columns_data.reshape(-1, n_peak_labels)
            additional_peak_columns_data[additional_peak_columns_data[:,2]<0.1, 0] = 0
            additional_peak_columns_data[additional_peak_columns_data[:,2]<0.1, 1] = 0
            additional_peak_columns_data = additional_peak_columns_data.reshape(batch_size, -1, n_peak_labels)
            other_peak_labels = additional_peak_columns_data[:,:,2:]
        exp_label = additional_peak_columns_data[:,:,0:2]
        exp_label = torch.from_numpy(exp_label) # B, R, C=2 RNA+,RNA-,ATAC
        other_peak_labels = torch.from_numpy(other_peak_labels)
    else:
        additional_peak_columns_data = torch.zeros(1)
        other_peak_labels = torch.zeros(1)

    return sample_track, sample_peak_sequence, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, exp_label, other_peak_labels
#%%
data_loader_train = torch.utils.data.DataLoader(
    pretrain,
    batch_size=8,
    num_workers=16,
    pin_memory=True,
    drop_last=True,
    collate_fn=get_rev_collate_fn,
    worker_init_fn=worker_init_fn_get,
)

loss_masked = nn.PoissonNLLLoss(log_input=False, reduce='mean')
#%%
model = GETFinetune(
        num_regions=500,
        num_res_block=0,
        motif_prior=False,
        embed_dim=768,
        num_layers=12,
        d_model=768,
        flash_attn=False,
        nhead=12,
        dropout=0.1,
        output_dim=2,
        pos_emb_components=[],
        atac_kernel_num = 161,
        atac_kernel_size = 3,
        joint_kernel_num = 161,
        final_bn = False,
    )
#%%
checkpoint = torch.load("/burg/pmg/users/xf2217/get_data/20240204-pretrain_conv50_depth4096_500_region_200bp-fetal-leaveout-Astrocyte-chr11-atpm-0.1.pth")
#%%
model.load_state_dict(checkpoint["model"], strict=True)

# checkpoint = torch.load('/pmglocal/alb2281/get_ckpts/checkpoint-135.pth')
# model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
model.cuda()

# for i, j in model.atac_attention.joint_conv.named_parameters():
#     print(i, j.requires_grad)
#     weight = j.detach().cpu().numpy()

# figsize = (10, 10)
# plt.imshow(weight[160,:,:], aspect=0.01)

#%%
# plot as six line plot
# plot as a panel horizontally
# fig, ax = plt.subplots(1, 10, figsize=(15, 2))
# for i in range(10):
#     # calculate reorder index
#     from scipy.cluster.hierarchy import linkage, dendrogram
#     Z = linkage((weight[i+10,:,:]), 'ward')
#     g = dendrogram(Z, no_plot=True)
#     ax[i].imshow((weight[i+10,:,:])[np.array(g['ivl']).astype('int')], aspect=0.01)
    
#%%
from get_model.dataset.zarr_dataset import get_mask_pos, get_padding_pos

def train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, atac_target, exp_target, criterion):
    device = peak_seq.device
    padding_mask = get_padding_pos(mask)
    mask_for_loss = 1-padding_mask
    padding_mask = padding_mask.to(device, non_blocking=True).bool()
    mask_for_loss = mask_for_loss.to(device, non_blocking=True).unsqueeze(-1)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        atac, exp = model(peak_seq, sample_track, mask_for_loss, padding_mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std)
    # loss_atac = criterion(atac, atac_target)
        

    exp = exp * mask_for_loss
    indices = torch.where(mask_for_loss==1)
    exp = exp[indices[0], indices[1], :].flatten()
    exp_target = exp_target * mask_for_loss
    exp_target = exp_target[indices[0], indices[1], :].flatten()
    loss_exp = criterion(exp, exp_target)
    # loss = loss_atac + loss_exp
    loss = loss_exp
    return loss, atac, exp, exp_target 

losses = []
preds = []
obs = []
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    for i, batch in tqdm(enumerate(data_loader_train)):
        if i > 200:
            break
        sample_track, peak_seq, sample_metadata, celltype_peaks, sample_track_boundary, sample_peak_sequence_boundary, chunk_size, mask, n_peaks, max_n_peaks, total_peak_len, motif_mean_std, labels_data, other_labels_data = batch
        if min(chunk_size)<0:
            continue
        device  = 'cuda'
        sample_track = sample_track.to(device, non_blocking=True).bfloat16()
        peak_seq = peak_seq.to(device, non_blocking=True).bfloat16()
        motif_mean_std = motif_mean_std.to(device, non_blocking=True).bfloat16()
        n_peaks = n_peaks.to(device, non_blocking=True)
        labels_data = labels_data.to(device, non_blocking=True).bfloat16()
        other_labels_data = other_labels_data.to(device, non_blocking=True).bfloat16()

        # compute output
        loss, atac, exp, exp_target = train_class_batch(model, peak_seq, sample_track, mask, chunk_size, n_peaks, max_n_peaks, motif_mean_std, None, labels_data, loss_masked)

        # other_labels_data is B, R, N where [:,:, 1] is TSS indicator
        # only append tss preds and obs
        print(other_labels_data.shape)
        print(exp.shape)
        preds.append(exp.reshape(8,500,2)[other_labels_data[:,:,1]==1, :].reshape(-1).detach().cpu().numpy())
        obs.append(exp_target.reshape(8,500,2)[other_labels_data[:,:,1]==1, :].reshape(-1).detach().cpu().numpy())

        # preds.append(exp.reshape(-1).detach().cpu().numpy())
        # obs.append(exp_target.reshape(-1).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0).reshape(-1)
    obs = np.concatenate(obs, axis=0).reshape(-1)
    # preds_atac = np.concatenate(preds_atac, axis=0).reshape(-1)
    # obs_atac = np.concatenate(obs_atac, axis=0).reshape(-1)
# %%
import seaborn as sns
# preds_ = preds[obs>0]
# obs_ = obs[obs>0]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(x=preds, y=obs, s=2, alpha=1, ax=ax)
# add correlation as text
# set x lim
# plt.xlim([0, 4])
# set y lim
# plt.ylim([0, 4])

from scipy.stats import spearmanr, pearsonr
# r2_score(preds, obs)
from sklearn.metrics import r2_score

corr = pearsonr(preds, obs)[0]
r2 = r2_score(preds, obs)
ax.text(0.5, 0.5, f'Pearson r={corr:.2f}\nR2={r2:.2f}', ha='center', va='center', transform=ax.transAxes)
# %%
# save the plot
fig.savefig('finetune_gbm.png')