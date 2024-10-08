#%%
from get_model.dataset.zarr_dataset import RegionMotifDataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cooler
#%%
from cooltools.lib.numutils import adaptive_coarsegrain
import cooler
import torch
def adaptive_coarsegrain_gpu(ar, countar, cutoff=5, max_levels=8, min_shape=8):
    """
    Adaptively coarsegrain a Hi-C matrix based on local neighborhood pooling
    of counts.

    Parameters
    ----------
    ar : torch.Tensor, shape (n, n)
        A square Hi-C matrix to coarsegrain. Usually this would be a balanced
        matrix.

    countar : torch.Tensor, shape (n, n)
        The raw count matrix for the same area. Has to be the same shape as the
        Hi-C matrix.

    cutoff : float, optional
        A minimum number of raw counts per pixel required to stop 2x2 pooling.
        Larger cutoff values would lead to a more coarse-grained, but smoother
        map. 3 is a good default value for display purposes, could be lowered
        to 1 or 2 to make the map less pixelated. Setting it to 1 will only
        ensure there are no zeros in the map.

    max_levels : int, optional
        How many levels of coarsening to perform. It is safe to keep this
        number large as very coarsened map will have large counts and no
        substitutions would be made at coarser levels.
    min_shape : int, optional
        Stop coarsegraining when coarsegrained array shape is less than that.

    Returns
    -------
    Smoothed array, shape (n, n)

    Notes
    -----
    The algorithm works as follows:

    First, it pads an array with NaNs to the nearest power of two. Second, it
    coarsens the array in powers of two until the size is less than minshape.

    Third, it starts with the most coarsened array, and goes one level up.
    It looks at all 4 pixels that make each pixel in the second-to-last
    coarsened array. If the raw counts for any valid (non-NaN) pixel are less
    than ``cutoff``, it replaces the values of the valid (4 or less) pixels
    with the NaN-aware average. It is then applied to the next
    (less coarsened) level until it reaches the original resolution.

    In the resulting matrix, there are guaranteed to be no zeros, unless very
    large zero-only areas were provided such that zeros were produced
    ``max_levels`` times when coarsening.

    Examples
    --------
    >>> c = cooler.Cooler("/path/to/some/cooler/at/about/2000bp/resolution")

    >>> # sample region of about 6000x6000
    >>> mat = c.matrix(balance=True).fetch("chr1:10000000-22000000")
    >>> mat_raw = c.matrix(balance=False).fetch("chr1:10000000-22000000")
    >>> mat_cg = adaptive_coarsegrain(mat, mat_raw)

    >>> plt.figure(figsize=(16,7))
    >>> ax = plt.subplot(121)
    >>> plt.imshow(np.log(mat), vmax=-3)
    >>> plt.colorbar()
    >>> plt.subplot(122, sharex=ax, sharey=ax)
    >>> plt.imshow(np.log(mat_cg), vmax=-3)
    >>> plt.colorbar()

    """
    #TODO: do this better without sideeffect
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    with torch.no_grad():
        def _coarsen(ar, operation=torch.sum, min_nan=False):
            """Coarsegrains an array by a factor of 2"""
            M = ar.shape[0] // 2
            newar = ar.reshape(M, 2, M, 2)
            if min_nan:
                newar = torch.nan_to_num(newar,nan=float('inf'))
                cg = operation(newar, axis=1)[0]
                cg = operation(cg, axis=2)[0]
            else:
                cg = operation(newar, axis=1)
                cg = operation(cg, axis=2)
            return cg

        def _expand(ar, counts=None):
            """
            Performs an inverse of nancoarsen
            """
            N = ar.shape[0] * 2
            newar = torch.zeros((N, N),dtype=ar.dtype)
            newar[::2, ::2] = ar
            newar[1::2, ::2] = ar
            newar[::2, 1::2] = ar
            newar[1::2, 1::2] = ar
            return newar

        # defining arrays, making sure they are floats
    #     ar = np.asarray(ar, float)
    #     ar = torch.from_numpy(ar)
    #     countar = np.asarray(countar, float)
    #     countar = torch.from_numpy(countar)
        # TODO: change this to the nearest shape correctly counting the smallest
        # shape the algorithm will reach
        Norig = ar.shape[0]
        Nlog = np.log2(Norig)
        if not np.allclose(Nlog, np.rint(Nlog)):
            newN = np.int(2 ** np.ceil(Nlog))  # next power-of-two sized matrix
            newar = torch.empty((newN, newN), dtype=torch.float)  # fitting things in there
            newar[:] = np.nan
            newcountar = torch.zeros((newN, newN), dtype=torch.float)
            newar[:Norig, :Norig] = torch.from_numpy(ar)
            newcountar[:Norig, :Norig] = torch.from_numpy(countar)
            ar = newar
            countar = newcountar

        armask = torch.isfinite(ar)  # mask of "valid" elements
        countar[~armask] = 0
        ar[~armask] = 0

        assert torch.isfinite(countar).all()
        assert countar.shape == ar.shape

        # We will be working with three arrays.
        ar_cg = [ar]  # actual Hi-C data
        countar_cg = [countar]  # counts contributing to Hi-C data (raw Hi-C reads)
        armask_cg = [armask]  # mask of "valid" pixels of the heatmap

        # 1. Forward pass: coarsegrain all 3 arrays
        for i in range(max_levels):
            if countar_cg[-1].shape[0] > min_shape:
                countar_cg.append(_coarsen(countar_cg[-1]))
                armask_cg.append(_coarsen(armask_cg[-1]))
                ar_cg.append(_coarsen(ar_cg[-1]))

        # Get the most coarsegrained array
        ar_cur = ar_cg.pop()
        countar_cur = countar_cg.pop()
        armask_cur = armask_cg.pop()

        # 2. Reverse pass: replace values starting with most coarsegrained array
        # We have 4 pixels that were coarsegrained to one pixel.
        # Let V be the array of values (ar), and C be the array of counts of
        # valid pixels. Then the coarsegrained values and valid pixel counts
        # are:
        # V_{cg} = V_{0,0} + V_{0,1} + V_{1,0} + V_{1,1}
        # C_{cg} = C_{0,0} + C_{0,1} + C_{1,0} + C_{1,1}
        # The average value at the coarser level is V_{cg} / C_{cg}
        # The average value at the finer level is V_{0,0} / C_{0,0}, etc.
        #
        # We would replace 4 values with the average if counts for either of the
        # 4 values are less than cutoff. To this end, we perform nanmin of raw
        # Hi-C counts in each 4 pixels
        # Because if counts are 0 due to this pixel being invalid - it's fine.
        # But if they are 0 in a valid pixel - we replace this pixel.
        # If we decide to replace the current 2x2 square with coarsegrained
        # values, we need to make it produce the same average value
        # To this end, we would replace V_{0,0} with V_{cg} * C_{0,0} / C_{cg} and
        # so on.
        for i in range(len(countar_cg)):
            ar_next = ar_cg.pop()
            countar_next = countar_cg.pop()
            armask_next = armask_cg.pop()

            # obtain current "average" value by dividing sum by the # of valid pixels
            val_cur = ar_cur / armask_cur
            # expand it so that it is the same shape as the previous level
            val_exp = _expand(val_cur)
            # create array of substitutions: multiply average value by counts
            addar_exp = val_exp * armask_next

            # make a copy of the raw Hi-C array at current level
            countar_next_mask = countar_next.clone()
            countar_next_mask[armask_next == 0] = np.nan  # fill nans
     
            countar_exp = _expand(_coarsen(countar_next, operation=torch.min,min_nan=True))

            curmask = countar_exp < cutoff  # replacement mask
            ar_next[curmask] = addar_exp[curmask]  # procedure of replacement
            ar_next[armask_next == 0] = 0  # now setting zeros at invalid pixels

            # prepare for the next level
            ar_cur = ar_next
            countar_cur = countar_next
            armask_cur = armask_next

        ar_next[armask_next == 0] = np.nan
        ar_next = ar_next[:Norig, :Norig]
        torch.set_default_tensor_type(torch.FloatTensor)
        return ar_next.detach().cpu().numpy()


def _adaptive_coarsegrain(ar, countar, max_levels=12, cuda=False):
    """
    Wrapper for cooltools adaptive coarse-graining to add support 
    for non-square input for interchromosomal predictions.
    """
    global adaptive_coarsegrain_fn
    if cuda:
        adaptive_coarsegrain_fn = adaptive_coarsegrain_gpu
    else:
        adaptive_coarsegrain_fn = adaptive_coarsegrain


    assert np.all(ar.shape == countar.shape)
    if ar.shape[0] < 9 and ar.shape[1] < 9:
        ar_padded = np.empty((9, 9))
        ar_padded.fill(np.nan)
        ar_padded[: ar.shape[0], : ar.shape[1]] = ar

        countar_padded = np.empty((9, 9))
        countar_padded.fill(np.nan)
        countar_padded[: countar.shape[0], : countar.shape[1]] = countar
        return adaptive_coarsegrain_fn(ar_padded, countar_padded, max_levels=max_levels)[
            : ar.shape[0], : ar.shape[1]
        ]

    if ar.shape[0] == ar.shape[1]:
        return adaptive_coarsegrain_fn(ar, countar, max_levels=max_levels)
    elif ar.shape[0] > ar.shape[1]:
        padding = np.empty((ar.shape[0], ar.shape[0] - ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain_fn(
            np.hstack([ar, padding]), np.hstack([countar, padding]), max_levels=max_levels
        )[:, : ar.shape[1]]
    elif ar.shape[0] < ar.shape[1]:
        padding = np.empty((ar.shape[1] - ar.shape[0], ar.shape[1]))
        padding.fill(np.nan)
        return adaptive_coarsegrain_fn(
            np.vstack([ar, padding]), np.vstack([countar, padding]), max_levels=max_levels
        )[: ar.shape[0], :]

background = np.load('/home/xf2217/Projects/get_data/resources/4DNFI643OYP9.rebinned.mcool.expected.res4000.npy')
normmat = np.exp(background[np.abs(np.arange(8000)[None, :] - np.arange(8000)[:, None])])

normmat_r1 = np.reshape(normmat[:250, :250], (250, 1, 250, 1)).mean(axis=1).mean(axis=2)
# normmat_r2 = np.reshape(normmat[:500, :500], (250, 2, 250, 2)).mean(axis=1).mean(axis=2)
# normmat_r4 = np.reshape(normmat[:1000, :1000], (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
# normmat_r8 = np.reshape(normmat[:2000, :2000], (250, 8, 250, 8)).mean(axis=1).mean(axis=2)
# normmat_r16 = (
#     np.reshape(normmat[:4000, :4000], (250, 16, 250, 16)).mean(axis=1).mean(axis=2)
# )
# normmat_r32 = (
#     np.reshape(normmat[:8000, :8000], (250, 32, 250, 32)).mean(axis=1).mean(axis=2)
# )
# normmat_r64 = (
#     np.reshape(normmat[:16000, :16000], (250, 64, 250, 64)).mean(axis=1).mean(axis=2)
# )
# normmat_r128 = (
#     np.reshape(normmat[:32000, :32000], (250, 128, 250, 128)).mean(axis=1).mean(axis=2)
# )
# normmat_r256 = (
#     np.reshape(normmat[:64000, :64000], (250, 256, 250, 256)).mean(axis=1).mean(axis=2)
# )
cool = cooler.Cooler('/home/xf2217/Projects/get_data/resources/4DNFI643OYP9.rebinned.mcool::/resolutions/4000')
mat = cool.matrix(balance=True).fetch('chr11:51000000-52000000')
mat_raw = cool.matrix(balance=False).fetch('chr11:51000000-52000000')
mat_cg = _adaptive_coarsegrain(mat, mat_raw) 
fig, ax = plt.subplots(figsize=(3, 3))
sns.heatmap(np.log(mat_cg)-np.log(normmat_r1), ax=ax, cbar=False)
#%%
# load the zarr file as a dataset. 
region_motif_dataset = RegionMotifDataset(
    "/home/xf2217/Projects/4dn_h1esc/peak_motif/original/H1ESC.4dn_h1esc.4dn_h1esc.peak_motif.zarr/",
    celltypes="H1ESC.4dn_h1esc.4dn_h1esc",
    quantitative_atac=True,
    num_region_per_sample=400,
    leave_out_celltypes=None,
    leave_out_chromosomes="chr11",
    is_train=False,
    hic_path="/home/xf2217/Projects/geneformer_nat/data/H1_ESC.hic",
    hic_resolution=5000,
    hic_method="observed",
    hic_normalization="KR"
)

#%%
mzd = region_motif_dataset.hic_obj.getMatrixZoomData('11', '11', 'observed', 'KR', "BP", 5000)
numpy_matrix = mzd.getRecordsAsMatrix(11000000, 15000000, 11000000, 15000000)
# numpy_matrix = np.nan_to_num(numpy_matrix)
numpy_matrix = np.log10(numpy_matrix+1)
fig, ax = plt.subplots(figsize=(3, 3))
sns.heatmap(numpy_matrix, ax=ax)
#%%
# side by side heatmap of the hic matrix and the distance matrix
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_sample(region_motif_dataset, sample_index):
    # Calculate distance matrix
    peak_coord_mean = region_motif_dataset[sample_index]['peak_coord'].mean(1)
    peak_coord_length = region_motif_dataset[sample_index]['peak_coord'][:, 1] - region_motif_dataset[sample_index]['peak_coord'][:, 0] 
    peak_coord_length = peak_coord_length / 100
    peak_coord_mean_col = peak_coord_mean.reshape(-1, 1)
    peak_coord_mean_row = peak_coord_mean.reshape(1, -1)
    distance = np.log10(np.abs((peak_coord_mean_col - peak_coord_mean_row)) + 1)

    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 4]}, sharex='col')

    # ATPM line plot for HIC Matrix
    atpm = region_motif_dataset[sample_index]['region_motif'][:, 16] / peak_coord_length.flatten()
    axs[0, 0].plot(atpm)
    axs[0, 0].set_title('ATPM')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_xlim(0, len(atpm) - 1)

    # HIC Matrix heatmap
    hic = region_motif_dataset[sample_index]['hic_matrix']
    sns.heatmap(hic, ax=axs[1, 0], cbar_ax=axs[1, 0].inset_axes([1.05, 0.2, 0.05, 0.6]), vmin=0, vmax=2, cmap='RdBu_r')
    axs[1, 0].set_title('HIC Matrix')

    # ATPM line plot for Distance Matrix
    axs[0, 1].plot(atpm)
    axs[0, 1].set_title('ATAC')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_xlim(0, len(atpm) - 1)

    # Distance Matrix heatmap
    sns.heatmap(distance, ax=axs[1, 1], cbar_ax=axs[1, 1].inset_axes([1.05, 0.2, 0.05, 0.6]))
    axs[1, 1].set_title('Distance Matrix')

    # Remove x-axis labels from top plots
    axs[0, 0].set_xlabel('')
    axs[0, 1].set_xlabel('')

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f'Sample {sample_index}', fontsize=16)
    plt.show()

# Example usage:
visualize_sample(region_motif_dataset, 50)
# %%
from get_model.config.config import load_config, pretty_print_config

from get_model.run_region import run_zarr as run

cfg = load_config('h1esc_hic_region_zarr_cnnk1_observe', './')
pretty_print_config(cfg)
#%%
cfg.stage = 'validate'
cfg.machine.batch_size=1
cfg.dataset.leave_out_chromosomes = 'chr11'
cfg.dataset.hic_method='observed'
cfg.dataset.normalize = True
# cfg.finetune.checkpoint = '/home/xf2217/output/h1esc_hic_region_zarr/debug_oe/checkpoints/last-v5.ckpt'
cfg.finetune.checkpoint = '/home/xf2217/output/h1esc_hic_region_zarr/cnnk3_observe_lr0.00015_cosine/checkpoints/last.ckpt'
# cfg.finetune.resume_ckpt = None
cfg.finetune.strict=True
cfg.finetune.use_lora=False
cfg.run.use_wandb = False
cfg.finetune.rename_config = {'model.': '', 'hic_header': 'head_hic'}
trainer = run(cfg)
#%%

#%%
import torch

def count_effective_params(model, threshold=0.05):
    total_params = 0
    effective_params = 0
    lora_params = []
    for name,param in model.named_parameters():
        total_params += param.numel()
        effective_params += (abs(param) > threshold).sum().item()
        if 'lora' in name:
            lora_params.append(param.numel())
    return total_params, effective_params, lora_params

# Usage
total, effective, lora_params = count_effective_params(trainer.model)
print(f"Total parameters: {total}")
print(f"Effective parameters: {effective}")
print(f"Lora parameters: {sum(lora_params)}")
# collect all parameters and plot histogram
import matplotlib.pyplot as plt
parameters_arr  = []
for param in trainer.model.parameters():
    parameters_arr.append(param.flatten().detach().cpu().numpy())
parameters_arr = np.concatenate(parameters_arr)
plt.hist(np.log(parameters_arr+1), bins=1000)
plt.show()
#%%
# replace parameters that abs < 0.1 with 0
for param in trainer.model.parameters():
    param.data[torch.abs(param.data) < 0.05] = torch.tensor(0.)

#%%
w = trainer.model.model.region_embed.embed.weight.data.cpu().numpy()
normalizing_factor = region_motif_dataset.region_motifs['H1ESC.4dn_h1esc.4dn_h1esc'].normalizing_factor
normalizing_factor = np.concatenate((normalizing_factor, [1, 1, 1]))
normalizing_factor.shape
# %%
w / normalizing_factor

#%%
trainer.model.model.region_embed.embed.weight.data = torch.tensor(w / normalizing_factor, dtype=torch.float32)

# %%
import torch
for i,batch in enumerate(trainer.val_dataloaders):
    if i == 37:
        print(batch['hic_matrix'].shape)
        print(batch['region_motif'].shape)
        # batch['region_motif'][:, 220:250, 282] = batch['region_motif'][:, 220:250, 282] * 0.01
        # batch['region_motif'][:, 220:250, 17:282] = batch['region_motif'][:, 220:250, 17:282] * 0.01
        trainer.model.model.to('cpu')
        input = trainer.model.model.get_input(batch)
        # input['distance_1d'][:,1:] = input['distance_1d'][:, 1:]
        # input['distance_1d'][:, 0:200] = input['distance_1d'][:, 0:200]
        # reverse input['distance_1d'][:, 200:]
        # input['distance_1d'][:, 200] = input['distance_1d'][:, 200] 
        # input['peak_length'][:, 200:300] = torch.flip(input['peak_length'][:, 200:300], dims=[1])
        # input['distance_1d'][:, 200:] = input['distance_1d'][:, 200:]+1
        pred = trainer.model(input)
        print(pred.shape)
        a = batch['hic_matrix'][0].cpu().numpy()
        b = pred[0].detach().squeeze().cpu().numpy()
        c = batch['distance_map'][0][0].squeeze().cpu().numpy()
        print(c.shape)
        break

# side by side
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, figsize=(8, 3), gridspec_kw={'height_ratios': [7, 1]})
axs = axs.flatten()
# top row is the hic matrix, the predicted hic matrix, and the distance map
# bottom row is the atpm, ctcf values, and length adjusted ctcf values

mask = (a==0)
mask_eye = np.eye(a.shape[0], dtype=bool)
mask_boundary = np.zeros_like(a, dtype=bool)
mask_boundary[0:10, :] = True
mask_boundary[:, 0:10] = True
mask_boundary[-10:, :] = True
mask_boundary[:, -10:] = True
mask = mask_boundary#|mask_eye|mask
a[mask] = 0
b[mask] = 0
sns.heatmap(a, ax=axs[0], vmin=0, vmax=2, cmap='viridis', cbar=False)
sns.heatmap(b, ax=axs[1], vmin=0, vmax=2, cmap='viridis', cbar=False)
sns.heatmap(c, ax=axs[2], vmin=3, vmax=6, cmap='viridis_r', cbar=False)
axs[0].set_title('Hi-C Matrix')
axs[1].set_title('Predicted HIC Matrix')
axs[2].set_title('Distance Map')
# set yticks of axs[0] to be the peak_coord
c_max = batch['peak_coord'].flatten().max()
c_min = batch['peak_coord'].flatten().min()
axs[0].set_yticks([0, 400])
axs[0].set_yticklabels([c_min.item(), c_max.item()])
axs[0].set_xticks([])
axs[0].set_ylabel(f'{batch["chromosome"][0]}')
axs[1].set_xticks([])
axs[2].set_xticks([])
axs[1].set_yticks([])
axs[2].set_yticks([])
# add atpm on bottom of heatmap as line plot
atpm = batch['region_motif'][0][:, 282].cpu().numpy().flatten()
ctcf = batch['region_motif'][0][:, 16].cpu().numpy().flatten()
def get_jacobian(model, batch):
    input = trainer.model.model.get_input(batch)
    from torch.autograd import grad
    input['region_motif'].requires_grad = True
    output = model(**input)[0]
    (output[~mask].detach()+1-output[~mask]).sum().backward(retain_graph=True)
    jacobian = input['region_motif'].grad
    return jacobian

jacobian = get_jacobian(trainer.model.model, batch)
jacobian = jacobian[0].detach().cpu().numpy()
axs[3].plot(ctcf)
axs[4].plot(np.abs(jacobian[:, 16]))
axs[5].plot(np.abs(jacobian).sum(1))
# make sure the x-axis is the same for all plots and they are all aligned
for ax in axs:
    ax.set_xticks([]) 
axs[3].set_xlabel('CTCF Motif')
axs[4].set_xlabel('CTCF Gradient')
axs[5].set_xlabel('Overall Gradient')
# xlim
axs[3].set_xlim(0, len(atpm) - 1)
axs[4].set_xlim(0, len(ctcf) - 1)
axs[5].set_xlim(0, len(ctcf) - 1)
# remove y-axis labels and ticks
for ax in axs[3:]:
    ax.set_yticklabels([])
    ax.set_yticks([])
plt.show()











#%%
def get_jacobian(model, batch):
    input = trainer.model.model.get_input(batch)
    recursive_cuda(input)
    from torch.autograd import grad
    input['region_motif'].requires_grad = True
    output = model(**input)[0]
    mask = batch['hic_matrix']==0
    mask_boundary = np.zeros_like(mask, dtype=bool)
    mask_boundary[0:10, :] = True
    mask_boundary[:, 0:10] = True
    mask_boundary[-10:, :] = True
    mask_boundary[:, -10:] = True   
    mask = mask | mask_boundary
    mask = mask.squeeze()
    (output[~mask].detach()+1-output[~mask]).sum().backward(retain_graph=True)
    jacobian = input['region_motif'].grad
    return jacobian

def get_jacobian_with_mask(model, batch, mask):
    input = trainer.model.model.get_input(batch)
    recursive_cuda(input)
    from torch.autograd import grad
    input['region_motif'].requires_grad = True
    output = model(**input)[0]
    (output[mask].detach()+1-output[mask]).sum().backward(retain_graph=True)
    jacobian = input['region_motif'].grad
    return jacobian.detach().cpu().numpy()

def get_full_jacobian(model, batch):
    """Get jacobian with mask for each element"""
    jacobians = []
    for k in range(batch['hic_matrix'].shape[0]):
        mask = batch['hic_matrix'][k]==0
        for i in tqdm(range(10, batch['hic_matrix'][k].shape[0]-10)):
            for j in tqdm(range(10, batch['hic_matrix'][k].shape[1]-10)):
                mask[i, j] = True
                mask[j, i] = True
                mask = mask.squeeze()
                jac = get_jacobian_with_mask(model, batch, mask)
                jacobians.append(jac)
                mask[i, j] = False
                mask[j, i] = False
        jacobians = np.concatenate(jacobians)
        jacobians = jacobians.reshape(batch['hic_matrix'].shape[0]-20, batch['hic_matrix'].shape[1]-20, 400, 283)
        return jacobians

preds = []
obs = []
distance = []
jacobians = []
from typing import Dict
def recursive_cuda(dict):
    for key in dict:
        if isinstance(dict[key], Dict):
            recursive_cuda(dict[key])
        else:
            dict[key] = dict[key].cuda()
from tqdm import tqdm
for i,batch in tqdm(enumerate(trainer.val_dataloaders)):
    if i>1:
        break
    trainer.model.model.to('cuda')
    input = trainer.model.model.get_input(batch)
    recursive_cuda(input)
    jacobian = get_full_jacobian(trainer.model.model, batch)
    # jacobian = jacobian
    jacobians.append(jacobian)
    pred = trainer.model(input)
    a = batch['hic_matrix'][0].cpu().numpy()
    b = pred[0].detach().squeeze().cpu().numpy()
    c = batch['distance_map'][0][0].squeeze().cpu().numpy()
    c = 10**c-1
    distance.append(c[10:-10, 10:-10])
    obs.append(a[10:-10, 10:-10])
    preds.append(b[10:-10, 10:-10])
#%%
jacobian_contact = jacobian[0,:,10:-10,16].reshape(-1,380)-jacobian[0,:,10:-10,16].reshape(-1,380).mean(0)
# plot jacobian_contact as heatmap with a in two panel
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
sns.heatmap(jacobian_contact+jacobian_contact.T, ax=ax[0], cmap='RdBu', vmin=-500, vmax=500)
sns.heatmap(a[10:-10, 10:-10], ax=ax[1], cmap='viridis')
plt.show()

#%%
# distance stratified pred-obs correlation
distance = np.concatenate(distance)
obs = np.concatenate(obs)
preds = np.concatenate(preds)
jacobians = np.concatenate(jacobians)
jacobians = jacobians.reshape(-1, 400, 283)
#%%
bins = np.linspace(1000, 3000000, 30)
# compute correlation for each bin
correlations = []
from tqdm import tqdm
for i, bin in tqdm(enumerate(bins)):
    if i<len(bins)-1:
        mask = (distance > bin) & (distance < bins[i+1])
        mask_boundary = np.zeros_like(mask, dtype=bool)
        mask_boundary[0:10, :] = True
        mask_boundary[:, 0:10] = True
        mask_boundary[-10:, :] = True
        mask_boundary[:, -10:] = True   
        mask = mask & (obs>0) & ~mask_boundary
        print(mask.sum())
        correlations.append(np.corrcoef(obs[mask].flatten(), preds[mask].flatten())[0, 1])
correlations = np.array(correlations)
# plot the correlation as a function of the distance
#%%
fig, ax = plt.subplots(figsize=(4, 2.5))
plt.plot(bins[1:], correlations)
plt.xlabel('Distance')
plt.ylabel('Pearson Correlation')
plt.xticks([0, 1000000, 2000000, 3000000], ['0', '1M', '2M', '3M'])
plt.show()
# %%
# scatter plot of the predicted hic matrix and the observed hic matrix
plt.scatter(obs[mask].flatten(), preds[mask].flatten(),s=0.5)
plt.show()
# compute the pearson correlation coefficient
from scipy.stats import pearsonr
pearsonr(obs[mask].flatten(), preds[mask].flatten())
#%%
# plot heatmap of the prediction at top, and atpm, ctcf, length_adjusted ctcf, and overall gradient as line plot at bottom
fig, axs = plt.subplots(5, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [12, 1, 1, 1, 1]})
axs = axs.flatten()
sns.heatmap(b, ax=axs[0], vmin=0, vmax=2, cmap='viridis', cbar=False)
axs[0].set_title('Predicted Hi-C Matrix')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[1].plot(atpm)
peak_coord_length = batch['peak_coord'][0][:, 1] - batch['peak_coord'][0][:,0]
peak_coord_length = peak_coord_length.cpu().numpy() / 100
axs[2].plot(ctcf)
axs[3].plot(np.abs(jacobian[:, 16]))
axs[4].plot(np.abs(jacobian).sum(1))
axs[1].set_ylabel('ATAC', fontsize=12, rotation=0, ha='right')
axs[2].set_ylabel('CTCF', fontsize=12, rotation=0, ha='right')
axs[3].set_ylabel('CTCF Gradient', fontsize=12, rotation=0, ha='right')
axs[4].set_ylabel('Overall Gradient', fontsize=12, rotation=0, ha='right')
axs[1].set_xlim(0, len(atpm) - 1)
axs[2].set_xlim(0, len(ctcf) - 1)
axs[3].set_xlim(0, len(ctcf) - 1)
axs[4].set_xlim(0, len(ctcf) - 1)
# remove y-axis labels and ticks
for ax in axs[1:]:
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel('')

plt.show()
#%%
def plot_hic_and_features(b, batch, jacobian, additional_gradients=None):
    num_plots = 5 + (len(additional_gradients) if additional_gradients else 0)
    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 9), 
                            gridspec_kw={'height_ratios': [12] + [1] * (num_plots - 1)})
    axs = axs.flatten()
    # jacobian = jacobian/l2 norm
    jacobian_norm = jacobian/np.linalg.norm(jacobian, axis=0)
    # jacobian_norm = jacobian_norm * batch['region_motif'].detach().cpu().numpy()[0]
    # Plot heatmap
    sns.heatmap(b, ax=axs[0], vmin=0, vmax=3, cmap='viridis', cbar=False)
    axs[0].set_title('Predicted Hi-C Matrix')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    atpm = batch['region_motif'][0][:, 282].cpu().numpy().flatten()
    ctcf = batch['region_motif'][0][:, 16].cpu().numpy().flatten()
    # Plot features
    features = [
        ('ATAC', atpm),
        ('CTCF Motif', ctcf),
        ('CTCF Gradient', np.abs(jacobian_norm[:, 16])),
        ('Overall Gradient', np.abs(jacobian_norm).sum(1))
    ]

    # Add additional gradients
    if additional_gradients:
        for name, grad in additional_gradients.items():
            if isinstance(grad, int):
                motif = batch['region_motif'][0][:, grad].cpu().numpy().flatten()
                grad = np.abs(jacobian_norm[:, grad])
            features.append((name, grad))
            # features.append((name.replace('Gradient', 'Motif'), motif))

    for i, (name, data) in enumerate(features, start=1):
        axs[i].plot(data)
        axs[i].set_ylabel(name, fontsize=12, rotation=0, ha='right')
        axs[i].set_xlim(0, len(data) - 1)
        axs[i].set_yticklabels([])
        axs[i].set_yticks([])
        # axs[i].set_ylim(0, np.abs(jacobian_norm).sum(1).max())
        axs[i].set_xticks([])
        axs[i].set_xlabel('')

    plt.tight_layout()
    plt.show()

#%%
import pandas as pd
motif_clusters = np.loadtxt('/home/xf2217/Projects/geneformer_esc/data/motif_cluster.txt', dtype=str)
# most important features (to the right)
motif_clusters[np.argsort(np.absolute(jacobian).mean(0))[-10:]]
# find top 20 region with largest overall gradient
jacobian_norm = jacobian * batch['region_motif'].detach().cpu().numpy()[0]
jacobian_df = pd.DataFrame(jacobian_norm, columns=motif_clusters).abs()
jacobian_df['overall'] = jacobian_df.mean(1)
jacobian_df['overall_but_ctcf_atac'] = jacobian_df.drop(columns=['CTCF', 'Accessibility']).mean(1)
jacobian_df = jacobian_df / jacobian_df.max(0)
plot_hic_and_features(b, batch, jacobian, {'POU Gradient': int(np.where(motif_clusters=="POU/1")[0]),
                                                'YY1 Gradient': int(np.where(motif_clusters=="YY1")[0])})
# %%

#%%
sns.scatterplot(data=jacobian_df, x='CTCF', y='overall')

jacobian_df.query('overall>CTCF*2 & overall>0.6').mean(0).sort_values().tail(10)
# %%
import numpy as np
import pandas as pd
# norm of each column
jacobian_norm = jacobians.reshape(-1, 283).mean(0)
# %%
jacobian_norm_df = pd.Series(jacobian_norm, index=motif_clusters)
# %%
jacobian_norm_df.sort_values().head(10)
# %%
# scatterplot of range(0, 283) and jacobian_norm
fig, ax = plt.subplots(figsize=(4, 2))
plt.scatter(range(283), jacobian_norm_df.sort_values())
for i, txt in enumerate(jacobian_norm_df.sort_values().tail(3).index):
    ax.text(i, jacobian_norm_df.sort_values()[txt], txt)
plt.xlabel('Rank Sorted Motifs')
plt.ylabel('Absolute Gradient')
plt.show()
# %%
jacobians.shape
# %%
sns.clustermap(np.absolute(jacobians).mean(1), cmap='viridis')
# %%
def get_jacobian(model, batch):
    all_outputs = []
    input = trainer.model.model.get_input(batch)
    recursive_cuda(input)   
    output = model(**input)[0]
    mask = batch['hic_matrix']==0
    mask_boundary = np.zeros_like(mask, dtype=bool)
    mask_boundary[0:10, :] = True
    mask_boundary[:, 0:10] = True
    mask_boundary[-10:, :] = True
    mask_boundary[:, -10:] = True   
    mask = mask | mask_boundary
    mask = mask.squeeze()
    output = output.detach().cpu().numpy()
    output[mask] = 0
    for i in tqdm(range(283)):

        input = trainer.model.model.get_input(batch)
        recursive_cuda(input)
        from torch.autograd import grad
        # input['region_motif'].requires_grad = True
        input['region_motif'][:, :, i] +=0.5
        output_alt = model(**input)[0]
        output_alt = output_alt.detach().cpu().numpy()
        output_alt[mask] = 0
        output_alt = output_alt-output
        all_outputs.append(output_alt)
    return np.array(all_outputs)
# %%
grad = get_jacobian(trainer.model.model.cuda(), batch)
# %%
grad.shape
# %%
sns.heatmap(grad[11], vmin=-1, vmax=1, cmap='coolwarm')

# %%
trainer.model.model.region_embed.embed.weight.data
#%%
w = trainer.model.model.region_embed.embed.weight.data.cpu().numpy()
# plot norm of each column vs rank
norm = np.absolute(w).sum(0)
plt.scatter(np.arange(len(norm)), norm[np.argsort(norm)])
for i, txt in enumerate(np.array(list(motif_clusters)+['peak_length', 'distance'])[np.argsort(norm)]):
    if i>280:
        plt.text(i, norm[np.argsort(norm)][i], txt)
plt.show()
#%%
# keep top 20 column 
w = pd.DataFrame(w, columns=list(motif_clusters)+['peak_length', 'distance'])
# w = w.iloc[:, np.argsort(np.absolute(w).sum(0))[-20:]]
#%%
sns.clustermap(w,  method='ward',cmap='RdBu', figsize=(10, 10), z_score=0)
# %%
w.mean(0)
# %%
sns.scatterplot(x=w.mean(0),y=np.absolute(w).max(0))
plt.xlabel('Gradient Mean')
plt.ylabel('Gradient Abs Max')
plt.show()
# %%
# %%
