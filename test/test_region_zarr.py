#%%
from get_model.dataset.zarr_dataset import RegionMotifDataset
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
sns.heatmap(region_motif_dataset[2]['hic_matrix'])
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
    sns.heatmap(10**hic-1, ax=axs[1, 0], cbar_ax=axs[1, 0].inset_axes([1.05, 0.2, 0.05, 0.6]), vmin=0, vmax=2, cmap='RdBu_r')
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

cfg = load_config('h1esc_hic_region_zarr')
pretty_print_config(cfg)
#%%
cfg.stage = 'validate'
cfg.machine.batch_size=2
cfg.dataset.leave_out_chromosomes = 'chr11'
cfg.finetune.resume_ckpt = None #'/home/xf2217/output/h1esc_hic_region_zarr/debug_observed_larger_lr_adamw/checkpoints/last-v3.ckpt'
cfg.finetune.strict=False
cfg.run.use_wandb = False
cfg.finetune.rename_config = {'model.': '', 'hic_header': 'head_hic', 'proj_distance': 'proj_distance_removed'}
trainer = run(cfg)



# %%

for i,batch in enumerate(trainer.val_dataloaders):
    if i == 78:
        print(batch['hic_matrix'].shape)
        print(batch['region_motif'].shape)
        trainer.model.model.to('cpu')
        pred = trainer.model(trainer.model.model.get_input(batch))
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
mask = mask_eye | mask_boundary
# a[mask] = 0
# b[mask] = 0
sns.heatmap(a, ax=axs[0], vmin=0, vmax=3, cmap='viridis', cbar=False)
sns.heatmap(b, ax=axs[1], vmin=0, vmax=3, cmap='viridis', cbar=False)
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
# %%
# scatter plot of the predicted hic matrix and the observed hic matrix
plt.scatter(a[~mask].flatten(), b[~mask].flatten(),s=0.5)
plt.show()
#%%
# plot heatmap of the prediction at top, and atpm, ctcf, length_adjusted ctcf, and overall gradient as line plot at bottom
fig, axs = plt.subplots(5, 1, figsize=(4, 6), gridspec_kw={'height_ratios': [12, 1, 1, 1, 1]})
axs = axs.flatten()
sns.heatmap(b, ax=axs[0], vmin=0, vmax=3, cmap='viridis', cbar=False)
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



# %%
motif_clusters = np.loadtxt('/home/xf2217/Projects/geneformer_esc/data/motif_cluster.txt', dtype=str)
# most important features (to the right)
motif_clusters[np.argsort(np.absolute(jacobian).mean(0))[-10:]]
#%%
# a function to convert a 2d contact map to a 3d point cloud
import numpy as np
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize

def normalize_contact_map(contact_map, max_distance=20.0, min_distance=3.8):
    """
    Normalize the contact frequency map to a distance map.
    
    Args:
    contact_map (np.array): 2D array of contact frequencies
    max_distance (float): Maximum distance for non-contacting residues
    min_distance (float): Minimum distance for contacting residues
    
    Returns:
    np.array: Normalized distance map
    """
    # Avoid division by zero
    max_freq = np.max(contact_map)
    normalized_map = 1 - (contact_map / max_freq)
    
    # Scale to distance range
    distance_map = normalized_map * (max_distance - min_distance) + min_distance
    
    return distance_map

def contact_map_to_point_cloud(contact_map, max_distance=20.0, min_distance=3.8, num_steps=1000):
    num_residues = contact_map.shape[0]
    
    # Normalize contact map to distance map
    distance_map = normalize_contact_map(contact_map, max_distance, min_distance)
    
    # Initialize random 3D coordinates
    initial_coords = np.random.rand(num_residues, 3) * max_distance
    
    def loss_function(coords):
        coords = coords.reshape(num_residues, 3)
        current_distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        
        # Calculate loss based on the difference between current distances and target distances
        loss = np.sum((current_distances - distance_map) ** 2)
        
        return loss
    
    # Optimize the coordinates add progress bar
    from tqdm import tqdm
    result = minimize(loss_function, initial_coords.flatten(), method='L-BFGS-B', options={'maxiter': num_steps}, callback=lambda x: tqdm(x, desc='Optimizing coordinates'))
    
    # Reshape the result back to (num_residues, 3)
    final_coords = result.x.reshape(num_residues, 3)
    
    return final_coords

# Function to apply random rotation (for rotation equivariance)
def random_rotation(coords):
    rotation_matrix = np.random.rand(3, 3)
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)
    return np.dot(coords, rotation_matrix)

# Example usage
contact_freq_map = a[10:-10, 10:-10]
point_cloud = contact_map_to_point_cloud(contact_freq_map, min_distance=1, max_distance=20, num_steps=5000)

# Apply random rotation for rotation equivariance
rotated_point_cloud = random_rotation(point_cloud)
# %%
rotated_point_cloud.shape
# %%
# plot the point cloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# use plotly to plot the point cloud
import plotly.graph_objects as go

fig = go.Figure()
# color by atpm, transparency by atpm
fig.add_trace(go.Scatter3d(x=rotated_point_cloud[:, 0], y=rotated_point_cloud[:, 1], z=rotated_point_cloud[:, 2], mode='markers', marker=dict(size=3, color=atpm[10:-10], colorscale='Viridis', opacity=0.5)))
# remove axis pane
fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
#%%
# compute the distance matrix of the point cloud
distance_matrix = np.linalg.norm(rotated_point_cloud[:, np.newaxis] - rotated_point_cloud, axis=2)
# plot the distance matrix as a heatmap
plt.imshow(a[10:-10, 10:-10], cmap='viridis')
plt.colorbar()
plt.show()
# %%