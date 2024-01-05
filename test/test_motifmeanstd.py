#%%
from get_model.dataset.zarr_dataset import MotifMeanStd

# %%
mms = MotifMeanStd('/pmglocal/xf2217/get_data/hg38.zarr')
# %%
mms.data_dict['chr1'][0:2].shape
# %%
