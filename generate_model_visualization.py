#%%
# a script to generate model visualization by reading the yaml config in get_model/config/model

from glob import glob
from get_model.config.config import load_config
#%%
model_configs = {}
for config_path in glob('get_model/config/model/*.yaml'):
    config_name = config_path.split('/')[-1].split('.')[0]
    try:
        c = load_config(config_name, 'model/')
        model_configs[c['model']['_target_'].split('.')[-1]] = c['model']['cfg']
    except Exception as e:        
        print(f'Error reading {config_path}: {e}')

# %%
model_configs
import omegaconf
# for each model, get the first level keys that has a dictionary value into a list
first_level_keys = []
for model_name, model_config in model_configs.items():
    for key, value in model_config.items():
        print(key, value)
        if isinstance(value, omegaconf.dictconfig.DictConfig):
            first_level_keys.append(key)

import numpy as np
count = np.unique(first_level_keys, return_counts=True)
count = np.vstack((count[0], count[1])).T
import pandas as pd
count = pd.DataFrame(count, columns=['key', 'count'])
count['count'] = count['count'].astype(int)
count = count.sort_values('count', ascending=False).query('~key.isin(["loss", "metrics", "mask_token"])')
# %%
module_list = count['key'].tolist()
# %%
adj_list = []
for model_name, model_config in model_configs.items():
    for key, value in model_config.items():
        if key in module_list:
            adj_list.append((model_name, key))
# %%
# to csv
import pandas as pd

df = pd.DataFrame(adj_list, columns=['model', 'component'])
df.to_csv('model_component.csv', index=False)
# %%
# for each model in get_model/model/model.py initialize with default config 
from get_model.model.model import *



# %%
info_json = {}
from tqdm import tqdm
for model_name, model_config in tqdm(model_configs.items()):
    model_class = globals()[model_name]
    model = model_class(model_config)
    info_json[model_name] = {}
    info_json[model_name]['name'] = model_name
    info_json[model_name]['doc'] = model.__doc__
    info_json[model_name]['config'] = omegaconf.OmegaConf.to_container(model_config)
    info_json[model_name]['layers'] = {}
# get class names for each layer
    for layer_name in model.get_layer_names():
        info_json[model_name][layer_name] = {}
        layer = model.get_layer(layer_name)
        info_json[model_name][layer_name]['name'] = layer.__class__.__name__
        info_json[model_name][layer_name]['doc'] = layer.__class__.__doc__
        info_json[model_name][layer_name]['config'] = omegaconf.OmegaConf.to_container(model_config[layer_name]) if layer_name in model_config else ''
# %%
# save to json
import json
with open('model_info.json', 'w') as f:
    json.dump(info_json, f, indent=4)
# %%
