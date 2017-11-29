# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:35:08 2017

@author: gshai
"""

from os import walk, path, makedirs
import shutil
import numpy as np

#%%

path_train = 'dataset/train'

dirpath, dirnames, filenames = next(walk(path_train))

plant_names = dirnames.copy()

plants_dict = dict()

for name in plant_names:
    directory = path_train + '/' + name
    dirpath, dirnames, filenames = next(walk(directory))
    plants_dict[name] = np.array(filenames)

#%%

plants_dict_new = dict()

for name, filenames in plants_dict.items():
    size = len(filenames)
    reorder = list(np.random.permutation(size))
    filenames = filenames[reorder]

    size_val = np.floor(size * 0.15).astype('int')
    filenames_train = filenames[:-size_val]
    filenames_val = filenames[-size_val:]

    plants_dict_new[name] = {'train': filenames_train, 'val': filenames_val}

    assert len(filenames_train) + len(filenames_val) == len(filenames)

#%%

path_val = 'dataset/val'

for name in plant_names:
    directory = path_val + '/' + name
    if not path.exists(directory):
        makedirs(directory)

#%%

for name, set_dict in plants_dict_new.items():
    folder_train = path_train + '/' + name
    folder_val = path_val + '/' + name
    for filename in set_dict['val']:
        shutil.move(folder_train + '/' + filename, folder_val)
