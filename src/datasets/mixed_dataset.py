import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import h5py

import torch
from torch.utils.data import Dataset


# My libraries
import datasets.cleargrasp_synthetic_dataset as cleargrasp_syn
import datasets.cleargrasp_dataset as cleargrasp
import datasets.omniverse_dataset as omniverse
import constants

class MixedDataset(Dataset):
    def __init__(self, cleargrasp_root_dir, omniverse_root_dir, params, exp_type):
        self.params = params
        self.exp_type = exp_type
        self.cleargrasp_syn_dataset = cleargrasp_syn.get_dataset(cleargrasp_root_dir, self.params, exp_type=self.exp_type)
        self.omniverse_dataset = omniverse.get_dataset(omniverse_root_dir, self.params, exp_type=self.exp_type)
        self.cleargrasp_syn_len = self.cleargrasp_syn_dataset.__len__()
        self.omniverse_len = self.omniverse_dataset.__len__()

    def __getitem__(self, idx):
        if idx < self.cleargrasp_syn_len:
            return self.cleargrasp_syn_dataset.__getitem__(idx)
        else:
            return self.omniverse_dataset.__getitem__(idx-self.cleargrasp_syn_len)


    def __len__(self):
        return self.cleargrasp_syn_len + self.omniverse_len


def get_dataset(cleargrasp_root_dir, omniverse_root_dir, params, exp_type):
    # set params
    params = params.copy()
    if exp_type != 'train':
        params['use_data_augmentation'] = False

    dataset = MixedDataset(cleargrasp_root_dir, omniverse_root_dir, params, exp_type)
    return dataset
