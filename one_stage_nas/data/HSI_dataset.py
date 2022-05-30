import os
import json
import h5py
import torch
import random
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt

def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        height, width = f['height'][:], f['width'][:]
        category_num = f['category_num'][:]
        train_map, val_map, test_map = f['train_label_map'][:], f['val_label_map'][:], f['test_label_map'][:]

    return height, width, category_num, train_map, val_map, test_map

class HSI_dataset(Dataset):
    def __init__(self, HSI_h5_dir, dist_h5_dir, data_dict, mode='train', aug=False, rand_crop=False, rand_map=False, crop_size=32):#31 48 96#hu 144
        self.data_dict = data_dict
        with h5py.File(HSI_h5_dir, 'r') as f:
            data = f['data'][:]
        self.data = data / data.max()
        if mode=='train':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['train_label_map'][0]
        elif mode=='val':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['val_label_map'][0]
        elif mode=='test':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['test_label_map'][0]
        self.label_map = label_map
        self.aug = aug
        self.rand_crop=rand_crop
        self.height, self.width = self.label_map.shape
        self.crop_size = crop_size
        self.rand_map = rand_map


    def __getitem__(self, idx):
        if self.rand_map:
            label_map_t = self.get_rand_map(self.label_map)
        else:
            label_map_t = self.label_map
        if self.rand_crop:
            flag=0
            while flag==0:
                x1 = random.randint(0, self.width - self.crop_size - 1)
                x2 = x1 + self.crop_size
                y1 = random.randint(0, self.height - self.crop_size - 1)
                y2 = y1 + self.crop_size
                if label_map_t[y1:y2, x1:x2].max() > 0:
                    flag=1

            input = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]

        else:
            patch_info = self.data_dict[idx]
            x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
            input = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]

        if self.aug:
            input, target = self.random_flip_lr(input, target)
            input, target = self.random_flip_tb(input, target)
            input, target = self.random_rot(input, target)
        return torch.from_numpy(input).float().permute(2, 0, 1).unsqueeze(dim=0), torch.from_numpy(target-1).long()

    def __len__(self):
        return len(self.data_dict)


    def random_flip_lr(self, input, target):
        if np.random.randint(0, 2):
            h, w, d = input.shape
            index = np.arange(w, 0, -1)-1
            return input[:, index, :], target[:, index]
        else:
            return input, target

    def random_flip_tb(self, input, target):
        if np.random.randint(0, 2):
            h, w, d = input.shape
            index = np.arange(h, 0, -1) - 1
            return input[index, :, :], target[index, :]
        else:
            return input, target

    def random_rot(self, input, target):
        rot_k = np.random.randint(0, 4)
        return np.rot90(input, rot_k, (0, 1)).copy(), np.rot90(target, rot_k, (0, 1)).copy()

    def get_rand_map(self, label_map, keep_ratio=0.6):
        label_map_t = label_map
        label_indices = np.where(label_map>0)
        label_num = len(label_indices[0])
        shuffle_indices = np.random.permutation(int(label_num))
        dis_num = int(label_num*(1-keep_ratio))
        dis_indices = (label_indices[0][shuffle_indices[:dis_num]], label_indices[1][shuffle_indices[:dis_num]])
        label_map_t[dis_indices]=0
        return label_map_t


class HSI_dataset_test(Dataset):
    def __init__(self, HSI_data, data_dict):
        self.HSI_data = HSI_data / HSI_data.max()
        self.data_dict = data_dict

    def __getitem__(self, idx):
        patch_info = self.data_dict[idx]
        x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
        input = self.HSI_data[y1:y2, x1:x2]
        return torch.from_numpy(input).float().permute(2, 0, 1).unsqueeze(dim=0), [x1, x2, y1, y2]

    def __len__(self):
        return len(self.data_dict)