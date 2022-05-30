import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset


def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)


# This class is build for loading different datasets in denoise tasks
class Dn_datasets(Dataset):
    def __init__(self, data_root, data_dict, transform, load_all=False, to_gray=False, s_factor=1, repeat_crop=1):
        self.data_root = data_root
        self.transform = transform
        self.load_all = load_all
        self.to_gray = to_gray
        self.repeat_crop = repeat_crop
        if self.load_all is False:
            self.data_dict = data_dict
        else:
            self.data_dict = []
            for sample_info in data_dict:
                sample_data = Image.open('/'.join((self.data_root, sample_info['path']))).copy()
                if sample_data.mode in ['RGBA']:
                    sample_data = sample_data.convert('RGB')
                width = sample_info['width']
                height = sample_info['height']
                sample = {
                    'data': sample_data,
                    'width': width,
                    'height': height
                }
                self.data_dict.append(sample)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        sample_info = self.data_dict[idx]
        if self.load_all is False:
            sample_data = Image.open('/'.join((self.data_root, sample_info['path'])))
            if sample_data.mode in ['RGBA']:
                sample_data = sample_data.convert('RGB')
        else:
            sample_data = sample_info['data']

        if self.to_gray:
            sample_data = sample_data.convert('L')

        # crop (w_start, h_start, w_end, h_end)
        image = sample_data
        target = sample_data

        sample = {'image': image, 'target': target}

        if self.repeat_crop != 1:
            image_stacks = []
            target_stacks = []

            for i in range(self.repeat_crop):
                sample_patch = self.transform(sample)
                image_stacks.append(sample_patch['image'])
                target_stacks.append(sample_patch['target'])
            return torch.stack(image_stacks), torch.stack(target_stacks)

        else:
            sample = self.transform(sample)
            return sample['image'], sample['target']



