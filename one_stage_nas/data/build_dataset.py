import numpy as np
import torch
import h5py
import random
import os
import matplotlib.pyplot as plt
from .HSI_dataset import HSI_dataset

def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        height, width = f['height'][0], f['width'][0]
        category_num = f['category_num'][0]
        train_map, val_map, test_map = f['train_label_map'][0], f['val_label_map'][0], f['test_label_map'][0]

    return height, width, category_num, train_map, val_map, test_map


def get_patches_list(height, width, crop_size, label_map, patches_num=1000, shuffle=True):
    patch_list = []
    count=0
    if shuffle:
        while count<patches_num:
            x1 = random.randint(0, width-crop_size-1)
            x2 = x1 + crop_size
            y1 = random.randint(0, height-crop_size-1)
            y2 = y1 + crop_size
            if label_map[y1:y2, x1:x2].max()>0:
                patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                patch_list.append(patch)
                count+=1
    else:
        slide_step = crop_size
        x1_list = list(range(0, width-crop_size, slide_step))
        y1_list = list(range(0, height-crop_size, slide_step))
        x1_list.append(width-crop_size)
        y1_list.append(height-crop_size)

        x2_list = [x+crop_size for x in x1_list]
        y2_list = [y+crop_size for y in y1_list]

        for x1, x2 in zip(x1_list, x2_list):
            for y1, y2 in zip(y1_list, y2_list):
                if label_map[y1:y2, x1:x2].max()>0:
                    patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                    patch_list.append(patch)

    return patch_list


def build_dataset(cfg):
    data_root = cfg.DATASET.DATA_ROOT
    data_set = cfg.DATASET.DATA_SET
    crop_size = cfg.DATASET.CROP_SIZE

    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN

    search_on = cfg.SEARCH.SEARCH_ON

    dist_dir = os.path.join(data_list_dir, '{}_dist_{}_train-{}_val-{}.h5'.
                            format(data_set,
                                   cfg.DATASET.DIST_MODE,
                                   float(cfg.DATASET.TRAIN_NUM),
                                   float(cfg.DATASET.VAL_NUM)))

    height, width, category_num, train_map, val_map, test_map = h5_dist_loader(dist_dir)

    if search_on:
        w_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        a_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        v_data_list = get_patches_list(height, width, crop_size, val_map, shuffle=False)

        dataset_w = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=w_data_list, mode='train')
        dataset_a = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=a_data_list, mode='train')
        dataset_v = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=v_data_list, mode='val')

        data_loader_w = torch.utils.data.DataLoader(
            dataset_w,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        data_loader_a = torch.utils.data.DataLoader(
            dataset_a,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        data_loader_v = torch.utils.data.DataLoader(
            dataset_v,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        return [data_loader_w, data_loader_a], data_loader_v
    else:
        tr_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM, shuffle=True)
        v_data_list = get_patches_list(height, width, crop_size, val_map, shuffle=False)
        te_data_list = get_patches_list(height, width, crop_size, test_map, shuffle=False)

        dataset_tr = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=tr_data_list, mode='train', aug=True, rand_crop=True, crop_size=crop_size)
        dataset_v = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=v_data_list, mode='val')
        dataset_te = HSI_dataset(HSI_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                 dist_h5_dir=dist_dir,
                                 data_dict=te_data_list, mode='test')


        data_loader_tr = torch.utils.data.DataLoader(
            dataset_tr,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        data_loader_v = torch.utils.data.DataLoader(
            dataset_v,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        data_loader_te = torch.utils.data.DataLoader(
            dataset_te,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)

        return data_loader_tr, data_loader_te

