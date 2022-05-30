"""
Searching script
"""

import argparse
import os
import h5py
import json
import torch
import sys
import PIL.Image as Image
import numpy as np
sys.path.append('..')
from one_stage_nas.config import cfg
from one_stage_nas.utils.misc import mkdir
from one_stage_nas.modeling.architectures import build_model
from one_stage_nas.data.HSI_dataset import HSI_dataset_test
from sklearn.metrics import cohen_kappa_score
from plot_result.color_dict import color_dict
import matplotlib.pyplot as plt
import time


def h5_data_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]

    return data, label

def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        height, width = f['height'][0], f['width'][0]
        category_num = f['category_num'][0]
        test_map = f['test_label_map'][0]

    return height, width, category_num, test_map


def get_patches_list(height, width, crop_size, overlap=False):
    patch_list = []
    if overlap:
        slide_step = crop_size // 2
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
            patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
            patch_list.append(patch)

    return patch_list


def OA_AA_K_cal(pre_label, tar_label):
    test_indices = np.where(tar_label>0)
    pre_label = pre_label[test_indices]
    tar_label = tar_label[test_indices]
    acc=[]
    samples_num = len(tar_label)
    category_num=tar_label.max()
    for i in range(1, int(category_num)+1):
        loc_i = np.where(tar_label==i)
        OA_i = np.array(pre_label[loc_i]==tar_label[loc_i], np.float32).sum()/len(loc_i[0])
        acc.append(OA_i)

    OA = np.array(pre_label==tar_label, np.float32).sum()/samples_num
    AA = np.average(np.array(acc))
    # c_matrix = confusion_matrix(tar_label, pre_label)
    # K = (samples_num*c_matrix.diagonal().sum())/(samples_num*samples_num - np.dot(sum(c_matrix,0), sum(c_matrix,1)))
    K = cohen_kappa_score(tar_label, pre_label)

    return OA, AA, K, np.array(acc)


def labelmap_2_img(color_list, label_map):
    h, w = label_map.shape
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            R,G,B = color_list[str(label_map[i, j])]
            img[i,j] = [R, G, B]
    return np.array(img, np.uint8)


def evaluation(cfg):

    print('model build')
    trained_model_dir = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'.
                                  format(cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                         cfg.SEARCH.TIE_CELL), 'train', 'models/model_best.pth'))
    if not os.path.exists(trained_model_dir):
        print('trained_model does not exist')
        return None, None
    model = build_model(cfg)
    model = torch.nn.DataParallel(model).cuda()

    model_state_dict = torch.load(trained_model_dir).pop("model")

    try:
        model.load_state_dict(model_state_dict)
    except:
        model.module.load_state_dict(model_state_dict)

    model.eval()

    print('load test set')
    data_root = cfg.DATASET.DATA_ROOT
    data_set = cfg.DATASET.DATA_SET
    batch_size = cfg.DATALOADER.BATCH_SIZE_TEST
    dataset_dir = '/'.join((data_root, '{}.h5'.format(data_set)))
    dataset_dist_dir = '/'.join((cfg.DATALOADER.DATA_LIST_DIR, '{}_dist_{}_train-{}_val-{}.h5'
                                 .format(data_set,cfg.DATASET.DIST_MODE,
                                   float(cfg.DATASET.TRAIN_NUM),
                                   float(cfg.DATASET.VAL_NUM))))
    test_data, label_map = h5_data_loader(dataset_dir)
    height, width, category_num, test_map = h5_dist_loader(dataset_dist_dir)

    result_save_dir = '/'.join((cfg.OUTPUT_DIR, '{}/Outline-{}c{}n_TC-{}'.
                                format(cfg.DATASET.DATA_SET, cfg.MODEL.NUM_LAYERS, cfg.MODEL.NUM_BLOCKS,
                                       cfg.SEARCH.TIE_CELL),
                                'eval_MS[{}]_OV[{}]'.format(cfg.DATASET.MULTI_SCALE, cfg.DATASET.OVERLAP)))
    mkdir(result_save_dir)

    if cfg.DATASET.MULTI_SCALE:
        overlap = cfg.DATASET.OVERLAP
        pred_score_map_c = torch.zeros(category_num, height, width)
        for crop_size in cfg.DATASET.MCROP_SIZE:
            test_patches_list = get_patches_list(height, width, crop_size, overlap)
            dataset_test = HSI_dataset_test(HSI_data=test_data, data_dict=test_patches_list)
            test_loader = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=batch_size,
                                                      pin_memory=True)

            print('dataset {} evaluation...'.format(data_set))
            pred_score_map = torch.zeros(category_num, height, width)
            pred_count_map = torch.zeros(height, width)
            time_sum=0
            count=0
            with torch.no_grad():
                for patches, indieces in test_loader:
                    count+=1
                    time_s = time.time()
                    pred = model(patches)
                    torch.cuda.synchronize()
                    time_e = time.time()
                    time_sum += (time_e-time_s)

                    for i, [x1, x2, y1, y2] in enumerate(zip(indieces[0], indieces[1], indieces[2], indieces[3])):
                        pred_patch = torch.softmax(pred[i].cpu(), dim=0)
                        pred_score_map[:, y1:y2, x1:x2] += pred_patch
                        pred_count_map[y1:y2, x1:x2] += 1
                pred_score_map = pred_score_map / pred_count_map.unsqueeze(dim=0)
            pred_score_map_c += pred_score_map

        pred_map = torch.argmax(pred_score_map_c, dim=0) + 1
        OA, AA, K, acc = OA_AA_K_cal(np.array(pred_map, np.float), test_map)

    else:
        crop_size = cfg.DATASET.CROP_SIZE
        overlap = cfg.DATASET.OVERLAP
        test_patches_list = get_patches_list(height, width, crop_size, overlap)
        dataset_test = HSI_dataset_test(HSI_data=test_data, data_dict=test_patches_list)
        test_loader = torch.utils.data.DataLoader( dataset_test, shuffle=False, batch_size=batch_size, pin_memory=True)

        print('dataset {} evaluation...'.format(data_set))
        pred_score_map = torch.zeros(category_num, height, width)
        pred_count_map = torch.zeros(height, width)
        time_sum = 0
        count = 0
        with torch.no_grad():
            for patches, indieces in test_loader:
                count += 1
                time_s = time.time()
                pred = model(patches)
                torch.cuda.synchronize()
                time_e = time.time()
                time_sum += (time_e - time_s)
                for i, [x1, x2, y1, y2] in enumerate(zip(indieces[0], indieces[1], indieces[2], indieces[3])):
                    pred_patch = torch.softmax(pred[i].cpu(), dim=0)
                    pred_score_map[:, y1:y2, x1:x2] += pred_patch
                    pred_count_map[y1:y2, x1:x2] += 1
            pred_score_map = pred_score_map/pred_count_map.unsqueeze(dim=0)
        pred_map = torch.argmax(pred_score_map, dim=0) + 1
        print('time_cost:{} cout:{}'.format(time_sum / count, count))
        print('done')
        OA, AA, K, acc = OA_AA_K_cal(np.array(pred_map, np.float), test_map)

    print('OA:{} AA:{} K:{}'.format(OA, AA, K))
    print(acc)

    with open(os.path.join(result_save_dir, 'evaluation_result.txt'), 'w') as f:
        f.write('OA: {}\n'.format(OA))
        f.write('AA: {}\n'.format(AA))
        f.write('K: {}\n'.format(K))
        for i in range(len(acc)):
            f.write('class {} acc: {}\n'.format(i+1, acc[i]))
    if not cfg.DATASET.SHOW_ALL:
        pred_map[np.where(label_map==0)]=0
    img_result = labelmap_2_img(color_dict[data_set], np.array(pred_map))
    img = Image.fromarray(img_result)
    img.save(os.path.join(result_save_dir, '{}.png'.format(data_set)))


def main():
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument(
        "--config-file",
        default="../configs/HoustonU/infe_ov.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--device",
        default='0',
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    evaluation(cfg)


if __name__ == "__main__":
    main()
