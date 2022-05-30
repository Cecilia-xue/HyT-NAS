import os
import json
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt


def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def h5_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]

    return data, label


def json_save(save_dir, dict_file):
    with open(save_dir, 'w') as f:
        json.dump(dict_file,f)


def get_dist_all(label_map, train_num, val_num, save_dir, dataset):
    label_num = sum(sum(label_map>0))
    train_ratio = train_num/label_num
    val_ratio = val_num/label_num
    category_num = label_map.max()
    train_dist = np.zeros(category_num)
    val_dist = np.zeros(category_num)
    cate_i_dist = np.zeros(category_num)

    for i in range(1, category_num):
        cate_i_num = sum(sum(label_map==i))
        train_dist[i-1]=max(2, np.around(train_ratio*cate_i_num))
        val_dist[i-1]=max(1, np.around(val_ratio*cate_i_num))
        cate_i_dist[i-1]=cate_i_num

    train_dist[-1] = train_num-train_dist.sum()
    val_dist[-1] = val_num-val_dist.sum()
    cate_i_dist[-1] = sum(sum(label_map==category_num))

    [m, n]=label_map.shape
    train_label_map = np.zeros((m,n))
    val_label_map = np.zeros((m,n))
    test_label_map = np.zeros((m,n))
    for i, [train_i_num, val_i_num, i_num] in enumerate(zip(train_dist, val_dist, cate_i_dist)):
        i_index = np.where(label_map==(i+1))
        shuffle_indices=np.random.permutation(int(i_num))
        train_indices=(i_index[0][shuffle_indices[:int(train_i_num)]], i_index[1][shuffle_indices[:int(train_i_num)]])
        val_indices=(i_index[0][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]], i_index[1][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]])
        test_indices=(i_index[0][shuffle_indices[int(train_i_num+val_i_num):]], i_index[1][shuffle_indices[int(train_i_num+val_i_num):]])

        train_label_map[train_indices]=i+1
        val_label_map[val_indices]=i+1
        test_label_map[test_indices]=i+1

    fig = plt.gcf()
    plt.subplot(1, 3, 1)
    plt.imshow(train_label_map * 25, )
    plt.title('train_set')
    plt.subplot(1, 3, 2)
    plt.imshow(val_label_map * 25)
    plt.title('val_set')
    plt.subplot(1, 3, 3)
    plt.imshow(test_label_map * 25)
    plt.title('test_set')
    fig.savefig(os.path.join(save_dir, dataset+'_dist_all_train-{}_val-{}.png'.format(train_num, val_num)), dpi=100)

    with h5py.File(os.path.join(save_dir, dataset + '_dist_all_train-{}_val-{}.h5'.format(train_num, val_num)), 'w') as f:
        f['height'] = m,
        f['width'] = n,
        f['train_label_map'] = train_label_map,
        f['val_label_map'] = val_label_map,
        f['test_label_map'] = test_label_map,
        f['category_num'] = category_num,


def get_dist_per(label_map, train_num, val_num, save_dir, dataset):
    category_num = label_map.max()
    train_per_num = np.around(train_num/category_num)
    val_per_num = np.around(val_num/category_num)
    train_dist = np.zeros(category_num)
    val_dist = np.zeros(category_num)
    cate_i_dist = np.zeros(category_num)

    for i in range(1, category_num):
        cate_i_num = sum(sum(label_map==i))
        train_dist[i-1]=int(min(np.floor(cate_i_num//2), train_per_num))
        val_dist[i-1]=int(min(np.floor(cate_i_num//2), val_per_num))
        cate_i_dist[i-1]=cate_i_num

    train_dist[-1] = train_num-train_dist.sum()
    val_dist[-1] = val_num-val_dist.sum()
    cate_i_dist[-1] = sum(sum(label_map==category_num))

    [m, n]=label_map.shape
    train_label_map = np.zeros((m,n))
    val_label_map = np.zeros((m,n))
    test_label_map = np.zeros((m,n))
    for i, [train_i_num, val_i_num, i_num] in enumerate(zip(train_dist, val_dist, cate_i_dist)):
        i_index = np.where(label_map==(i+1))
        shuffle_indices=np.random.permutation(int(i_num))
        train_indices=(i_index[0][shuffle_indices[:int(train_i_num)]], i_index[1][shuffle_indices[:int(train_i_num)]])
        val_indices=(i_index[0][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]], i_index[1][shuffle_indices[int(train_i_num):int(train_i_num+val_i_num)]])
        test_indices=(i_index[0][shuffle_indices[int(train_i_num+val_i_num):]], i_index[1][shuffle_indices[int(train_i_num+val_i_num):]])

        train_label_map[train_indices]=i+1
        val_label_map[val_indices]=i+1
        test_label_map[test_indices]=i+1

    fig = plt.gcf()
    plt.subplot(1, 3, 1)
    plt.imshow(train_label_map * 25, )
    plt.title('train_set')
    plt.subplot(1, 3, 2)
    plt.imshow(val_label_map * 25)
    plt.title('val_set')
    plt.subplot(1, 3, 3)
    plt.imshow(test_label_map * 25)
    plt.title('test_set')
    fig.savefig(os.path.join(save_dir, dataset+'_dist_per_train-{}_val-{}.png'.format(train_per_num, val_per_num)), dpi=100)

    with h5py.File(os.path.join(save_dir, dataset + '_dist_per_train-{}_val-{}.h5'.format(train_per_num, val_per_num)), 'w') as f:
        f['height'] = m,
        f['width'] = n,
        f['train_label_map'] = train_label_map,
        f['val_label_map'] = val_label_map,
        f['test_label_map'] = test_label_map,
        f['category_num'] = category_num,


def main():
    parser = argparse.ArgumentParser(description='HSI samples distribution')
    # parser.add_argument('--data_root', type=str, default='/home/hkzhang/Documents/data/HSIs/h5')
    parser.add_argument('--data_root', type=str, default='/public/datasets/hkzhang/data/h5')
    parser.add_argument('--dist_dir', type=str, default='./dataset_db')
    parser.add_argument('--dataset', type=str, default='HoustonU')
    parser.add_argument('--train_num', type=int, default=450)
    parser.add_argument('--val_num', type=int, default=150)
    parser.add_argument('--dist', type=str, default='per',
                        help='two sample distribution strategys: '
                             '1) all, randomly extraction samples from all labelled sampels '
                             '2) per, randomly extract samples of equal amount from each class')

    args = parser.parse_args()
    data_dir = os.path.join(args.data_root, args.dataset + '.h5')
    data, label = h5_loader(data_dir)
    make_if_not_exist(args.dist_dir)
    if args.dist=='all':
        get_dist_all(label, args.train_num, args.val_num, args.dist_dir, args.dataset)
    elif args.dist=='per':
        get_dist_per(label, args.train_num, args.val_num, args.dist_dir, args.dataset)


if __name__ == '__main__':
    main()





