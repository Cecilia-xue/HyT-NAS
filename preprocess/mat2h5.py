import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import h5py

dataset = 'Pavia'
dataset_HSI = 'pavia'
dataset_gt = 'pavia_gt'

data_mat_dir = '/data/data2/zhk218/data/HSIs/mat/'
data_h5_dir = '/data/data2/zhk218/data/HSIs/h5/'

dataset_mat_dir = data_mat_dir + '{}/{}.mat'.format(dataset, dataset)
dataset_gt_dir = data_mat_dir + '{}/{}_gt.mat'.format(dataset, dataset)
dataset_h5_save_dir = data_h5_dir + '{}.h5'.format(dataset)

if dataset == 'HoustonU':
    mat_file = h5py.File(dataset_mat_dir)
    HSI_data = np.array(mat_file['IGRSS_2013']).swapaxes(0, 2)
    mat_file = h5py.File(dataset_gt_dir)
    HSI_gt = np.array(mat_file['IGRSS_2013_gt']).swapaxes(0,1)

else:
    HSI_data = sio.loadmat(dataset_mat_dir)[dataset_HSI]
    HSI_gt = sio.loadmat(dataset_gt_dir)[dataset_gt]


with h5py.File(dataset_h5_save_dir, 'w') as f:
    f['data'] = HSI_data
    f['label'] = HSI_gt