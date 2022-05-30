from PIL import Image
import numpy as np
import h5py
from collections import Counter
import json
import matplotlib.pyplot as plt

data_set = 'KSC'
color_source_dir = './color_source'
mat_gt_dir = '/data/data2/path/data/HSIs/h5'


with h5py.File('/'.join((mat_gt_dir, '{}.h5'.format(data_set))), 'r') as f:
    label_map=f['label'][:]
m, n = label_map.shape
img = np.array(Image.open('/'.join((color_source_dir, '{}.jpg'.format(data_set)))).convert('RGB').resize((n, m)))
category_num = label_map.max()

color_dict = []
for i in range(1, category_num+1):
    indices=np.where(label_map==i)
    R_arr = img[indices][:, 0]
    G_arr = img[indices][:, 1]
    B_arr = img[indices][:, 2]
    print('label:{} RGB:{}'.format(i, [Counter(R_arr).most_common(1)[0][0],
                                    Counter(G_arr).most_common(1)[0][0],
                                    Counter(B_arr).most_common(1)[0][0]]))



