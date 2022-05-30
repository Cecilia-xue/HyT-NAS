import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

atten_dir = '/home/path/atten.npy'

atten = np.load(atten_dir)
map_cn = np.zeros((32, 32))
map_cn[0:3, 0:3] = 1.0/9

map_cn_2 = np.zeros((32, 32))
map_cn_2[14:17, 14:17] = 1.0/9

# fig, ax = plt.subplots()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
p1 = sns.heatmap(map_cn_2, ax=ax)

s1 = p1.get_figure()
s1.savefig('./map_cn_2.png', bbox_inches='tight')
# sns.heatmap(map1, annot=True, ax=ax)



print('done')
