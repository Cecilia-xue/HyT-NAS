# Grafting Transformer on Automatically Designed Convolutional Neural Network for Hyperspectral Image Classification

[Xizhe Xue](https://cecilia-xue.github.io/), [Haokui Zhang](https://github.com/hkzhang91), Bei Fang, Zongwen Bai, Ying Li

[[`arXiv`](https://arxiv.org/abs/2110.11084)] [[`BibTeX`](#CitingHyT-NAS)]

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/Cecilia-xue/image-hosting@main/20220529/overrall.450fm2jgj3s0.webp" width="100%" height="100%"/>
</div><br/>
### Features
* Auto-searched structure for HSI Classification.
* Support  Hyperspectral Image Classification dataset: Houston University, Pavia University, Pavia Center.

## Installation

- `pip install -r requirements.txt`

## Datasets preparing

The current version HyT-NAS has support for a few datasets. Due to Policy constraints, we are not able to directly provide and host HSI images. However, we share the pre-processed HSI images in .h5 and .mat files. Datasets can be downloaded by accessing [Google Drive](https://drive.google.com/drive/folders/1DM_I__KRbyzV88De8Y4lL8k4VDPYgTTz?usp=sharing).

We have provided random  sample assignment files on different dataset in the 'preprocess/dataset_db', which can be used directly.
If you would like to generate them by yourself, run the samples_extraction.py script to assign the training, test and val samples.
- `python samples_extraction.py --data_root data_dir --dist_dir output_dir --dataset dataset_name --train_num number_training_samples --val_num number_val_samples`

Then, you should set the path of sample assignment files e,g("HoustonU_dist_per_train-20.0_val-10.0.h5") in the config files.

## Architucture Searching

- `cd ./tools python search.py --config-file '../configs/Pavia/search_ad.yaml' --device '0'`
- `cd ./tools python search.py --config-file '../configs/PaviaU/search_ad.yaml' --device '0'`
- `cd ./tools python search.py --config-file '../configs/HoustonU/search_ad.yaml' --device '0'`

## Model Training

- `cd ./tools python train.py --config-file '../configs/Pavia/train_ad.yaml' --device '0'`
- `cd ./tools python train.py --config-file '../configs/PaviaU/train_ad.yaml' --device '0'`
- `cd ./tools python train.py --config-file '../configs/HoustonU/train_ad.yaml' --device '0'`

### Inference with Pre-trained Models
- `cd ./tools python infe.py --config-file '../configs/Pavia/infe_ov.yaml' --device '0'`
- `cd ./tools python infe.py --config-file '../configs/PaviaU/infe_ov.yaml' --device '0'`
- `cd ./tools python infe.py --config-file '../configs/HoustonU/infe_ov.yaml' --device '0'`



Pick a model from [model zoo](MODEL_ZOO.md).
## Model Zoo
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">Number of training samples</th>
<th valign="bottom">OA%</th>
<th valign="bottom">AA%</th>
<th valign="bottom">K%</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: maskformer2_R50_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Pavia Center</td>
<td align="center">20 pixel/class</td>
<td align="center">99.28</td>
<td align="center">98.05</td>
<td align="center">98.98</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1pYnmbRKPU_gnoHAjSTrXquOsZfE8MrLB?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_R101_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Pavia Center</td>
<td align="center">30 pixel/class</td>
<td align="center">99.49</td>
<td align="center">98.99</td>
<td align="center">99.28</td>
<td align="center"><a href="https://drive.google.com/drive/folders/121I7OtWNXxbxT6--qJ3ONUxcVRfvIbQU?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_tiny_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Pavia University</td>
<td align="center">20 pixel/class</td>
<td align="center">98.77</td>
<td align="center">98.81</td>
<td align="center">98.37</td>
<td align="center"><a href="https://drive.google.com/drive/folders/101MAXQ53Ge6AN60gyy3Uz-Hsho05MgPq?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_small_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Pavia University</td>
<td align="center">30 pixel/class</td>
<td align="center">99.52</td>
<td align="center">99.51</td>
<td align="center">99.37</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1ztxBzyCLanrr22cvV9tL90ITsK28xJIB?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_384_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Houston University</td>
<td align="center">20 pixel/class</td>
<td align="center">87.11</td>
<td align="center">88.84</td>
<td align="center">86.07</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1wvMwGalu-b44x2aSs6W3xdaSGiBvhXZk?usp=sharing">model</a></td>
</tr>
<!-- ROW: maskformer2_swin_base_IN21k_384_bs16_50ep -->
 <tr><td align="center">HyT-NAS</td>
<td align="center">Houston University</td>
<td align="center">30 pixel/class</td>
<td align="center">91.14</td>
<td align="center">92.66</td>
<td align="center">90.42</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1pXStCBTEARGirE6OnuIuzwoQfUMPmb4W?usp=sharing">model</a></td>
</tr>
</tbody></table>

## <a name="CitingHyT-NAS"></a>Citing HyT-NAS

If you use HyT-NAS in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@ARTICLE{9791305,
  author={Xue, Xizhe and Zhang, Haokui and Fang, Bei and Bai, Zongwen and Li, Ying},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Grafting Transformer on Automatically Designed Convolutional Neural Network for Hyperspectral Image Classification}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2022.3180685}}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@article{zhang20213,
  title={3-D-ANAS: 3-D Asymmetric Neural Architecture Search for Fast Hyperspectral Image Classification},
  author={Zhang, Haokui and Gong, Chengrong and Bai, Yunpeng and Bai, Zongwen and Li, Ying},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--19},
  year={2021},
  publisher={IEEE}
}
```
