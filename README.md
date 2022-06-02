# Grafting Transformer on Automatically Designed Convolutional Neural Network for Hyperspectral Image Classification

[Xizhe Xue](https://cecilia-xue.github.io/), [Haokui Zhang](https://github.com/hkzhang91), Bei Fang, Zongwen Bai, Ying Li

[[`arXiv`](https://arxiv.org/abs/2110.11084)] [[`BibTeX`](#CitingHyT-NAS)]

<div align="center">
  <img src="https://cdn.jsdelivr.net/gh/Cecilia-xue/image-hosting@main/20220529/overrall.450fm2jgj3s0.webp" width="100%" height="100%"/>
</div><br/>
### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Support  Hyperspectral Image Classification dataset: Houston University, Pavia University, Pavia Center.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

Replicate web demo and docker image is available here: [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingHyT-NAS"></a>Citing HyT-NAS

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@article{xue20213d,
  title={ Grafting Transformer Module on Automatically Designed ConvNet for Hyperspectral Image Classification},
  author={Xue, Xizhe and Zhang, Haokui and Fang, Bei and Bai, Zongwen and Li, Ying},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
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