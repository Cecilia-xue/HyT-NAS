# IR_NAS: Neural Architecture Search for Image Restoration

Official implementation of the IR_NAS (PyTorch) 

# Installation

The code is tested with Python 3.6, PyTorch 1.0.0 and Cuda 9.0, Cudnn7.1.

Further requirments can be installed with 

  ```
    pip install -r requirements.txt
  ```

# data preprocess
Download the BSD300 and BSD200 and put the datasets in 'your_dir/nas_data/derain/'
cd ./preprocess

python denoise_preprocess.py --data_root 'your_dir/nas_data/' --task 'denoise'



# inference
Using the pretrained models to inference on BSD200. We provided pretrained models of three noise levels, which are put in 'tools/output/denoise/'. Change the DATA_ROOTs in .yaml files into your dir.

cd ./tools
python denoise_eval.py --config_file '../configs/denoise/BSD500_4c5n_g/inference.yaml'


# search
cd ./tools
python search.py --config_file '../configs/denoise/BSD500_4c5n_g/search.yaml'

# train your own models
cd ./tools
python train.py --config_file '../configs/denoise/BSD500_4c5n_g/train_sigma<30|50|70>.yaml'

## Citation
```

@article{zhang2019ir-nas,
  title={IR_NAS: Neural Architecture Search for Image Restoration},
  author={Zhang, Haokui and Li, Ying and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint  arXiv:1909.08228v1},
  year={2019}
}
```
