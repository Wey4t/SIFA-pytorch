﻿# SIFA-pytorch
This is a PyTorch implementation of SIFA for 'Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation.'


### 1. Dataset

If you wish to utilize the provided UnpairedDataset, please prepare your dataset in the following format. Please note that each individual data unit should be stored in an NPZ file, where '[arr_0]' contains the image data, and '[arr_1]' contains the corresponding labels:
```
your/data_root/
       source_domain/
          s001.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          s002.npz
          ...

       target_domain/
          t001.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          t002.npz
          ...
       test/
          t101.npz
            ['arr_0']:imgae_arr
            ['arr_1']:label_arr
          t102.npz
          ...
```

### 2. Perform experimental settings in ```config/train.cfg```

### 3. Train SIFA
```
CUDA_LAUNCH_BLOCKING=0 python train.py
```

### 4. Test SIFA
```
CUDA_LAUNCH_BLOCKING=0 python test.py
```


#### If you find the code useful, please consider citing the following article (with [code](https://github.com/HiLab-git/FPL-plus)):

```bibtex
@article{wu2024fpl+,
  author={Wu, Jianghao and Guo, Dong and Wang, Guotai and Yue, Qiang and Yu, Huijun and Li, Kang and Zhang, Shaoting},
  journal={IEEE Transactions on Medical Imaging}, 
  title={FPL+: Filtered Pseudo Label-Based Unsupervised Cross-Modality Adaptation for 3D Medical Image Segmentation}, 
  year={2024},
  volume={43},
  number={9},
  pages={3098-3109}
}


```
#### Furthermore, Source-Free Domain Adaptation is a more advanced domain adaptation task that does not require source domain data for adaptation. Please refer to the following paper (with [code](https://github.com/HiLab-git/UPL-SFDA)):
```bibtex
@ARTICLE{10261231,
  author={Wu, Jianghao and Wang, Guotai and Gu, Ran and Lu, Tao and Chen, Yinan and Zhu, Wentao and Vercauteren, Tom and Ourselin, Sébastien and Zhang, Shaoting},
  journal={IEEE Transactions on Medical Imaging}, 
  title={UPL-SFDA: Uncertainty-Aware Pseudo Label Guided Source-Free Domain Adaptation for Medical Image Segmentation}, 
  year={2023},
  volume={42},
  number={12},
  pages={3932-3943}

```
