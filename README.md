# Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images
This repository contains the source code for our paper, [YNet](https://arxiv.org/abs/1806.01313), which is accepted for publication at [MICCAI'18](https://www.miccai2018.org/en/).

## Sample output of Y-Net

Y-Net  identified correctly classified tissues that were not important for diagnosis. For example, stroma was identified as an important tissue, but blood was not. Stroma is an important tissue label for diagnosing breast cancer [1] and removing information about stroma decreased the diagnostic classification accuracy by about  4\%. See paper for more details.

[1] Beck, Andrew H., et al. "Systematic analysis of breast cancer morphology uncovers stromal features associated with survival." Science translational medicine 3.108 (2011): 108ra113-108ra113.

![Results](/images/results.png)

Some segmentation results (Left: RGB WSI, Middle: Ground truth, Right: Predictions by Y-Net)

![Results](/images/results.gif)

## Structure of this repository
YNet is trained in two stages:
* [stage1](/stage1/) This directory contains the source code for training the stage 1 in Y-Net. Stage 1 is nothing but a segmentation brach.
* [stage2](/stage2/) This directory contains the source code for training the stage 2 in Y-Net. Stage 2 is jointly learning the segmentation and classification.
* [seg_eval](/seg_eval/) This directory contains the source code for producing the segmentation masks. 

## Pre-requisite

To run this code, you need to have following libraries:
* [OpenCV](https://opencv.org/) - We tested our code with version 3.3.0. If you are using other versions, please change the source code accordingly.
* [PyTorch](http://pytorch.org/) - We tested with v0.2.0_4. If you are using other versions, please change the source code accordingly.
* Python - We tested our code with Python 3.6.2 (Anaconda custom 64-bit). If you are using other Python versions, please feel free to make necessary changes to the code. 

We recommend to use [Anaconda](https://conda.io/docs/user-guide/install/linux.html). We have tested our code on Ubuntu 16.04.


## Citation
If Y-Net is useful for your research, then please cite our paper.
```
@inproceedings{mehta2018ynet,
  title={{Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images}},
  author={Sachin Mehta and Ezgi Mercan and Jamen Bartlett and Donald Weaver and Joann  Elmore and Linda Shapiro},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  year={2018},
  organization={Springer}
}

@article{mehta2018espnet,
  title={ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation},
  author={Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, and Hannaneh Hajishirzi},
  journal={arXiv preprint arXiv:1803.06815},
  year={2018}
}
```

## License
This code is released under the same license terms as [ESPNet](https://github.com/sacmehta/ESPNet).
