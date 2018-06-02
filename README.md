# Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images
This repository contains the source code for our paper, YNet, which is accepted for publication at [MICCAI'18](https://www.miccai2018.org/en/) paper

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
If ESPNet is useful for your research, then please cite our paper.
```
@inproceedings{mehta2018ynet,
  title={{Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images}},
  author={Sachin Mehta and Ezgi Mercan and Jamen Bartlett and Donald Weaver and Joann  Elmore and Linda Shapiro},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  year={2018},
  organization={Springer}
}
```

## License
This code is released under the same license terms as [ESPNet](https://github.com/sacmehta/ESPNet).
