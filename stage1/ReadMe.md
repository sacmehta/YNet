# Y-Net: Joint Segmentation and Classification for Diagnosis of Breast Biopsy Images
This repository contains the source code of segmentation branch of Y-Net.


## Structure of this repository
This repository is organized as:
* [data](/stage1/data/) This directory should contain the training and validation data.
* [pretrained_models_st1](/stage1/pretrained_models_st1/) This directory contains the pre-trained models. We only provide the models for Y-Net with ESP as encoding blocks and PSP as decoding blocks.
* Python files - These files contain the source code that we used to train the data

## Getting Started

### Training Y-Net segmentation branch

You can start training the model using below command:

```
python main.py 
```

Please see the command line arguments for more details.

**Note 1:** Currently, we support only single GPU training. If you want to train the model on multiple-GPUs, you can use **nn.DataParallel** api provided by PyTorch.

**Note 2:** To train on a specific GPU (single), you can specify the GPU_ID using the CUDA_VISIBLE_DEVICES as:

```
CUDA_VISIBLE_DEVICES=2 python main.py
```

This will run the training program on GPU with ID 2.
