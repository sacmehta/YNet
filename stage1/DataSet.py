#
#author: Sachin Mehta
#Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file is used to create data tuples
#==============================================================================

import cv2
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        if self.transform:
            [image, label] = self.transform(image, label)
        return (image, label)