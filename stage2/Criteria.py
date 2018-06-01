#
#author          :Sachin Mehta
#Description     : This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
#File Description: This file implements the Cross entropy loss for 2D data.
#==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)