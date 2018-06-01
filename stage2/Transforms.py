#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the source code for different types of augmentation and numpy to Tensor Conversion.
# ==============================================================================

import numpy as np
import torch
import random
import cv2


class Zoom(object):

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, label, label1):
        h1, w1 = img.shape[:2]
        startH = random.randint(0, int(abs(self.h - h1)/2))
        startW = random.randint(0, int(abs(self.w - w1)/2))
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        img = img[startH:startH + h1, startW:startW + w1]
        label = label[startH:startH + h1, startW:startW + w1]
        return [img, label, label1]

class RandomCropResize(object):
    """
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label, label1):
        if random.random() < 0.5:
            w, h = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w,h), interpolation=cv2.INTER_NEAREST)
            return [img_crop, label_crop, label1]
        else:
            return [img, label, label1]

class RandomCrop(object):
    def __init__(self, scale):
        self.crop = scale

    def __call__(self, img, label, label1):

        if random.random() < 0.5:
            w, h = img.shape[:2]
            img_crop = img[self.crop:h-self.crop, self.crop:w-self.crop]
            label_crop = label[self.crop:h-self.crop, self.crop:w-self.crop]
            return [img_crop, label_crop, label1]
        else:
            return [img, label, label1]



class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, image, label, label1):

        if random.random() < 0.5:
            x1 = random.randint(0, 1)
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
        return [image, label, label1]


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label, label1):
        image = image.astype(np.float32)
        for i in range(3):
            image[:,:,i] -= self.mean[i]
        for i in range(3):
            image[:,:, i] /= self.std[i]

        return [image, label, label1]

class ToTensor(object):

    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label, label1):

        if self.scale != 1:
            w, h = label.shape[:2]
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), interpolation=cv2.INTER_NEAREST)

        image = image.transpose((2,0,1))
        image = image.astype(np.float32)

        image_tensor = torch.from_numpy(image).div(255)
        label_tensor =  torch.LongTensor(np.array(label, dtype=np.int)) #torch.from_numpy(label)

        return [image_tensor, label_tensor, label1]

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
