#
# author: Sachin Mehta
# Project Description: This repository contains source code for semantically segmenting WSIs; however, it could be easily
#                   adapted for other domains such as natural image segmentation
# File Description: This file contains the CNN models
# ==============================================================================

import torch
import torch.nn as nn


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False)

    def forward(self, input):
        output = self.conv(input)
        return output


class BasicResidualBlock(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        self.c1 = CBR(nIn, nOut, 3, 1)
        self.c2 = CB(nOut, nOut, 3, 1)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output = self.c1(input)
        output = self.c2(output)
        output = input + output
        # output = self.drop(output)
        output = self.act(output)
        return output


class DownSamplerA(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = CBR(nIn, nOut, 3, 2)

    def forward(self, input):
        output = self.conv(input)
        return output


class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-03)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedParllelResidualBlockB1(nn.Module):  # with k=4
    '''
    ESP Block from ESPNet. See details here: ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
    Link: https://arxiv.org/abs/1803.06815
    '''
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        k = 4 # we implemented with K=4 only
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        self.c1 = C(nIn, n, 3, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        output = self.act(output)
        return output


class PSPDec(nn.Module):
    '''
    Inspired or Adapted from Pyramid Scene Network paper
    Link: https://arxiv.org/abs/1612.01105
    '''
    def __init__(self, nIn, nOut, downSize, upSize=48):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(downSize), # NOTE: we trained our network at fixed size. If you want to train the network at variable size,
                                            #use the below version.
            nn.Conv2d(nIn, nOut, 1, bias=False),
            nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Upsample(size=upSize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)


# class PSPDec(nn.Module):
#     '''
#     Inspired or Adapted from Pyramid Scene Network paper
#     '''
#
#     def __init__(self, nIn, nOut, downSize):
#         super().__init__()
#         self.scale = downSize
#         self.features = nn.Sequential(
#             nn.Conv2d(nIn, nOut, 1, bias=False),
#             nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         assert x.dim() == 4
#         inp_size = x.size()
#         out_dim1, out_dim2 = int(inp_size[2] * self.scale), int(inp_size[3] * self.scale)
#         x_down = F.adaptive_avg_pool2d(x, output_size=(out_dim1, out_dim2))
#         return F.upsample(self.features(x_down), size=(inp_size[2], inp_size[3]), mode='bilinear')

class ResNetC1(nn.Module):
    '''
    This model uses ESP blocks for encoding and PSP blocks for decoding
    '''
    def __init__(self, classes):
        super().__init__()
        self.level1 = CBR(3, 16, 7, 2)

        self.p01 = PSPDec(16 + classes, classes, 160, 192)
        self.p02 = PSPDec(16 + classes, classes, 128, 192)
        self.p03 = PSPDec(16 + classes, classes, 96, 192)
        self.p04 = PSPDec(16 + classes, classes, 72, 192)

        self.class_0 = nn.Sequential(
            nn.Conv2d(16 + 5 * classes, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = DilatedParllelResidualBlockB1(128, 128)
        self.level2_1 = DilatedParllelResidualBlockB1(128, 128)

        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)

        self.level4_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_3 = DilatedParllelResidualBlockB1(256, 256, 0.3)

        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 4 * 128, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input1):
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)
        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)

        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))
        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1))

        output3 = self.upsample_3(output3)

        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23), self.p30(combine_up_23),
             self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        combine_up = torch.cat([output0, output23_hook], 1)

        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up), self.p03(combine_up), self.p04(combine_up)], 1))
        classifier = self.upsample_1(output0_hook)

        return classifier


class ResNetD1(nn.Module):
    '''
        This model uses ResNet blocks for encoding and PSP blocks for decoding
        '''
    def __init__(self, classes):
        super().__init__()
        self.level1 = CBR(3, 16, 7, 2)  # 384 x 384

        self.p01 = PSPDec(16 + classes, classes, 160, 192)
        self.p02 = PSPDec(16 + classes, classes, 128, 192)
        self.p03 = PSPDec(16 + classes, classes, 96, 192)
        self.p04 = PSPDec(16 + classes, classes, 72, 192)

        self.class_0 = nn.Sequential(
            nn.Conv2d(16 + 5 * classes, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = BasicResidualBlock(128, 128)
        self.level2_1 = BasicResidualBlock(128, 128)

        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = BasicResidualBlock(256, 256, 0.3)
        self.level3_2 = BasicResidualBlock(256, 256, 0.3)  # 256 x 128

        self.level4_1 = BasicResidualBlock(256, 256, 0.3)
        self.level4_2 = BasicResidualBlock(256, 256, 0.3)
        self.level4_3 = BasicResidualBlock(256, 256, 0.3)  # 128 x 64

        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 128 * 4, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, input1):
        # input1 = self.cmlrn(input)
        output0 = self.level1(input1)
        output1_0 = self.level2(output0)
        output1 = self.level2_0(output1_0)
        output1 = self.level2_1(output1)

        output1 = self.br_2(torch.cat([output1_0, output1], 1))

        output2_0 = self.level3_0(output1)
        output2 = self.level3_1(output2_0)
        output2 = self.level3_2(output2)

        output3 = self.level4_1(output2)
        output3 = self.level4_2(output3)

        output3 = self.level4_3(output3)
        output3 = self.br_4(torch.cat([output2_0, output3], 1))
        output3 = self.classifier(
            torch.cat([output3, self.p1(output3), self.p2(output3), self.p3(output3), self.p4(output3)], 1))

        output3 = self.upsample_3(output3)
        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.class_1(torch.cat(
            [combine_up_23, self.p10(combine_up_23), self.p20(combine_up_23), self.p30(combine_up_23),
             self.p40(combine_up_23)], 1))
        output23_hook = self.upsample_2(output23_hook)

        combine_up = torch.cat([output23_hook, output0], 1)

        output0_hook = self.class_0(torch.cat(
            [combine_up, self.p01(combine_up), self.p02(combine_up), self.p03(combine_up), self.p04(combine_up)], 1))
        classifier = self.upsample_1(output0_hook)

        return classifier
