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


class DownSampler(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.conv = nn.Conv2d(nIn, nOut - nIn, 3, stride=2, padding=1, bias=False)
        self.pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        output = self.act(output)
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


class CDilated1(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)
        self.br = BR(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.br(output)


class DilatedParllelResidualBlockB(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        # output = self.drop(output)
        output = self.act(output)
        return output


class DilatedParllelResidualBlockB1(nn.Module):
    def __init__(self, nIn, nOut, prob=0.03):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 3, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3)
        self.act = nn.ReLU(True)  # nn.PReLU(nOut)
        # self.drop = nn.Dropout2d(p=prob)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        # d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        # add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3], 1)
        combine_in_out = input + combine
        output = self.bn(combine_in_out)
        # output = self.drop(output)
        output = self.act(output)
        return output


class PSPDec(nn.Module):
    def __init__(self, nIn, nOut, downSize, upSize=48):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(downSize),
            nn.Conv2d(nIn, nOut, 1, bias=False),
            nn.BatchNorm2d(nOut, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(nOut),
            nn.Upsample(size=upSize, mode='bilinear')
        )

    def forward(self, x):
        return self.features(x)


class ResNetC1(nn.Module):
    '''
        Segmentation model with ESP as the encoding block.
        This is the same as in stage 1
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
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(classes, classes, 7, padding=3, bias=False)
        )

        self.level2 = DownSamplerA(16, 128)
        self.level2_0 = DilatedParllelResidualBlockB1(128, 128)
        self.level2_1 = DilatedParllelResidualBlockB1(128, 128)  # 512 x 256

        self.p10 = PSPDec(8 + 256, 64, 80, 96)
        self.p20 = PSPDec(8 + 256, 64, 64, 96)
        self.p30 = PSPDec(8 + 256, 64, 48, 96)
        self.p40 = PSPDec(8 + 256, 64, 36, 96)

        self.class_1 = nn.Sequential(
            nn.Conv2d(8 + 256 + 64 * 4, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )

        self.br_2 = BR(256)

        self.level3_0 = DownSamplerA(256, 256)
        self.level3_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level3_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)  # 256 x 128

        self.level4_1 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(256, 256, 0.3)
        self.level4_3 = DilatedParllelResidualBlockB1(256, 256, 0.3)  # 128 x 64

        self.p1 = PSPDec(512, 128, 40)
        self.p2 = PSPDec(512, 128, 32)
        self.p3 = PSPDec(512, 128, 24)
        self.p4 = PSPDec(512, 128, 18)

        self.br_4 = BR(512)

        self.classifier = nn.Sequential(
            nn.Conv2d(512 + 4 * 128, 128, 1, padding=0, bias=False),
            nn.BatchNorm2d(128, momentum=0.95, eps=1e-3),
            nn.ReLU(True),  # nn.PReLU(classes),
            # nn.Dropout2d(.1),
            nn.Conv2d(128, classes, 3, padding=1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True),
            nn.Conv2d(classes, classes, 1, bias=False),
            nn.BatchNorm2d(classes, momentum=0.95, eps=1e-3),
            nn.ReLU(True)
        )
        # C(320, classes, 7, 1)

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

        #        output3 = output2_0 + output3

        #        classifier = self.classifier(output3)
        classifier = self.upsample_1(output0_hook)

        return classifier


class ResNetC1_YNet(nn.Module):
    '''
    Jointly learning the segmentation and classification with ESP as encoding blocks
    '''

    def __init__(self, classes, diagClasses, segNetFile=None):
        super().__init__()

        self.level4_0 = DownSamplerA(512, 128)
        self.level4_1 = DilatedParllelResidualBlockB1(128, 128, 0.3)
        self.level4_2 = DilatedParllelResidualBlockB1(128, 128, 0.3)

        self.br_con_4 = BR(256)

        self.level5_0 = DownSamplerA(256, 64)
        self.level5_1 = DilatedParllelResidualBlockB1(64, 64, 0.3)
        self.level5_2 = DilatedParllelResidualBlockB1(64, 64, 0.3)

        self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, diagClasses)

        # segmentation model
        self.segNet = ResNetC1(classes)
        if segNetFile is not None:
            print('Loading pre-trained segmentation model')
            self.segNet.load_state_dict(torch.load(segNetFile))
        self.modules = []
        for i, m in enumerate(self.segNet.children()):
            self.modules.append(m)

    def forward(self, input1):
        output0 = self.modules[0](input1)

        output1_0 = self.modules[6](output0)  # downsample
        output1 = self.modules[7](output1_0)
        output1 = self.modules[8](output1)

        output1 = self.modules[14](torch.cat([output1_0, output1], 1))

        output2_0 = self.modules[15](output1)  # downsample
        output2 = self.modules[16](output2_0)
        output2 = self.modules[17](output2)
        output3 = self.modules[18](output2)
        output3 = self.modules[19](output3)
        output3 = self.modules[20](output3)

        output3_hook = self.modules[25](torch.cat([output2_0, output3], 1))
        output3 = self.modules[26](
            torch.cat([output3_hook, self.modules[21](output3_hook), self.modules[22](output3_hook),
                       self.modules[23](output3_hook), self.modules[24](output3_hook)], 1))

        output3 = self.modules[29](output3)

        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.modules[13](torch.cat(
            [combine_up_23, self.modules[9](combine_up_23), self.modules[10](combine_up_23),
             self.modules[11](combine_up_23),
             self.modules[12](combine_up_23)], 1))
        output23_hook = self.modules[28](output23_hook)

        combine_up = torch.cat([output0, output23_hook], 1)

        output0_hook = self.modules[5](torch.cat(
            [combine_up, self.modules[1](combine_up), self.modules[2](combine_up), self.modules[3](combine_up),
             self.modules[4](combine_up)], 1))

        # segmentation classsifier
        classifier = self.modules[27](output0_hook)

        # diagnostic branch
        l5_0 = self.level4_0(output3_hook)
        l5_1 = self.level4_1(l5_0)
        l5_2 = self.level4_2(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1))

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1))

        glbAvg = self.global_Avg(l6_con)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        diagClass = self.fc2(fc1)

        return classifier, diagClass


class ResNetD1(nn.Module):
    '''
        Segmentation model with RCB as encoding blocks.
        This is the same as in Stage 1
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
        self.level2_1 = BasicResidualBlock(128, 128)  # 512 x 256

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
        self.level3_2 = BasicResidualBlock(256, 256, 0.3)

        self.level4_1 = BasicResidualBlock(256, 256, 0.3)
        self.level4_2 = BasicResidualBlock(256, 256, 0.3)
        self.level4_3 = BasicResidualBlock(256, 256, 0.3)

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


class ResNetD1_YNet(nn.Module):
    '''
        Jointly learning the segmentation and classification with RCB as encoding blocks
        '''

    def __init__(self, classes, diagClasses, segNetFile=None):
        super().__init__()

        self.level4_0 = DownSamplerA(512, 128)  # 24x24
        self.level4_1 = BasicResidualBlock(128, 128, 0.3)
        self.level4_2 = BasicResidualBlock(128, 128, 0.3)

        self.br_con_4 = BR(256)

        self.level5_0 = DownSamplerA(256, 64)  # 12x12
        self.level5_1 = BasicResidualBlock(64, 64, 0.3)
        self.level5_2 = BasicResidualBlock(64, 64, 0.3)

        self.br_con_5 = BR(128)

        self.global_Avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, diagClasses)

        self.segNet = ResNetD1(classes)  # 384 x 384
        if segNetFile is not None:
            print('Loading segmentation pre-trained model')
            self.segNet.load_state_dict(torch.load(segNetFile))
        self.modules = []
        for i, m in enumerate(self.segNet.children()):
            self.modules.append(m)
            # print(i, m)

    def forward(self, input1):
        output0 = self.modules[0](input1)

        output1_0 = self.modules[6](output0)  # downsample
        output1 = self.modules[7](output1_0)
        output1 = self.modules[8](output1)

        output1 = self.modules[14](torch.cat([output1_0, output1], 1))

        output2_0 = self.modules[15](output1)  # downsample
        output2 = self.modules[16](output2_0)
        output2 = self.modules[17](output2)
        output3 = self.modules[18](output2)
        output3 = self.modules[19](output3)
        output3 = self.modules[20](output3)

        output3_hook = self.modules[25](torch.cat([output2_0, output3], 1))
        output3 = self.modules[26](
            torch.cat([output3_hook, self.modules[21](output3_hook), self.modules[22](output3_hook),
                       self.modules[23](output3_hook), self.modules[24](output3_hook)], 1))

        output3 = self.modules[29](output3)

        combine_up_23 = torch.cat([output3, output1], 1)
        output23_hook = self.modules[13](torch.cat(
            [combine_up_23, self.modules[9](combine_up_23), self.modules[10](combine_up_23),
             self.modules[11](combine_up_23),
             self.modules[12](combine_up_23)], 1))
        output23_hook = self.modules[28](output23_hook)

        combine_up = torch.cat([output0, output23_hook], 1)

        output0_hook = self.modules[5](torch.cat(
            [combine_up, self.modules[1](combine_up), self.modules[2](combine_up), self.modules[3](combine_up),
             self.modules[4](combine_up)], 1))

        # segmentation classsifier
        classifier = self.modules[27](output0_hook)

        # diagnostic branch
        l5_0 = self.level4_0(output3_hook)
        l5_1 = self.level4_1(l5_0)
        l5_2 = self.level4_2(l5_1)
        l5_con = self.br_con_4(torch.cat([l5_0, l5_2], 1))

        l6_0 = self.level5_0(l5_con)
        l6_1 = self.level5_1(l6_0)
        l6_2 = self.level5_2(l6_1)
        l6_con = self.br_con_5(torch.cat([l6_0, l6_2], 1))

        glbAvg = self.global_Avg(l6_con)
        flatten = glbAvg.view(glbAvg.size(0), -1)
        fc1 = self.fc1(flatten)
        diagClass = self.fc2(fc1)

        return classifier, diagClass
