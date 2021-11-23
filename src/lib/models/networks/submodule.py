from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.ops import RoIAlign, RoIPool 
from models.decode import bbox_decode

from .feature_extraction_dla34 import feature_extraction_dla34
from .pointNet import PointNetDetector

BN_MOMENTUM = 0.1

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=pad, bias=False)

class cost_volume(nn.Module):
    def __init__(self, hg):
        super(cost_volume, self).__init__()
        self.hg = hg
        if self.hg:
            self.dres1 = nn.Sequential(convbn_3d(96, 64, 3, 1, 1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(64, 128, 3, 2, 1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(inplace=True)) 
            self.dres2 = nn.Sequential(convbn_3d(128, 128, 3, 2, 1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(128, 128, 3, 1, 1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(inplace=True))

            self.dres3 = nn.Sequential(
                nn.ConvTranspose3d(128, 128, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                nn.BatchNorm3d(128))
            self.dres4 = nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                nn.BatchNorm3d(64))
        else:
            self.dres1 = nn.Sequential(convbn_3d(96, 64, 3, 1, 1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(64, 64, 3, 2, 1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True)) 
            self.dres2 = nn.Sequential(convbn_3d(64, 64, 3, 2, 1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(64, 64, 3, 1, 1),
                                    nn.BatchNorm3d(64))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, cost):
        cost = cost.contiguous()

        if self.hg:
            cost0 = self.dres1(cost)
            cost = self.dres2(cost0)

            cost = self.dres3(cost) + cost0
            cost = self.dres4(cost)
        else:
            cost = self.dres1(cost)
            cost = self.dres2(cost)

        return cost

class PointNetfeat_strAM(nn.Module):
    def __init__(self, input_c):
        super(PointNetfeat_strAM, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_c, 256, 1)
        self.conv2 = torch.nn.Conv1d(256, 512, 1)
        self.conv3 = torch.nn.Conv1d(512, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 1024, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.strAM_2D=torch.nn.Conv2d(1024,1024,3,1,1)

    def forward(self, x, res):
        x = F.relu(self.bn1((self.conv1(x))))
        x = F.relu((self.bn2(self.conv2(x))))
        x = self.bn3(self.conv3(x))

        isp_cube=x.view(x.size(0),x.size(1), res, res, res)
        isp=torch.mean(isp_cube,dim=3)
        isp=torch.sigmoid(self.strAM_2D(isp)).unsqueeze(3)
        isp=isp.expand_as(isp_cube)
        isp=isp*isp_cube
        isp=isp.view(x.size(0), x.size(1), res*res*res)

        x = F.relu(self.bn4(self.conv4(isp)))+x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x

class PointNetDetector(nn.Module):
    def __init__(self, input_c):
        super(PointNetDetector, self).__init__()
        self.feat_all = PointNetfeat_strAM(input_c)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.depth = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, input_data, res):
        xa = self.feat_all(input_data, res)
        x = self.fc1(xa)
        x = x if x.shape[0] <= 1 else self.bn1(x)
        x = F.relu(x)

        
        x = self.dropout(self.fc2(x))
        x = x if x.shape[0] <= 1 else self.bn2(x)
        x = F.relu(x)
        depth = self.depth(x)

        return depth

# class PointNetDetector(nn.Module):
#     def __init__(self, input_c):
#         super(PointNetDetector, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(input_c*10, 1024, 3, 1, 1),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True))

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(1024, 512, 3, 1, 1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True))
        
#         self.pool1 = nn.MaxPool2d((2, 2))
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(512, 256, 3, 1, 1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True))
        
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(256, 128, 3, 1, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True))
        
#         self.pool2 = nn.MaxPool2d((5, 5))
        
#         self.depth = nn.Conv2d(128, 1, 1, 1, 0)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
    
#     def forward(self, Voxel, res):
#         Voxel = Voxel.contiguous()

#         N, _, W, H, L = Voxel.shape
#         Voxel = Voxel.permute(0, 1, 3, 2, 4).reshape(N, -1, W, L).contiguous()

#         Voxel_BEV = self.conv1(Voxel)
#         Voxel_BEV = self.conv2(Voxel_BEV)
#         Voxel_BEV = self.pool1(Voxel_BEV)

#         Voxel_BEV = self.conv3(Voxel_BEV)
#         Voxel_BEV = self.conv4(Voxel_BEV)
#         Voxel_BEV = self.pool2(Voxel_BEV)

#         dep = self.depth(Voxel_BEV)
#         dep = dep.squeeze(3).squeeze(2) 

#         return dep