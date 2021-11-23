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
input_h, input_w = 384., 1280.

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=pad, bias=False)

def get_proposal_shift(left_boxes, right_boxes, depth_rate, fbs, trans_invs):
    depth_max = 87.0
    # D = depth_rate - 1
    # depth_bin_rate = [float(i*(i + 1))/float(D*(D + 1)) for i in range(depth_rate)]
    depth_bin_rate = [float(i)/(depth_rate - 1) for i in range(depth_rate)]
    depth_bin_rate = torch.tensor(depth_bin_rate).type(torch.float32).cuda()

    proposals_left_list = []
    proposals_right_list = []
    depth_bin_list = []

    batch_size = fbs.shape[0]
    for b in range(batch_size):
        index = left_boxes[:, 0] == b
        ind = left_boxes[index, 0]
        if ind.shape[0] == 0:
            continue
    
        fb = fbs[b]

        xmin = torch.min(left_boxes[index, 1], right_boxes[index, 1])
        ymin = torch.min(left_boxes[index, 2], right_boxes[index, 2])
        xmax = torch.max(left_boxes[index, 3], right_boxes[index, 3])
        ymax = torch.max(left_boxes[index, 4], right_boxes[index, 4])

        depth_bin_per_image_min = (fb/((xmax - xmin)*0.9*4)).view(-1,1)
        depth_bin_per_image_min = torch.clamp(depth_bin_per_image_min, min=1.0, max=87.0)
        depth_bin_per_image = depth_max - (depth_max - depth_bin_per_image_min) * depth_bin_rate
        disp_bin_per_image = fb / depth_bin_per_image / 8
        depth_bin_list.append(depth_bin_per_image)

        objnum = xmin.shape[0]
        bbox_shift_left_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)
        bbox_shift_right_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)

        for i in range(len(depth_bin_rate)):
            xmin_shift_left = torch.clamp(xmin + disp_bin_per_image[:, i], max=input_w//4 - 1.)
            xmax_shift_left = torch.clamp(xmax + disp_bin_per_image[:, i], max=input_w//4 - 1.)
            bbox_shift_left = torch.stack((ind, xmin_shift_left, ymin, xmax_shift_left, ymax), dim = 1)
            bbox_shift_left_per_depth[i, :, :] = bbox_shift_left

            xmin_shift_right = torch.clamp(xmin - disp_bin_per_image[:, i], min=0.)
            xmax_shift_right = torch.clamp(xmax - disp_bin_per_image[:, i], min=0.)
            bbox_shift_right = torch.stack((ind, xmin_shift_right, ymin, xmax_shift_right, ymax), dim = 1)
            bbox_shift_right_per_depth[i, :, :] = bbox_shift_right
        
        proposals_left_list.append(bbox_shift_left_per_depth)
        proposals_right_list.append(bbox_shift_right_per_depth)
    
    # for b in range(batch_size):
    #     index = left_boxes[:, 0] == b
    #     ind = left_boxes[index, 0]
    #     if ind.shape[0] == 0:
    #         continue
    
    #     fb = fbs[b]

    #     left_boxes_keep = left_boxes[index, :]
    #     right_boxes_keep = right_boxes[index, :]

    #     center_left = (left_boxes_keep[:, 1] + left_boxes_keep[:, 3])/2
    #     center_right = (right_boxes_keep[:, 1] + right_boxes_keep[:, 3])/2
    #     center_disp = center_left - center_right

    #     x1, x2 = left_boxes[index, 1], left_boxes[index, 3]
    #     y1, y2 = left_boxes[index, 2], left_boxes[index, 4]

    #     depth_bin_per_image_min = ((fb/(center_disp*4)) - 12.5).view(-1,1)
    #     depth_bin_per_image_max = ((fb/(center_disp*4)) + 12.5).view(-1,1)
    #     depth_bin_per_image_min = torch.clamp(depth_bin_per_image_min, min=0.0, max=90.0)
    #     depth_bin_per_image_max = torch.clamp(depth_bin_per_image_max, min=0.0, max=90.0)
    #     depth_bin_per_image = depth_bin_per_image_max - (depth_bin_per_image_max - depth_bin_per_image_min) * depth_bin_rate
    #     disp_bin_per_image = fb / depth_bin_per_image / 4
    #     depth_bin_list.append(depth_bin_per_image)

    #     objnum = x1.shape[0]
    #     bbox_shift_left_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)
    #     bbox_shift_right_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)

    #     for i in range(len(depth_bin_rate)):
    #         bbox_shift_left = torch.stack((ind, x1, y1, x2, y2), dim = 1)
    #         bbox_shift_left_per_depth[i, :, :] = bbox_shift_left

    #         x1_shift_right = torch.clamp(x1 - disp_bin_per_image[:, i], min=0)
    #         x2_shift_right = torch.clamp(x2 - disp_bin_per_image[:, i], min=0)
    #         bbox_shift_right = torch.stack((ind, x1_shift_right, y1, x2_shift_right, y2), dim = 1)
    #         bbox_shift_right_per_depth[i, :, :] = bbox_shift_right
        
    #     proposals_left_list.append(bbox_shift_left_per_depth)
    #     proposals_right_list.append(bbox_shift_right_per_depth)

    depth_bin = depth_bin_list[0]
    proposals_left = proposals_left_list[0]
    proposals_right = proposals_right_list[0]
    for i in range(1, len(depth_bin_list)):
        depth_bin = torch.cat((depth_bin, depth_bin_list[i]), dim=0) 
        proposals_left = torch.cat((proposals_left, proposals_left_list[i]), dim=1) 
        proposals_right = torch.cat((proposals_right, proposals_right_list[i]), dim=1) 

    return proposals_left.cuda(), proposals_right.cuda(), depth_bin.cuda()

class cost_volume(nn.Module):
    def __init__(self, inChannel):
        super(cost_volume, self).__init__()

        self.dres0 = nn.Sequential(convbn_3d(96, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True)) 

        self.strAM_2D= nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, 1),
                                     nn.BatchNorm2d(64)) 

        self.dres1 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   nn.BatchNorm3d(64),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(64, 128, 3, 1, 1),
                                   nn.BatchNorm3d(128),
                                   nn.ReLU(inplace=True)) 
        
        self.max_pool1 = nn.MaxPool3d((1,2,2)) #TODO
        # self.max_pool1 = nn.MaxPool3d((2,2,2)) #TODO

        self.dres2 = nn.Sequential(convbn_3d(128, 128, 3, 1, 1),
                                   nn.BatchNorm3d(128),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(128, 128, 3, 1, 1),
                                   nn.BatchNorm3d(128),
                                   nn.ReLU(inplace=True))

        self.max_pool2 = nn.MaxPool3d((1,2,2))
 
        self.classify = nn.Sequential(convbn_3d(128, 64, 3, 1, 1),
                                      nn.BatchNorm3d(64),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(64, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        self.avg_pool = nn.AvgPool2d(4,4)

        # self.pointNet = PointNetDetector(input_c=128)
        # self.down_ratio = 2

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
    
    def forward(self, cost, maxdisp, depth_bin):
        cost = cost.contiguous()

        num_channels = 32
        x_l_norm = torch.sqrt(torch.sum(cost[:, :num_channels,:,:,:]*cost[:, :num_channels,:,:,:],(1,3,4))) 
        x_r_norm = torch.sqrt(torch.sum(cost[:, num_channels:num_channels*2,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(1,3,4)))
        x_cross  = torch.sum(cost[:, :num_channels,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(1,3,4))/torch.clamp(x_l_norm*x_r_norm,min=0.01)
        x_cross = x_cross.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        cost = cost*x_cross

        cost = self.dres0(cost)
        
        isp = torch.mean(cost, dim=3)
        isp = torch.sigmoid(self.strAM_2D(isp)).unsqueeze(3)
        isp = isp.expand_as(cost)
        cost = isp*cost

        # num_channels = 32
        # x_l_norm = torch.sqrt(torch.sum(cost[:, :num_channels,:,:,:]*cost[:, :num_channels,:,:,:],(3,4))) 
        # x_r_norm = torch.sqrt(torch.sum(cost[:, num_channels:num_channels*2,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(3,4)))
        # x_cross  = torch.sum(cost[:, :num_channels,:,:,:]*cost[:, num_channels:num_channels*2,:,:,:],(3,4))/torch.clamp(x_l_norm*x_r_norm,min=0.01)
        # x_cross = x_cross.unsqueeze(3).unsqueeze(4).repeat(1, 3, 1, 1, 1)
        # x_cross = torch.clamp(x_cross, min=0.01, max=1)
        # cost = cost*x_cross

        cost = self.dres1(cost)
        cost = self.max_pool1(cost)

        # cost = cost*x_cross
        
        cost = self.dres2(cost) + cost
        cost = self.max_pool2(cost)

        cost = self.classify(cost)
        cost = torch.squeeze(cost, 1)
        cost = self.avg_pool(cost)

        pred = F.softmax(cost.squeeze().view(-1, maxdisp), dim=1)
        disp = Variable(torch.FloatTensor(pred.size()[0]).zero_()).cuda()
        for i in range(maxdisp):
            disp += pred[:,i] * depth_bin[:,i]
        disp = disp.contiguous()

        # res = maxdisp//self.down_ratio
        # pred = self.pointNet(cost.view(cost.shape[0], cost.shape[1], res*res*res), res)
        # disp = Variable(torch.FloatTensor(pred.size()[0]).zero_()).cuda()
        # disp = (depth_bin[:, 0] + depth_bin[:, -1])/2 + pred[:, 0]
        # disp = disp.contiguous()

        return disp

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def fill_reduce_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class stereo_network(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(stereo_network, self).__init__()
        self.down_ratio = down_ratio
        self.first_level = int(np.log2(down_ratio))
        self.feature_extraction = feature_extraction_dla34(base_name, pretrained=True,
                                                            down_ratio=down_ratio, last_level=5)
        channels = self.feature_extraction.channels
        self.roiSize = 16
        self.RoI = RoIAlign((self.roiSize, self.roiSize), spatial_scale=1, sampling_ratio=2)

        self.feaRuduce = nn.Sequential(
            nn.Conv2d(channels[self.first_level], 32, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.reduced_channel = 32
        self.depth_estimator = cost_volume(channels[self.first_level])
        
        self.left_only = ['kept_type']
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head in self.left_only:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level]*2, 256, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
                )
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

        # self.left_only = ['hm', 'kept_type']
        # self.heads = heads
        # for head in self.heads:
        #     ratio = 1 if head in self.left_only else 2
        #     classes = self.heads[head]
        #     if head_conv > 0:
        #         fc = nn.Sequential(
        #           nn.Conv2d(channels[self.first_level]*ratio, head_conv,kernel_size=3, padding=1, bias=True),
        #           nn.ReLU(inplace=True),
        #           nn.Conv2d(head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True))
        #         if 'hm' in head:
        #             fc[-1].bias.data.fill_(-2.19)
        #         else:
        #             fill_fc_weights(fc)
        #     else:
        #         fc = nn.Conv2d(channels[self.first_level]*ratio, classes, kernel_size=1, stride=1, padding=0, bias=True)
        #         if 'hm' in head:
        #             fc.bias.data.fill_(-2.19)
        #         else:
        #             fill_fc_weights(fc)
        #     self.__setattr__(head, fc)

    def forward(self, batch, useCostVolume=True, target=None, wh_scale=1.0):
        left, right = batch['input'], batch['input_right']

        imgfea_left = self.feature_extraction(left)
        imgfea_right = self.feature_extraction(right)

        z = {}
        for head in self.heads:
            if head in self.left_only:
                z[head] = self.__getattr__(head)(imgfea_left)
            else:
                z[head] = self.__getattr__(head)(torch.cat((imgfea_left, imgfea_right), 1))
        
        if useCostVolume:
            fb, p2, p3 = batch['fb'], batch['p2'], batch['p3']
            trans, trans_inv = batch['trans'], batch['trans_inv']

            imgfeaReduce_left = self.feaRuduce(imgfea_left)
            imgfeaReduce_right = self.feaRuduce(imgfea_right)

            if target is not None:
                bbox_keep, bbox_right_keep, bboxShape = target
            else:
                bbox_keep, bbox_right_keep, bboxShape = bbox_decode(z['hm'], z['wh']*wh_scale, z['reg'])

            batch_size, max_obj, _ = bboxShape
            depth = torch.zeros((batch_size, max_obj, 1), dtype=torch.float32)
    
            if bbox_keep.shape[0] != 0:
                pro_left, pro_right, depth_bin = get_proposal_shift(bbox_keep, bbox_right_keep, \
                                                                        self.roiSize, fb, trans_inv) 
                num_channels = self.reduced_channel
                cost = Variable(torch.FloatTensor(depth_bin.size()[0], num_channels*3, \
                                    self.roiSize, self.roiSize, self.roiSize).zero_()).cuda()
                for ind in range(self.roiSize):
                    roi_left = self.RoI(imgfeaReduce_left, pro_left[ind, :, :])
                    roi_right = self.RoI(imgfeaReduce_right, pro_right[ind, :, :])
                    cost[:, :num_channels, ind, :, :] = roi_left
                    cost[:, num_channels : num_channels*2, ind, :, :] = roi_right
                    cost[:, num_channels*2 : num_channels*3, ind, :, :] = roi_left - roi_right
                disp = self.depth_estimator(cost, self.roiSize, depth_bin)
                for b in range(batch_size):
                    index = bbox_keep[:, 0] == b
                    sum_num = disp[index].shape[0]
                    depth[b, :sum_num, 0] = disp[index]

            z.update({"depth": depth.cuda()})
        #     return [z], depth.cuda()
        # else:
        return [z]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = stereo_network('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model
