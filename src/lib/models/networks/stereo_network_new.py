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

from torchvision.ops import RoIAlign, RoIPool 
from models.decode import bbox_decode

from .submodule import *
from .feature_extraction_dla34 import feature_extraction_dla34

BN_MOMENTUM = 0.1
input_h, input_w = 384., 1280.

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False)


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                        padding=pad, bias=False)

def project_rect_to_image(pts_3d_rect, P):
    n = pts_3d_rect.shape[0]
    ones = torch.ones((n,1))
    if pts_3d_rect.is_cuda:
        ones = ones.cuda()
    pts_3d_rect = torch.cat([pts_3d_rect, ones], dim=1)
    pts_2d = torch.mm(pts_3d_rect, torch.transpose(P, 0, 1)) # nx3
    # pts_2d[:,0] /= pts_2d[:,2]
    # pts_2d[:,1] /= pts_2d[:,2]
    pts_2d = pts_2d/pts_2d[:, 2:]
    return pts_2d

def get_proposal_shift(left_boxes, right_boxes, depth_rate, fbs, trans_invs):
    depth_max = 87
    # D = depth_rate - 1
    # depth_bin_rate = [float(i*(i + 1))/float(D*(D + 1)) for i in range(depth_rate)]
    depth_bin_rate = [float(i)/(depth_rate - 1) for i in range(depth_rate)]
    depth_bin_rate = torch.tensor(depth_bin_rate).type(torch.float32)

    proposals_left_list = []
    proposals_right_list = []
    depth_bin_list = []

    batch_size = fbs.shape[0]
    # for b in range(batch_size):
    #     index = left_boxes[:, 0] == b
    #     ind = left_boxes[index, 0]
    #     if ind.shape[0] == 0:
    #         continue

    #     xmin = torch.min(left_boxes[index, 1], right_boxes[index, 1])
    #     ymin = torch.min(left_boxes[index, 2], right_boxes[index, 2])
    #     xmax = torch.max(left_boxes[index, 3], right_boxes[index, 3])
    #     ymax = torch.max(left_boxes[index, 4], right_boxes[index, 4])

    #     depth_bin_per_image_min = (fbs[b]/((xmax - xmin)*0.9*4)).view(-1,1)
    #     depth_bin_per_image_min = torch.clamp(depth_bin_per_image_min, min=1.0, max=90.0)
    #     depth_bin_per_image = depth_max - (depth_max - depth_bin_per_image_min) * depth_bin_rate
    #     disp_bin_per_image = fbs[b] / depth_bin_per_image / 4
    #     depth_bin_list.append(depth_bin_per_image)

    #     objnum = xmin.shape[0]
    #     bbox_shift_left_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)
    #     bbox_shift_right_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)

    #     for i in range(len(depth_bin_rate)):
    #         xmin_shift_left = torch.clamp(xmin, max=input_w/4 - 1.)
    #         xmax_shift_left = torch.clamp(xmax, max=input_w/4 - 1.)
    #         bbox_shift_left = torch.stack((ind, xmin_shift_left, ymin, xmax_shift_left, ymax), dim = 1)
    #         bbox_shift_left_per_depth[i, :, :] = bbox_shift_left

    #         xmin_shift_right = torch.clamp(xmin - disp_bin_per_image[:, i], min=0.)
    #         xmax_shift_right = torch.clamp(xmax - disp_bin_per_image[:, i], min=0.)
    #         bbox_shift_right = torch.stack((ind, xmin_shift_right, ymin, xmax_shift_right, ymax), dim = 1)
    #         bbox_shift_right_per_depth[i, :, :] = bbox_shift_right
        
    #     proposals_left_list.append(bbox_shift_left_per_depth)
    #     proposals_right_list.append(bbox_shift_right_per_depth)
    
    for b in range(batch_size):
        index = left_boxes[:, 0] == b
        ind = left_boxes[index, 0]
        if ind.shape[0] == 0:
            continue
    
        fb = fbs[b].detach().cpu().type(torch.float32)
        trans_inv = trans_invs[b].detach().cpu().type(torch.float32)

        left_boxes_keep = left_boxes[index, :].detach().cpu().type(torch.float32)
        right_boxes_keep = right_boxes[index, :].detach().cpu().type(torch.float32)
        ones = torch.ones((left_boxes_keep.shape[0], 1))

        pt1 = torch.cat([left_boxes_keep[:, 1:3], ones], dim=1)
        pt1 = torch.mm(pt1, trans_inv.transpose(0, 1))
        pt2 = torch.cat([left_boxes_keep[:, 3:5], ones], dim=1)
        pt2 = torch.mm(pt2, trans_inv.transpose(0, 1))
        left_boxes_ori = torch.cat([pt1, pt2], dim=1)

        pt1_right = torch.cat([right_boxes_keep[:, 1:3], ones], dim=1)
        pt1_right = torch.mm(pt1_right, trans_inv.transpose(0, 1))
        pt2_right = torch.cat([right_boxes_keep[:, 3:5], ones], dim=1)
        pt2_right = torch.mm(pt2_right, trans_inv.transpose(0, 1))
        right_boxes_ori = torch.cat([pt1_right, pt2_right], dim=1)

        center_left = (left_boxes_ori[:, 0] + left_boxes_ori[:, 2])/2
        center_right = (right_boxes_ori[:, 0] + right_boxes_ori[:, 2])/2
        center_disp = center_left - center_right

        x1, x2 = left_boxes[index, 1], left_boxes[index, 3]
        y1, y2 = left_boxes[index, 2], left_boxes[index, 4]

        depth_bin_per_image_min = ((fb/center_disp) - 12.5).view(-1,1)
        depth_bin_per_image_max = ((fb/center_disp) + 12.5).view(-1,1)
        depth_bin_per_image_min = torch.clamp(depth_bin_per_image_min, min=1.0, max=90.0)
        depth_bin_per_image_max = torch.clamp(depth_bin_per_image_max, min=1.0, max=90.0)
        depth_bin_per_image = depth_bin_per_image_max - (depth_bin_per_image_max - depth_bin_per_image_min) * depth_bin_rate
        disp_bin_per_image = fb / depth_bin_per_image / 4
        disp_bin_per_image = disp_bin_per_image.cuda()
        depth_bin_list.append(depth_bin_per_image)

        objnum = x1.shape[0]
        bbox_shift_left_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)
        bbox_shift_right_per_depth = torch.zeros((len(depth_bin_rate), objnum, 5), dtype=torch.float32)

        for i in range(len(depth_bin_rate)):
            bbox_shift_left = torch.stack((ind, x1, y1, x2, y2), dim = 1)
            bbox_shift_left_per_depth[i, :, :] = bbox_shift_left

            x1_shift_right = torch.clamp(x1 - disp_bin_per_image[:, i], min=0)
            x2_shift_right = torch.clamp(x2 - disp_bin_per_image[:, i], min=0)
            bbox_shift_right = torch.stack((ind, x1_shift_right, y1, x2_shift_right, y2), dim = 1)
            bbox_shift_right_per_depth[i, :, :] = bbox_shift_right
        
        proposals_left_list.append(bbox_shift_left_per_depth)
        proposals_right_list.append(bbox_shift_right_per_depth)

    depth_bin = depth_bin_list[0]
    proposals_left = proposals_left_list[0]
    proposals_right = proposals_right_list[0]
    for i in range(1, len(depth_bin_list)):
        depth_bin = torch.cat((depth_bin, depth_bin_list[i]), dim=0) 
        proposals_left = torch.cat((proposals_left, proposals_left_list[i]), dim=1) 
        proposals_right = torch.cat((proposals_right, proposals_right_list[i]), dim=1) 

    return proposals_left.cuda(), proposals_right.cuda(), depth_bin.cuda()

def get_voxel(left_boxes, right_boxes, p2s, p3s, fbs, depth_bins, trans, trans_invs):
    stride = 0.5

    batch_size = fbs.shape[0]
    norm_coord_imgs = []
    norm_coord_left_imgs2ds = []
    norm_coord_right_imgs2ds = []
    depth_ori_list = []
    for b in range(batch_size):
        index = left_boxes[:, 0] == b
        ind = left_boxes[index, 0]
        if ind.shape[0] == 0:
            continue

        p2 = p2s[b].detach().cpu().type(torch.float32)
        p3 = p3s[b].detach().cpu().type(torch.float32)
        fb = fbs[b].detach().cpu().type(torch.float32)
        trans_inv = trans_invs[b].detach().cpu().type(torch.float32)
        tran = trans[b].detach().cpu().type(torch.float32)

        left_boxes_keep = left_boxes[index, :].detach().cpu().type(torch.float32)
        right_boxes_keep = right_boxes[index, :].detach().cpu().type(torch.float32)
        ones = torch.ones((left_boxes_keep.shape[0], 1))

        pt1 = torch.cat([left_boxes_keep[:, 1:3], ones], dim=1)
        pt1 = torch.mm(pt1, trans_inv.transpose(0, 1))
        pt2 = torch.cat([left_boxes_keep[:, 3:5], ones], dim=1)
        pt2 = torch.mm(pt2, trans_inv.transpose(0, 1))
        left_boxes_ori = torch.cat([pt1, pt2], dim=1)

        pt1_right = torch.cat([right_boxes_keep[:, 1:3], ones], dim=1)
        pt1_right = torch.mm(pt1_right, trans_inv.transpose(0, 1))
        pt2_right = torch.cat([right_boxes_keep[:, 3:5], ones], dim=1)
        pt2_right = torch.mm(pt2_right, trans_inv.transpose(0, 1))
        right_boxes_ori = torch.cat([pt1_right, pt2_right], dim=1)

        center_x = (left_boxes_ori[:, 0] + left_boxes_ori[:, 2])/2
        center_y = (left_boxes_ori[:, 1] + left_boxes_ori[:, 3])/2
        center_x_right = (right_boxes_ori[:, 0] + right_boxes_ori[:, 2])/2
        depth = fb/(center_x - center_x_right)
        depth_ori_list.append(depth)

        # center_x = (left_boxes_keep[:, 0] + left_boxes_keep[:, 2])/2
        # center_y = (left_boxes_keep[:, 1] + left_boxes_keep[:, 3])/2
        # center_x_right = (left_boxes_keep[:, 0] + left_boxes_keep[:, 2])/2
        # depth = fb/((center_x - center_x_right)*4)
        # depth_ori_list.append(depth)

        for i in range(depth.shape[0]):
            z = depth[i] - p2[2, 3]
            x = (center_x[i] * depth[i] - p2[0, 3] - p2[0, 2] * z) / p2[0, 0]
            y = (center_y[i] * depth[i] - p2[1, 3] - p2[1, 2] * z) / p2[1, 1]

            zs = torch.arange(-5., 5., 1.) + 0.5 + z
            ys = torch.arange(-2.5, 2.5, stride) + stride/2 + y
            xs = torch.arange(-2.5, 2.5, stride) + stride/2 + x
            xs, ys, zs = torch.meshgrid(xs, ys, zs)
            coord_rect = torch.stack([xs, ys, zs], dim=-1)

            coord_img = torch.as_tensor(
                project_rect_to_image(
                    coord_rect.reshape(-1, 3), p2
                    ).reshape(*coord_rect.shape[:3], 3), dtype=torch.float32)
            coord_img = torch.mm(coord_img.view(-1, 3), tran.transpose(0, 1))
            coord_img = torch.cat([coord_img.reshape(*coord_rect.shape[:3], 2), coord_rect[..., 2:]], dim=-1)
            # coord_img = torch.cat([coord_img[..., :2]/4, coord_rect[..., 2:]], dim=-1)

            u_min, u_max = left_boxes[index, 1][i], left_boxes[index, 3][i]
            v_min, v_max = left_boxes[index, 2][i], left_boxes[index, 4][i]

            # u_min = torch.min(left_boxes[index, 1][i], right_boxes[index, 1][i])
            # v_min = torch.min(left_boxes[index, 2][i], right_boxes[index, 2][i])
            # u_max = torch.max(left_boxes[index, 3][i], right_boxes[index, 3][i])
            # v_max = torch.max(left_boxes[index, 4][i], right_boxes[index, 4][i])
            depth_bin = depth_bins[index][i].detach().cpu() 
            d_min, d_max = torch.min(depth_bin), torch.max(depth_bin)

            norm_coord_img = (coord_img - torch.as_tensor([u_min, v_min, d_min])[None, None, None, :]) / \
                (torch.as_tensor([u_max, v_max, d_max]) - torch.as_tensor([u_min, v_min, d_min]))[None, None, None, :]
            norm_coord_img = norm_coord_img * 2. - 1.
            norm_coord_imgs.append(norm_coord_img)

            ########################################################
            u_min, u_max = 0., input_w/4 - 1.
            v_min, v_max = 0., input_h/4 - 1.

            norm_coord_left_imgs2d = (coord_img[..., :2] - torch.as_tensor([u_min, v_min])[None, None, :]) / \
                (torch.as_tensor([u_max, v_max]) - torch.as_tensor([u_min, v_min]))[None, None, :]
            norm_coord_left_imgs2d = norm_coord_left_imgs2d * 2. - 1.
            norm_coord_left_imgs2ds.append(norm_coord_left_imgs2d)

            coord_img_right = torch.as_tensor(
                project_rect_to_image(
                    coord_rect.reshape(-1, 3), p3
                    ).reshape(*coord_rect.shape[:3], 3), dtype=torch.float32)
            coord_img_right = torch.mm(coord_img_right.view(-1, 3), tran.transpose(0, 1))
            coord_img_right = coord_img_right.reshape(*coord_rect.shape[:3], 2)
            # coord_img_right = coord_img_right[..., :2]/4
            
            norm_coord_img_right = (coord_img_right - torch.as_tensor([u_min, v_min])[None, None, :]) / \
                (torch.as_tensor([u_max, v_max]) - torch.as_tensor([u_min, v_min]))[None, None, :]
            norm_coord_img_right = norm_coord_img_right * 2. - 1.
            norm_coord_right_imgs2ds.append(norm_coord_img_right)

    norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
    valids = (norm_coord_imgs[..., 0] >= -1.) & (norm_coord_imgs[..., 0] <= 1.) & \
                (norm_coord_imgs[..., 1] >= -1.) & (norm_coord_imgs[..., 1] <= 1.) & \
                (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
    valids = valids.float()

    norm_coord_left_imgs2ds = torch.stack(norm_coord_left_imgs2ds, dim=0)
    valids_left = (norm_coord_left_imgs2ds[..., 0] >= -1.) & (norm_coord_left_imgs2ds[..., 0] <= 1.) & \
                (norm_coord_left_imgs2ds[..., 1] >= -1.) & (norm_coord_left_imgs2ds[..., 1] <= 1.)
    valids_left = valids_left.float()

    norm_coord_right_imgs2ds = torch.stack(norm_coord_right_imgs2ds, dim=0)
    valids_right = (norm_coord_right_imgs2ds[..., 0] >= -1.) & (norm_coord_right_imgs2ds[..., 0] <= 1.) & \
                (norm_coord_right_imgs2ds[..., 1] >= -1.) & (norm_coord_right_imgs2ds[..., 1] <= 1.)
    valids_right = valids_right.float()

    depth_ori_list = torch.cat(depth_ori_list, dim=0)

    return norm_coord_imgs.cuda(), valids.cuda(), norm_coord_left_imgs2ds.cuda(), valids_left.cuda(), \
        norm_coord_right_imgs2ds.cuda(), valids_right.cuda(), depth_ori_list.cuda()

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
        self.roiSize = 20
        # self.RoI = RoIAlign((self.roiSize, self.roiSize), spatial_scale=1, sampling_ratio=2)

        # self.feaRuduce = nn.Sequential(
        #     nn.Conv2d(channels[self.first_level], 32, kernel_size=1, padding=0, bias=False),
        #     nn.BatchNorm2d(32, momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True)
        # )

        self.feaRuduce = nn.Sequential(
            nn.Conv2d(channels[self.first_level], 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # self.reduced_channel = 32

        # self.hg = False
        # self.depth_estimator = cost_volume(self.hg)

        self.pointNet = PointNetDetector(input_c=192)
        
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

    def forward(self, batch, useCostVolume=True, target=None):
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
                bbox_keep, bbox_right_keep, bboxShape = bbox_decode(z['hm'], z['wh'], z['reg'])

            batch_size, max_obj, _ = bboxShape
            depth = torch.zeros((batch_size, max_obj, 1), dtype=torch.float32)
    
            if bbox_keep.shape[0] != 0:
                pro_left, pro_right, depth_bin = get_proposal_shift(bbox_keep, bbox_right_keep, \
                                                                        self.roiSize, fb, trans_inv) 
                # num_channels = self.reduced_channel
                # cost = Variable(torch.FloatTensor(depth_bin.size()[0], num_channels*3, \
                #                     self.roiSize, self.roiSize, self.roiSize).zero_()).cuda()
                # for ind in range(self.roiSize):
                #     roi_left = self.RoI(imgfeaReduce_left, pro_left[ind, :, :])
                #     roi_right = self.RoI(imgfeaReduce_right, pro_right[ind, :, :])
                #     cost[:, :num_channels, ind, :, :] = roi_left
                #     cost[:, num_channels : num_channels*2, ind, :, :] = roi_right
                #     cost[:, num_channels*2 : num_channels*3, ind, :, :] = roi_left - roi_right
                # cost = self.depth_estimator(cost)

                ###########################################################################

                norm_coord_imgs, valids, norm_coord_left_imgs2ds, valids_left, \
                    norm_coord_right_imgs2ds, valids_right, depth_ori = \
                        get_voxel(bbox_keep, bbox_right_keep, p2, p3, fb, depth_bin, trans, trans_inv)
                
                # norm_coord_imgs = norm_coord_imgs*valids.unsqueeze(4)
                norm_coord_left_imgs2ds = norm_coord_left_imgs2ds*valids_left.unsqueeze(4)
                norm_coord_right_imgs2ds = norm_coord_right_imgs2ds*valids_right.unsqueeze(4)
                
                ba, res = norm_coord_imgs.shape[0], norm_coord_imgs.shape[1]
                # voxel = F.grid_sample(cost, norm_coord_imgs)
                # voxel = voxel*valids.unsqueeze(1)

                voxel2d = []
                voxel2d_right = []
                for b in range(batch_size):
                    index = bbox_keep[:, 0] == b
                    N = index.int().sum()
                    if N == 0:
                        continue
                    
                    norm_coord_left_imgs2d = norm_coord_left_imgs2ds[index]
                    norm_coord_left_imgs2d = norm_coord_left_imgs2d.reshape(N, -1, 2).unsqueeze(0)
                    voxel2d_fea = F.grid_sample(imgfeaReduce_left[b:b+1, :, :, :], norm_coord_left_imgs2d)
                    voxel2d_fea = voxel2d_fea.squeeze(0).transpose(0, 1)
                    voxel2d.append(voxel2d_fea)
                    
                    norm_coord_right_imgs2d = norm_coord_right_imgs2ds[index]
                    norm_coord_right_imgs2d = norm_coord_right_imgs2d.reshape(N, -1, 2).unsqueeze(0)
                    voxel2d_fea_right = F.grid_sample(imgfeaReduce_right[b:b+1, :, :, :], norm_coord_right_imgs2d)
                    voxel2d_fea_right = voxel2d_fea_right.squeeze(0).transpose(0, 1)
                    voxel2d_right.append(voxel2d_fea_right)
                voxel2d = torch.cat(voxel2d, dim=0)
                voxel2d = voxel2d.reshape(ba, -1, res, res, res)
                voxel2d = voxel2d * valids_left.unsqueeze(1)

                voxel2d_right = torch.cat(voxel2d_right, dim=0)
                voxel2d_right = voxel2d_right.reshape(ba, -1, res, res, res)
                voxel2d_right = voxel2d_right * valids_right.unsqueeze(1)

                # voxel = torch.cat([voxel, voxel2d, voxel2d_right], dim=1)
                voxel = torch.cat([voxel2d - voxel2d_right, voxel2d, voxel2d_right], dim=1)

                # res = voxel.shape[2]
                voxel = voxel.reshape(voxel.shape[0], voxel.shape[1], -1)

                disp = self.pointNet(voxel, res=res)
                ############################################################################

                for b in range(batch_size):
                    index = bbox_keep[:, 0] == b
                    N = index.int().sum()
                    if N == 0:
                        continue

                    depth[b, :N, 0] = depth_ori[index] + disp[index, 0]
                    # depth[b, :N, 0] = disp[index, 0]

            z.update({"depth": depth.cuda()})
        return [z]


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
  model = stereo_network('dla{}'.format(num_layers), heads,
                 pretrained=True,
                 down_ratio=down_ratio,
                 final_kernel=1,
                 last_level=5,
                 head_conv=head_conv)
  return model
