from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _transpose_and_gather_feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ddd_decode(heat, kept, dim, orien, wh, reg, grid_size, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    xs_right = torch.zeros_like(xs)
    
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 3)
    xs_right = xs.view(batch, K, 1) + reg[:, :, 1:2]
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 2:3]

    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    orien = _transpose_and_gather_feat(orien, inds)
    orien = orien.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 3)

    kept_offset = _transpose_and_gather_feat(kept[:, :4*grid_size, :, :], inds)
    kept_offset = kept_offset.view(batch, K, 4*grid_size)
    _, kept_offset = torch.max(kept_offset, dim=2)
    kept_type = (kept_offset/grid_size).type(torch.cuda.FloatTensor).unsqueeze(2)
    kept_offset = (kept_offset%grid_size).type(torch.cuda.FloatTensor).unsqueeze(2)

    borderLeft_offset = _transpose_and_gather_feat(kept[:, 4*grid_size:5*grid_size, :, :], inds)
    borderLeft_offset = borderLeft_offset.view(batch, K, grid_size)
    _, borderLeft_offset = torch.max(borderLeft_offset, dim=2)
    borderLeft_offset = borderLeft_offset.type(torch.cuda.FloatTensor).unsqueeze(2)

    borderRight_offset = _transpose_and_gather_feat(kept[:, 5*grid_size:, :, :], inds)
    borderRight_offset = borderRight_offset.view(batch, K, grid_size)
    _, borderRight_offset = torch.max(borderRight_offset, dim=2)
    borderRight_offset = borderRight_offset.type(torch.cuda.FloatTensor).unsqueeze(2)

    detections = torch.cat(
        [xs, ys, wh[:, : , [0, 2]], scores, clses], dim=2)
    detections_right = torch.cat(
        [xs_right, ys, wh[:, : , [1, 2]], scores, clses], dim=2)
    info_3d = torch.cat(
        [dim, orien, borderLeft_offset, borderRight_offset, kept_offset, kept_type], dim=2)

    # detections = torch.cat(
    #     [xs, ys, wh[:, : , [0, 2]], scores,  dim_orien, depth, clses], dim=2)
    # detections_right = torch.cat(
    #     [xs_right, ys, wh[:, : , [1, 2]], scores, clses], dim=2)
      
    return detections, detections_right, info_3d

def bbox_decode(heat, wh, reg, K=100):
    batch, cat, height, width = heat.size()
    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    xs_right = torch.zeros_like(xs)
    
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 3)
    xs_right = xs.view(batch, K, 1) + reg[:, :, 1:2]
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 2:3]
    
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 3)
    
    center = torch.cat([xs, ys], dim=2)
    center_right = torch.cat([xs_right, ys], dim=2)
    batch_index = torch.tensor([num for num in range(xs.shape[0])], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    batch_index = batch_index.repeat(1, xs.shape[1], 1)

    bbox = torch.zeros((xs.shape[0], xs.shape[1], 5), dtype=torch.float32)
    bbox[:, :, 1:3] = center - 0.5*wh[:, :, [0, 2]]
    bbox[:, :, 3:5] = center + 0.5*wh[:, :, [0, 2]]
    bbox_right = torch.zeros((xs.shape[0], xs.shape[1], 5), dtype=torch.float32)
    bbox_right[:, :, 1:3] = center_right - 0.5*wh[:, :, [1, 2]]
    bbox_right[:, :, 3:5] = center_right + 0.5*wh[:, :, [1, 2]]
    bbox[:, :, 0:1], bbox_right[:, :, 0:1] = batch_index, batch_index

    bbox_keep, bbox_right_keep = bbox.view(-1, 5), bbox_right.view(-1, 5)
    keep = torch.sum(bbox_keep[:, 1:5], dim=1) > 0
    bbox_keep, bbox_right_keep = bbox_keep[keep, :], bbox_right_keep[keep, :]
      
    return bbox_keep.cuda(), bbox_right_keep.cuda(), bbox.shape

