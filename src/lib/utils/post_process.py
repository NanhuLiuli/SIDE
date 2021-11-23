from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math as m
import torch
from .image import transform_preds
from .stereo_utils import read_obj_calibration
from .box_estimator import solve_x_y_theta_from_kpt, solve_x_y_z_theta_from_kpt

from dense_align.dense_align import align_parallel

def get_alpha(rot):
  sin_alpha = rot[:, 0]
  cos_alpha = rot[:, 1]
  return np.arctan2(sin_alpha, cos_alpha)

def post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  bbox = np.zeros((dets.shape[0], dets.shape[1], 5), dtype=np.float32)
  bbox[:, :, :2] = dets[:, :, :2] - 0.5*dets[:, :, 2:4]
  bbox[:, :, 2:4] = dets[:, :, :2] + 0.5*dets[:, :, 2:4]
  bbox[:, :, 4] = dets[:, :, 4]
  for i in range(dets.shape[0]):
    top_preds = {}
    bbox[i, :, :2] = transform_preds(
          bbox[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    bbox[i, :, 2:4] = transform_preds(
          bbox[i, :, 2:4], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = bbox[i, inds, :5].astype(np.float32)
    ret.append(top_preds)
  return ret

def post_process_info(info_3d, dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  bbox = np.zeros((dets.shape[0], dets.shape[1], 5), dtype=np.float32)
  bbox[:, :, :2] = dets[:, :, :2] - 0.5*dets[:, :, 2:4]
  bbox[:, :, 2:4] = dets[:, :, :2] + 0.5*dets[:, :, 2:4]

  border_kept = np.zeros((dets.shape[0], dets.shape[1], 3), dtype=np.float32)
  border_kept[:, :, :3] = info_3d[:, :, 5:8]
  for i in range(dets.shape[0]):
    top_preds = {}
    bbox[i, :, :2] = transform_preds(
          bbox[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    bbox[i, :, 2:4] = transform_preds(
          bbox[i, :, 2:4], c[i], s[i], (opt.output_w, opt.output_h))
    width = bbox[i, :, 2:3] - bbox[i, :, 0:1]
    start = bbox[i, :, 0:1]
    border_kept[i, :, :] = start + border_kept[i, :, :]*width/opt.grid
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = border_kept[i, inds, :3].astype(np.float32)
      ##############################################
      top_preds[j + 1] = np.concatenate([top_preds[j + 1], info_3d[i, inds, 8:9], \
                              info_3d[i, inds, :3], get_alpha(info_3d[i, inds, 3:5])[:, np.newaxis]], axis=1)
      if opt.cost_volume:
        top_preds[j + 1] = np.concatenate([top_preds[j + 1], info_3d[i, inds, 9:10]], axis=1)
      # disparity = np.zeros((top_preds[j + 1].shape[0], 1), dtype=np.float32)
      # top_preds[j + 1] = np.concatenate([top_preds[j + 1], disparity], axis=1)
    ret.append(top_preds)
  return ret

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    keep = rot_y > np.pi
    rot_y[keep] = rot_y[keep] - 2 * np.pi
    keep = rot_y < -np.pi
    rot_y[keep] = rot_y[keep] + 2 * np.pi
    # if rot_y > np.pi:
    #   rot_y -= 2 * np.pi
    # if rot_y < -np.pi:
    #   rot_y += 2 * np.pi
    return rot_y

def post_process_3d(dets2d, dets2d_right, info_3d, s, calibs, opt):
  ret = []
  for i in range(len(dets2d)):
    preds = {}

    calib = read_obj_calibration(calibs[i])
    f = calib.p2[0,0]
    cx, cy = calib.p2[0,2], calib.p2[1,2]
    bl = (calib.p2[0,3] - calib.p3[0,3])/f
    x_shift = (calib.p2[0, 3] - calib.p0[0, 3])/f
    y_shift = (calib.p2[1, 3] - calib.p0[1, 3])/f 
    z_shift = (calib.p2[2, 3] - calib.p0[2, 3])/f

    for cls_id in range(1, opt.num_classes + 1):
      preds[cls_id] = []
      dets2d_it = dets2d[i][cls_id]
      dets2d_it_right = dets2d_right[i][cls_id]
      info_3d_it = info_3d[i][cls_id]

      box_left = dets2d_it[:, 0:4]
      box_right = dets2d_it_right[:, 0:4]
      scores = dets2d_it[:, 4:5]

      dim = info_3d_it[:, 4:7]
      alpha = info_3d_it[:, 7:8]
      
      # h, w, l = dim[:, 0:1], dim[:, 1:2], dim[:, 2:3] 
      # w, h, l = dim[0], dim[1], dim[2] 
      center_x = (box_left[:, 0:1] + box_left[:, 2:3])/2
      center_y = (box_left[:, 1:2] + box_left[:, 3:4])/2
      center_x_right = (box_right[:, 0:1] + box_right[:, 2:3])/2
      
      if opt.cost_volume:
        depth = info_3d_it[:, 8:9]
      else:
        disp = center_x - center_x_right
        depth = f*bl/disp
      
      z = depth - calib.p2[2, 3]
      x = (center_x * depth - calib.p2[0, 3] - calib.p2[0, 2] * z) / calib.p2[0, 0]
      y = (center_y * depth - calib.p2[1, 3] - calib.p2[1, 2] * z) / calib.p2[1, 1] + dim[:, 0:1]/2
      theta = alpha2rot_y(alpha, center_x, calib.p2[0, 2], calib.p2[0, 0])
      # theta = alpha - np.arctan2(-x, z)

      pred = np.concatenate([alpha, box_left, dim, x, y, z, theta, scores], axis=1)
      keep = pred[:, -1] > opt.peak_thresh
      preds[cls_id] = pred[keep, :]

      for detect_idx in range(dets2d_it.shape[0]):
        if dets2d_it[detect_idx, -1] > opt.peak_thresh:
          box_left_it = box_left[detect_idx, :]
          box_right_it = box_right[detect_idx, :]
          dim_it = dim[detect_idx, :]
          alpha_it = alpha[detect_idx, 0]
          depth_it = depth[detect_idx, 0]

          # status, state_rect = solve_x_y_z_theta_from_kpt(s[i], calib, \
          #   alpha_it, (dim_it[1], dim_it[0], dim_it[2]), box_left_it, box_right_it, depth_it, info_3d_it[detect_idx, :4])
          # if status == 1:
          #   x_rect = state_rect[0] - x_shift
          #   y_rect = state_rect[1] - y_shift
          #   z_rect = state_rect[2] - z_shift
          #   theta_rect = state_rect[3] - m.pi/2
          #   preds[cls_id][detect_idx, 8:12] = x_rect, y_rect, z_rect, theta_rect
          
          state_rect, z = solve_x_y_theta_from_kpt(s[i], calib, alpha_it, (dim_it[1], dim_it[0], dim_it[2]), \
                                box_left_it, f*bl/depth_it, info_3d_it[detect_idx, :4])
          x_rect = state_rect[0] - x_shift
          y_rect = state_rect[1] - y_shift
          z_rect = z - z_shift
          theta_rect = state_rect[2] - m.pi/2
          preds[cls_id][detect_idx, 8:12] = x_rect, y_rect, z_rect, theta_rect
          
    ret.append(preds)
  return ret

def ddd_post_process(dets, dets_right, info_3d, c, s, calibs, opt, img, img_right):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  # dets2d, info_3d= post_process_2d_left(dets, c, s, opt)
  # dets3d = post_process_3d(dets2d, info_3d, s, calibs, opt)

  dets2d= post_process_2d(dets, c, s, opt)
  dets2d_right = post_process_2d(dets_right, c, s, opt)
  info_3d= post_process_info(info_3d, dets, c, s, opt)

  dets3d = post_process_3d(dets2d, dets2d_right, info_3d, s, calibs, opt)

  for i in range(len(dets3d)):
    for cls_id in range(1, opt.num_classes + 1):
      det = dets3d[i][cls_id]
      if det.shape[0] == 0:
        continue
      calib = read_obj_calibration(calibs[i])
      f = calib.p2[0,0]
      x_shift = (calib.p2[0, 3] - calib.p0[0, 3])/f
      y_shift = (calib.p2[1, 3] - calib.p0[1, 3])/f 
      z_shift = (calib.p2[2, 3] - calib.p0[2, 3])/f
      f = calib.p2[0,0]
      bl = (calib.p2[0,3] - calib.p3[0,3])/f

      bbox, dim, theta = det[:, 1:5], det[:, 5:8], det[:, 11:12] + m.pi/2
      x, y, z = det[:, 8:9] + x_shift, det[:, 9:10] + y_shift, det[:, 10:11] + z_shift
      pose = np.concatenate([x, y, z, dim[:, 1:2], dim[:, :1], dim[:, 2:], theta], axis=1)
      bbox, pose = torch.from_numpy(bbox), torch.from_numpy(pose)
      succ, dis_final = align_parallel(calib, opt, img, img_right, bbox, info_3d[i][cls_id][:, :2], pose)

      for solved_idx in range(succ.size(0)):
        if succ[solved_idx] > 0:
          # dets3d[i][cls_id][solved_idx, 10] = f*bl/dis_final[solved_idx].cpu().numpy()
          state_rect, z = solve_x_y_theta_from_kpt(s[i], calib, det[solved_idx,0], pose[solved_idx,3:6].cpu().numpy(), \
                                bbox[solved_idx, :].cpu().numpy(), dis_final[solved_idx].cpu().numpy(), \
                                info_3d[i][cls_id][solved_idx, :4])
          xyzTheta = np.array([state_rect[0] - x_shift, state_rect[1] - y_shift, z - z_shift, state_rect[2] - m.pi/2])
          dets3d[i][cls_id][solved_idx, 8:12] = xyzTheta
            
  return dets3d, info_3d
