from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import pycocotools.coco as coco
import numpy as np
import math as m
import torch
import json
import cv2
import os
import math

from utils.image import color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.stereo_utils import read_obj_data, read_obj_calibration

class StereoDataset(data.Dataset):
  num_classes = 3
  default_resolution = [384, 1280]
  mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
  dim_exp = np.array([3.88, 1.63, 1.53], np.float32)

  def __init__(self, opt, split):
    super(StereoDataset, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'kitti')
    self.img_dir = os.path.join(self.data_dir, 'training', 'image_2')
    self.img_right_dir = os.path.join(self.data_dir, 'training', 'image_3')
    self.annot_path = os.path.join(self.data_dir, 'annotations_3d', 'kitti_{}_{}.json').format(opt.kitti_split, split)
    
    self.max_objs = 50
    self.class_name = [
      '__background__', 'Car',  'Van', 'Truck']
    #'Car',  'Van', 'Truck'
    self.cat_to_id = {name: i-1 for i, name in enumerate(self.class_name)}
    
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    self.split = split
    self.opt = opt
    self.alpha_in_degree = False

    print('==> initializing kitti {}, {} data.'.format(opt.kitti_split, split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.ori_samples =  len(self.images)
    if opt.flip_train and split == 'train':
      self.images = self.images*2
      print("use flip data augmentation!")
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _convert_alpha(self, alpha):
    return math.radians(alpha + 45) if self.alpha_in_degree else alpha

  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.coco.loadImgs(ids=[img_id])[0]

    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img_right_path = os.path.join(self.img_right_dir, img_info['file_name'])

    if self.opt.flip_train and index > self.ori_samples - 1:
      img = cv2.imread(img_right_path)
      img = img[:, ::-1, :].copy()
      img_right = cv2.imread(img_path)
      img_right = img_right[:, ::-1, :].copy()
    else:
      img = cv2.imread(img_path)
      img_right = cv2.imread(img_right_path)

    if 'calib' in img_info:
      calib = img_info['calib']
    else:
      calib = self.calib

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    if self.opt.keep_res:
      s = np.array([self.opt.input_w, self.opt.input_h], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)
    
    aug = False
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
      aug = True
      sf = self.opt.scale
      cf = self.opt.shift
      s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      c[0] += img.shape[1] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
      c[1] += img.shape[0] * np.clip(np.random.randn()*cf, -2*cf, 2*cf)

    trans_input = get_affine_transform(
      c, s, 0, [self.opt.input_w, self.opt.input_h])
    # inp = img
    inp = cv2.warpAffine(img, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    inp_right = cv2.warpAffine(img_right, trans_input, 
                         (self.opt.input_w, self.opt.input_h),
                         flags=cv2.INTER_LINEAR)
    inp_right = (inp_right.astype(np.float32) / 255.)
    if self.split == 'train' and np.random.random() < self.opt.aug_ddd:
        color_aug(self._data_rng, inp_right, self._eig_val, self._eig_vec)
    inp_right = (inp_right - self.mean) / self.std
    inp_right = inp_right.transpose(2, 0, 1)

    num_classes = self.opt.num_classes
    trans_output = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])
    
    hm = np.zeros(
      (num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 3), dtype=np.float32)
    reg = np.zeros((self.max_objs, 3), dtype=np.float32)
    # dim_orien = np.zeros((self.max_objs, 5), dtype=np.float32)
    dim = np.zeros((self.max_objs, 3), dtype=np.float32)
    orien = np.zeros((self.max_objs, 2), dtype=np.float32)
    depth = np.zeros((self.max_objs, 1), dtype=np.float32)
    kept = np.zeros((self.max_objs, 6), dtype=np.float32) # (kept1, kept2, kept3, kept4, border_left, border_right)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    ind_float = np.zeros((self.max_objs), dtype=np.float32)
    rot_mask = np.zeros((self.max_objs), dtype=np.uint8)

    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    objects = read_obj_data(anns, calib, self.class_name[1:], img.shape) ##########
    num_objs = min(len(objects), self.max_objs)
    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    gt_det = []

    # print('------ ' + str(index) + ' ---------')
    kk = 0
    for k in range(num_objs):
      object_it = objects[k]
      # print(object_it.pos[2])

      cls_id = self.cat_to_id[object_it.cls]

      if self.opt.flip_train and index > self.ori_samples - 1:
        bbox = np.array(object_it.boxes[1].box, dtype=np.float32)
        bbox_right = np.array(object_it.boxes[0].box, dtype=np.float32)

        img_w, img_right_w = img.shape[1], img_right.shape[1]
        oldx1, oldx2 = bbox[0], bbox[2]
        oldx1_right, oldx2_right = bbox_right[0], bbox_right[2]

        bbox[0] = img_w - oldx2 - 1
        bbox[2] = img_w - oldx1 - 1
        bbox_right[0] = img_right_w - oldx2_right - 1
        bbox_right[2] = img_right_w - oldx1_right - 1
      else:
        bbox = np.array(object_it.boxes[0].box, dtype=np.float32)
        bbox_right = np.array(object_it.boxes[1].box, dtype=np.float32)

      bbox[:2] = affine_transform(bbox[:2], trans_output)
      bbox[2:] = affine_transform(bbox[2:], trans_output)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)

      bbox_right[:2] = affine_transform(bbox_right[:2], trans_output)
      bbox_right[2:] = affine_transform(bbox_right[2:], trans_output)
      bbox_right[[0, 2]] = np.clip(bbox_right[[0, 2]], 0, self.opt.output_w - 1)
      bbox_right[[1, 3]] = np.clip(bbox_right[[1, 3]], 0, self.opt.output_h - 1)

      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
      h_right, w_right = bbox_right[3] - bbox_right[1], bbox_right[2] - bbox_right[0]

      keypoints = [0, 0, 0, 0, 0, 0]
      keypoints[0] = affine_transform((object_it.boxes[0].keypoints[0], object_it.boxes[0].box[3]), trans_output)[0]
      keypoints[1] = affine_transform((object_it.boxes[0].keypoints[1], object_it.boxes[0].box[3]), trans_output)[0]
      keypoints[2] = affine_transform((object_it.boxes[0].keypoints[2], object_it.boxes[0].box[3]), trans_output)[0]
      keypoints[3] = affine_transform((object_it.boxes[0].keypoints[3], object_it.boxes[0].box[3]), trans_output)[0]
      keypoints[4] = affine_transform((object_it.boxes[0].visible_left, object_it.boxes[0].box[3]), trans_output)[0]
      keypoints[5] = affine_transform((object_it.boxes[0].visible_right, object_it.boxes[0].box[3]), trans_output)[0]

      if self.opt.flip_train and index > self.ori_samples - 1:
        img_w = img.shape[1]

        kpts0_old = object_it.boxes[1].keypoints[0]
        kpts1_old = object_it.boxes[1].keypoints[1]
        kpts2_old = object_it.boxes[1].keypoints[2]
        kpts3_old = object_it.boxes[1].keypoints[3]
        kpts4_old = object_it.boxes[1].visible_left
        kpts5_old = object_it.boxes[1].visible_right

        keypoints[0] = -1 if kpts3_old == -1 else img_w - kpts3_old - 1
        keypoints[1] = -1 if kpts2_old == -1 else img_w - kpts2_old - 1
        keypoints[2] = -1 if kpts1_old == -1 else img_w - kpts1_old - 1
        keypoints[3] = -1 if kpts0_old == -1 else img_w - kpts0_old - 1
        keypoints[4] = -1 if kpts5_old == -1 else img_w - kpts5_old - 1
        keypoints[5] = -1 if kpts4_old == -1 else img_w - kpts4_old - 1

        keypoints[0] = affine_transform((keypoints[0], object_it.boxes[1].box[3]), trans_output)[0]
        keypoints[1] = affine_transform((keypoints[1], object_it.boxes[1].box[3]), trans_output)[0]
        keypoints[2] = affine_transform((keypoints[2], object_it.boxes[1].box[3]), trans_output)[0]
        keypoints[3] = affine_transform((keypoints[3], object_it.boxes[1].box[3]), trans_output)[0]
        keypoints[4] = affine_transform((keypoints[4], object_it.boxes[1].box[3]), trans_output)[0]
        keypoints[5] = affine_transform((keypoints[5], object_it.boxes[1].box[3]), trans_output)[0]
      
      keypoints = np.array(keypoints, dtype=np.float32)
      keypoints = np.clip(keypoints, -1, self.opt.output_w - 1)

      if h > 0 and w > 0:
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_right = np.array(
          [(bbox_right[0] + bbox_right[2]) / 2, (bbox_right[1] + bbox_right[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_gaussian(hm[cls_id], ct, radius)

        gt_det.append([ct[0], ct[1], 1] + \
                      self._alpha_to_8(self._convert_alpha(object_it.alpha)) + \
                      [object_it.pos[2]] + (np.array(object_it.dim) / 1).tolist() + [cls_id])
        if self.opt.reg_bbox:
          gt_det[-1] = gt_det[-1][:-1] + [w, h] + [gt_det[-1][-1]]

        wh[k] = 1. * w, 1. * w_right, 1. * h
        ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
        reg_left, reg_right = ct - ct_int, ct_right - ct_int
        reg[k] = reg_left[0], reg_right[0], reg_left[1]
        dim_it, alpha_it = object_it.dim, object_it.alpha
        if self.opt.flip_train and index > self.ori_samples - 1:
          if alpha_it > m.pi:
            alpha_it = alpha_it - 2.0*m.pi
          elif alpha_it < -m.pi:
            alpha_it = alpha_it + 2.0*m.pi
          alpha_it = (m.pi - alpha_it) if alpha_it >= 0 else (-m.pi - alpha_it)
        # dim_orien[k] = dim[0], dim[1], dim[2], math.sin(alpha), math.cos(alpha)
        dim[k] = dim_it[0], dim_it[1], dim_it[2]
        orien[k] = math.sin(alpha_it), math.cos(alpha_it)
        depth[kk] = object_it.pos[2]
        ind_float[k] = ind[k].astype(np.float32)
        rot_mask[k] = 1
        kept[k] = keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4], keypoints[5]
        kept[k] = kept[k] - bbox[0]
        kk = kk + 1
      # else:
      #   print(h, w)
      #   print(bbox)


    # print('gt_det', gt_det)
    # print('')
    # print(depth)
    
    ret = {'input': inp, 'input_right': inp_right, 'hm': hm, 'ind': ind, 'dim': dim, 'orien':orien, \
      'depth': depth, 'kept':kept, 'ind_float': ind_float, 'rot_mask': rot_mask}

    calibration = read_obj_calibration(calib)
    p2, p3 = calibration.p2, calibration.p3
    f = calibration.p2[0,0]
    bl = (calibration.p2[0,3] - calibration.p3[0,3])/f

    trans = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h])
    trans_inv = get_affine_transform(
      c, s, 0, [self.opt.output_w, self.opt.output_h], inv=1)
    
    ret.update({'fb':f*bl, 'p2':p2, 'p3': p3, 'trans': trans, 'trans_inv':trans_inv})

    if self.opt.reg_bbox:
      ret.update({'wh': wh})
    if self.opt.reg_offset:
      ret.update({'reg': reg})
    if self.opt.debug > 0 or not ('train' in self.split):
      gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
               np.zeros((1, 18), dtype=np.float32)
      meta = {'c': c, 's': s, 'gt_det': gt_det, 'calib': calib,
              'image_path': img_path, 'image_right': img_right_path, 'img_id': img_id, 'flipped': False}
      if self.opt.flip_train and index > self.ori_samples - 1:
        meta['flipped'] = True
      ret['meta'] = meta
    
    return ret

  def _alpha_to_8(self, alpha):
    # return [alpha, 0, 0, 0, 0, 0, 0, 0]
    ret = [0, 0, 0, 1, 0, 0, 0, 1]
    if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
      r = alpha - (-0.5 * np.pi)
      ret[1] = 1
      ret[2], ret[3] = np.sin(r), np.cos(r)
    if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
      r = alpha - (0.5 * np.pi)
      ret[5] = 1
      ret[6], ret[7] = np.sin(r), np.cos(r)
    return ret

  def save_results(self, results, save_dir):
    results_dir = os.path.join(save_dir, 'results')
    if not os.path.exists(results_dir):
      os.mkdir(results_dir)
    for img_id in results.keys():
      # if len(results[img_id][1]) == 0:
      #   continue
      out_path = os.path.join(results_dir, '{:06d}.txt'.format(img_id))
      f = open(out_path, 'w')
      for cls_ind in results[img_id]:
        for j in range(len(results[img_id][cls_ind])):
          class_name = self.class_name[cls_ind]
          f.write('{} 0.0 0'.format(class_name))
          for i in range(len(results[img_id][cls_ind][j])):
            f.write(' {:.2f}'.format(results[img_id][cls_ind][j][i]))
          f.write('\n')
      f.close()

  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    os.system('./tools/kitti_eval/evaluate_object_3d_offline ' + \
              '../data/kitti/training/label_2 ' + \
              '{}/results/'.format(save_dir))

