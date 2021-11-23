from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math as m
import cv2
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d
from .stereo_utils import read_obj_calibration

class Debugger(object):
  def __init__(self, ipynb=False, theme='black', 
               num_classes=-1, dataset=None, down_ratio=4):
    self.ipynb = ipynb
    if not self.ipynb:
      import matplotlib.pyplot as plt
      self.plt = plt
    self.imgs = {}
    self.theme = theme
    colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]
    self.colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)
    if self.theme == 'white':
      self.colors = self.colors.reshape(-1)[::-1].reshape(len(colors), 1, 1, 3)
      self.colors = np.clip(self.colors, 0., 0.6 * 255).astype(np.uint8)
    self.dim_scale = 1
    if num_classes == 3 or dataset == 'kitti':
      self.names = kitti_class_name
      self.focal_length = 721.5377
      self.W = 1242
      self.H = 375
    num_classes = len(self.names)
    self.down_ratio=down_ratio
    # for bird view
    self.world_size = 64
    self.out_size = 384

  def add_img(self, img, img_id='default', revert_color=False):
    if revert_color:
      img = 255 - img
    self.imgs[img_id] = img.copy()
  
  def add_mask(self, mask, bg, imgId = 'default', trans = 0.8):
    self.imgs[imgId] = (mask.reshape(
      mask.shape[0], mask.shape[1], 1) * 255 * trans + \
      bg * (1 - trans)).astype(np.uint8)
  
  def show_img(self, pause = False, imgId = 'default'):
    cv2.imshow('{}'.format(imgId), self.imgs[imgId])
    if pause:
      cv2.waitKey()
  
  def add_blend_img(self, back, fore, img_id='blend', trans=0.7):
    if self.theme == 'white':
      fore = 255 - fore
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    self.imgs[img_id] = (back * (1. - trans) + fore * trans)
    self.imgs[img_id][self.imgs[img_id] > 255] = 255
    self.imgs[img_id][self.imgs[img_id] < 0] = 0
    self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

  '''
  # slow version
  def gen_colormap(self, img, output_res=None):
    # num_classes = len(self.colors)
    img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl = self.colors[i] if not (self.theme == 'white') \
           else 255 - self.colors[i]
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
    '''

  
  def gen_colormap(self, img, output_res=None):
    img = img.copy()
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map
    
  '''
  # slow
  def gen_colormap_hp(self, img, output_res=None):
    # num_classes = len(self.colors)
    # img[img < 0] = 0
    h, w = img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    color_map = np.zeros((output_res[0], output_res[1], 3), dtype=np.uint8)
    for i in range(img.shape[0]):
      resized = cv2.resize(img[i], (output_res[1], output_res[0]))
      resized = resized.reshape(output_res[0], output_res[1], 1)
      cl =  self.colors_hp[i] if not (self.theme == 'white') else \
        (255 - np.array(self.colors_hp[i]))
      color_map = np.maximum(color_map, (resized * cl).astype(np.uint8))
    return color_map
  '''
  def gen_colormap_hp(self, img, output_res=None):
    c, h, w = img.shape[0], img.shape[1], img.shape[2]
    if output_res is None:
      output_res = (h * self.down_ratio, w * self.down_ratio)
    img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
    colors = np.array(
      self.colors_hp, dtype=np.float32).reshape(-1, 3)[:c].reshape(1, 1, c, 3)
    if self.theme == 'white':
      colors = 255 - colors
    color_map = (img * colors).max(axis=2).astype(np.uint8)
    color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
    return color_map


  def add_rect(self, rect1, rect2, c, conf=1, img_id='default'): 
    cv2.rectangle(
      self.imgs[img_id], (rect1[0], rect1[1]), (rect2[0], rect2[1]), c, 2)
    if conf < 1:
      cv2.circle(self.imgs[img_id], (rect1[0], rect1[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect1[0], rect2[1]), int(10 * conf), c, 1)
      cv2.circle(self.imgs[img_id], (rect2[0], rect1[1]), int(10 * conf), c, 1)

  def add_coco_bbox(self, bbox, cat, conf=1, show_txt=True, img_id='default'): 
    bbox = np.array(bbox, dtype=np.int32)
    # cat = (int(cat) + 1) % 80
    cat = int(cat)
    # print('cat', cat, self.names[cat])
    c = self.colors[cat][0][0].tolist()
    if self.theme == 'white':
      c = (255 - np.array(c)).tolist()
    txt = '{}{:.1f}'.format(self.names[cat], conf)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
    cv2.rectangle(
      self.imgs[img_id], (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
    if show_txt:
      cv2.rectangle(self.imgs[img_id],
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2), c, -1)
      cv2.putText(self.imgs[img_id], txt, (bbox[0], bbox[1] - 2), 
                  font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

  def add_coco_hp(self, points, img_id='default'): 
    points = np.array(points, dtype=np.int32).reshape(self.num_joints, 2)
    for j in range(self.num_joints):
      cv2.circle(self.imgs[img_id],
                 (points[j, 0], points[j, 1]), 3, self.colors_hp[j], -1)
    for j, e in enumerate(self.edges):
      if points[e].min() > 0:
        cv2.line(self.imgs[img_id], (points[e[0], 0], points[e[0], 1]),
                      (points[e[1], 0], points[e[1], 1]), self.ec[j], 2,
                      lineType=cv2.LINE_AA)

  def add_points(self, points, img_id='default'):
    num_classes = len(points)
    # assert num_classes == len(self.colors)
    for i in range(num_classes):
      for j in range(len(points[i])):
        c = self.colors[i, 0, 0]
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio, 
                                       points[i][j][1] * self.down_ratio),
                   5, (255, 255, 255), -1)
        cv2.circle(self.imgs[img_id], (points[i][j][0] * self.down_ratio,
                                       points[i][j][1] * self.down_ratio),
                   3, (int(c[0]), int(c[1]), int(c[2])), -1)

  def show_all_imgs(self, pause=False, time=0):
    if not self.ipynb:
      for i, v in self.imgs.items():
        cv2.imshow('{}'.format(i), v)
      if cv2.waitKey(0 if pause else 1) == 27:
        import sys
        sys.exit(0)
    else:
      self.ax = None
      nImgs = len(self.imgs)
      fig=self.plt.figure(figsize=(nImgs * 10,10))
      nCols = nImgs
      nRows = nImgs // nCols
      for i, (k, v) in enumerate(self.imgs.items()):
        fig.add_subplot(1, nImgs, i + 1)
        if len(v.shape) == 3:
          self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
        else:
          self.plt.imshow(v)
      self.plt.show()

  def save_img(self, imgId='default', path='./cache/debug/'):
    cv2.imwrite(path + '{}.png'.format(imgId), self.imgs[imgId])
    
  def save_all_imgs(self, path='./cache/debug/', prefix='', genID=False):
    if genID:
      try:
        idx = int(np.loadtxt(path + '/id.txt'))
      except:
        idx = 0
      prefix=idx
      np.savetxt(path + '/id.txt', np.ones(1) * (idx + 1), fmt='%d')
    for i, v in self.imgs.items():
      cv2.imwrite(path + '/{}{}.png'.format(prefix, i), v)

  def remove_side(self, img_id, img):
    if not (img_id in self.imgs):
      return
    ws = img.sum(axis=2).sum(axis=0)
    l = 0
    while ws[l] == 0 and l < len(ws):
      l+= 1
    r = ws.shape[0] - 1
    while ws[r] == 0 and r > 0:
      r -= 1
    hs = img.sum(axis=2).sum(axis=1)
    t = 0
    while hs[t] == 0 and t < len(hs):
      t += 1
    b = hs.shape[0] - 1
    while hs[b] == 0 and b > 0:
      b -= 1
    self.imgs[img_id] = self.imgs[img_id][t:b+1, l:r+1].copy()

  def project_3d_to_bird(self, pt):
    pt[0] += self.world_size / 2
    pt[1] = self.world_size - pt[1]
    pt = pt * self.out_size / self.world_size
    return pt.astype(np.int32)

  def add_ct_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    if self.imgs.get(img_id) is None:
      self.imgs[img_id] = img.copy()
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            bbox = dets[cat][i, 1:5]
            self.add_coco_bbox(
              bbox, cat - 1, dets[cat][i, -1], 
              show_txt=show_txt, img_id=img_id)
    else:
      assert 0, "add_ct_detection error!"

  def add_kept(
    self, dets, info_3d, center_thresh=0.5, img_id='det'):
    # dets: max_preds x 5
    if type(dets) == type({}):
      for cat in dets:
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            point0 = (int(info_3d[cat][i, 0]), int(dets[cat][i, 4]))
            cv2.circle(self.imgs[img_id], point0, 3, (255, 0, 0), -1) #blue
            point1 = (int(info_3d[cat][i, 1]), int(dets[cat][i, 4]))
            cv2.circle(self.imgs[img_id], point1, 4, (0, 255, 0), -1) #green
            point2 = (int(info_3d[cat][i, 2]), int(dets[cat][i, 4]))
            cv2.circle(self.imgs[img_id], point2, 5, (0, 0, 255), -1) #red
    else:
      assert 0, "add_kept error!"


  def add_3d_detection(
    self, image_or_path, dets, calib, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      if self.imgs.get(img_id) is None:
        self.imgs[img_id] = image_or_path.copy()
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d.astype(np.int32), cl)
  
  def add_init_3d_detection(
    self, image_or_path, dets, info_3d, calibs, show_txt=False, 
    center_thresh=0.5, img_id='det'):
    if isinstance(image_or_path, np.ndarray):
      if self.imgs.get(img_id) is None:
        self.imgs[img_id] = image_or_path.copy()
    else: 
      self.imgs[img_id] = cv2.imread(image_or_path)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat + 3, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          bbox = dets[cat][i, 1:5]
          alpha = dets[cat][i, 0]
          h, w, l = dim[0], dim[1], dim[2] 
          # w, h, l = dim[0], dim[1], dim[2] 
          ul, ur, vt, vb = bbox[0], bbox[2], bbox[1], bbox[3]

          calib = read_obj_calibration(calibs)
          f = calib.p2[0,0]
          cx, cy = calib.p2[0,2], calib.p2[1,2]
          bl = (calib.p2[0,3] - calib.p3[0,3])/f
          x_shift = (calib.p2[0, 3] - calib.p0[0, 3])/f
          y_shift = (calib.p2[1, 3] - calib.p0[1, 3])/f 
          z_shift = (calib.p2[2, 3] - calib.p0[2, 3])/f

          left_u = (ul - cx)/f
          right_u = (ur - cx)/f 
          top_v = (vt - cy)/f
          bottom_v = (vb - cy)/f 

          disparity = info_3d[cat][i, -1]
          z = f*bl/disparity - z_shift
          x = z*(left_u+right_u)/2.0 - x_shift
          y = z*(bottom_v+top_v)/2.0 + h/2.0 - y_shift
          rot_y = alpha - m.atan2(-x, z)
          loc = [x, y, z]

          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib.p2)
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d.astype(np.int32), cl)

  def compose_vis_add(
    self, img_path, dets, calib,
    center_thresh, pred, bev, img_id='out'):
    self.imgs[img_id] = cv2.imread(img_path)
    # h, w = self.imgs[img_id].shape[:2]
    # pred = cv2.resize(pred, (h, w))
    h, w = pred.shape[:2]
    hs, ws = self.imgs[img_id].shape[0] / h, self.imgs[img_id].shape[1] / w
    self.imgs[img_id] = cv2.resize(self.imgs[img_id], (w, h))
    self.add_blend_img(self.imgs[img_id], pred, img_id)
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          # loc[1] = loc[1] - dim[0] / 2 + dim[0] / 2 / self.dim_scale
          # dim = dim / self.dim_scale
          if loc[2] > 1:
            box_3d = compute_box_3d(dim, loc, rot_y)
            box_2d = project_to_image(box_3d, calib)
            box_2d[:, 0] /= hs
            box_2d[:, 1] /= ws
            self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, cl)
    self.imgs[img_id] = np.concatenate(
      [self.imgs[img_id], self.imgs[bev]], axis=1)

  def add_2d_detection(
    self, img, dets, show_box=False, show_txt=True, 
    center_thresh=0.5, img_id='det'):
    self.imgs[img_id] = img
    for cat in dets:
      for i in range(len(dets[cat])):
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        if dets[cat][i, -1] > center_thresh:
          bbox = dets[cat][i, 1:5]
          self.add_coco_bbox(
            bbox, cat - 1, dets[cat][i, -1], 
            show_txt=show_txt, img_id=img_id)

  def add_bird_view(self, dets, center_thresh=0.3, img_id='bird'):
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for cat in dets:
      cl = (self.colors[cat - 1, 0, 0]).tolist()
      lc = (250, 152, 12)
      for i in range(len(dets[cat])):
        if dets[cat][i, -1] > center_thresh:
          dim = dets[cat][i, 5:8]
          loc  = dets[cat][i, 8:11]
          rot_y = dets[cat][i, 11]
          rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
          for k in range(4):
            rect[k] = self.project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
          cv2.polylines(
              bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
              True,lc,2,lineType=cv2.LINE_AA)
          for e in [[0, 1]]:
            t = 4 if e == [0, 1] else 1
            cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                    (rect[e[1]][0], rect[e[1]][1]), lc, t,
                    lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view

  def add_bird_views(self, dets_dt, dets_gt, center_thresh=0.3, img_id='bird'):
    alpha = 0.5
    bird_view = np.ones((self.out_size, self.out_size, 3), dtype=np.uint8) * 230
    for ii, (dets, lc, cc) in enumerate(
      [(dets_gt, (12, 49, 250), (0, 0, 255)), 
       (dets_dt, (250, 152, 12), (255, 0, 0))]):
      # cc = np.array(lc, dtype=np.uint8).reshape(1, 1, 3)
      for cat in dets:
        cl = (self.colors[cat - 1, 0, 0]).tolist()
        for i in range(len(dets[cat])):
          if dets[cat][i, -1] > center_thresh:
            dim = dets[cat][i, 5:8]
            loc  = dets[cat][i, 8:11]
            rot_y = dets[cat][i, 11]
            rect = compute_box_3d(dim, loc, rot_y)[:4, [0, 2]]
            for k in range(4):
              rect[k] = self.project_3d_to_bird(rect[k])
            if ii == 0:
              cv2.fillPoly(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                lc,lineType=cv2.LINE_AA)
            else:
              cv2.polylines(
                bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
                True,lc,2,lineType=cv2.LINE_AA)
            # for e in [[0, 1], [1, 2], [2, 3], [3, 0]]:
            for e in [[0, 1]]:
              t = 4 if e == [0, 1] else 1
              cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
                      (rect[e[1]][0], rect[e[1]][1]), lc, t,
                      lineType=cv2.LINE_AA)
    self.imgs[img_id] = bird_view


kitti_class_name = [
  'Car', 'Van' ,'Truck', 'Misc'
]

color_list = np.array(
        [
            1.000, 1.000, 1.000,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
