from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch


from models.decode import ddd_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ddd_post_process
from utils.debugger import Debugger
from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

from models.model import create_model, load_model

class StereoDectector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    # self.model = torch.nn.DataParallel(self.model, device_ids=[0])
    self.model = self.model.to(opt.device)
    self.model.eval()
    # self.model.train()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True

  def pre_process(self, image, image_right, calib):
    height, width = image.shape[0:2]
    
    inp_height, inp_width = self.opt.input_h, self.opt.input_w
    c = np.array([width / 2, height / 2], dtype=np.float32)
    if self.opt.keep_res:
      s = np.array([inp_width, inp_height], dtype=np.int32)
    else:
      s = np.array([width, height], dtype=np.int32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp = cv2.warpAffine(
      image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...]

    inp_right = cv2.warpAffine(
        image_right, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_right = (inp_right.astype(np.float32) / 255.)
    inp_right = (inp_right - self.mean) / self.std
    inp_right = inp_right.transpose(2, 0, 1)[np.newaxis, ...]

    inp = torch.from_numpy(inp)
    inp_right = torch.from_numpy(inp_right)

    trans = get_affine_transform(
      c,  s, 0, [self.opt.output_w, self.opt.output_h])
    trans_inv = get_affine_transform(
      c,  s, 0, [self.opt.output_w, self.opt.output_h], inv=1)

    meta = {'c': c, 's': s, 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio,
            'calib': calib, 'trans': trans, 'trans_inv':trans_inv}
    return inp, inp_right, meta
  
  def process(self, batch, return_time=False):
    with torch.no_grad():
      torch.cuda.synchronize()
      output = self.model(batch, useCostVolume=self.opt.cost_volume, wh_scale=self.opt.wh_scale)[-1]
      output['hm'] = output['hm'].sigmoid_()
      wh = output['wh'] if self.opt.reg_bbox else None
      reg = output['reg'] if self.opt.reg_offset else None
      torch.cuda.synchronize()
      forward_time = time.time()
      
      dets, dets_right, info_3d = ddd_decode(output['hm'], output['kept_type'], output['dim'], output['orien'], \
                                          wh=wh, reg=reg, grid_size=self.opt.grid, K=self.opt.K)
      if self.opt.cost_volume:
        depth = output['depth']
        info_3d = torch.cat([info_3d, depth], dim=2)
    
    if return_time:
      return output, dets, dets_right, info_3d, forward_time
    else:
      return output, dets, dets_right, info_3d

  def post_process(self, dets, dets_right, info_3d, meta, images, images_right):
    dets = dets.detach().cpu().numpy()
    dets_right = dets_right.detach().cpu().numpy()
    info_3d = info_3d.detach().cpu().numpy()
    
    detections, info_3d = ddd_post_process(
      dets.copy(), dets_right.copy(), info_3d.copy(), \
      meta['c'].numpy(), meta['s'].numpy(), [meta['calib']], \
      self.opt, images, images_right)
    
    self.this_calib = meta['calib'][2]
    return detections[0], info_3d[0]

  def merge_outputs(self, detections):
    results = detections[0]
    for j in range(1, self.num_classes + 1):
      if len(results[j] > 0):
        keep_inds = (results[j][:, -1] > self.opt.peak_thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, inp, output, image, dets, info_3d):
    # dets = dets.detach().cpu().numpy()
    img = inp[0].detach().cpu().numpy().transpose(1, 2, 0)
    img = ((img * self.std + self.mean) * 255).astype(np.uint8)
    pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
    # debugger.add_blend_img(img, pred, 'pred_hm')
    debugger.add_ct_detection(
      image, dets, show_box=self.opt.reg_bbox, 
      center_thresh=self.opt.vis_thresh, img_id='add_pred')
    debugger.add_kept(dets, info_3d, center_thresh=self.opt.vis_thresh, img_id='add_pred')
  
  def show_results(self, debugger, image, results, image_id):
    debugger.add_3d_detection(
      image, results, self.this_calib,
      center_thresh=self.opt.vis_thresh, img_id='add_pred')
    debugger.add_bird_view(
      results, center_thresh=self.opt.vis_thresh, img_id='bird_pred')
    # debugger.show_all_imgs(pause=self.pause)
    debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(image_id))

  def run(self, image_or_path_or_tensor, image_id, meta_calib=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor[0]
      image_right = image_or_path_or_tensor[1]
    elif type(image_or_path_or_tensor) == type (list()) and type(image_or_path_or_tensor[0]) == type (''): 
      image = cv2.imread(image_or_path_or_tensor[0])
      image_right = cv2.imread(image_or_path_or_tensor[1])
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      image_right = image_or_path_or_tensor['image_right'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    
    scale_start_time = time.time()
    if not pre_processed:
      inp, inp_right, meta = self.pre_process(image, image_right, meta_calib)
    else:
      # import pdb; pdb.set_trace()
      inp = pre_processed_images['inp'][0]
      inp_right = pre_processed_images['inp_right'][0]
      meta = pre_processed_images['meta']
    
    fb = pre_processed_images['fb'].to(self.opt.device)
    p2 = pre_processed_images['p2'].to(self.opt.device)
    p3 = pre_processed_images['p3'].to(self.opt.device)
    trans = meta['trans'].to(self.opt.device)
    trans_inv = meta['trans_inv'].to(self.opt.device)
    inp = inp.to(self.opt.device)
    inp_right = inp_right.to(self.opt.device)

    torch.cuda.synchronize()
    pre_process_time = time.time()
    pre_time += pre_process_time - scale_start_time
    batch = {'input':inp, 'input_right':inp_right, \
              'fb':fb, 'p2':p2, 'p3':p3, 'trans':trans, 'trans_inv':trans_inv}
    output, dets, dets_right, info_3d, forward_time = self.process(batch, return_time=True)

    torch.cuda.synchronize()
    net_time += forward_time - pre_process_time
    decode_time = time.time()
    dec_time += decode_time - forward_time
    
    dets, info_3d = self.post_process(dets, dets_right, info_3d, meta, image, image_right) # TODO
    torch.cuda.synchronize()
    post_process_time = time.time()
    post_time += post_process_time - decode_time

    detections.append(dets)
    
    if self.opt.debug >= 2:
      self.debug(debugger, inp, output, image, dets, info_3d)
    
    results = self.merge_outputs(detections)
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, image_id)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}