from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter

from modules.stereoDataset import StereoDataset
from modules.stereoDetector import StereoDectector
from utils.debugger import Debugger

from utils.stereo_utils import read_obj_calibration

class PrefetchDataset(torch.utils.data.Dataset):
  def __init__(self, opt, dataset, pre_process_func):
    self.images = dataset.images
    self.load_image_func = dataset.coco.loadImgs
    self.img_dir = dataset.img_dir
    self.img_right_dir = dataset.img_right_dir
    self.pre_process_func = pre_process_func
    self.opt = opt
  
  def __getitem__(self, index):
    img_id = self.images[index]
    img_info = self.load_image_func(ids=[img_id])[0]
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img_right_path = os.path.join(self.img_right_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    image_right = cv2.imread(img_right_path)

    calibration = read_obj_calibration(img_info['calib'])
    p2, p3 = calibration.p2, calibration.p3
    f = calibration.p2[0,0]
    bl = (calibration.p2[0,3] - calibration.p3[0,3])/f

    inp, inp_right, meta = self.pre_process_func(image, image_right, img_info['calib'])
        
    return img_id, {'inp': inp, 'inp_right': inp_right, 'fb':f*bl, 'p2':p2, 'p3':p3, \
                        'image': image, 'image_right':image_right, 'calib':img_info['calib'], 'meta': meta}

  def __len__(self):
    return len(self.images)
  
def show_results(opt, image, calib, results, img_id):
  debugger = Debugger(dataset=opt.dataset, ipynb=False,
                      theme=opt.debugger_theme)
  image = image.numpy()[0]
  debugger.add_3d_detection(
    image, results, calib[2],
    center_thresh=opt.vis_thresh, img_id='add_pred')
  debugger.add_bird_view(
    results, center_thresh=opt.vis_thresh, img_id='bird_pred')
  debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(img_id.numpy().astype(np.int32)[0]))

def prefetch_test(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

  opt = opts().update_dataset_info_and_set_heads(opt, StereoDataset)
  Logger(opt)
  
  split = 'val' if not opt.trainval else 'test'
  dataset = StereoDataset(opt, split)
  detector = StereoDectector(opt)
  
  data_loader = torch.utils.data.DataLoader(
    PrefetchDataset(opt, dataset, detector.pre_process), 
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

  results = {}
  num_iters = len(dataset)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}
  for ind, (img_id, pre_processed_images) in enumerate(data_loader):
    ret = detector.run(pre_processed_images, img_id.numpy().astype(np.int32)[0])
    # show_results(opt, pre_processed_images["image"], pre_processed_images["calib"], ret['results'], img_id)

    results[img_id.numpy().astype(np.int32)[0]] = ret['results']
    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {tm.val:.3f}s ({tm.avg:.3f}s) '.format(
        t, tm = avg_time_stats[t])
    bar.next()
    # if ind == 50:
    #   break
  bar.finish()
  # with open("test.txt", 'w') as f:
  #   for k, v in results.items():
  #     f.write(str(k) + '\n')
  #     f.write(str(v) + '\n')
  # print("Done!")
  dataset.run_eval(results, opt.save_dir)

if __name__ == '__main__':
    opt = opts().parse()
    prefetch_test(opt)