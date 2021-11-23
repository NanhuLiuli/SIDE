from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
import numpy as np
import cv2

from opts import opts
from modules.stereoDataset import StereoDataset
from utils.image import transform_preds

def main(opt):
  opt = opts().update_dataset_info_and_set_heads(opt, StereoDataset)
  # print(opt)

  print('Setting up data...')
  ori_samples = StereoDataset(opt, 'val').ori_samples
  val_loader = torch.utils.data.DataLoader(
      StereoDataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  # train_loader = torch.utils.data.DataLoader(
  #     StereoDataset(opt, 'train'), 
  #     batch_size=opt.batch_size, 
  #     shuffle=True,
  #     num_workers=opt.num_workers,
  #     pin_memory=True,
  #     drop_last=True
  # )

  print('Starting training...')
  print(ori_samples)
  for epoch in range(1):
    for iter_id, batch in enumerate(val_loader):
      if iter_id >= 20 and iter_id < ori_samples:
        print(iter_id)
        continue

      img_id = batch['meta']['img_id'][0]
      if iter_id < ori_samples:
        img_left = cv2.imread(batch['meta']['image_path'][0])
      else:
        img_left = cv2.imread(batch['meta']['image_right'][0])
        img_left = img_left[:, ::-1, :].copy()
      # img_left = batch['input'][0].numpy().astype(np.int32)
      c, s = batch['meta']['c'][0].numpy(), batch['meta']['s'][0].numpy()
      gt_det = batch['meta']['gt_det'][0].numpy()
      bbox = np.zeros((gt_det.shape[0], 4), dtype=np.float32)
      bbox[:, :2] = gt_det[:, :2] - 0.5*gt_det[:, 15:17]
      bbox[:, 2:4] = gt_det[:, :2] + 0.5*gt_det[:, 15:17]
      kept = batch['kept'][0][:gt_det.shape[0], :]
      kept = kept.numpy() + bbox[:, :1] 
      bbox[:, :2] = transform_preds(bbox[:, :2], c, s, (opt.output_w, opt.output_h))
      bbox[:, 2:4] = transform_preds(bbox[:, 2:4], c, s, (opt.output_w, opt.output_h))
      kept[:, 4] = transform_preds(np.concatenate([kept[:, 4:5], bbox[:, 3:]], axis=1), c, s, (opt.output_w, opt.output_h))[:, 0]
      kept[:, 5] = transform_preds(np.concatenate([kept[:, 5:], bbox[:, 3:]], axis=1), c, s, (opt.output_w, opt.output_h))[:, 0]
      for i in range(bbox.shape[0]):
        box = bbox[i].astype(np.int32)
        # print(box)
        img_left = cv2.rectangle(img_left, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
        for j in range(4):
          kept[i, j] = transform_preds(np.concatenate([kept[i:i+1, j:j+1], bbox[i:i+1, 3:]], axis=1), c, s, (opt.output_w, opt.output_h))[0, 0]
      #     if kept[i, j] == kept[i, 4] or kept[i, j] == kept[i, 5] or kept[i, j] < 0:
      #       continue
      #     else:
          point = (int(kept[i, j]), int(bbox[i, 3]))
          img_left = cv2.circle(img_left, point, 3, (0, 0, 255), -1) #red
        # print(kept[i].astype(np.int32))
        point1 = (int(kept[i, 4]), int(bbox[i, 3]))
        img_left = cv2.circle(img_left, point1, 4, (255, 0, 0), -1) #blue
        point2 = (int(kept[i, 5]), int(bbox[i, 3]))
        img_left = cv2.circle(img_left, point2, 5, (0, 255, 0), -1) #green
      img_name = "./test/{}.png".format(img_id) if iter_id <= 20 else "./test/{}_flip.png".format(img_id)
      print(iter_id)
      cv2.imwrite(img_name, img_left)
        # cv2.imshow("test", img_left)
        # cv2.waitKey(0)
        
      if iter_id == ori_samples + 20:
        break

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)