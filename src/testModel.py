from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from modules.stereoDataset import StereoDataset


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  opt = opts().update_dataset_info_and_set_heads(opt, StereoDataset)

  dummy_input = torch.rand(1, 3, 384, 384).to(torch.device('cuda'))
  print('Creating model...')
  # with SummaryWriter(comment='stereo') as s:
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  model = model.to(torch.device('cuda'))
  # s.add_graph(model, (dummy_input, dummy_input), verbose=True)
  output = model(dummy_input, dummy_input)
  # print(model)
  print(output[0].keys())
  for key in output[0].keys():
    print(key + " size :")
    print(output[0][key].size())
  print("done!")
  

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)