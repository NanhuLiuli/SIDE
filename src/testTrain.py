from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from torch.autograd import Variable
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger

from modules.stereoDataset import StereoDataset
from modules.stereoTrainer import StereoTrainer


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  opt = opts().update_dataset_info_and_set_heads(opt, StereoDataset)
  # print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  if opt.uncert:
    opt.lossWeight = torch.rand(7)
    torch.nn.init.constant_(opt.lossWeight, -1)
    opt.lossWeight = torch.nn.Parameter(opt.lossWeight, requires_grad=True)
    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
          params += [{'params':[value]}]
    params += [{'params':[opt.lossWeight]}]

    optimizer = torch.optim.Adam(params, opt.lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, opt.lossWeight, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.lossWeight, opt.resume, opt.lr, opt.lr_step)

  trainer = StereoTrainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      StereoDataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    # val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      StereoDataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer, opt.lossWeight)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer, opt.lossWeight)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer, opt.lossWeight)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch)))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    # elif epoch <= 15:
    #   ori_lr = 1.25e-4
    #   lr = opt.lr + (ori_lr - opt.lr)*(float(epoch)/20.)
    #   print('Drop LR to', lr)
    #   for param_group in optimizer.param_groups:
    #       param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)