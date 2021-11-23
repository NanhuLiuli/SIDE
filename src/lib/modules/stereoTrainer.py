from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
import torch.nn.functional as F

from models.losses import FocalLoss, L1Loss, CrossLoss, SmoothL1Loss
from models.decode import ddd_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ddd_post_process

from progress.bar import Bar
from models.data_parallel import DataParallel
from utils.utils import AverageMeter

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

class ModelWithLoss(torch.nn.Module):
  def __init__(self, model, loss, opt):
    super(ModelWithLoss, self).__init__()
    self.model = model
    self.loss = loss
    self.opt = opt
  
  def forward(self, batch):
    xs, ys = batch['ind_float'] % self.opt.output_w, batch['ind_float'] // self.opt.output_w
    wh, reg = batch['wh'], batch['reg']
    xs_right = xs + reg[:, :, 1]
    xs, ys = xs + reg[:, :, 0], ys + reg[:, :, 2]
    center = torch.cat([xs.unsqueeze(2), ys.unsqueeze(2)], dim=2)
    center_right = torch.cat([xs_right.unsqueeze(2), ys.unsqueeze(2)], dim=2)

    batch_index = torch.tensor([num for num in range(xs.shape[0])], dtype=torch.float32).unsqueeze(1).unsqueeze(2)
    batch_index = batch_index.repeat(1, xs.shape[1], 1)

    bbox = torch.zeros((xs.shape[0], xs.shape[1], 5), dtype=torch.float32)
    bbox[:, :, 1:3] = center - 0.5*wh[:, :, [0, 2]]*self.opt.wh_scale
    bbox[:, :, 3:5] = center + 0.5*wh[:, :, [0, 2]]*self.opt.wh_scale
    bbox_right = torch.zeros((xs.shape[0], xs.shape[1], 5), dtype=torch.float32)
    bbox_right[:, :, 1:3] = center_right - 0.5*wh[:, :, [1, 2]]*self.opt.wh_scale
    bbox_right[:, :, 3:5] = center_right + 0.5*wh[:, :, [1, 2]]*self.opt.wh_scale
    bbox[:, :, 0:1], bbox_right[:, :, 0:1] = batch_index, batch_index

    bbox_keep, bbox_right_keep = bbox.view(-1, 5), bbox_right.view(-1, 5)
    keep = torch.sum(bbox_keep[:, 1:5], dim=1) > 0
    bbox_keep, bbox_right_keep = bbox_keep[keep, :], bbox_right_keep[keep, :]

    outputs = self.model(batch, self.opt.cost_volume, (bbox_keep.cuda(), bbox_right_keep.cuda(), bbox.shape))
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats

class StereoLoss(torch.nn.Module):
  def __init__(self, opt):
    super(StereoLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = L1Loss()
    # self.crit_reg = SmoothL1Loss()
    self.crit_cross = CrossLoss()
    self.lossWeight = opt.lossWeight
    self.opt = opt

  def computeKeptLabel(self, kept, wh):
        assert kept.size(1) == kept.size(1)
        assert wh.size(2) == 3
        assert kept.size(2) == 6

        grid_size = self.opt.grid
        width = wh[:,:,0] + 1
        width = width.unsqueeze(2).expand(-1,-1,6) 
        target = torch.round(kept*grid_size/width) 
        target[target < 0] = -225 
        target[target > grid_size - 1] = -225 
        kpts_pos, kpts_type = torch.max(target[:,:,:4], 2)
        kpts_pos = kpts_pos.unsqueeze(2) 
        kpts_type = kpts_type.unsqueeze(2) 
        target = torch.cat((kpts_type.type(torch.cuda.FloatTensor)*grid_size+kpts_pos,\
                            target[:,:,4:].type(torch.cuda.FloatTensor)),2)
        target[target < 0] = 0
        
        return target.type(torch.cuda.LongTensor)

  def forward(self, outputs, batch):
    hm_loss, wh_loss, off_loss, dim_loss, orien_loss, depth_loss = 0, 0, 0, 0, 0, 0
    kept_type_loss, border_left_loss, border_right_loss = 0, 0, 0
    
    output = outputs[-1]
    if self.opt.cost_volume:
      depth_loss = F.l1_loss(output['depth'], batch['depth'], reduction='mean')
    
    output['hm'] = _sigmoid(output['hm'])
    hm_loss += self.crit(output['hm'], batch['hm'])
    dim_loss += self.crit_reg(output['dim'], batch['rot_mask'], 
                                    batch['ind'], batch['dim'])
    orien_loss += self.crit_reg(output['orien'], batch['rot_mask'], 
                                    batch['ind'], batch['orien'])
    
    target = self.computeKeptLabel(batch['kept'], batch['wh'])
    grid_size = self.opt.grid
    kept_type_loss += self.crit_cross(output['kept_type'][:, :4*grid_size, :, :], batch['rot_mask'], 
                                    batch['ind'], target[:, :, 0].unsqueeze(2))
    border_left_loss += self.crit_cross(output['kept_type'][:, 4*grid_size:5*grid_size, :, :], batch['rot_mask'], 
                                    batch['ind'], target[:, :, 1].unsqueeze(2))
    border_right_loss += self.crit_cross(output['kept_type'][:, 5*grid_size:, :, :], batch['rot_mask'], 
                                    batch['ind'], target[:, :, 2].unsqueeze(2))
    kept_loss = (kept_type_loss + border_left_loss + border_right_loss)/3
  
    wh_loss += self.crit_reg(output['wh'], batch['rot_mask'],
                                batch['ind'], batch['wh'])
    off_loss += self.crit_reg(output['reg'], batch['rot_mask'],
                                batch['ind'], batch['reg'])
    
    loss = torch.Tensor([0.0]).float().cuda()
    if self.opt.uncert:
      loss = hm_loss * torch.exp(-self.lossWeight[0]) + self.lossWeight[0] +\
             wh_loss * torch.exp(-self.lossWeight[1]) + self.lossWeight[1] +\
             off_loss * torch.exp(-self.lossWeight[2]) + self.lossWeight[2] +\
             depth_loss * torch.exp(-self.lossWeight[3]) + self.lossWeight[3] +\
             dim_loss * torch.exp(-self.lossWeight[4]) + self.lossWeight[4]+\
             orien_loss * torch.exp(-self.lossWeight[5]) + self.lossWeight[5]+\
             kept_loss * torch.exp(-self.lossWeight[6]) + self.lossWeight[6]
    else:
      loss = self.lossWeight[0] * hm_loss + self.lossWeight[1] * wh_loss + self.lossWeight[2] * off_loss + \
          self.lossWeight[3] * depth_loss + self.lossWeight[4] * dim_loss  + self.lossWeight[5] * orien_loss + self.lossWeight[6] * kept_loss
           
    loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'wh_loss': wh_loss, 'off_loss': off_loss, 
                  'dim_loss': dim_loss, 'orien_loss': orien_loss, 'kept_loss' : kept_loss}
    if self.opt.cost_volume:
      loss_stats.update({'depth_loss': depth_loss})
    return loss, loss_stats

class StereoTrainer(object):
  def __init__(self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = self._get_losses(opt)
    self.model_with_loss = ModelWithLoss(model, self.loss, self.opt)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss',  'wh_loss', 'off_loss', 'dim_loss', 'orien_loss', 'kept_loss']
    if self.opt.cost_volume:
      loss_states.append('depth_loss')
    loss = StereoLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
      opt = self.opt
      wh = output['wh'] if opt.reg_bbox else None
      reg = output['reg'] if opt.reg_offset else None
      dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], output['orien'], wh=wh, reg=reg, K=opt.K)

      # x, y, score, r1-r8, depth, dim1-dim3, cls
      dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
      calib = batch['meta']['calib'].detach().numpy()
      # x, y, score, rot, depth, dim1, dim2, dim3
      # if opt.dataset == 'gta':
      #   dets[:, 12:15] /= 3
      dets_pred = ddd_post_process(
        dets.copy(), batch['meta']['c'].detach().numpy(), 
        batch['meta']['s'].detach().numpy(), calib, opt)
      dets_gt = ddd_post_process(
        batch['meta']['gt_det'].detach().numpy().copy(),
        batch['meta']['c'].detach().numpy(), 
        batch['meta']['s'].detach().numpy(), calib, opt)
      #for i in range(input.size(0)):
      for i in range(1):
        debugger = Debugger(dataset=opt.dataset, ipynb=(opt.debug==3),
                            theme=opt.debugger_theme)
        img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
        img = ((img * self.opt.std + self.opt.mean) * 255.).astype(np.uint8)
        pred = debugger.gen_colormap(
          output['hm'][i].detach().cpu().numpy())
        gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'hm_pred')
        debugger.add_blend_img(img, gt, 'hm_gt')
        # decode
        debugger.add_ct_detection(
          img, dets[i], show_box=opt.reg_bbox, center_thresh=opt.center_thresh, 
          img_id='det_pred')
        debugger.add_ct_detection(
          img, batch['meta']['gt_det'][i].cpu().numpy().copy(), 
          show_box=opt.reg_bbox, img_id='det_gt')
        debugger.add_3d_detection(
          batch['meta']['image_path'][i], dets_pred[i], calib[i],
          center_thresh=opt.center_thresh, img_id='add_pred')
        debugger.add_3d_detection(
          batch['meta']['image_path'][i], dets_gt[i], calib[i],
          center_thresh=opt.center_thresh, img_id='add_gt')
        # debugger.add_bird_view(
        #   dets_pred[i], center_thresh=opt.center_thresh, img_id='bird_pred')
        # debugger.add_bird_view(dets_gt[i], img_id='bird_gt')
        debugger.add_bird_views(
          dets_pred[i], dets_gt[i], 
          center_thresh=opt.center_thresh, img_id='bird_pred_gt')
        
        # debugger.add_blend_img(img, pred, 'out', white=True)
        debugger.compose_vis_add(
          batch['meta']['image_path'][i], dets_pred[i], calib[i],
          opt.center_thresh, pred, 'bird_pred_gt', img_id='out')
        # debugger.add_img(img, img_id='out')
        if opt.debug ==4:
          debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
        else:
          debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    opt = self.opt
    wh = output['wh'] if opt.reg_bbox else None
    reg = output['reg'] if opt.reg_offset else None
    dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                        output['dim'], output['orien'], wh=wh, reg=reg, K=opt.K)

    # x, y, score, r1-r8, depth, dim1-dim3, cls
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    calib = batch['meta']['calib'].detach().numpy()
    # x, y, score, rot, depth, dim1, dim2, dim3
    dets_pred = ddd_post_process(
      dets.copy(), batch['meta']['c'].detach().numpy(), 
      batch['meta']['s'].detach().numpy(), calib, opt)
    img_id = batch['meta']['img_id'].detach().numpy()[0]
    results[img_id] = dets_pred[0]
    for j in range(1, opt.num_classes + 1):
      keep_inds = (results[img_id][j][:, -1] > opt.center_thresh)
      results[img_id][j] = results[img_id][j][keep_inds]

  def set_device(self, gpus, chunk_sizes, device):
    if len(gpus) > 1:
      self.model_with_loss = DataParallel(
        self.model_with_loss, device_ids=gpus, 
        chunk_sizes=chunk_sizes).to(device)
    else:
      self.model_with_loss = self.model_with_loss.to(device)
    
    for state in self.optimizer.state.values():
      for k, v in state.items():
        if isinstance(v, torch.Tensor):
          state[k] = v.to(device=device, non_blocking=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      # model_with_loss = torch.nn.DataParallel(model_with_loss,device_ids=[0])
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
    bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break
      data_time.update(time.time() - end)

      for k in batch:
        if k != 'meta':
          batch[k] = batch[k].to(device=opt.device, non_blocking=True) 
      output, loss, loss_stats = model_with_loss(batch)
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        # clip_gradient(model_with_loss, 10)
        self.optimizer.step()
      batch_time.update(time.time() - end)
      end = time.time()

      Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
        epoch, iter_id, num_iters, phase=phase,
        total=bar.elapsed_td, eta=bar.eta_td)
      for l in avg_loss_stats:
        avg_loss_stats[l].update(
          loss_stats[l].mean().item(), batch['input'].size(0))
        Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
      uncert_data = opt.lossWeight
      Bar.suffix = Bar.suffix + '|%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' \
                %(uncert_data[0], uncert_data[1], uncert_data[2], uncert_data[3], uncert_data[4], uncert_data[5], uncert_data[6])
      if not opt.hide_data_time:
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
      if opt.print_iter > 0:
        if iter_id % opt.print_iter == 0:
          print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
      else:
        bar.next()
      
      if opt.debug > 0:
        self.debug(batch, output, iter_id)
      
      # if opt.test:
      #   self.save_result(output, batch, results)
      del output, loss, loss_stats
    
    bar.finish()
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    ret['time'] = bar.elapsed_td.total_seconds() / 60.
    return ret, results
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)