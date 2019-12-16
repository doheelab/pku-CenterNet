from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
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
                 epoch, model, optimizer)
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
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

import argparse

if __name__ == '__main__':
  #opt = opts().parse()
  opt=argparse.Namespace(K=100, aggr_weight=0.0, agnostic_ex=False, arch='dla_34', aug_ddd=0.5, aug_rot=0, batch_size=8, cat_spec_wh=False, center_thresh=0.1, chunk_sizes=[7], data_dir='/home/dohee/kaggle/CenterNet/src/lib/../../data', dataset='kitti', debug=0, debug_dir='/home/dohee/kaggle/CenterNet/src/lib/../../exp/ddd/3dop/debug', debugger_theme='white', demo='', dense_hp=False, dense_wh=False, dep_weight=1, dim_weight=1, down_ratio=4, eval_oracle_dep=False, eval_oracle_hm=False, eval_oracle_hmhp=False, eval_oracle_hp_offset=False, eval_oracle_kps=False, eval_oracle_offset=False, eval_oracle_wh=False, exp_dir='/home/dohee/kaggle/CenterNet/src/lib/../../exp/ddd', exp_id='3dop', fix_res=True, flip=0.5, flip_test=False, gpus=[0], gpus_str='0', head_conv=256, hide_data_time=False, hm_hp=True, hm_hp_weight=1, hm_weight=1, hp_weight=1, input_h=-1, input_res=-1, input_w=-1, keep_res=False, kitti_split='3dop', load_model='', lr=0.000125, lr_step=[45, 60], master_batch_size=7, metric='loss', mse_loss=False, nms=False, no_color_aug=False, norm_wh=False, not_cuda_benchmark=False, not_hm_hp=False, not_prefetch_test=False, not_rand_crop=False, not_reg_bbox=False, not_reg_hp_offset=False, not_reg_offset=False, num_epochs=70, num_iters=-1, num_stacks=1, num_workers=4, off_weight=1, pad=31, peak_thresh=0.2, print_iter=0, rect_mask=False, reg_bbox=True, reg_hp_offset=True, reg_loss='l1', reg_offset=True, resume=False, root_dir='/home/dohee/kaggle/CenterNet/src/lib/../..', rot_weight=1, rotate=0, save_all=False, save_dir='/home/dohee/kaggle/CenterNet/src/lib/../../exp/ddd/3dop', scale=0.4, scores_thresh=0.1, seed=317, shift=0.1, task='ddd', test=False, test_scales=[1.0], trainval=False, val_intervals=5, vis_thresh=0.3, wh_weight=0.1)
  main(opt)