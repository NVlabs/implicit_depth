import os
import os.path as osp
from glob import glob
import shutil
import time
import json
import random
import pickle
import csv
import importlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

import datasets.cleargrasp_synthetic_dataset as cleargrasp_syn
import datasets.cleargrasp_dataset as cleargrasp
import datasets.omniverse_dataset as omniverse
import datasets.mixed_dataset as mixed_dataset
import constants
import models.pipeline as pipeline
import utils.point_utils as point_utils
import utils.optimizer_utils as optimizer_utils
import utils.vis_utils as vis_utils
from utils.training_utils import *

class TrainRefine(object):
    def __init__(self, opt):
        super(TrainRefine, self).__init__()
        self.opt = opt
        if self.opt.dist.ddp:
            print('Use GPU {} in Node {} for training'.format(self.opt.gpu_id, self.opt.dist.node_rank))
        # set device as local gpu id.
        torch.cuda.set_device(self.opt.gpu_id)
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_id))
        self.setup_misc()
        if self.opt.dist.ddp:
            dist.barrier()
        self.setup_model()
        self.setup_data()
        # sync all processes at the end of init
        if self.opt.dist.ddp:
            dist.barrier()

    def setup_model(self):
        print('===> Building models, GPU {}'.format(self.opt.gpu_id))
        self.lidf = pipeline.LIDF(self.opt, self.device)
        self.refine_net = pipeline.RefineNet(self.opt, self.device)
        

        if self.opt.lidf_ckpt_path is not None and osp.isfile(self.opt.lidf_ckpt_path):
            loc = 'cuda:{}'.format(self.opt.gpu_id)
            checkpoint = torch.load(self.opt.lidf_ckpt_path, map_location=loc)
            restore(self.lidf.resnet_model, checkpoint['resnet_model'])
            restore(self.lidf.pnet_model, checkpoint['pnet_model'])
            restore(self.lidf.offset_dec, checkpoint['offset_dec'])
            restore(self.lidf.prob_dec, checkpoint['prob_dec'])
            print('Loaded checkpoint at epoch {} from {}.'.format(checkpoint['epoch'], self.opt.lidf_ckpt_path))
        else:
            raise ValueError('LIDF should be pretrained!')

        # freeze lidf
        for param in self.lidf.parameters():
             param.requires_grad = False

        
        # optimizer for training
        if self.opt.exp_type in ['train', 'valid']:
            model_params = list(self.refine_net.pnet_model.parameters()) + list(self.refine_net.offset_dec.parameters())
            self.optimizer = getattr(optimizer_utils, self.opt.training.optimizer_name)(model_params, lr=self.opt.training.lr)
            # lr scheduler
            if self.opt.training.scheduler_name == 'StepLR':
                self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.opt.training.nepoch_decay, 
                                    gamma=self.opt.training.decay_gamma)
            else:
                raise NotImplementedError('Does not support scheduler type: {}'.format(self.opt.training.scheduler_name))
            print('optimizer at GPU {}'.format(self.opt.gpu_id))

        # load checkpoint
        if self.opt.checkpoint_path is not None and osp.isfile(self.opt.checkpoint_path):
            loc = 'cuda:{}'.format(self.opt.gpu_id)
            checkpoint = torch.load(self.opt.checkpoint_path, map_location=loc)
            restore(self.refine_net.pnet_model, checkpoint['pnet_model_refine'])
            restore(self.refine_net.offset_dec, checkpoint['offset_dec_refine'])
            print('Loaded checkpoint at epoch {} from {}.'.format(checkpoint['epoch'], self.opt.checkpoint_path))
        if self.opt.exp_type in ['test'] and self.opt.checkpoint_path is None:
            raise ValueError('Should identify checkpoint_path for testing!')

        if self.opt.exp_type in ['train', 'valid']:
            # only process 0 handles ckpt save, so only process 0 can have min_*_error
            if self.opt.gpu_id == 0:
                self.min_err, self.max_acc, self.min_angle_err = 1e5, -1, 1e5
                print("=> Setup min error for GPU {}".format(self.opt.gpu_id))
        # resume training
        if self.opt.exp_type == 'train':
            self.start_epoch = 0
            self.opt.resume = osp.join(self.ckpt_dir, self.opt.resume)
            if self.opt.resume is not None and osp.isfile(self.opt.resume):
                loc = 'cuda:{}'.format(self.opt.gpu_id)
                checkpoint = torch.load(self.opt.resume, map_location=loc)
                restore(self.refine_net.pnet_model, checkpoint['pnet_model_refine'])
                restore(self.refine_net.offset_dec, checkpoint['offset_dec_refine'])
                if 'epoch' in checkpoint.keys():
                    self.start_epoch = checkpoint['epoch']+1
                if 'optimizer' in checkpoint.keys():
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                if self.opt.gpu_id == 0:
                    if 'min_err' in checkpoint.keys():
                        self.min_err = checkpoint['min_err']
                    if 'max_acc' in checkpoint.keys():
                        self.max_acc = checkpoint['max_acc']
                    if 'min_angle_err' in checkpoint.keys():
                        self.min_angle_err = checkpoint['min_angle_err']
                    print("=> Loaded min error for GPU {} at loc {}".format(self.opt.gpu_id, loc))
                print("=> Continue Training from '{}' for GPU {} at loc {}".format(self.opt.resume, self.opt.gpu_id, loc))

        # ddp setting
        if self.opt.dist.ddp:
            # batchnorm to syncbatchnorm
            self.refine_net = nn.SyncBatchNorm.convert_sync_batchnorm(self.refine_net)
            print('sync batchnorm at GPU {}'.format(self.opt.gpu_id))
            # distributed data parallel
            self.refine_net = nn.parallel.DistributedDataParallel(self.refine_net, device_ids=[self.opt.gpu_id], find_unused_parameters=True)
            print('DistributedDataParallel at GPU {}'.format(self.opt.gpu_id))


    def setup_data(self):
        print('===> Setup data loader')
        # prepare dataset and loader
        params = {
            'img_width': self.opt.dataset.img_width,
            'img_height': self.opt.dataset.img_height,
            'use_data_augmentation': self.opt.dataset.use_data_augmentation,
            'split_ratio': self.opt.dataset.split_ratio,
            'omni_corrupt_all': self.opt.dataset.omni_corrupt_all,
            'gamma_shape': self.opt.dataset.gamma_shape,
            'gamma_scale': self.opt.dataset.gamma_scale,
            'gaussian_scale': self.opt.dataset.gaussian_scale,
            'gp_rescale_factor': self.opt.dataset.gp_rescale_factor,
            'ellipse_dropout_mean': self.opt.dataset.ellipse_dropout_mean,
            'ellipse_gamma_shape': self.opt.dataset.ellipse_gamma_shape,
            'ellipse_gamma_scale': self.opt.dataset.ellipse_gamma_scale,
            'corrupt_table': self.opt.dataset.corrupt_table,
            'depth_aug': self.opt.dataset.depth_aug,
            'corrupt_all_pix': self.opt.dataset.corrupt_all_pix,
            'max_depth': self.opt.dataset.max_depth,
        }
        get_dataloader = lambda dataset: DataLoader(dataset, 
                            batch_size=self.opt.training.valid_batch_size,
                            shuffle=False,
                            num_workers=self.opt.training.num_workers,
                            pin_memory=self.opt.training.pin_memory,
                        )
        # TRAIN
        if self.opt.exp_type == 'train':
            if self.opt.dataset.type == 'cleargrasp':
                train_dataset = cleargrasp_syn.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='train')
            elif self.opt.dataset.type == 'omniverse':
                train_dataset = omniverse.get_dataset(self.opt.dataset.omniverse_root_dir, params, exp_type='train')
            elif self.opt.dataset.type == 'mixed':
                train_dataset = mixed_dataset.get_dataset(self.opt.dataset.cleargrasp_root_dir, 
                                    self.opt.dataset.omniverse_root_dir, params, exp_type='train')
            
            # set train loader
            if self.opt.dist.ddp:
                train_batch_size = int(self.opt.training.batch_size / self.opt.dist.ngpus_per_node)
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            else:
                train_batch_size = self.opt.training.batch_size
                self.train_sampler = None
            self.train_data_loader = DataLoader(train_dataset,
                                batch_size=train_batch_size,
                                shuffle=(self.train_sampler is None),
                                num_workers=self.opt.training.num_workers,
                                pin_memory=self.opt.training.pin_memory,
                                sampler=self.train_sampler,
                                drop_last=True,
                            )
            print('train_data_loader at GPU {}, BN: {}, WORKERS: {}'.format(
                    self.opt.gpu_id, self.train_data_loader.batch_size, self.train_data_loader.num_workers))
            ''' Valid set, only for process 0'''
            if self.opt.gpu_id == 0:
                # prepare for valid only
                cg_syn_novel_test_dataset = cleargrasp_syn.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='valid', obj_type='novel')
                self.cg_syn_novel_test_data_loader = get_dataloader(cg_syn_novel_test_dataset)
                cg_syn_known_test_dataset = cleargrasp_syn.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='valid', obj_type='known')
                self.cg_syn_known_test_data_loader = get_dataloader(cg_syn_known_test_dataset)

                cg_real_novel_test_dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='valid', obj_type='novel')
                self.cg_real_novel_test_data_loader = get_dataloader(cg_real_novel_test_dataset)
                cg_real_known_test_dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='valid', obj_type='known')
                self.cg_real_known_test_data_loader = get_dataloader(cg_real_known_test_dataset)
        
        # TEST
        elif self.opt.exp_type == 'test':
            if self.opt.dataset.type == 'cleargrasp':
                syn_novel_test_dataset = cleargrasp_syn.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='test', obj_type='novel')
                self.syn_novel_test_data_loader = get_dataloader(syn_novel_test_dataset)
                syn_known_test_dataset = cleargrasp_syn.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='test', obj_type='known')
                self.syn_known_test_data_loader = get_dataloader(syn_known_test_dataset)

                real_novel_test_dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='test', obj_type='novel')
                self.real_novel_test_data_loader = get_dataloader(real_novel_test_dataset)
                real_known_test_dataset = cleargrasp.get_dataset(self.opt.dataset.cleargrasp_root_dir, params, exp_type='test', obj_type='known')
                self.real_known_test_data_loader = get_dataloader(real_known_test_dataset)



    def setup_misc(self):
        ''' Only process 0 will handle This part '''
        print('===> Setup miscs')
        # prepare log name
        if self.opt.log_name is None:
            print('Using default log name')
            # training setting
            self.opt.log_name = 'refine_bn{}_lr{}'.format(self.opt.training.batch_size, self.opt.training.lr)
            self.opt.log_name += '_nepo{}'.format(self.opt.training.nepochs)
            self.opt.log_name += '_{}'.format(self.opt.training.optimizer_name)
            # grid setting
            self.opt.log_name += '_gres{}'.format(self.opt.grid.res)
            self.opt.log_name += '_msn{}_vsn{}'.format(self.opt.grid.miss_sample_num, self.opt.grid.valid_sample_num)
            if self.opt.grid.offset_range[0] != 0 or self.opt.grid.offset_range[1] != 1:
                self.opt.log_name += '_os{}_oe{}'.format(self.opt.grid.offset_range[0], self.opt.grid.offset_range[1])
            # refine setting
            self.opt.log_name += '_refine'
            self.opt.log_name += '_forward{}'.format(self.opt.refine.forward_times)
            if self.opt.refine.perturb:
                self.opt.log_name += '_perturb{}'.format(self.opt.refine.perturb_prob)
            if self.opt.refine.pnet_pos_type != 'rel':
                self.opt.log_name += '_validAbs' 
            if not self.opt.refine.pos_encode:
                self.opt.log_name += '_noPosEnc'
            if self.opt.refine.intersect_pos_type == 'rel':
                self.opt.log_name += '_intersectRel'
            self.opt.log_name += '_s{}_e{}'.format(self.opt.refine.offset_range[0], self.opt.refine.offset_range[1])
            # loss setting
            self.opt.log_name += '_prob_{}'.format(self.opt.loss.prob_loss_type)
            if self.opt.loss.surf_norm_w > 0:
                self.opt.log_name += '_sn{}_epo{}'.format(self.opt.loss.surf_norm_w, self.opt.loss.surf_norm_epo)
            if self.opt.loss.smooth_w > 0:
                self.opt.log_name += '_smooth{}_epo{}'.format(self.opt.loss.smooth_w, self.opt.loss.smooth_epo)
            if self.opt.loss.hard_neg:
                self.opt.log_name += '_hardneg'
            # dataset_setting
            self.opt.log_name += '_{}'.format(self.opt.dataset.type)
            if self.opt.custom_postfix != '':
                self.opt.log_name += '_' + self.opt.custom_postfix

        # prepare directory
        self.ckpt_dir = osp.join(self.opt.base_log_dir, 'ckpt', self.opt.log_name)
        self.result_dir = osp.join(self.opt.base_log_dir, 'result', self.opt.log_name)
        if self.opt.gpu_id != 0:
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # meters to record stats for validation and testing
        self.train_loss = AverageValueMeter()
        self.val_loss = AverageValueMeter()
        self.xyz_err = AverageValueMeter()
        self.end_acc = AverageValueMeter()
        self.angle_err = AverageValueMeter()
        self.a1 = AverageValueMeter()
        self.a2 = AverageValueMeter()
        self.a3 = AverageValueMeter()
        self.rmse = AverageValueMeter()
        self.rmse_log = AverageValueMeter()
        self.abs_rel = AverageValueMeter()
        self.mae = AverageValueMeter()
        self.sq_rel = AverageValueMeter()

        # vis dir
        vis_dir = create_dir(osp.join(self.result_dir, '{}_vis'.format(self.opt.exp_type)))
        setattr(self, '{}_vis_dir'.format(self.opt.exp_type), vis_dir)
        # log path
        log_path = osp.join(self.result_dir, '{}_log.txt'.format(self.opt.exp_type))
        setattr(self, 'log_path', log_path)

        if self.opt.exp_type in ['train', 'valid']:
            self.valid_log_path = osp.join(self.result_dir, 'valid_log.txt')

        # write config to file
        if self.opt.exp_type == 'train':
            self.valid_vis_dir = create_dir(osp.join(self.result_dir, 'valid_vis'))
            all_keys = self.opt.get_all_keys(all_keys={}, dic=self.opt.dict, parent_key='')
            with open(osp.join(self.result_dir, 'config.txt'), 'w') as file:
                for k,v in all_keys.items():
                    file.write('{}: {}\n'.format(k,v))
            with open(osp.join(self.ckpt_dir, 'config.txt'), 'w') as file:
                for k,v in all_keys.items():
                    file.write('{}: {}\n'.format(k,v))
        
    def train_epoch(self):
        for epoch in range(self.start_epoch, self.opt.training.nepochs):
            debug_print("=> Epoch {} for GPU {}".format(epoch, self.opt.gpu_id), self.opt.debug)
            if self.opt.dist.ddp:
                self.train_sampler.set_epoch(epoch)
            # train
            self.train(self.train_data_loader, 'train', epoch)

            # valid and logging is only done by process 0
            if self.opt.gpu_id == 0:
                if self.opt.training.do_valid:
                    with open(self.valid_log_path, 'a') as file:
                        file.write('Epoch[{}]:\n'.format(epoch))
                    self.validate(self.cg_syn_known_test_data_loader, 'valid', epoch, 'cg_syn_known')
                    self.validate(self.cg_syn_novel_test_data_loader, 'valid', epoch, 'cg_syn_novel')
                    self.validate(self.cg_real_known_test_data_loader, 'valid', epoch, 'cg_real_known')
                    self.validate(self.cg_real_novel_test_data_loader, 'valid', epoch, 'cg_real_novel')
                # save ckpt and write log
                self.save_ckpt_and_log(epoch)
            # lr scheduler
            self.scheduler.step()

    def save_ckpt_and_log(self, epoch):
        if self.xyz_err.avg < self.min_err:
            self.min_err = self.xyz_err.avg
        if self.end_acc.avg > self.max_acc:
            self.max_acc = self.end_acc.avg
        if self.angle_err.avg < self.min_angle_err:
            self.min_angle_err = self.angle_err.avg
        # dump stats in log file
        log_table = {
            'epoch' : epoch,
            'train_loss' : self.train_loss.avg,
            'val_loss' : self.val_loss.avg,
            'xyz_err': self.xyz_err.avg,
            'min_err': self.min_err,
            'end_acc': self.end_acc.avg,
            'max_acc': self.max_acc,
            'angle_err': self.angle_err.avg,
            'min_angle_err': self.min_angle_err,
            'a1': self.a1.avg,
            'a2': self.a2.avg,
            'a3': self.a3.avg,
            'rmse': self.rmse.avg,
            'rmse_log': self.rmse_log.avg,
            'abs_rel': self.abs_rel.avg,
            'mae': self.mae.avg,
            'sq_rel': self.sq_rel.avg,
        }
        with open(self.log_path, 'a') as file:
            if self.opt.exp_type == 'train':
                file.write(json.dumps(log_table))
                file.write('\n')

        # save checkpoints
        if self.opt.dist.ddp:
            pnet_refine_state_dict = self.refine_net.module.pnet_model.state_dict()
            offset_dec_refine_state_dict = self.refine_net.module.offset_dec.state_dict()
        else:
            pnet_refine_state_dict = self.refine_net.pnet_model.state_dict()
            offset_dec_refine_state_dict = self.refine_net.offset_dec.state_dict()

        ckpt_dict = log_table.copy()
        ckpt_dict.update({
            'pnet_model_refine': pnet_refine_state_dict,
            'offset_dec_refine': offset_dec_refine_state_dict,
            'optimizer': self.optimizer.state_dict(),
        })

        torch.save(ckpt_dict, osp.join(self.ckpt_dir,'latest_network.pth'))
        if epoch % self.opt.training.nepoch_ckpt == 0:
            torch.save(ckpt_dict, osp.join(self.ckpt_dir,'epoch{:03d}_network.pth'.format(epoch)))

    def run_iteration(self, epoch, iteration, iter_len, exp_type, vis_iter, batch):
        pred_mask = None
        success_flag, data_dict, loss_dict = self.lidf(batch, exp_type, epoch, pred_mask)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([success_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                self.optimizer.zero_grad()
                return
        elif not success_flag:
            if exp_type == 'train':
                self.optimizer.zero_grad()
            return

        # 2nd stage forward pass
        data_dict, loss_dict_refine = self.refine_net(exp_type, epoch, data_dict)

        # Backward pass
        if exp_type == 'train':
            self.optimizer.zero_grad()
            loss_dict_refine['loss_net'].backward()
            self.optimizer.step()

        # Reduction across GPUs
        if exp_type == 'train'  and self.opt.dist.ddp:
            reduced_loss_net = reduce_tensor(loss_dict_refine['loss_net'], reduction='mean')
            reduced_pos_loss = reduce_tensor(loss_dict_refine['pos_loss'], reduction='mean')
            reduced_prob_loss = reduce_tensor(loss_dict['prob_loss'], reduction='mean')
            reduced_surf_norm_loss = reduce_tensor(loss_dict_refine['surf_norm_loss'], reduction='mean')
            reduced_smooth_loss = reduce_tensor(loss_dict_refine['smooth_loss'], reduction='mean')
            reduced_err = reduce_tensor(loss_dict_refine['err'], reduction='mean')
            reduced_acc = reduce_tensor(loss_dict['acc'], reduction='mean')
            reduced_angle_err = reduce_tensor(loss_dict_refine['angle_err'], reduction='mean')
        
        # Metrics update
        if exp_type != 'train':
            self.val_loss.update(loss_dict_refine['loss_net'].item())
            self.xyz_err.update(loss_dict_refine['err'].item())
            self.end_acc.update(loss_dict['acc'].item())
            self.angle_err.update(loss_dict_refine['angle_err'].item())
            self.a1.update(loss_dict_refine['a1'].item())
            self.a2.update(loss_dict_refine['a2'].item())
            self.a3.update(loss_dict_refine['a3'].item())
            self.rmse.update(loss_dict_refine['rmse'].item())
            self.rmse_log.update(loss_dict_refine['rmse_log'].item())
            self.abs_rel.update(loss_dict_refine['abs_rel'].item())
            self.mae.update(loss_dict_refine['mae'].item())
            self.sq_rel.update(loss_dict_refine['sq_rel'].item())
        elif self.opt.gpu_id == 0:
            if self.opt.dist.ddp:
                self.train_loss.update(reduced_loss_net.item())
            else:
                self.train_loss.update(loss_dict_refine['loss_net'].item())

        # Logging
        log_cond1 = (iteration % self.opt.training.log_interval == 0) and (exp_type != 'train')
        log_cond2 = (iteration % self.opt.training.log_interval == 0) and (self.opt.gpu_id == 0)
        if log_cond1:
            err_log = '===> Epoch[{}]({}/{}): rmse: {:.3f}, abs_rel: {:.3f}, mae: {:.3f}, a1: {:.2f}, a2: {:.2f}, a3: {:.2f}'.format(
                    epoch, iteration, iter_len, 
                    loss_dict_refine['rmse'].item(), loss_dict_refine['abs_rel'].item(), loss_dict_refine['mae'].item(), 
                    loss_dict_refine['a1'].item()*100, loss_dict_refine['a2'].item()*100, loss_dict_refine['a3'].item()*100)
            print(err_log)
        elif log_cond2:
            # set loss for log
            if exp_type == 'train' and self.opt.dist.ddp:
                loss_net_4log = reduced_loss_net.item()
                pos_loss_4log = reduced_pos_loss.item()
                prob_loss_4log = reduced_prob_loss.item()
                surf_norm_loss_4log = reduced_surf_norm_loss.item()
                smooth_loss_4log = reduced_smooth_loss.item()
                err_4log = reduced_err.item()
                acc_4log = reduced_acc.item()
                angle_err_4log = reduced_angle_err.item()
            else:
                loss_net_4log = loss_dict_refine['loss_net'].item()
                pos_loss_4log = loss_dict_refine['pos_loss'].item()
                prob_loss_4log = loss_dict['prob_loss'].item()
                surf_norm_loss_4log = loss_dict_refine['surf_norm_loss'].item()
                smooth_loss_4log = loss_dict_refine['smooth_loss'].item()
                err_4log = loss_dict_refine['err'].item()
                acc_4log = loss_dict['acc'].item()
                angle_err_4log = loss_dict_refine['angle_err'].item()
            log_info = '===> Epoch[{}]({}/{}):  Loss: {:.5f}, Pos Loss: {:.5f}, Prob Loss: {:.5f}, Surf Loss: {:.5f}, Smooth Loss: {:.5f}'.format(
                    epoch, iteration, iter_len, 
                    loss_net_4log, pos_loss_4log, prob_loss_4log,
                    surf_norm_loss_4log, smooth_loss_4log)
            log_info += ', Err: {:.5f}, Acc: {:.5f}, Angle Err: {:.5f}'.format(
                    err_4log, acc_4log, angle_err_4log)
            print(log_info)

        # Visualization
        vis_bid = 0
        vis_cond1 = (exp_type != 'test') and (self.opt.gpu_id == 0) and (iteration % (iter_len//vis_iter-1) == 0)
        vis_cond2 = (exp_type == 'test') and (iteration % vis_iter == 0)
        if vis_cond1 or vis_cond2:
            result_dir = getattr(self, '{}_vis_dir'.format(exp_type))
            self.visualize(data_dict, exp_type, epoch, iteration, vis_bid, result_dir)

        # write detailed err for test
        if exp_type == 'test':
            with open(self.csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.field_names, delimiter=',')
                row_data = [
                    iteration, round(loss_dict_refine['rmse'].item(),3), round(loss_dict_refine['abs_rel'].item(),3), round(loss_dict_refine['mae'].item(),3),
                    round(loss_dict_refine['a1'].item()*100,2), round(loss_dict_refine['a2'].item()*100,2), round(loss_dict_refine['a3'].item()*100,2)
                ]
                writer.writerow(dict(zip(self.field_names, row_data)))

    def visualize(self, data_dict, exp_type, epoch, iteration, vis_bid, result_dir):
        # untransform and save rgb
        rgb_img = data_dict['rgb_img'][vis_bid].detach().cpu().numpy() # (3,h,w), float, (0,1)
        mean=np.array(constants.IMG_MEAN).reshape(-1,1,1)
        std=np.array(constants.IMG_NORM).reshape(-1,1,1)
        rgb_img = rgb_img * std + mean
        rgb_img = np.transpose(rgb_img,(1,2,0))*255.0
        rgb_img = rgb_img.astype(np.uint8) # (h,w,3), uint8, (0,255)
        color = rgb_img.reshape(-1,3)

        # undo transform of pred mask
        corrupt_mask_img = data_dict['corrupt_mask'][vis_bid].cpu().numpy()
        corrupt_mask_img = (corrupt_mask_img*255).astype(np.uint8)
        # undo transform of gt mask
        valid_mask_img = data_dict['valid_mask'][vis_bid].cpu().numpy()
        valid_mask_img = (valid_mask_img*255).astype(np.uint8)

        # Setup a figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        axes = axes.flat
        for ax in axes:
            ax.axis("off")
        # Inp RGB
        axes[0].set_title("Inp RGB")
        axes[0].imshow(rgb_img)
        # Corrupt Mask
        axes[1].set_title("Corrupt Mask")
        axes[1].imshow(corrupt_mask_img, cmap='gray', vmin=0, vmax=255)
        # Valid Mask
        axes[2].set_title("Valid Mask")
        axes[2].imshow(valid_mask_img, cmap='gray', vmin=0, vmax=255)
        # Save figure
        plt.tight_layout()
        # name setting
        dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_img.png'.format(epoch, iteration))
        plt.savefig(dst_path)
        plt.close(fig)
        
        # save inp pcl
        vis_inp_pcl = data_dict['xyz_corrupt_flat'].clone()
        vis_inp_pcl = vis_inp_pcl[vis_bid].detach().cpu().numpy()
        dst_path = os.path.join(result_dir, 'epo{:03d}_iter{:05d}_inp.ply'.format(epoch, iteration))
        vis_utils.save_point_cloud(vis_inp_pcl, color, dst_path)
        
        # save gt pcl
        vis_gt_pcl = data_dict['xyz_corrupt_flat'].clone()
        vis_gt_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['gt_pos']
        vis_gt_pcl = vis_gt_pcl[vis_bid].detach().cpu().numpy()
        dst_path = os.path.join(result_dir, 'epo{:03d}_iter{:05d}_gt.ply'.format(epoch, iteration))
        vis_utils.save_point_cloud(vis_gt_pcl, color, dst_path)
        
        # save pred pcl
        vis_pred_pcl = data_dict['xyz_corrupt_flat'].clone()
        vis_pred_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos_refine']
        corrupt_mask_flat = data_dict['corrupt_mask'].view(data_dict['rgb_img'].shape[0],-1)
        miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
        miss_bid = miss_idx[:,0]
        miss_flat_img_id = miss_idx[:,1]
        transparent_pred = vis_pred_pcl[miss_bid, miss_flat_img_id]
        vis_pred_pcl = vis_pred_pcl[vis_bid].detach().cpu().numpy()
        dst_path = os.path.join(result_dir, 'epo{:03d}_iter{:05d}_pred.ply'.format(epoch, iteration))
        vis_utils.save_point_cloud(vis_pred_pcl, color, dst_path)
        vis_transparent_pred_pcl = data_dict['xyz_corrupt_flat'].clone()
        vis_transparent_pred_pcl[miss_bid, miss_flat_img_id] = transparent_pred
        vis_transparent_pred_pcl = vis_transparent_pred_pcl[vis_bid].detach().cpu().numpy()
        dst_path = os.path.join(result_dir, 'epo{:03d}_iter{:05d}_pred_transparent.ply'.format(epoch, iteration))
        vis_utils.save_point_cloud(vis_transparent_pred_pcl, color, dst_path)

        # visualize surface normal image
        if self.opt.loss.surf_norm_w > 0:
            gt_surf_norm_img_np = data_dict['gt_surf_norm_img'][vis_bid].permute(1,2,0).contiguous().detach().cpu().numpy()
            gt_surf_norm_img_np  = (gt_surf_norm_img_np+1) / 2 * 255.
            gt_surf_norm_img_np = gt_surf_norm_img_np.astype(np.uint8)
            pred_surf_norm_img_np = data_dict['pred_surf_norm_img_refine'][vis_bid].permute(1,2,0).contiguous().detach().cpu().numpy()
            pred_surf_norm_img_np  = (pred_surf_norm_img_np+1) / 2 * 255.
            pred_surf_norm_img_np = pred_surf_norm_img_np.astype(np.uint8)

            # Setup a figure
            fig, axes = plt.subplots(1, 2, figsize=(20, 7))
            axes = axes.flat
            for ax in axes:
                ax.axis("off")
            # Pred SN
            axes[0].set_title("Pred Surface Normal")
            axes[0].imshow(pred_surf_norm_img_np)
            # GT SN
            axes[1].set_title("GT Surface Normal")
            axes[1].imshow(gt_surf_norm_img_np)
            # Save figure
            plt.tight_layout()
            # name setting
            dst_path = osp.join(result_dir, 'epo{:03d}_iter{:05d}_sn.png'.format(epoch, iteration))
            plt.savefig(dst_path)
            plt.close(fig)


    def train(self, data_loader, exp_type, epoch):
        if self.opt.gpu_id == 0:
            self.train_loss.reset()

        self.lidf.train()
        self.refine_net.train()
        for iteration, batch in enumerate(data_loader):
            debug_print("=> Iter {} for GPU {}".format(iteration, self.opt.gpu_id), self.opt.debug)
            self.run_iteration(epoch, iteration, len(data_loader), exp_type, 
                self.opt.training.train_vis_iter, batch)
            if self.opt.debug and iteration == 5:
                break

    def validate(self, data_loader, exp_type, epoch, dataset_name='valid'):
        with torch.no_grad():
            self.val_loss.reset()
            self.xyz_err.reset()
            self.end_acc.reset()
            self.angle_err.reset()
            self.a1.reset()
            self.a2.reset()
            self.a3.reset()
            self.rmse.reset()
            self.rmse_log.reset()
            self.abs_rel.reset()
            self.mae.reset()
            self.sq_rel.reset()

            self.lidf.eval()
            self.refine_net.eval()
            for iteration, batch in enumerate(data_loader):
                self.run_iteration(epoch, iteration, len(data_loader), exp_type, 
                    self.opt.training.val_vis_iter, batch)
                if self.opt.debug and iteration == 5:
                    break
            err_log = '        {}: rmse: {:.3f}, abs_rel: {:.3f}, mae: {:.3f}, a1: {:.2f}, a2: {:.2f}, a3: {:.2f}\n'.format(
                    dataset_name, self.rmse.avg, self.abs_rel.avg, self.mae.avg, self.a1.avg*100, self.a2.avg*100, self.a3.avg*100)
            print(err_log)
            with open(self.valid_log_path, 'a') as file:
                file.write(err_log)


    def test(self, data_loader, exp_type, dataset_type, epoch=0):
        # Create CSV File to store error metrics
        self.csv_filename = osp.join(self.result_dir, 'detailed_errors', '{}_{}.csv'.format(self.opt.dataset.type, dataset_type))
        os.makedirs(osp.dirname(self.csv_filename), exist_ok=True)
        self.field_names = ["Image Num", "RMSE", "REL", "MAE", "Delta 1.05", "Delta 1.10", "Delta 1.25"]
        with open(self.csv_filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names, delimiter=',')
            writer.writeheader()

        vis_iter = self.opt.training.test_vis_iter
        with torch.no_grad():
            self.val_loss.reset()
            self.xyz_err.reset()
            self.end_acc.reset()
            self.angle_err.reset()
            self.a1.reset()
            self.a2.reset()
            self.a3.reset()
            self.rmse.reset()
            self.rmse_log.reset()
            self.abs_rel.reset()
            self.mae.reset()
            self.sq_rel.reset()

            self.lidf.eval()
            self.refine_net.eval()
            for iteration, batch in enumerate(data_loader):
                self.run_iteration(epoch, iteration, len(data_loader), exp_type, vis_iter, batch)

        err_log = '===> {}: rmse: {:.3f}, abs_rel: {:.3f}, mae: {:.3f}, a1: {:.2f}, a2: {:.2f}, a3: {:.2f}\n'.format(
                dataset_type, self.rmse.avg, self.abs_rel.avg, self.mae.avg, self.a1.avg*100, self.a2.avg*100, self.a3.avg*100)
        print(err_log)
        with open(self.log_path, 'a') as file:
            file.write(err_log)
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names, delimiter=',')
            row_data = ['MEAN', round(self.rmse.avg,3), round(self.abs_rel.avg,3), round(self.mae.avg,3),
            round(self.a1.avg*100,2), round(self.a2.avg*100,2), round(self.a3.avg*100,2)]
            writer.writerow(dict(zip(self.field_names, row_data)))

def createAndRunTrainer(gpu_id, opt):
    '''
        gpu_id: gpu id for current trainer. same as process local rank. handled by spawn for multiprocessing, or 
        manually passed in for single GPU training
        opt: option data
    '''

    # local gpu id for current process in residing node.
    opt.gpu_id = gpu_id
    if opt.dist.ddp:
        # global gpu id for current process 
        opt.dist.global_gpu_id = opt.dist.node_rank * opt.dist.ngpus_per_node + gpu_id
        # total GPU number
        opt.dist.world_size = opt.dist.ngpus_per_node * opt.dist.nodes_num
        dist.init_process_group(backend=opt.dist.dist_backend, init_method=opt.dist.dist_url,
                                world_size=opt.dist.world_size, rank=opt.dist.global_gpu_id)
    # init trainer
    trainer = TrainRefine(opt)

    # Training
    if opt.exp_type == 'train':
        print("=> Start Training")
        trainer.train_epoch()
    
    # Testing
    elif opt.exp_type == 'test':
        if opt.dist.ddp:
            raise ValueError('Testing should use single GPU')
        print("=> Start Testing")
        # vis dir
        trainer.test_vis_dir = create_dir(osp.join(trainer.result_dir, '{}_test_vis'.format(opt.dataset.type)))
        # log path
        trainer.log_path = osp.join(trainer.result_dir, '{}_test_log.txt'.format(opt.dataset.type))
        with open(trainer.log_path, 'a') as file:
            file.write('Mask Type: {}\n'.format(opt.mask_type))
            file.write('checkpoint path: {}\n'.format(opt.checkpoint_path.split('/')[-1]))
        if opt.dataset.type == 'cleargrasp':
            trainer.test(trainer.syn_known_test_data_loader, 'test', 'syn_known', epoch=0)
            trainer.test(trainer.syn_novel_test_data_loader, 'test', 'syn_novel', epoch=1)
            trainer.test(trainer.real_known_test_data_loader, 'test', 'real_known', epoch=2)
            trainer.test(trainer.real_novel_test_data_loader, 'test', 'real_novel', epoch=3)
    else:
        raise NotImplementedError('{} not supported'.format(opt.exp_type))