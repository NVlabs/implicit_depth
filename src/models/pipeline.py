import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import torch
import torch.nn as nn
import torchvision.ops as tv_ops
import torch.nn.functional as F
import torchvision.transforms as transforms


from torch_scatter import scatter, scatter_softmax, scatter_max, scatter_log_softmax
from extensions.ray_aabb.jit import ray_aabb
from extensions.pcl_aabb.jit import pcl_aabb


import constants
import models.pointnet as pnet
import models.resnet_dilated as resnet_dilated
import models.implicit_net as im_net
import utils.point_utils as point_utils
import utils.vis_utils as vis_utils
import utils.loss_utils as loss_utils
from utils.training_utils import *

class LIDF(nn.Module):
    def __init__(self, opt, device):
        super(LIDF, self).__init__()
        self.opt = opt
        self.device = device
        # build models
        self.build_model()

    def build_model(self):
        # positional embedding
        if self.opt.model.pos_encode:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.model.multires)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.model.multires_views)
        else:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.model.multires, i=-1)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.model.multires_views, i=-1)
            assert embed_ch == embeddirs_ch == 3
        
        # rgb model
        if self.opt.model.rgb_model_type == 'resnet':
            self.resnet_model = resnet_dilated.Resnet34_8s(inp_ch=self.opt.model.rgb_in, out_ch=self.opt.model.rgb_out).to(self.device)
        else:
            raise NotImplementedError('Does not support RGB model: {}'.format(self.opt.model.rgb_model_type))
        
        # pointnet model
        if self.opt.model.pnet_model_type == 'twostage':
            self.pnet_model = pnet.PointNet2Stage(input_channels=self.opt.model.pnet_in,
                                    output_channels=self.opt.model.pnet_out, gf_dim=self.opt.model.pnet_gf).to(self.device)
        else:
            raise NotImplementedError('Does not support PNET model: {}'.format(self.opt.model.pnet_model_type))
        
        # decoder input dim
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            dec_inp_dim = self.opt.model.pnet_out + self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2) \
                            + 2 * embed_ch + embeddirs_ch
        else:
            raise NotImplementedError('Does not support RGB embedding: {}'.format(self.opt.model.rgb_embedding_type))
        # offset decoder
        if self.opt.model.offdec_type == 'IMNET':
            self.offset_dec = im_net.IMNet(inp_dim=dec_inp_dim, out_dim=1, 
                                    gf_dim=self.opt.model.imnet_gf, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        elif self.opt.model.offdec_type == 'IEF':
            self.offset_dec = im_net.IEF(self.device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=self.opt.model.imnet_gf, 
                                    n_iter=self.opt.model.n_iter, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        else:
            raise NotImplementedError('Does not support Offset Decoder Type: {}'.format(self.opt.model.offdec_type))
        
        # prob decoder
        if self.opt.loss.prob_loss_type == 'ray':
            prob_out_dim = 1
        if self.opt.model.probdec_type == 'IMNET':
            self.prob_dec = im_net.IMNet(inp_dim=dec_inp_dim, out_dim=prob_out_dim, 
                                gf_dim=self.opt.model.imnet_gf, use_sigmoid=self.opt.model.use_sigmoid).to(self.device)
        else:
            raise NotImplementedError('Does not support Prob Decoder Type: {}'.format(self.opt.model.probdec_type))

        # loss function
        self.pos_loss_fn = nn.L1Loss()
        print('loss_fn at GPU {}'.format(self.opt.gpu_id))

    def prepare_data(self, batch, exp_type, pred_mask):
        # fetch data
        batch = to_gpu(batch, self.device)
        rgb_img = batch['rgb']
        bs = rgb_img.shape[0]
        h,w = rgb_img.shape[2],rgb_img.shape[3]
        corrupt_mask = batch['corrupt_mask'].squeeze(1)
        xyz = batch['xyz']
        xyz_corrupt = batch['xyz_corrupt']
        if 'valid_mask' in batch.keys():
            valid_mask = batch['valid_mask'].squeeze(1)
        else:
            valid_mask = 1 - corrupt_mask
        
        # flat h and w dim
        xyz_flat = xyz.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)
        xyz_corrupt_flat = xyz_corrupt.permute(0, 2, 3, 1).contiguous().reshape(bs,-1,3)

        # arrange data in a dictionary
        data_dict = {
            'bs': bs,
            'h': h,
            'w': w,
            'rgb_img': rgb_img,
            'corrupt_mask': corrupt_mask,
            'valid_mask': valid_mask,
            'xyz_flat': xyz_flat,
            'xyz_corrupt_flat': xyz_corrupt_flat,
            'fx': batch['fx'].float(),
            'fy': batch['fy'].float(),
            'cx': batch['cx'].float(),
            'cy': batch['cy'].float(),
            'item_path': batch['item_path'],
        }
        # add pred_mask
        if exp_type != 'train':
            if self.opt.mask_type == 'pred':
                data_dict['pred_mask'] = pred_mask
                data_dict['valid_mask'] = 1 - pred_mask
            elif self.opt.mask_type == 'all':
                data_dict['pred_mask'] = torch.ones_like(data_dict['corrupt_mask'])
                inp_zero_mask = (batch['depth_corrupt'] == 0).squeeze(1).float()
                data_dict['valid_mask'] = 1 - inp_zero_mask

        return data_dict

    def get_valid_points(self, data_dict):
        '''
            If valid_sample_num == -1, use all valid points. Otherwise uniformly sample valid points in a small block.
            valid_idx: (valid_point_num,2), 1st dim is batch idx, 2nd dim is flattened img idx.
        '''
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        if self.opt.grid.valid_sample_num != -1: # sample valid points
            valid_idx = point_utils.sample_valid_points(data_dict['valid_mask'], self.opt.grid.valid_sample_num, block_x=8, block_y=8)
        else: # get all valid points
            valid_mask_flat = data_dict['valid_mask'].reshape(bs,-1)
            valid_idx = torch.nonzero(valid_mask_flat, as_tuple=False)
        valid_bid = valid_idx[:,0]
        valid_flat_img_id = valid_idx[:,1]
        # get rgb and xyz for valid points.
        valid_xyz = data_dict['xyz_corrupt_flat'][valid_bid, valid_flat_img_id]
        rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(bs,-1,3)
        valid_rgb = rgb_img_flat[valid_bid, valid_flat_img_id]
        # update intermediate data in data_dict
        data_dict.update({
            'valid_bid': valid_bid,
            'valid_flat_img_id': valid_flat_img_id,
            'valid_xyz': valid_xyz,
            'valid_rgb': valid_rgb,
        })

    def get_occ_vox_bound(self, data_dict):
        ##################################
        #  Get occupied voxel in a batch
        ##################################
        # setup grid properties
        xmin = torch.Tensor(constants.XMIN).float().to(self.device)
        xmax = torch.Tensor(constants.XMAX).float().to(self.device)
        min_bb = torch.min(xmax- xmin).item()
        part_size = min_bb / self.opt.grid.res
        # we need half voxel margin on each side
        xmin = xmin - 0.5 * part_size
        xmax = xmax + 0.5 * part_size
        # get occupied grid
        occ_vox_bid_global_coord, revidx, valid_v_pid, \
        valid_v_rel_coord, idx_grid = point_utils.batch_get_occupied_idx(
                    data_dict['valid_xyz'], data_dict['valid_bid'].unsqueeze(-1),
                    xmin=xmin, xmax=xmax, 
                    crop_size=part_size, overlap=False)
        # images in current minibatch do not have occupied voxels
        if occ_vox_bid_global_coord.shape[0] == 0:
            print('No occupied voxel', data_dict['item_path'])
            return False
        occ_vox_bid = occ_vox_bid_global_coord[:,0]
        occ_vox_global_coord = occ_vox_bid_global_coord[:,1:]
        ''' compute occupied voxel bound '''
        bound_min = xmin.unsqueeze(0) + occ_vox_global_coord * part_size
        bound_max = bound_min + part_size
        voxel_bound = torch.cat((bound_min,bound_max),1)
        # update data_dict
        data_dict.update({
            'xmin': xmin,
            'part_size': part_size,
            'revidx': revidx,
            'valid_v_pid': valid_v_pid,
            'valid_v_rel_coord': valid_v_rel_coord,
            'occ_vox_bid': occ_vox_bid,
            'occ_vox_global_coord': occ_vox_global_coord,
            'voxel_bound': voxel_bound,    
        })
        return True

    def get_miss_ray(self, data_dict, exp_type):
        #####################################
        # compute ray dir and img grid index 
        #####################################
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        fx,fy = data_dict['fx'], data_dict['fy']
        cx,cy = data_dict['cx'], data_dict['cy']
        y_ind, x_ind = torch.meshgrid(torch.arange(h), torch.arange(w))
        x_ind = x_ind.unsqueeze(0).repeat(bs,1,1).float().to(self.device)
        y_ind = y_ind.unsqueeze(0).repeat(bs,1,1).float().to(self.device)
        # img grid index, (bs,h*w,2)
        img_ind_flat = torch.stack((x_ind,y_ind),-1).reshape(bs,h*w,2).long()
        cam_x = x_ind - cx.reshape(-1,1,1)
        cam_y = (y_ind - cy.reshape(-1,1,1)) * fx.reshape(-1,1,1) / fy.reshape(-1,1,1)
        cam_z = fx.reshape(-1,1,1).repeat(1,h,w)
        ray_dir = torch.stack((cam_x,cam_y,cam_z),-1)
        ray_dir = ray_dir / torch.norm(ray_dir,dim=-1,keepdim=True)
        ray_dir_flat = ray_dir.reshape(bs,-1,3)
        
        ###################################
        # sample miss points 
        # (miss_point_num,2): 1st dim is batch idx, second dim is flatted img idx.
        ###################################
        if exp_type != 'train' and self.opt.mask_type in ['pred', 'all']:
            pred_mask_flat = data_dict['pred_mask'].view(bs,-1)
            miss_idx = torch.nonzero(pred_mask_flat, as_tuple=False)
        else:
            corrupt_mask_flat = data_dict['corrupt_mask'].view(bs,-1)
            miss_idx = torch.nonzero(corrupt_mask_flat, as_tuple=False)
        if exp_type == 'train' and self.opt.grid.miss_sample_num != -1 and bs*self.opt.grid.miss_sample_num < miss_idx.shape[0]:            
            ''' randomly sample miss point. make them as continuous as possible '''
            miss_bid = miss_idx[:,0]
            # get max miss ray cnt for all examples inside a minibatch
            miss_bid_nodup, _, miss_bid_cnt = torch.unique_consecutive(miss_bid,dim=0,return_counts=True,return_inverse=True)
            # make sure cnt is sorted and fill in zero if non exist
            miss_bid_cnt_sorted = scatter(miss_bid_cnt, miss_bid_nodup, 
                            dim=0, dim_size=bs, reduce="sum")
            miss_bid_sid_eid = torch.cumsum(miss_bid_cnt_sorted, 0)
            miss_bid_sid_eid = torch.cat((torch.Tensor([0]).long().to(self.device), miss_bid_sid_eid),0)
            sample_list = []
            # iterate over examples in a batch
            for i in range(miss_bid_sid_eid.shape[0]-1):
                cur_sid = miss_bid_sid_eid[i].item()
                cur_eid = miss_bid_sid_eid[i+1].item()
                cur_cnt = miss_bid_cnt_sorted[i].item()
                if cur_cnt > self.opt.grid.miss_sample_num: # sample random miss points
                    start_range = cur_cnt - self.opt.grid.miss_sample_num + 1
                    start_id = np.random.choice(start_range) + cur_sid
                    sample_list.append(miss_idx[start_id:start_id+self.opt.grid.miss_sample_num])
                else: # add all miss points
                    sample_list.append(miss_idx[cur_sid:cur_eid])
            miss_idx = torch.cat(sample_list,0)
        
        total_miss_sample_num = miss_idx.shape[0]
        miss_bid = miss_idx[:,0]
        miss_flat_img_id = miss_idx[:,1]
        # get ray dir and img index for sampled miss point
        miss_ray_dir = ray_dir_flat[miss_bid, miss_flat_img_id]
        miss_img_ind = img_ind_flat[miss_bid, miss_flat_img_id]
        # update data_dict
        data_dict.update({
            'miss_bid': miss_bid,
            'miss_flat_img_id': miss_flat_img_id,
            'miss_ray_dir': miss_ray_dir,
            'miss_img_ind': miss_img_ind,
            'total_miss_sample_num': total_miss_sample_num
        })

    def compute_ray_aabb(self, data_dict):
        ################################## 
        #    Run ray AABB slab test
        #    mask: (occ_vox_num_in_batch, miss_ray_num_in_batch)
        #    dist: (occ_vox_num_in_batch, miss_ray_num_in_batch,2). store in voxel dist and out voxel dist
        ##################################
        mask, dist = ray_aabb.forward(data_dict['miss_ray_dir'], data_dict['voxel_bound'], 
                            data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
        mask = mask.long()
        dist = dist.float()

        # get idx of ray-voxel intersect pair
        intersect_idx = torch.nonzero(mask, as_tuple=False)
        occ_vox_intersect_idx = intersect_idx[:,0]
        miss_ray_intersect_idx = intersect_idx[:,1]
        # images in current mini batch do not have ray occ vox intersection pair.
        if intersect_idx.shape[0] == 0:
            print('No miss ray and occ vox intersection pair', data_dict['item_path'])
            return False
        data_dict.update({
            'mask': mask,
            'dist': dist,
            'occ_vox_intersect_idx': occ_vox_intersect_idx,
            'miss_ray_intersect_idx': miss_ray_intersect_idx,
        })
        return True

    def compute_gt(self, data_dict):
        ###########################################
        #    Compute Groundtruth for position and ray termination label
        ###########################################
        # get gt pos for sampled missing point
        gt_pos = data_dict['xyz_flat'][data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        # pcl_mask(i,j) indicates if j-th missing point gt pos inside i-th voxel
        pcl_mask = pcl_aabb.forward(gt_pos, data_dict['voxel_bound'], data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
        pcl_mask = pcl_mask.long()
        # compute gt label for ray termination
        pcl_label = pcl_mask[data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
        pcl_label_float = pcl_label.float()

        # get intersected voxels
        unique_intersect_vox_idx, occ_vox_intersect_idx_nodup2dup = torch.unique(data_dict['occ_vox_intersect_idx'], sorted=True, dim=0, return_inverse=True)
        intersect_voxel_bound = data_dict['voxel_bound'][unique_intersect_vox_idx]
        intersect_vox_bid = data_dict['occ_vox_bid'][unique_intersect_vox_idx]
        # get sampled valid pcl inside intersected voxels
        valid_intersect_mask = pcl_aabb.forward(data_dict['valid_xyz'], intersect_voxel_bound.contiguous(), data_dict['valid_bid'].int(), intersect_vox_bid.int().contiguous())
        valid_intersect_mask = valid_intersect_mask.long()
        try:
            valid_intersect_nonzero_idx = torch.nonzero(valid_intersect_mask, as_tuple=False)
        except:
            print(data_dict['valid_xyz'].shape)
            print(valid_intersect_mask.shape)
            print(unique_intersect_vox_idx.shape, intersect_voxel_bound.shape)
            print(data_dict['item_path'])
        valid_xyz_in_intersect = data_dict['valid_xyz'][valid_intersect_nonzero_idx[:,1]]
        valid_rgb_in_intersect = data_dict['valid_rgb'][valid_intersect_nonzero_idx[:,1]]
        valid_bid_in_intersect = data_dict['valid_bid'][valid_intersect_nonzero_idx[:,1]]
        # update data_dict
        data_dict.update({
            'gt_pos': gt_pos,
            'pcl_label': pcl_label,
            'pcl_label_float': pcl_label_float,
            'valid_xyz_in_intersect': valid_xyz_in_intersect,
            'valid_rgb_in_intersect': valid_rgb_in_intersect,
            'valid_bid_in_intersect': valid_bid_in_intersect
        })

    def get_embedding(self, data_dict):
        ########################### 
        #   Get embedding
        ##########################
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        ''' Positional Encoding '''
        # compute intersect pos
        intersect_dist = data_dict['dist'][data_dict['occ_vox_intersect_idx'], data_dict['miss_ray_intersect_idx']]
        intersect_enter_dist, intersect_leave_dist = intersect_dist[:,0], intersect_dist[:,1]

        intersect_dir = data_dict['miss_ray_dir'][data_dict['miss_ray_intersect_idx']]
        intersect_enter_pos = intersect_dir * intersect_enter_dist.unsqueeze(-1)
        intersect_leave_pos = intersect_dir * intersect_leave_dist.unsqueeze(-1)


        intersect_voxel_bound = data_dict['voxel_bound'][data_dict['occ_vox_intersect_idx']]
        intersect_voxel_center = (intersect_voxel_bound[:,:3] + intersect_voxel_bound[:,3:]) / 2.
        if self.opt.model.intersect_pos_type == 'rel':
            inp_enter_pos = intersect_enter_pos - intersect_voxel_center
            inp_leave_pos = intersect_leave_pos - intersect_voxel_center
        else:
            inp_enter_pos = intersect_enter_pos
            inp_leave_pos = intersect_leave_pos

        # positional encoding
        intersect_enter_pos_embed = self.embed_fn(inp_enter_pos)
        intersect_leave_pos_embed = self.embed_fn(inp_leave_pos)
        intersect_dir_embed = self.embeddirs_fn(intersect_dir)    
        
        ''' RGB Embedding ''' 
        miss_ray_intersect_img_ind = data_dict['miss_img_ind'][data_dict['miss_ray_intersect_idx']]
        miss_ray_intersect_bid = data_dict['miss_bid'][data_dict['miss_ray_intersect_idx']]
        full_rgb_feat = self.resnet_model(data_dict['rgb_img'])
        # ROIAlign to pool features
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            # compute input boxes for ROI Align
            miss_ray_intersect_ul = miss_ray_intersect_img_ind - self.opt.model.roi_inp_bbox // 2
            miss_ray_intersect_br = miss_ray_intersect_img_ind + self.opt.model.roi_inp_bbox // 2
            # clamp is done in original image coords
            miss_ray_intersect_ul[:,0] = torch.clamp(miss_ray_intersect_ul[:,0], min=0., max=w-1)
            miss_ray_intersect_ul[:,1] = torch.clamp(miss_ray_intersect_ul[:,1], min=0., max=h-1)
            miss_ray_intersect_br[:,0] = torch.clamp(miss_ray_intersect_br[:,0], min=0., max=w-1)
            miss_ray_intersect_br[:,1] = torch.clamp(miss_ray_intersect_br[:,1], min=0., max=h-1)
            roi_boxes = torch.cat((miss_ray_intersect_bid.unsqueeze(-1), miss_ray_intersect_ul, miss_ray_intersect_br),-1).float()
            # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
            spatial_scale = 1.0
            intersect_rgb_feat = tv_ops.roi_align(full_rgb_feat, roi_boxes, 
                                    output_size=self.opt.model.roi_out_bbox,
                                    spatial_scale=spatial_scale,
                                    aligned=True)
            try:
                intersect_rgb_feat = intersect_rgb_feat.reshape(intersect_rgb_feat.shape[0],-1)
            except:
                print(intersect_rgb_feat.shape)
                print(roi_boxes.shape)
                print(data_dict['miss_ray_intersect_idx'].shape, miss_ray_intersect_bid.shape, miss_ray_intersect_img_ind.shape)
                print(data_dict['total_miss_sample_num'])
                print(data_dict['item_path'])
        else:
            raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))

        '''  Voxel Embedding '''
        valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
        if self.opt.model.pnet_pos_type == 'rel': # relative position w.r.t voxel center
            pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
        else:
            raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.model.pnet_pos_type))
        # pointnet forward
        if self.opt.model.pnet_model_type == 'twostage':
            occ_voxel_feat = self.pnet_model(inp_feat=pnet_inp, vox2point_idx=data_dict['revidx'])
        else:
            raise NotImplementedError('Does not support pnet model type: {}'.format(self.opt.model.pnet_model_type))
        intersect_voxel_feat = occ_voxel_feat[data_dict['occ_vox_intersect_idx']]

        # update data_dict
        data_dict.update({
            'intersect_dir': intersect_dir,
            'intersect_enter_dist': intersect_enter_dist,
            'intersect_leave_dist': intersect_leave_dist,
            'intersect_enter_pos': intersect_enter_pos,
            'intersect_leave_pos': intersect_leave_pos,
            'intersect_enter_pos_embed': intersect_enter_pos_embed,
            'intersect_leave_pos_embed': intersect_leave_pos_embed,
            'intersect_dir_embed': intersect_dir_embed,
            'full_rgb_feat': full_rgb_feat,
            'intersect_rgb_feat': intersect_rgb_feat,
            'intersect_voxel_feat': intersect_voxel_feat
        })

    def get_pred(self, data_dict, exp_type, epoch):
        ######################################################## 
        # Concat embedding and send to decoder 
        ########################################################
        inp_embed = torch.cat(( data_dict['intersect_voxel_feat'].contiguous(), data_dict['intersect_rgb_feat'].contiguous(),
                                data_dict['intersect_enter_pos_embed'].contiguous(),
                                data_dict['intersect_leave_pos_embed'].contiguous(), data_dict['intersect_dir_embed'].contiguous()),-1)
        pred_offset = self.offset_dec(inp_embed)
        pred_prob_end = self.prob_dec(inp_embed)
        # scale pred_offset from (0,1) to (offset_range[0], offset_range[1]).
        pred_scaled_offset = pred_offset * (self.opt.grid.offset_range[1] - self.opt.grid.offset_range[0]) + self.opt.grid.offset_range[0]
        pred_scaled_offset = pred_scaled_offset * np.sqrt(3) * data_dict['part_size']
        pair_pred_pos = data_dict['intersect_enter_pos'] + pred_scaled_offset * data_dict['intersect_dir']
        # we detach the pred_prob_end. we don't want pos loss to affect ray terminate score.
        if self.opt.loss.prob_loss_type == 'ray':
            pred_prob_end_softmax = scatter_softmax(pred_prob_end.detach()[:,0], data_dict['miss_ray_intersect_idx'])
        # training uses GT pcl_label to get max_pair_id (voxel with largest prob)
        if exp_type == 'train' and epoch < self.opt.model.maxpool_label_epo:
            _, max_pair_id = scatter_max(data_dict['pcl_label_float'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        # test/valid uses pred_prob_end_softmax to get max_pair_id (voxel with largest prob)
        else:
            _, max_pair_id = scatter_max(pred_prob_end_softmax, data_dict['miss_ray_intersect_idx'],
                            dim_size=data_dict['total_miss_sample_num'])
        if self.opt.model.scatter_type == 'Maxpool':
            dummy_pos = torch.zeros([1,3]).float().to(self.device)
            pair_pred_pos_dummy = torch.cat((pair_pred_pos, dummy_pos),0)
            pred_pos = pair_pred_pos_dummy[max_pair_id]    
        else:
            raise NotImplementedError('Does not support Scatter Type: {}'.format(self.opt.model.scatter_type))
        
        assert pred_pos.shape[0] == data_dict['total_miss_sample_num']
        # update data_dict
        data_dict.update({
            'pair_pred_pos': pair_pred_pos,
            'max_pair_id': max_pair_id,
            'pred_prob_end': pred_prob_end,
            'pred_prob_end_softmax': pred_prob_end_softmax,
            'pred_pos': pred_pos,
        })

    def compute_loss(self, data_dict, exp_type, epoch):
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        ''' position loss '''
        if self.opt.loss.pos_loss_type == 'single':
            if not self.opt.loss.hard_neg:
                pos_loss = self.pos_loss_fn(data_dict['pred_pos'], data_dict['gt_pos'])
            else:
                pos_loss_unreduce = torch.mean((data_dict['pred_pos'] - data_dict['gt_pos']).abs(),-1)
                k = int(pos_loss_unreduce.shape[0] * self.opt.loss.hard_neg_ratio)
                pos_loss_topk,_ = torch.topk(pos_loss_unreduce, k)
                pos_loss = torch.mean(pos_loss_topk)

        ''' Ending probability loss '''
        if self.opt.loss.prob_loss_type == 'ray':
            pred_prob_end_log_softmax = scatter_log_softmax(data_dict['pred_prob_end'][:,0], data_dict['miss_ray_intersect_idx'])
            pcl_label_idx = torch.nonzero(data_dict['pcl_label'], as_tuple=False).reshape(-1)
            prob_loss_unreduce = -1*pred_prob_end_log_softmax[pcl_label_idx]
            if not self.opt.loss.hard_neg:
                prob_loss = torch.mean(prob_loss_unreduce)
            else:
                k = int(prob_loss_unreduce.shape[0] * self.opt.loss.hard_neg_ratio)
                prob_loss_topk,_ = torch.topk(prob_loss_unreduce, k)
                prob_loss = torch.mean(prob_loss_topk)
            
        ''' surface normal loss '''
        if exp_type == 'train':
            gt_pcl = data_dict['xyz_flat'].clone()
            pred_pcl = data_dict['xyz_flat'].clone()
        else:
            gt_pcl = data_dict['xyz_corrupt_flat'].clone()
            pred_pcl = data_dict['xyz_corrupt_flat'].clone()
        gt_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['gt_pos']
        gt_pcl = gt_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        gt_surf_norm_img,_,_ = point_utils.get_surface_normal(gt_pcl)
        gt_surf_norm_flat = gt_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        gt_surf_norm = gt_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        pred_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
        pred_pcl = pred_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        pred_surf_norm_img, dx, dy = point_utils.get_surface_normal(pred_pcl)
        pred_surf_norm_flat = pred_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        pred_surf_norm = pred_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        # surface normal loss
        cosine_val = F.cosine_similarity(pred_surf_norm, gt_surf_norm, dim=-1)
        surf_norm_dist = (1 - cosine_val) / 2.
        if not self.opt.loss.hard_neg:
            surf_norm_loss = torch.mean(surf_norm_dist)
        else:
            k = int(surf_norm_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            surf_norm_dist_topk,_ = torch.topk(surf_norm_dist, k)
            surf_norm_loss = torch.mean(surf_norm_dist_topk)
        # angle err
        angle_err = torch.mean(torch.acos(torch.clamp(cosine_val,min=-1,max=1)))
        angle_err = angle_err / np.pi * 180.

        # smooth loss
        dx_dist = torch.sum(dx*dx,1)
        dx_dist_flat = dx_dist.reshape(bs,h*w)
        miss_dx_dist = dx_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        dy_dist = torch.sum(dy*dy,1)
        dy_dist_flat = dy_dist.reshape(bs,h*w)
        miss_dy_dist = dy_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        if not self.opt.loss.hard_neg:
            smooth_loss = torch.mean(miss_dx_dist) + torch.mean(miss_dy_dist)
        else:
            k = int(miss_dx_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            miss_dx_dist_topk,_ = torch.topk(miss_dx_dist, k)
            miss_dy_dist_topk,_ = torch.topk(miss_dy_dist, k)
            smooth_loss = torch.mean(miss_dx_dist_topk) + torch.mean(miss_dy_dist_topk)
        
        ''' loss net '''
        loss_net = self.opt.loss.pos_w * pos_loss + self.opt.loss.prob_w * prob_loss
        if self.opt.loss.surf_norm_w > 0 and epoch >= self.opt.loss.surf_norm_epo:
            loss_net += self.opt.loss.surf_norm_w * surf_norm_loss
        if self.opt.loss.smooth_w > 0 and epoch >= self.opt.loss.smooth_epo:
            loss_net += self.opt.loss.smooth_w * smooth_loss

        
        #######################
        # Evaluation Metric
        #######################
        # ending accuracy for missing point
        _, pred_label = scatter_max(data_dict['pred_prob_end_softmax'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        _, gt_label = scatter_max(data_dict['pcl_label'], data_dict['miss_ray_intersect_idx'],
                                dim_size=data_dict['total_miss_sample_num'])
        acc = torch.sum(torch.eq(pred_label, gt_label).float()) / torch.numel(pred_label)

        # position L2 error: we don't want to consider 0 depth point in the position L2 error.
        zero_mask = torch.sum(data_dict['gt_pos'].abs(),dim=-1)
        zero_mask[zero_mask!=0] = 1.
        elem_num = torch.sum(zero_mask)
        if elem_num.item() == 0:
            err = torch.Tensor([0]).float().to(self.device)
        else:
            err = torch.sum(torch.sqrt(torch.sum((data_dict['pred_pos'] - data_dict['gt_pos'])**2,-1))*zero_mask) / elem_num
        # compute depth errors following cleargrasp
        zero_mask_idx = torch.nonzero(zero_mask, as_tuple=False).reshape(-1)


        if exp_type != 'train':
            if bs != 1:
                pred_depth = data_dict['pred_pos'][:,2]
                gt_depth = data_dict['gt_pos'][:,2]
                pred = pred_depth[zero_mask_idx]
                gt = gt_depth[zero_mask_idx]
            else:
                # scale image to make sure it is same as cleargrasp eval metric
                gt_xyz = data_dict['xyz_flat'].clone()
                gt_xyz = gt_xyz.reshape(bs,h,w,3).cpu().numpy()
                gt_depth = gt_xyz[0,:,:,2]
                gt_depth = cv2.resize(gt_depth, (256, 144), interpolation=cv2.INTER_NEAREST)
                gt_depth[np.isnan(gt_depth)] = 0
                gt_depth[np.isinf(gt_depth)] = 0
                mask_valid_region = (gt_depth > 0)

                seg_mask = data_dict['corrupt_mask'].cpu().numpy()
                seg_mask = seg_mask[0].astype(np.uint8)
                seg_mask = cv2.resize(seg_mask, (256, 144), interpolation=cv2.INTER_NEAREST)
                mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
                mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)

                pred_xyz = data_dict['xyz_corrupt_flat'].clone()
                pred_xyz[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos']
                pred_xyz = pred_xyz.reshape(bs,h,w,3).cpu().numpy()
                pred_depth = pred_xyz[0,:,:,2]
                pred_depth = cv2.resize(pred_depth, (256, 144), interpolation=cv2.INTER_NEAREST)

                gt = torch.from_numpy(gt_depth).float().to(self.device)
                pred = torch.from_numpy(pred_depth).float().to(self.device)
                mask = torch.from_numpy(mask_valid_region).bool().to(self.device)
                gt = gt[mask]
                pred = pred[mask]

            # compute metrics
            safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
            safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
            thresh = torch.max(gt / pred, pred / gt)
            a1 = (thresh < 1.05).float().mean()
            a2 = (thresh < 1.10).float().mean()
            a3 = (thresh < 1.25).float().mean()

            rmse = ((gt - pred)**2).mean().sqrt()
            rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
            log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
            abs_rel = ((gt - pred).abs() / gt).mean()
            mae = (gt - pred).abs().mean()
            sq_rel = ((gt - pred)**2 / gt).mean()

        # update data_dict
        data_dict.update({
            'zero_mask_idx': zero_mask_idx,
            'gt_surf_norm_img': gt_surf_norm_img,
            'pred_surf_norm_img': pred_surf_norm_img
        })

        # loss dict
        loss_dict = {
            'pos_loss': pos_loss,
            'prob_loss': prob_loss,
            'surf_norm_loss': surf_norm_loss,
            'smooth_loss': smooth_loss,
            'loss_net': loss_net,
            'acc': acc,
            'err': err,
            'angle_err': angle_err,
        }
        if exp_type != 'train':
            loss_dict.update({
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'rmse': rmse,
                'rmse_log': rmse_log,
                'log10': log10,
                'abs_rel': abs_rel,
                'mae': mae,
                'sq_rel': sq_rel,
            })
        return loss_dict

    def forward(self, batch, exp_type, epoch, pred_mask=None):
        loss_dict = {}
        # prepare input and gt data
        data_dict = self.prepare_data(batch, exp_type, pred_mask)
        
        # get valid points data
        self.get_valid_points(data_dict)
        
        # get occupied voxel data
        occ_vox_flag = self.get_occ_vox_bound(data_dict)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([occ_vox_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not occ_vox_flag:
            return False, data_dict, loss_dict
        
        # get miss ray data
        self.get_miss_ray(data_dict, exp_type)
        miss_sample_flag = (data_dict['total_miss_sample_num'] != 0)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([miss_sample_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not miss_sample_flag:
            return False, data_dict, loss_dict

        # ray AABB slab test
        intersect_pair_flag = self.compute_ray_aabb(data_dict)
        if exp_type == 'train' and self.opt.dist.ddp:
            # have to set barrier to wait for all processes finished forward pass
            dist.barrier()
            success_num = torch.Tensor([intersect_pair_flag]).to(self.device)
            dist.all_reduce(success_num, op=dist.ReduceOp.SUM)
            # at least one gpu fails: clear grad buffer and return
            if success_num[0] < self.opt.dist.ngpus_per_node:
                print('gpu {}: {}'.format(self.opt.gpu_id, success_num[0]))
                return False, data_dict, loss_dict
        elif not intersect_pair_flag:
            return False, data_dict, loss_dict
        
        # compute gt
        self.compute_gt(data_dict)
        # get embedding
        self.get_embedding(data_dict)
        # get prediction
        self.get_pred(data_dict, exp_type, epoch)
        # compute loss
        loss_dict = self.compute_loss(data_dict, exp_type, epoch)
        return True, data_dict, loss_dict


class RefineNet(nn.Module):
    def __init__(self, opt, device):
        super(RefineNet, self).__init__()
        self.opt = opt
        self.device = device
        # build models
        self.build_model()

    def build_model(self):
        # positional embedding
        if self.opt.refine.pos_encode:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.refine.multires)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.refine.multires_views)
        else:
            self.embed_fn, embed_ch = im_net.get_embedder(self.opt.refine.multires, i=-1)
            self.embeddirs_fn, embeddirs_ch = im_net.get_embedder(self.opt.refine.multires_views, i=-1)
            assert embed_ch == embeddirs_ch == 3
        
        # pointnet
        if self.opt.refine.pnet_model_type == 'twostage':
            self.pnet_model = pnet.PointNet2Stage(input_channels=self.opt.refine.pnet_in,
                                    output_channels=self.opt.refine.pnet_out, gf_dim=self.opt.refine.pnet_gf).to(self.device)
        else:
            raise NotImplementedError('Does not support Pnet type for RefineNet: {}'.format(self.opt.refine.pnet_model_type))
        
        # decoder input dim
        dec_inp_dim = self.opt.refine.pnet_out + embed_ch + embeddirs_ch
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            dec_inp_dim += self.opt.model.rgb_out * (self.opt.model.roi_out_bbox**2)
        else:
            raise NotImplementedError('Does not support RGB embedding: {}'.format(self.opt.model.rgb_embedding_type))

        # offset decoder
        if self.opt.refine.offdec_type == 'IMNET':
            self.offset_dec = im_net.IMNet(inp_dim=dec_inp_dim, out_dim=1, 
                                    gf_dim=self.opt.refine.imnet_gf, use_sigmoid=self.opt.refine.use_sigmoid).to(self.device)
        elif self.opt.refine.offdec_type == 'IEF':
            self.offset_dec = im_net.IEF(self.device, inp_dim=dec_inp_dim, out_dim=1, gf_dim=self.opt.refine.imnet_gf, 
                                    n_iter=self.opt.refine.n_iter, use_sigmoid=self.opt.refine.use_sigmoid).to(self.device)
        else:
            raise NotImplementedError('Does not support Offset Decoder Type: {}'.format(self.opt.refine.offdec_type))

        # loss function
        self.pos_loss_fn = nn.L1Loss()
        print('loss_fn at GPU {}'.format(self.opt.gpu_id))

    def compute_loss(self, data_dict, exp_type, epoch):
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        ''' position loss '''
        if self.opt.loss.pos_loss_type == 'single':
            if not self.opt.loss.hard_neg:
                pos_loss = self.pos_loss_fn(data_dict['pred_pos_refine'], data_dict['gt_pos'])
            else:
                pos_loss_unreduce = torch.mean((data_dict['pred_pos_refine'] - data_dict['gt_pos']).abs(),-1)
                k = int(pos_loss_unreduce.shape[0] * self.opt.loss.hard_neg_ratio)
                pos_loss_topk,_ = torch.topk(pos_loss_unreduce, k)
                pos_loss = torch.mean(pos_loss_topk)
        else:
            raise NotImplementedError('Does not support pos_loss_type for refine model'.format(self.opt.loss.pos_loss_type))
            
        ''' surface normal loss '''
        if exp_type == 'train':
            gt_pcl = data_dict['xyz_flat'].clone()
            pred_pcl = data_dict['xyz_flat'].clone()
        else:
            gt_pcl = data_dict['xyz_corrupt_flat'].clone()
            pred_pcl = data_dict['xyz_corrupt_flat'].clone()
        gt_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['gt_pos']
        gt_pcl = gt_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        gt_surf_norm_img,_,_ = point_utils.get_surface_normal(gt_pcl)
        gt_surf_norm_flat = gt_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        gt_surf_norm = gt_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        pred_pcl[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos_refine']
        pred_pcl = pred_pcl.reshape(bs,h,w,3).permute(0,3,1,2).contiguous()
        pred_surf_norm_img, dx, dy = point_utils.get_surface_normal(pred_pcl)
        pred_surf_norm_flat = pred_surf_norm_img.permute(0,2,3,1).contiguous().reshape(bs,h*w,3)
        pred_surf_norm = pred_surf_norm_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]

        # surface normal loss
        cosine_val = F.cosine_similarity(pred_surf_norm, gt_surf_norm, dim=-1)
        surf_norm_dist = (1 - cosine_val) / 2.
        if not self.opt.loss.hard_neg:
            surf_norm_loss = torch.mean(surf_norm_dist)
        else:
            k = int(surf_norm_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            surf_norm_dist_topk,_ = torch.topk(surf_norm_dist, k)
            surf_norm_loss = torch.mean(surf_norm_dist_topk)
        # angle err
        angle_err = torch.mean(torch.acos(torch.clamp(cosine_val,min=-1,max=1)))
        angle_err = angle_err / np.pi * 180.

        # smooth loss
        dx_dist = torch.sum(dx*dx,1)
        dx_dist_flat = dx_dist.reshape(bs,h*w)
        miss_dx_dist = dx_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        dy_dist = torch.sum(dy*dy,1)
        dy_dist_flat = dy_dist.reshape(bs,h*w)
        miss_dy_dist = dy_dist_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        
        if not self.opt.loss.hard_neg:
            smooth_loss = torch.mean(miss_dx_dist) + torch.mean(miss_dy_dist)
        else:
            k = int(miss_dx_dist.shape[0] * self.opt.loss.hard_neg_ratio)
            miss_dx_dist_topk,_ = torch.topk(miss_dx_dist, k)
            miss_dy_dist_topk,_ = torch.topk(miss_dy_dist, k)
            smooth_loss = torch.mean(miss_dx_dist_topk) + torch.mean(miss_dy_dist_topk)
        
        ''' loss net '''
        loss_net = self.opt.loss.pos_w * pos_loss
        if self.opt.loss.surf_norm_w > 0 and epoch >= self.opt.loss.surf_norm_epo:
            loss_net += self.opt.loss.surf_norm_w * surf_norm_loss
        if self.opt.loss.smooth_w > 0 and epoch >= self.opt.loss.smooth_epo:
            loss_net += self.opt.loss.smooth_w * smooth_loss
        
        #######################
        # Evaluation Metric
        #######################
        # position L2 error: we don't want to consider 0 depth point in the position L2 error.
        zero_mask = torch.sum(data_dict['gt_pos'].abs(),dim=-1)
        zero_mask[zero_mask!=0] = 1.
        elem_num = torch.sum(zero_mask)
        if elem_num.item() == 0:
            err = torch.Tensor([0]).float().to(self.device)
        else:
            err = torch.sum(torch.sqrt(torch.sum((data_dict['pred_pos_refine'] - data_dict['gt_pos'])**2,-1))*zero_mask) / elem_num
        # compute depth errors following cleargrasp
        zero_mask_idx = torch.nonzero(zero_mask, as_tuple=False).reshape(-1)
        if exp_type != 'train':
            if bs != 1:
                pred_depth = data_dict['pred_pos_refine'][:,2]
                gt_depth = data_dict['gt_pos'][:,2]
                pred = pred_depth[zero_mask_idx]
                gt = gt_depth[zero_mask_idx]
            else:
                # scale image to make sure it is same as cleargrasp eval metric
                gt_xyz = data_dict['xyz_flat'].clone()
                gt_xyz = gt_xyz.reshape(bs,h,w,3).cpu().numpy()
                gt_depth = gt_xyz[0,:,:,2]
                # same size as cleargrasp for fair comparison
                gt_depth = cv2.resize(gt_depth, (256, 144), interpolation=cv2.INTER_NEAREST)
                gt_depth[np.isnan(gt_depth)] = 0
                gt_depth[np.isinf(gt_depth)] = 0
                mask_valid_region = (gt_depth > 0)
                seg_mask = data_dict['corrupt_mask'].cpu().numpy()
                seg_mask = seg_mask[0].astype(np.uint8)
                seg_mask = cv2.resize(seg_mask, (256, 144), interpolation=cv2.INTER_NEAREST)
                mask_valid_region = np.logical_and(mask_valid_region, seg_mask)
                mask_valid_region = (mask_valid_region.astype(np.uint8) * 255)

                pred_xyz = data_dict['xyz_corrupt_flat'].clone()
                pred_xyz[data_dict['miss_bid'], data_dict['miss_flat_img_id']] = data_dict['pred_pos_refine']
                pred_xyz = pred_xyz.reshape(bs,h,w,3).cpu().numpy()
                pred_depth = pred_xyz[0,:,:,2]
                pred_depth = cv2.resize(pred_depth, (256, 144), interpolation=cv2.INTER_NEAREST)

                gt = torch.from_numpy(gt_depth).float().to(self.device)
                pred = torch.from_numpy(pred_depth).float().to(self.device)
                mask = torch.from_numpy(mask_valid_region).bool().to(self.device)
                gt = gt[mask]
                pred = pred[mask]

            # compute metrics
            safe_log = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
            safe_log10 = lambda x: torch.log(torch.clamp(x, 1e-6, 1e6))
            thresh = torch.max(gt / pred, pred / gt)
            a1 = (thresh < 1.05).float().mean()
            a2 = (thresh < 1.10).float().mean()
            a3 = (thresh < 1.25).float().mean()
            rmse = ((gt - pred)**2).mean().sqrt()
            rmse_log = ((safe_log(gt) - safe_log(pred))**2).mean().sqrt()
            log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean()
            abs_rel = ((gt - pred).abs() / gt).mean()
            mae = (gt - pred).abs().mean()
            sq_rel = ((gt - pred)**2 / gt).mean()

        # update data_dict
        data_dict.update({
            'zero_mask_idx': zero_mask_idx,
            'pred_surf_norm_img_refine': pred_surf_norm_img
        })

        # loss dict
        loss_dict = {
            'pos_loss': pos_loss,
            'surf_norm_loss': surf_norm_loss,
            'smooth_loss': smooth_loss,
            'loss_net': loss_net,
            'err': err,
            'angle_err': angle_err,
        }
        if exp_type != 'train':
            loss_dict.update({
                'a1': a1,
                'a2': a2,
                'a3': a3,
                'rmse': rmse,
                'rmse_log': rmse_log,
                'log10': log10,
                'abs_rel': abs_rel,
                'mae': mae,
                'sq_rel': sq_rel,
            })

        return loss_dict


    def get_pred_refine(self, data_dict, pred_pos, exp_type, cur_iter):
        bs,h,w = data_dict['bs'], data_dict['h'], data_dict['w']
        concat_dummy = lambda feat: torch.cat((feat, torch.zeros([1,feat.shape[1]]).to(feat.dtype).to(self.device)),0)
        # manually perturb prediction by adding noise, we only perturb in 1st iter.
        if exp_type == 'train' and self.opt.refine.perturb and cur_iter == 0 and np.random.random() < self.opt.refine.perturb_prob:
            prob = np.random.random()
            if prob < 0.5:
                noise = np.random.random() * (0 + 0.05) - 0.05
            elif prob < 0.8:
                noise = np.random.random() * (0.05 - 0)
            elif prob < 0.9:
                noise = np.random.random() * (-0.05 + 0.1) - 0.1
            else:
                noise = np.random.random() * (0.1 - 0.05) + 0.05

            pred_pos = pred_pos + noise * data_dict['miss_ray_dir']
        # recompute voxel ending id
        pred_occ_mask = pcl_aabb.forward(pred_pos, data_dict['voxel_bound'], data_dict['miss_bid'].int(), data_dict['occ_vox_bid'].int())
        pred_occ_mask = pred_occ_mask.long()
        pred_occ_mask_idx = torch.nonzero(pred_occ_mask, as_tuple=False)
        occ_vox_intersect_idx_dummy = concat_dummy(data_dict['occ_vox_intersect_idx'].unsqueeze(-1))
        end_voxel_id = occ_vox_intersect_idx_dummy[data_dict['max_pair_id']].reshape(-1)
        scatter(pred_occ_mask_idx[:,0], pred_occ_mask_idx[:,1], out=end_voxel_id, reduce='max')

        # dir embed
        intersect_dir_embed_end = self.embeddirs_fn(data_dict['miss_ray_dir'])    
        # rgb embed 
        miss_ray_img_ind = data_dict['miss_img_ind']
        miss_ray_bid = data_dict['miss_bid']
        # ROIAlign to pool features
        if self.opt.model.rgb_embedding_type == 'ROIAlign':
            # compute input boxes for ROI Align
            miss_ray_ul = miss_ray_img_ind - self.opt.model.roi_inp_bbox // 2
            miss_ray_br = miss_ray_img_ind + self.opt.model.roi_inp_bbox // 2
            # clamp is done in original image coords
            miss_ray_ul[:,0] = torch.clamp(miss_ray_ul[:,0], min=0., max=w-1)
            miss_ray_ul[:,1] = torch.clamp(miss_ray_ul[:,1], min=0., max=h-1)
            miss_ray_br[:,0] = torch.clamp(miss_ray_br[:,0], min=0., max=w-1)
            miss_ray_br[:,1] = torch.clamp(miss_ray_br[:,1], min=0., max=h-1)
            roi_boxes = torch.cat((miss_ray_bid.unsqueeze(-1), miss_ray_ul, miss_ray_br),-1).float()
            # sampled rgb features for ray-voxel intersect pair. (pair num,rgb_feat_len,roi_out_bbox,roi_out_bbox)
            spatial_scale = 1.0
            intersect_rgb_feat_end = tv_ops.roi_align(data_dict['full_rgb_feat'], roi_boxes, 
                                    output_size=self.opt.model.roi_out_bbox,
                                    spatial_scale=spatial_scale,
                                    aligned=True)
            try:
                intersect_rgb_feat_end = intersect_rgb_feat_end.reshape(intersect_rgb_feat_end.shape[0],-1)
            except:
                print(data_dict['item_path'])
        else:
            raise NotImplementedError('Does not support RGB embedding type: {}'.format(self.opt.model.rgb_embedding_type))
        # miss point rgb
        rgb_img_flat = data_dict['rgb_img'].permute(0,2,3,1).contiguous().reshape(data_dict['bs'],-1,3)
        miss_rgb = rgb_img_flat[data_dict['miss_bid'], data_dict['miss_flat_img_id']]
        # get miss point predicted ending voxel center
        occ_voxel_bound = data_dict['voxel_bound']
        end_voxel_bound = occ_voxel_bound[end_voxel_id]
        end_voxel_center = (end_voxel_bound[:, :3] + end_voxel_bound[:, 3:]) / 2.
        # prep_inp
        if self.opt.refine.pnet_pos_type == 'rel':
            pred_rel_xyz = pred_pos - end_voxel_center
            pred_inp = torch.cat((pred_rel_xyz, miss_rgb),1)
        else:
            pred_inp = torch.cat((pred_pos, miss_rgb),1)
        if exp_type != 'train' and self.opt.mask_type == 'all' and self.opt.refine.use_all_pix == False:
            inp_zero_mask = 1 - data_dict['valid_mask']
            zero_pixel_idx = torch.nonzero(inp_zero_mask, as_tuple=False)
            pred_inp_img = pred_inp.reshape(bs,h,w,pred_inp.shape[-1])
            new_pred_inp = pred_inp_img[zero_pixel_idx[:,0],zero_pixel_idx[:,1],zero_pixel_idx[:,2]]
            end_voxel_id_img = end_voxel_id.reshape(bs,h,w)
            new_end_voxel_id = end_voxel_id_img[zero_pixel_idx[:,0],zero_pixel_idx[:,1],zero_pixel_idx[:,2]]
        else:
            new_pred_inp = pred_inp
            new_end_voxel_id = end_voxel_id
        

        # pnet inp
        valid_v_rgb = data_dict['valid_rgb'][data_dict['valid_v_pid']]
        if self.opt.refine.pnet_pos_type == 'rel': # relative position w.r.t voxel center
            pnet_inp = torch.cat((data_dict['valid_v_rel_coord'], valid_v_rgb),-1)
        elif self.opt.refine.pnet_pos_type == 'abs': # absolute position
            valid_v_xyz = data_dict['valid_xyz'][data_dict['valid_v_pid']]
            pnet_inp = torch.cat((valid_v_xyz, valid_v_rgb),-1)
        else:
            raise NotImplementedError('Does not support Pnet pos type: {}'.format(self.opt.refine.pnet_pos_type))
        # concat pnet_inp and pred_inp, and update revidx
        final_pnet_inp = torch.cat((pnet_inp, new_pred_inp),0)
        final_revidx = torch.cat((data_dict['revidx'],new_end_voxel_id),0)
        # pointnet forward
        if self.opt.refine.pnet_model_type == 'twostage':
            occ_voxel_feat = self.pnet_model(inp_feat=final_pnet_inp, vox2point_idx=final_revidx)
        else:
            raise NotImplementedError('Does not support Pnet model type: {}'.format(self.opt.refine.pnet_model_type))
        intersect_voxel_feat_end = occ_voxel_feat[end_voxel_id]

        # pos embed
        if self.opt.refine.intersect_pos_type == 'rel':
            enter_pos = pred_pos - end_voxel_center
        else:
            enter_pos = pred_pos
        intersect_pos_embed_end = self.embed_fn(enter_pos)
        # concat inp
        inp_embed = torch.cat((intersect_voxel_feat_end, intersect_rgb_feat_end, 
                        intersect_pos_embed_end, intersect_dir_embed_end),-1)
        pred_refine_offset = self.offset_dec(inp_embed)
        pred_scaled_refine_offset = pred_refine_offset * (self.opt.refine.offset_range[1] - self.opt.refine.offset_range[0]) + self.opt.refine.offset_range[0]
        pred_pos_refine = pred_pos + pred_scaled_refine_offset * data_dict['miss_ray_dir']
        return pred_pos_refine

    def forward(self, exp_type, epoch, data_dict):
        for cur_iter in range(self.opt.refine.forward_times):
            if cur_iter == 0:
                pred_pos_refine = self.get_pred_refine(data_dict, data_dict['pred_pos'], exp_type, cur_iter)
            else:
                pred_pos_refine = self.get_pred_refine(data_dict, pred_pos_refine, exp_type, cur_iter)

        data_dict['pred_pos_refine'] = pred_pos_refine
        loss_dict_refine = self.compute_loss(data_dict, exp_type, epoch)
        return data_dict, loss_dict_refine
