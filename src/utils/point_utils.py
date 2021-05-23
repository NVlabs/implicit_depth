import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import torch
import torch.nn.functional as F
import constants


def batch_get_occupied_idx(v, batch_id,
    xmin=(0., 0., 0.),
    xmax=(1., 1., 1.),
    crop_size=.125, overlap=False):

    if not torch.is_tensor(xmin):
        xmin = torch.Tensor(xmin).float().to(v.device)
    if not torch.is_tensor(xmax):
        xmax = torch.Tensor(xmax).float().to(v.device)
    # get coords of valid point w.r.t full global grid
    v = v.clone()-xmin.unsqueeze(0)
    # get resolution of voxel grids
    r = torch.ceil((xmax-xmin)/crop_size)
    # if overlap, we need to add r-1 voxel cells in between
    rr = r.long() if not overlap else (2*r-1).long()

    # create index grid
    idx_grid = torch.stack(torch.meshgrid(torch.arange(rr[0]),
                                    torch.arange(rr[1]),
                                    torch.arange(rr[2])), dim=-1).to(v.device)

    # shift_idxs for each overlapping grid: shape (1, 1, 3) for non-overlap; (1, 8, 3) for overlap after reshaping 
    shift_idxs = torch.stack(
                    torch.meshgrid(torch.arange(int(overlap)+1),
                    torch.arange(int(overlap)+1),
                    torch.arange(int(overlap)+1)), dim=-1).to(v.device)
    shift_idxs = shift_idxs.reshape(-1,3).unsqueeze(0)

    # get coords of valid point w.r.t each overlapping voxel grid. (np,1 or 8,3)
    v_xyz = v.unsqueeze(1) - shift_idxs * crop_size * 0.5
    v_xmin = v.unsqueeze(1).repeat(1,shift_idxs.shape[1],1)
    # get local voxel coord of voxel of valid point. (np, 1 or 8, 3)
    v_local_coord = torch.floor(v_xyz / crop_size).long()
    # get global voxel coord of voxel of valid point. (np, 1 or 8,3)
    if overlap:
        v_global_coord = 2 * v_local_coord + shift_idxs
        v_voxel_center = v_global_coord * crop_size * 0.5 + 0.5 * crop_size
    else:
        v_global_coord = v_local_coord.clone()
        v_voxel_center = v_global_coord * crop_size + 0.5 * crop_size
    v_rel_coord = v_xmin - v_voxel_center
    # get batch id of voxel of valid point. (np, 1 or 8, 1)
    v_bid = batch_id.clone().unsqueeze(1).repeat(1,shift_idxs.shape[1],1)
    #  we need to build a valid point id tensor so that we can accumulate the features from valid points
    v_pid = torch.arange(v_global_coord.shape[0]).to(v.device)
    v_pid = v_pid.unsqueeze(-1).repeat(1,v_global_coord.shape[1]).unsqueeze(-1).long()
    # check if every voxel of valid point is inside the full global grid.
    valid_mask = torch.ones(v_global_coord.shape[0], v_global_coord.shape[1]).bool().to(v.device)
    for i in range(3):
        valid_mask = torch.logical_and(valid_mask, v_global_coord[:,:, i] >= 0)
        valid_mask = torch.logical_and(valid_mask, v_global_coord[:,:, i] < idx_grid.shape[i])
    # the global voxel coord of valid voxel of valid point, (valid_vox_num, 3)
    valid_v_global_coord = v_global_coord[valid_mask]
    # the valid point index of valid voxel of valid point, (valid_vox_num, 1)
    valid_v_pid = v_pid[valid_mask]
    # the batch id of valid voxel of valid point, (valid_vox_num, 1)
    valid_v_bid = v_bid[valid_mask]
    valid_v_rel_coord = v_rel_coord[valid_mask]
    # concatenate batch id and point grid index before using unique. This step is necessary as we want to make sure
    # same grid index from diff batch id will not be filtered
    valid_v_bid_global_coord = torch.cat((valid_v_bid, valid_v_global_coord), dim=-1)
    # using torch.unique to get occupied voxel coord, and a reverse index. 
    # occ_bid_global_coord[revidx] = valid_v_bid_global_coord
    occ_bid_global_coord, revidx = torch.unique(valid_v_bid_global_coord, dim=0, return_inverse=True)
    return occ_bid_global_coord, revidx, valid_v_pid.reshape(-1), valid_v_rel_coord, idx_grid

    
def sample_valid_points(valid_mask, sample_num, block_x=8, block_y=8):
    bs,h,w = valid_mask.shape
    assert h % block_y == 0
    assert w % block_x == 0
    # reshape valid mask to make sure non zero returns in the block order other than column order.
    valid_mask = valid_mask.reshape(bs,h//block_y,block_y,w).permute(0,1,3,2).contiguous()
    valid_mask = valid_mask.reshape(bs,h//block_y,w//block_x,block_x,block_y).permute(0,1,2,4,3).contiguous()
    valid_idx = torch.nonzero(valid_mask, as_tuple=False)
    valid_bid = valid_idx[:,0]
    # since nonzero return in c seq. we can make sure valid_bid is sorted
    # use torch.unique_consecutive to avoid sorting
    _, example_cnt = torch.unique_consecutive(valid_bid, return_counts=True)
    bid_interval = torch.cumsum(example_cnt,0)
    bid_interval = torch.cat((torch.Tensor([0]).long().to(valid_mask.device), bid_interval),0)
    # Now we use for loop over batch dim. can be accelerated by cuda kernal
    tmp_list = []
    for i in range(bid_interval.shape[0]-1):
        sid = bid_interval[i]
        eid = bid_interval[i+1]
        cur_cnt = eid - sid
        if cur_cnt < sample_num:
            mult = np.ceil(float(sample_num)/float(cur_cnt)) - 1
            cur_points_idx = torch.arange(sid,eid).long().to(valid_mask.device)
            rand_pool = cur_points_idx.repeat(int(mult))
            nextra = sample_num - cur_cnt
            rand_pool_idx = np.random.choice(rand_pool.shape[0], nextra, replace=False)
            extra_idx = rand_pool[rand_pool_idx]
            sample_idx = torch.cat([cur_points_idx, extra_idx], dim=0)
        else:
            sample_step = cur_cnt // sample_num
            interval_num = cur_cnt // sample_step
            sample_offset = torch.randint(low=0,high=sample_step,size=(interval_num,)).to(valid_mask.device)
            sample_idx = sid + sample_offset + sample_step * torch.arange(interval_num).long().to(valid_mask.device)
            if sample_num <= sample_idx.shape[0]:
                tmp_idx = torch.randperm(sample_idx.shape[0])[:sample_num].long().to(valid_mask.device)
                sample_idx = sample_idx[tmp_idx]
            else:
                raise ValueError('Should be samller')
        
        tmp_list.append(valid_idx[sample_idx])
    sampled_valid_idx = torch.cat(tmp_list,0)
    sampled_flat_img_id = (sampled_valid_idx[:,1] * block_y + sampled_valid_idx[:,3]) * w \
                        + sampled_valid_idx[:,2] * block_x + sampled_valid_idx[:,4]
    sampled_bid = sampled_valid_idx[:,0]
    sampled_valid_idx = torch.stack((sampled_bid,sampled_flat_img_id),-1)
    assert sampled_valid_idx.shape[0] == bs * sample_num
    return sampled_valid_idx


def vis_voxel(occ_vox_bid, valid_bid, miss_bid, valid_xyz, valid_rgb,
    overlap, align, xmin, part_size, occ_vox_global_coord, mask, dst_path, cur_bid=0):

    ''' visualize and save data '''

    # steup matplotlib figure
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = plt.axes(projection='3d')
    if not align:
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.gca().invert_zaxis()
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # draw voxels, be careful about yz swap
    def draw_voxels(bound, ax, color='b'):
        from itertools import product, combinations
        for cur_bound in bound:
            xlim = cur_bound[0].tolist()
            ylim = cur_bound[1].tolist()
            zlim = cur_bound[2].tolist()
            for s, e in combinations(np.array(list(product(xlim, ylim, zlim))), 2):
                if np.sum(np.abs(s-e)) == xlim[1]-xlim[0]:
                    ax.plot3D(*zip(s, e), color=color)
    
    # transform from int voxel coord to float cam coord
    if overlap:
        bound_min = xmin.unsqueeze(0) + occ_vox_global_coord * part_size * 0.5
    else:
        bound_min = xmin.unsqueeze(0) + occ_vox_global_coord * part_size
    bound_max = bound_min + part_size
    if not align:
        bound_min = torch.cat((bound_min[:,0:1],bound_min[:,2:3],bound_min[:,1:2]),-1)
        bound_max = torch.cat((bound_max[:,0:1],bound_max[:,2:3],bound_max[:,1:2]),-1)


    # get idx of occ voxels whose bid=cur_bid, in all occ voxels among minibatch
    occ_vox_bid_mask = (occ_vox_bid == cur_bid)
    occ_curbid_idx = torch.nonzero(occ_vox_bid_mask,as_tuple=False).reshape(-1)
    # get idx of occ voxels who intersect with at least one ray, in all occ voxels among minibatch
    occ_vox_intersect_mask = (torch.sum(mask,1) != 0)
    occ_vox_mask = torch.logical_and(occ_vox_bid_mask, occ_vox_intersect_mask)
    occ_intersect_curbid_idx = torch.nonzero(occ_vox_mask, as_tuple=False).reshape(-1)

    # draw occupied and intersected voxels
    occ_curbid_bound_min = np.expand_dims(bound_min[occ_curbid_idx].cpu().numpy(),-1)
    occ_curbid_bound_max = np.expand_dims(bound_max[occ_curbid_idx].cpu().numpy(),-1)
    occ_curbid_bound = np.concatenate((occ_curbid_bound_min,occ_curbid_bound_max),-1)
    draw_voxels(occ_curbid_bound, ax, color='r')

    # draw occupied and intersected voxels
    occ_intersect_curbid_bound_min = np.expand_dims(bound_min[occ_intersect_curbid_idx].cpu().numpy(),-1)
    occ_intersect_curbid_bound_max = np.expand_dims(bound_max[occ_intersect_curbid_idx].cpu().numpy(),-1)
    occ_intersect_curbid_bound = np.concatenate((occ_intersect_curbid_bound_min,occ_intersect_curbid_bound_max),-1)
    draw_voxels(occ_intersect_curbid_bound, ax, color='b')

    # get valid xyz and color of current example
    valid_curbid_idx = torch.nonzero(valid_bid == cur_bid,as_tuple=False).reshape(-1)
    valid_xyz_curbid = valid_xyz[valid_curbid_idx].cpu().numpy()
    valid_rgb_curbid = valid_rgb[valid_curbid_idx].cpu().numpy()
    mean=np.array(constants.IMG_MEAN).reshape(1,-1)
    std=np.array(constants.IMG_NORM).reshape(1,-1)
    valid_rgb_curbid = valid_rgb_curbid * std + mean

    if not align:
        valid_xyz_curbid = np.concatenate((valid_xyz_curbid[:,0:1],valid_xyz_curbid[:,2:3],valid_xyz_curbid[:,1:2]),-1)

    # be careful about yz swap
    xs = valid_xyz_curbid[:,0]
    ys = valid_xyz_curbid[:,1]
    zs = valid_xyz_curbid[:,2]
    c = np.clip(valid_rgb_curbid,0,1)
    ax.scatter(xs, ys, zs, s=1, c=c)

    plt.savefig(dst_path)
    plt.close(fig)


def gradient(x):
    # idea from tf.image.image_gradients(image)
    # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
    # x: (b,c,h,w), float32 or float64
    # dx, dy: (b,c,h,w)

    # gradient step=1
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
    dx, dy = right - left, bottom - top 
    # dx will always have zeros in the last column, right-left
    # dy will always have zeros in the last row,    bottom-top
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def get_surface_normal(x):
    dx,dy = gradient(x)
    surface_normal = torch.cross(dx, dy, dim=1)
    surface_normal = surface_normal / (torch.norm(surface_normal,dim=1,keepdim=True)+1e-8)
    return surface_normal, dx, dy