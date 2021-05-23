import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(pred, gt, reduction='mean'):
    return F.mse_loss(pred, gt, reduction=reduction)

def l1_loss(pred, gt, reduction='mean'):
    return F.l1_loss(pred, gt, reduction=reduction)

def masked_mse_loss(pred, gt, mask, reduction='mean'):
    ''' pred, gt, mask should be broadcastable, mask is 0-1 '''
    diff = (pred - gt)**2
    if reduction == 'mean':
        ele_num = torch.sum(mask)
        # avoid divide by 0
        if ele_num.item() == 0:
            ele_num += 1e-8
        return torch.sum(mask * diff) / ele_num
    else:
        return torch.sum(mask * diff)

def masked_l1_loss(pred, gt, mask, reduction='mean'):
    ''' pred, gt, mask should be broadcastable, mask is 0-1 '''
    diff = torch.abs(pred - gt)
    if reduction == 'mean':
        ele_num = torch.sum(mask)
        # avoid divide by 0
        if ele_num.item() == 0:
            ele_num += 1e-8
        return torch.sum(mask * diff) / ele_num
    else:
        return torch.sum(mask * diff)

def rmse_depth(pred, gt):
    '''pred, gt: (N,H,W) '''
    diff = (pred - gt)**2
    rmse_batch = torch.sqrt(torch.mean(diff, [1,2]))
    rmse_error = torch.mean(rmse_batch)
    return rmse_error

def masked_rmse_depth(pred, gt, mask):
    '''pred, gt, mask: (N,H,W) '''
    diff = (pred - gt)**2
    ele_num = torch.sum(mask, [1,2])
    rmse_batch = torch.sqrt(torch.sum(diff*mask, [1,2]) / (ele_num+1e-8))
    rmse_error = torch.mean(rmse_batch)
    return rmse_error