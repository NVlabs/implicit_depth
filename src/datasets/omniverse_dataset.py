import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import h5py

import torch
from torch.utils.data import Dataset

# My libraries
import utils.data_augmentation as data_augmentation
import constants




class Omniverse_Dataset(Dataset):
    def __init__(self, root_dir, exp_type, params):
        self.root_dir = root_dir
        self.exp_type = exp_type
        self.params = params

        h5_paths = sorted(glob(osp.join(self.root_dir, '*/*.h5')))
        idx = int(len(h5_paths)*self.params['split_ratio'])
        if self.exp_type == 'train':
            self.h5_paths = h5_paths[:idx]
        elif self.exp_type == 'valid':
            self.h5_paths = h5_paths[idx:]
        elif self.exp_type == 'test':
            self.h5_paths = h5_paths

        print('{} images for Omniverse {} dataset'.format(len(self.h5_paths), self.exp_type))


    def process_rgb(self, rgb_img):
        """ Process RGB image
                - random color warping
        """
        if self.exp_type == 'train' and self.params['use_data_augmentation'] and np.random.random() > 0.2:
            rgb_img = data_augmentation.chromatic_transform(rgb_img)
            rgb_img = data_augmentation.add_noise(rgb_img)
        
        rgb_img = cv2.resize(rgb_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_LINEAR)
        # BGR to RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # normalize by mean and std
        rgb_img = data_augmentation.standardize_image(rgb_img)
        rgb_img = data_augmentation.array_to_tensor(rgb_img) # Shape: [3 x H x W]

        return rgb_img

    def process_depth(self, f, camera_params, corrupt_mask_float):
        disparity = f['depth'][:]
        depth_img = 1. / (disparity+1e-8) * 0.01
        depth_img = np.clip(depth_img, 0, 4)
        xyz_img = data_augmentation.compute_xyz(depth_img, camera_params)
        inp_depth_img = depth_img.copy()
        if self.exp_type == 'train' and self.params['depth_aug']:
            inp_depth_img = data_augmentation.add_noise_to_depth(inp_depth_img, self.params)
        inp_xyz_img = data_augmentation.compute_xyz(inp_depth_img, camera_params)
        if self.exp_type == 'train' and self.params['depth_aug']:
            inp_xyz_img = data_augmentation.add_noise_to_xyz(inp_xyz_img, inp_depth_img, self.params)

        depth_img = cv2.resize(depth_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        xyz_img = cv2.resize(xyz_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        inp_depth_img = cv2.resize(inp_depth_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        inp_xyz_img = cv2.resize(inp_xyz_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        # transform to tensor
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).float()
        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()
        inp_depth_img = torch.from_numpy(inp_depth_img).unsqueeze(0).float()
        inp_xyz_img = torch.from_numpy(inp_xyz_img).permute(2, 0, 1).float()

        # get corrupt depth and xyz
        depth_corrupt_img = inp_depth_img * (1 - corrupt_mask_float)
        xyz_corrupt_img = inp_xyz_img * (1 - corrupt_mask_float)
        return depth_img, xyz_img, depth_corrupt_img, xyz_corrupt_img

    def get_corrupt_mask(self, instance_mask, semantic_mask, instance_num, corrupt_all=False, ratio_low=0.4, ratio_high=0.8):
        rng = np.random.default_rng()
        corrupt_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1]))
        if self.exp_type == 'train':
            if corrupt_all:
                corrupt_obj_num = instance_num
                corrupt_obj_ids = np.arange(instance_num)
            else:
                # randomly select corrupted objects number
                corrupt_obj_num = rng.choice(np.arange(1,instance_num+1), 1, replace=False)[0]
                # randomly select corrupted objects ids
                corrupt_obj_ids = rng.choice(instance_num, corrupt_obj_num, replace=False)
            for cur_obj_id in corrupt_obj_ids:
                cur_obj_id = cur_obj_id + 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask==cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0,0],nonzero_idx[0,1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: select partial pixels.
                else:
                    ratio = np.random.random() * (ratio_high - ratio_low) + ratio_low
                    sample_num = int(nonzero_idx.shape[0] * ratio)
                    sample_start_idx = rng.choice(nonzero_idx.shape[0]-sample_num, 1, replace=False)[0]
                    sampled_nonzero_idx = nonzero_idx[sample_start_idx:sample_start_idx+sample_num]
                corrupt_mask[sampled_nonzero_idx[:,0],sampled_nonzero_idx[:,1]] = 1
        else:
            for cur_obj_id in range(instance_num):
                cur_obj_id += 1
                nonzero_idx = np.transpose(np.nonzero(instance_mask==cur_obj_id))
                if nonzero_idx.shape[0] == 0:
                    continue
                # transparent objects: select all pixels
                if semantic_mask[nonzero_idx[0,0],nonzero_idx[0,1]] == 2:
                    sampled_nonzero_idx = nonzero_idx
                # opaque objects: skip
                else:
                    continue
                corrupt_mask[sampled_nonzero_idx[:,0],sampled_nonzero_idx[:,1]] = 1
        
        return corrupt_mask


    def get_cam_params(self, cam_dataset, img_size):
        # camera
        img_h, img_w = img_size
        cam_params = {}
        cam2world = cam_dataset['pose'][:].T
        cam2world[0:3,-1] *= 0.01
        cam_params['cam2world'] = cam2world
        cam_params['rot_mat'] = cam2world[0:3,0:3]
        focal_length = cam_dataset['focal_length'][:][0]
        horizontal_aperture = cam_dataset['horizontal_aperture'][:][0]
        vertical_aperture = cam_dataset['vertical_aperture'][:][0]
        cam_params['fx'] = focal_length / horizontal_aperture * img_w
        cam_params['fy'] = focal_length / vertical_aperture * img_h
        cam_params['cx'] = img_w // 2
        cam_params['cy'] = img_h // 2
        cam_params['xres'] = img_w
        cam_params['yres'] = img_h

        return cam_params

    def __getitem__(self, idx):

        f = h5py.File(self.h5_paths[idx], "r")

        # rgb
        rgb_img = f['rgb_glass'][:]
        # RGB to BGR, consistent with cv2
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        
        img_size = (rgb_img.shape[0], rgb_img.shape[1])
        # get image scale, (x_s, y_s)
        scale = (self.params['img_width'] / rgb_img.shape[1], self.params['img_height'] / rgb_img.shape[0])
        # RGB image processing
        rgb_img = self.process_rgb(rgb_img)

        # segmentation
        instance_seg = f['instance_seg'][:]
        instance_id = np.arange(1,instance_seg.shape[0]+1).reshape(-1,1,1)
        instance_mask = np.sum(instance_seg * instance_id,0).astype(np.uint8)
        instance_mask = cv2.resize(instance_mask, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)

        semantic_seg = f['semantic_seg'][:]
        semantic_id = np.arange(1,semantic_seg.shape[0]+1).reshape(-1,1,1)
        semantic_mask = np.sum(semantic_seg * semantic_id,0).astype(np.uint8)
        semantic_mask = cv2.resize(semantic_mask, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)

        corrupt_mask = self.get_corrupt_mask(instance_mask, semantic_mask, instance_seg.shape[0], corrupt_all=self.params['omni_corrupt_all'], ratio_low=0.3, ratio_high=0.7)
        corrupt_mask_float = torch.from_numpy(corrupt_mask).unsqueeze(0).float()
        corrupt_mask_label = torch.from_numpy(corrupt_mask.copy()).long()
        
        # load cam data before scaling
        camera_params = self.get_cam_params(f['camera'], img_size)
        
        # depth
        depth_img, xyz_img, \
        depth_corrupt_img, xyz_corrupt_img = self.process_depth(f, camera_params, corrupt_mask_float)
        
        # valid mask
        valid_mask = 1 - corrupt_mask.copy()
        if self.exp_type == 'train' and self.params['use_data_augmentation'] and np.random.random() > 0.2:
            valid_mask = data_augmentation.dropout_random_ellipses_4mask(valid_mask, self.params)
        valid_mask_float = torch.from_numpy(valid_mask).unsqueeze(0).float()
        valid_mask_label = torch.from_numpy(valid_mask).long()

        if self.exp_type == 'train':
            if self.params['corrupt_table']:
                corrupt_mask = data_augmentation.dropout_random_ellipses_4corruptmask(corrupt_mask, self.params)
                # prepare corrupt mask
                corrupt_mask_float = torch.from_numpy(corrupt_mask).unsqueeze(0).float()
                corrupt_mask_label = torch.from_numpy(corrupt_mask).long()
            elif self.params['corrupt_all_pix']:
                new_corrupt_mask = np.ones_like(corrupt_mask)
                # prepare corrupt mask
                corrupt_mask_float = torch.from_numpy(new_corrupt_mask).unsqueeze(0).float()
                corrupt_mask_label = torch.from_numpy(new_corrupt_mask).long()

        # scale affect fx, fy, cx, cy
        camera_params['fx'] *= scale[0]
        camera_params['fy'] *= scale[1]
        camera_params['cx'] *= scale[0]
        camera_params['cy'] *= scale[1]

        item_path = self.h5_paths[idx]
        sample = {
            'rgb': rgb_img,
            'depth_corrupt': depth_corrupt_img,
            'xyz_corrupt': xyz_corrupt_img,
            'depth': depth_img,
            'xyz' : xyz_img,
            'corrupt_mask': corrupt_mask_float,
            'corrupt_mask_label': corrupt_mask_label,
            'valid_mask': valid_mask_float,
            'valid_mask_label': valid_mask_label,
            'fx': camera_params['fx'],
            'fy': camera_params['fy'],
            'cx': camera_params['cx'],
            'cy': camera_params['cy'],
            'cam_rot': camera_params['rot_mat'],
            'item_path': item_path,
        }
        return sample


    def __len__(self):
        return len(self.h5_paths)


def get_dataset(root_dir, params, exp_type):
    # set params
    params = params.copy()
    if exp_type != 'train':
        params['use_data_augmentation'] = False
    
    if exp_type == 'train':
        dataset_dir = osp.join(root_dir, 'train')
    elif exp_type == 'valid':
        dataset_dir = osp.join(root_dir, 'train')
    elif exp_type == 'test':
        dataset_dir = osp.join(root_dir, 'small_test')

    dataset = Omniverse_Dataset(dataset_dir, exp_type, params)
    return dataset
