import os
import os.path as osp
from glob import glob
import json
import numpy as np
import cv2
from scipy.ndimage.measurements import label as connected_components
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset

# My libraries
import utils.seg_utils as seg_utils
import utils.data_augmentation as data_augmentation
import constants


class ClearGrasp_Syn_Object_Dataset(Dataset):
    def __init__(self, dataset_subdir, exp_type, params):
        self.dataset_subdir = dataset_subdir
        self.exp_type = exp_type
        self.params = params

        image_paths, mask_paths, depth_paths, json_paths = self.list_dataset(self.dataset_subdir)
        idx = int(len(image_paths)*self.params['split_ratio'])
        if exp_type == 'train':
            self.image_paths = image_paths[:idx]
            self.mask_paths = mask_paths[:idx]
            self.depth_paths = depth_paths[:idx]
            self.json_paths = json_paths[:idx]
        elif exp_type in ['valid','test']:
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            self.depth_paths = depth_paths
            self.json_paths = json_paths

        print('{} images for cleargrasp synthetic dataset'.format(len(self.image_paths)))


    def list_dataset(self, data_path):
        image_paths = []
        mask_paths = []
        depth_paths = []
        json_paths = []
        for i in range(len(data_path)):
            # collect transparent rgb, mask, depth paths
            cur_img_paths = sorted( glob(osp.join(data_path[i], '*', 'rgb-imgs', '*-rgb.jpg')) )
            cur_mask_paths = [p.replace('rgb-imgs', 'segmentation-masks').replace('-rgb.jpg', '-segmentation-mask.png') for p in cur_img_paths]
            cur_depth_paths = [p.replace('rgb-imgs', 'depth-imgs-rectified').replace('-rgb.jpg', '-depth-rectified.exr') for p in cur_img_paths]
            cur_json_paths = [p.replace('rgb-imgs', 'json-files').replace('-rgb.jpg', '-masks.json') for p in cur_img_paths]
            image_paths += cur_img_paths
            mask_paths += cur_mask_paths
            depth_paths += cur_depth_paths
            json_paths += cur_json_paths

        return image_paths, mask_paths, depth_paths, json_paths

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

    def process_depth(self, depth_filename, camera_params, corrupt_mask_float):
        depth_img = data_augmentation.exr_loader(depth_filename, 1)
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
        # get corrupt depth and xyz. 
        depth_corrupt_img = inp_depth_img * (1 - corrupt_mask_float)
        xyz_corrupt_img = inp_xyz_img * (1 - corrupt_mask_float)
        
        return depth_img, xyz_img, depth_corrupt_img, xyz_corrupt_img

    def process_label(self, mask):
        """ Process foreground_labels
        """
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        foreground_labels, num_components = connected_components(mask == 255)

        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels

        foreground_labels = cv2.resize(foreground_labels, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)

        return foreground_labels

    def get_cam_params(self, json_filename, img_size):
        meta_data = json.load(open(json_filename, 'r'))
        # If the pixel is square, then fx=fy. Also note this is the cam params before scaling
        if 'camera' not in meta_data.keys() or 'field_of_view' not in meta_data['camera'].keys():
            fov_x = 1.2112585306167603
            fov_y = 0.7428327202796936
        else:
            fov_x = meta_data['camera']['field_of_view']['x_axis_rads']
            fov_y = meta_data['camera']['field_of_view']['y_axis_rads']
        if 'image' not in meta_data.keys():
            img_h = img_size[0]
            img_w = img_size[1]
        else:
            img_h = meta_data['image']['height_px']
            img_w = meta_data['image']['width_px']

        if 'camera' not in meta_data.keys() or 'world_pose' not in meta_data['camera'].keys() or \
            'rotation' not in meta_data['camera']['world_pose'].keys() or \
            'quaternion' not in meta_data['camera']['world_pose']['rotation'].keys():
            raise ValueError('No quaternion: {}'.format(json_filename))
        else:
            q = meta_data['camera']['world_pose']['rotation']['quaternion']
            quaternion = np.array([q[1],q[2],q[3],q[0]])
            r = R.from_quat(quaternion)
            rot_from_q = r.as_matrix().astype(np.float32)
            world_pose  = np.array(meta_data['camera']['world_pose']['matrix_4x4']).astype(np.float32)
        
        fx = img_w*0.5 / np.tan(fov_x*0.5)
        fy = img_h*0.5 / np.tan(fov_y*0.5)
        cx = img_w*0.5
        cy = img_h*0.5
        camera_params = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'yres': img_h,
            'xres': img_w,
            'world_pose': world_pose,
            'rot_mat': rot_from_q,
        }
        return camera_params

    def __getitem__(self, idx):

        
        # RGB image
        rgb_filename = str(self.image_paths[idx])
        rgb_img = cv2.imread(rgb_filename)
        img_size = (rgb_img.shape[0], rgb_img.shape[1])
        # get image scale, (x_s, y_s)
        scale = (self.params['img_width'] / rgb_img.shape[1], self.params['img_height'] / rgb_img.shape[0])
        # RGB image processing
        rgb_img = self.process_rgb(rgb_img)

        # read transparent mask
        labels_filename = str(self.mask_paths[idx])
        mask = seg_utils.imread_indexed(labels_filename)
        foreground_labels = self.process_label(mask)
        corrupt_mask = foreground_labels.copy()
        corrupt_mask[corrupt_mask!=0] = 1
        # prepare corrupt mask
        corrupt_mask_float = torch.from_numpy(corrupt_mask).unsqueeze(0).float()
        corrupt_mask_label = torch.from_numpy(corrupt_mask).long()

        # load cam data before scaling
        json_filename = str(self.json_paths[idx])
        camera_params = self.get_cam_params(json_filename, img_size)

        # Process Depth image
        depth_filename = str(self.depth_paths[idx])
        depth_img, xyz_img,\
        depth_corrupt_img, xyz_corrupt_img = self.process_depth(depth_filename, camera_params, corrupt_mask_float)

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
        
        # Used for evaluation
        item_path = rgb_filename
        
        assert rgb_filename.split('/')[-3] == labels_filename.split('/')[-3] == \
                json_filename.split('/')[-3] == depth_filename.split('/')[-3],'{}, {}, {}, {}'.format(
                    rgb_filename, labels_filename, json_filename, depth_filename)
        
        assert rgb_filename.split('/')[-1].split('-')[0] == labels_filename.split('/')[-1].split('-')[0] == \
                json_filename.split('/')[-1].split('-')[0] == depth_filename.split('/')[-1].split('-')[0],'{}, {}, {}, {}'.format(
                    rgb_filename, labels_filename, json_filename, depth_filename)

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
        return len(self.image_paths)


def get_dataset(dataset_dir, params, exp_type, obj_type='known'):
    # get dir list
    dataset_subdir = []
    if exp_type == 'train':
        dataset_subdir.append(osp.join(dataset_dir, 'cleargrasp-dataset-train'))
    elif exp_type in ['valid', 'test']:
        if obj_type == 'novel':
            dataset_subdir.append(osp.join(dataset_dir, 'cleargrasp-dataset-test-val', 'synthetic-test'))
        elif obj_type == 'known':
            dataset_subdir.append(osp.join(dataset_dir, 'cleargrasp-dataset-test-val', 'synthetic-val'))

    # set params
    params = params.copy()
    if exp_type != 'train':
        params['use_data_augmentation'] = False
    dataset = ClearGrasp_Syn_Object_Dataset(dataset_subdir, exp_type, params)
    return dataset
