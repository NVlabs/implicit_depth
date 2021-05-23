import os
import os.path as osp
from glob import glob
import yaml
from easydict import EasyDict as edict
import numpy as np
import cv2
from scipy.ndimage.measurements import label as connected_components

import torch
from torch.utils.data import Dataset

# My libraries
import utils.seg_utils as seg_utils
import utils.data_augmentation as data_augmentation
import constants


class ClearGrasp_Object_Dataset(Dataset):
    def __init__(self, dataset_subdir, exp_type, params):
        self.dataset_subdir = dataset_subdir
        self.exp_type = exp_type
        self.params = params

        self.image_paths, self.mask_paths, self.transparent_depth_paths, \
        self.opaque_depth_paths, self.camera_intrinsics = self.list_dataset(self.dataset_subdir)

        print('{} images for cleargrasp dataset'.format(len(self.image_paths)))


    def list_dataset(self, data_path):
        image_paths = []
        mask_paths = []
        transparent_depth_paths = []
        opaque_depth_paths = []
        camera_intrinsics = {}
        for i in range(len(data_path)):
            for camera in ['d415', 'd435']:
                dirpath = osp.join(data_path[i], camera)
                if not osp.exists(dirpath):
                    continue
                # collect transparent rgb, mask, depth paths
                cur_image_paths = sorted( glob(osp.join(dirpath, '*-transparent-rgb-img.jpg')) )
                cur_mask_paths = [p.replace('-transparent-rgb-img.jpg', '-mask.png') for p in cur_image_paths]
                cur_transparent_depth_paths = [p.replace('-transparent-rgb-img.jpg', '-transparent-depth-img.exr') for p in cur_image_paths]
                cur_opaque_depth_paths = [p.replace('-transparent-rgb-img.jpg', '-opaque-depth-img.exr') for p in cur_image_paths]
                
                image_paths += cur_image_paths
                mask_paths += cur_mask_paths
                transparent_depth_paths += cur_transparent_depth_paths
                opaque_depth_paths += cur_opaque_depth_paths

                # camera intrinsics
                if camera not in camera_intrinsics.keys():
                    filename = osp.join(dirpath, 'camera_intrinsics.yaml')
                    with open(filename, 'r') as f:
                        intrinsics = edict(yaml.load(f, Loader=yaml.FullLoader))
                    camera_intrinsics[camera] = intrinsics

        return image_paths, mask_paths, transparent_depth_paths, opaque_depth_paths, camera_intrinsics

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


    def __getitem__(self, idx):
        
        rgb_filename = str(self.image_paths[idx])
        rgb_img = cv2.imread(rgb_filename)
        img_size = (rgb_img.shape[0], rgb_img.shape[1])
        # get image scale, (x_s, y_s)
        scale = (self.params['img_width'] / rgb_img.shape[1], self.params['img_height'] / rgb_img.shape[0])
        # RGB image processing
        rgb_img = self.process_rgb(rgb_img)

        # Label
        labels_filename = str(self.mask_paths[idx])
        mask = seg_utils.imread_indexed(labels_filename)
        foreground_labels = self.process_label(mask)
        corrupt_mask = foreground_labels.copy()
        corrupt_mask[corrupt_mask!=0] = 1
        corrupt_mask_float = torch.from_numpy(corrupt_mask).unsqueeze(0).float()
        corrupt_mask_label = torch.from_numpy(corrupt_mask).long()

        # load cam before scaling
        if 'd415' in rgb_filename:
            camera_params = self.camera_intrinsics['d415']
        else:
            camera_params = self.camera_intrinsics['d435']

        # Transparent Depth image
        transparent_depth_filename = str(self.transparent_depth_paths[idx])
        depth_corrupt_img = data_augmentation.exr_loader(transparent_depth_filename, 1)
        # clean NaN value
        depth_corrupt_img[np.isnan(depth_corrupt_img)] = 0.
        xyz_corrupt_img = data_augmentation.compute_xyz(depth_corrupt_img, camera_params)
        # scale depth and xyz
        depth_corrupt_img = cv2.resize(depth_corrupt_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        xyz_corrupt_img = cv2.resize(xyz_corrupt_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        
        # generate corrupt mask
        valid_mask = 1 - corrupt_mask
        valid_mask[depth_corrupt_img==0] = 0
        valid_mask_float = torch.from_numpy(valid_mask).unsqueeze(0).float()
        valid_mask_label = torch.from_numpy(valid_mask).long()

        depth_corrupt_img = torch.from_numpy(depth_corrupt_img).unsqueeze(0).float()
        xyz_corrupt_img = torch.from_numpy(xyz_corrupt_img).permute(2, 0, 1).float()

        # Opaque Depth image (GT depth image)
        opaque_depth_filename = str(self.opaque_depth_paths[idx])
        depth_img = data_augmentation.exr_loader(opaque_depth_filename, 1)
        # clean NaN value
        depth_img[np.isnan(depth_img)] = 0.
        xyz_img = data_augmentation.compute_xyz(depth_img, camera_params)
        depth_img = cv2.resize(depth_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        xyz_img = cv2.resize(xyz_img, (self.params['img_width'], self.params['img_height']), interpolation=cv2.INTER_NEAREST)
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).float()
        xyz_img = torch.from_numpy(xyz_img).permute(2, 0, 1).float()

        # scale affect fx, fy, cx, cy
        scaled_fx = camera_params['fx'] * scale[0]
        scaled_fy = camera_params['fy'] * scale[1]
        scaled_cx = camera_params['cx'] * scale[0]
        scaled_cy = camera_params['cy'] * scale[1]
        
        # Used for evaluation
        dir_type = rgb_filename.split('/')[-3]
        camera_type = rgb_filename.split('/')[-2]
        img_id = rgb_filename.split('/')[-1].split('-')[0]
        item_path = '{}_{}_{}'.format(dir_type, camera_type,img_id)
        
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
            'fx': scaled_fx,
            'fy': scaled_fy,
            'cx': scaled_cx,
            'cy': scaled_cy,
            'item_path': item_path,
        }
        return sample


    def __len__(self):
        return len(self.image_paths)


def get_dataset(dataset_dir, params, exp_type, obj_type='known'):
    # get dir list
    dataset_subdir = []
    if exp_type == 'train':
        raise NotImplementedError('Real Data does not support train data.')
    else:
        if obj_type == 'novel':
            dataset_subdir.append(osp.join(dataset_dir, 'cleargrasp-dataset-test-val', 'real-test'))
        elif obj_type == 'known':
            dataset_subdir.append(osp.join(dataset_dir, 'cleargrasp-dataset-test-val', 'real-val'))
        else:
            raise NotImplementedError('OBJ type not supported.')

    # set params
    params = params.copy()
    if exp_type != 'train':
        params['use_data_augmentation'] = False
    dataset = ClearGrasp_Object_Dataset(dataset_subdir, exp_type, params)
    return dataset
