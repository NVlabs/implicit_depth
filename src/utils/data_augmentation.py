import torch
import random
import numpy as np
import numbers
import OpenEXR, Imath
from PIL import Image # PyTorch likes PIL instead of cv2
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import utils.seg_utils as seg_utils
import constants


##### Useful Utilities #####

def exr_loader(exr_path, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        exr_path: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr

def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    fx = camera_params['fx']
    fy = camera_params['fy']
    x_offset = camera_params['cx']
    y_offset = camera_params['cy']
    indices = seg_utils.build_matrix_of_indices(camera_params['yres'], camera_params['xres'])
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor



##### Depth Augmentations #####
def dropout_random_ellipses_4corruptmask(mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    dropout_mask = mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    zero_pixel_indices = np.array(np.where(dropout_mask == 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(zero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = zero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(dropout_mask)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_mask[tmp_mask == 1] = 1

    return dropout_mask

def dropout_random_ellipses_4mask(valid_mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    dropout_valid_mask = valid_mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(dropout_valid_mask > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        mask = np.zeros_like(dropout_valid_mask)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_valid_mask[mask == 1] = 0

    return dropout_valid_mask



def add_noise_to_depth(depth_img, noise_params):
    """ Distort depth image with multiplicative gamma noise.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img

    return depth_img

def add_noise_to_xyz(xyz_img, depth_img, noise_params):
    """ Add (approximate) Gaussian Process noise to ordered point cloud.
        This is adapted from the DexNet 2.0 codebase.

        @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    small_H, small_W = (np.array([H, W]) / noise_params['gp_rescale_factor']).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

    return xyz_img

def dropout_random_ellipses(depth_img, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    if not noise_params['enable_ellipse']:
        return depth_img, np.zeros_like(depth_img).astype(np.uint8)
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    corrupt_mask = np.zeros_like(depth_img).astype(np.uint8)
    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        mask = np.zeros_like(depth_img)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        depth_img[mask == 1] = 0
        corrupt_mask[mask == 1] = 1

    return depth_img, corrupt_mask

def dropout_random_objects(depth_img, seg_img, noise_params):
    # init depth map and corruption mask 
    depth_img = depth_img.copy()
    object_mask = np.zeros_like(depth_img).astype(np.uint8)
    # get rid of background
    obj_ids = np.unique(seg_img)
     # get rid of background
    if obj_ids[0] == 0:
        obj_ids = obj_ids[1:] 
    # no objects in the scene, return original depth
    if obj_ids.shape[0] == 0:
        return depth_img, object_mask
     # get rid of table
    if obj_ids[0] == 1:
        obj_ids = obj_ids[1:]
    # no objects in the scene, return original depth
    if obj_ids.shape[0] == 0:
        return depth_img, object_mask
    # permute obj_ids in case duplicate examples
    obj_ids = np.random.permutation(obj_ids)
    if noise_params['enable_obj_remove']:
        depth_img, object_mask_remove, obj_ids = remove_object_depth(depth_img, seg_img, obj_ids, noise_params)
        object_mask = ((object_mask + object_mask_remove) > 0)
    if noise_params['enable_obj_swap']:
        depth_img, object_mask_swap, obj_ids = swap_object_depth(depth_img, seg_img, obj_ids, noise_params)
        object_mask = ((object_mask + object_mask_swap) > 0)
    
    return depth_img, object_mask

def remove_object_depth(depth_img, seg_img, obj_ids, noise_params):
    original_obj_ids = obj_ids.copy()
    # no objects in the scene, return original depth
    if obj_ids.shape[0] < 1:
        return depth_img, np.zeros_like(depth_img).astype(np.uint8), original_obj_ids
    ''' 
        select a random object with pixel area larger than num_pixels_thre,
        if all objects are smaller than num_pixels_thre, select the largest obj
    ''' 
    depth_min, depth_max = 0., np.amax(depth_img)
    additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'])
    num_pixels, num_pixels_max = 0, 0
    object_mask = np.zeros_like(depth_img).astype(np.uint8)
    selected_obj_id = -1
    while True:
        # no objects in the scene: break
        if obj_ids.shape[0] == 0:
            break
        # select a random object
        index = np.random.choice(obj_ids.shape[0])
        obj_id = obj_ids[index]
        tmp_mask = (seg_img == obj_id).astype(np.uint8)
        num_pixels = np.count_nonzero(tmp_mask)
        if num_pixels > noise_params['num_pixels_thre']:
            object_mask = tmp_mask
            selected_obj_id = obj_id
            break
        if num_pixels > num_pixels_max:
            num_pixels_max = num_pixels
            object_mask = tmp_mask
            selected_obj_id = obj_id
        # remove selected obj_id from obj_ids to avoid duplicate selection
        obj_ids = np.delete(obj_ids, index)

    selected_obj_index = np.nonzero(original_obj_ids==selected_obj_id)[0]
    original_obj_ids = np.delete(original_obj_ids, selected_obj_index)
    if np.random.random() < noise_params['assign_prob']:
        depth_img[object_mask==1] = depth_min + additive_noise
    else:
        depth_img[object_mask==1] = depth_max + additive_noise

    return depth_img, object_mask, original_obj_ids


def swap_object_depth(depth_img, seg_img, obj_ids, noise_params):
    original_obj_ids = obj_ids.copy()
    # less than 2 objects in the scene, can't switch depth, return original depth
    if obj_ids.shape[0] < 2:
        return depth_img, np.zeros_like(depth_img).astype(np.uint8), original_obj_ids
    ''' 
        select 2 random objects satisfying rel depth larger than depth threshold. 
        If rel depth is smaller than depth threshold, return swap min depth and max depth of all objects
    ''' 
    min_depth, max_depth = 1e5, -1
    min_mask, max_mask = None, None
    min_obj_id, max_obj_id = -1,-1
    for i in range(obj_ids.shape[0]):
        cur_obj_id = obj_ids[i]
        tmp_mask = (seg_img == cur_obj_id).astype(np.uint8)
        # num_pixels = np.count_nonzero(tmp_mask)
        # if num_pixels < num_pixels_thre:
        #     continue
        mean_depth = np.mean(depth_img[tmp_mask==1]).copy()
        # update min and max depth
        if mean_depth < min_depth:
            min_depth = mean_depth
            min_mask = tmp_mask
            min_obj_id = cur_obj_id
        if mean_depth > max_depth:
            max_depth = mean_depth
            max_mask = tmp_mask
            max_obj_id = cur_obj_id
        # if rel depth larger than thre, early break
        if max_depth - min_depth > noise_params['rel_depth_thre']:
            break
    # if min_mask or max_mask does not exist, directly return
    if min_mask is None or max_mask is None:
        return depth_img, np.zeros_like(depth_img).astype(np.uint8), original_obj_ids
    # set depth of object with minimal mean depth to max depth
    depth_img[min_mask==1] = max_depth
    # set depth of object with maximal mean depth to min depth
    depth_img[max_mask==1] = min_depth
    object_mask = ((min_mask + max_mask) > 0)
    # delete min obj and max obj
    selected_obj_index = np.nonzero(original_obj_ids==min_obj_id)[0]
    original_obj_ids = np.delete(original_obj_ids, selected_obj_index)
    selected_obj_index = np.nonzero(original_obj_ids==max_obj_id)[0]
    original_obj_ids = np.delete(original_obj_ids, selected_obj_index)

    return depth_img, object_mask, original_obj_ids



##### RGB Augmentations #####

def get_rgb_aug():
    seq = iaa.Sequential([
        # # Geometric Augs
        # iaa.Resize({
        #     "height": config.train.imgHeight,
        #     "width": config.train.imgWidth
        # }, interpolation='nearest'),
        # # iaa.Fliplr(0.5),
        # # iaa.Flipud(0.5),
        # # iaa.Rot90((0, 4)),

        # Bright Patches
        iaa.Sometimes(
            0.1,
            iaa.BlendAlpha(factor=(0.2, 0.7),
                            foreground=iaa.BlendAlphaSimplexNoise(foreground=iaa.Multiply((1.5, 3.0), per_channel=False),
                                                              upscale_method='cubic',
                                                              iterations=(1, 2)),
                            name="simplex-blend")),

        # Color Space Mods
        iaa.Sometimes(
            0.3,
            iaa.OneOf([
                iaa.Add((20, 20), per_channel=0.7, name="add"),
                iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
                iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV,
                                   from_colorspace=iaa.CSPACE_RGB,
                                   children=iaa.WithChannels(0, iaa.Add((-200, 200))),
                                   name="hue"),
                iaa.WithColorspace(to_colorspace=iaa.CSPACE_HSV,
                                   from_colorspace=iaa.CSPACE_RGB,
                                   children=iaa.WithChannels(1, iaa.Add((-20, 20))),
                                   name="sat"),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.2, name="norm"),
                # iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
                iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
            ])),

        # Blur and Noise
        iaa.Sometimes(
            0.2,
            iaa.SomeOf((1, None), [
                iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
                           iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
                iaa.OneOf([
                    iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
                    iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
                    iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
                    iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
                    iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
                ]),
            ],
                       random_order=True)),

        # Colored Blocks
        iaa.Sometimes(0.2, iaa.CoarseDropout(0.02, size_px=(4, 16), per_channel=0.5, name="cdropout")),
    ])
    return seq

def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image, level = 0.1):

    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row,col,ch= image.shape
        mean = 0
        noise_level = random.uniform(0, level)
        sigma = np.random.rand(1) * noise_level * 256
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)

    return noisy.astype('uint8')



def standardize_image(image):
    """ Convert a numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes

        @return: a [H x W x 3] numpy array of np.float32
    """
    image_standardized = np.zeros_like(image).astype(np.float32)

    mean = constants.IMG_MEAN
    std = constants.IMG_NORM
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255. - mean[i]) / std[i]

    return image_standardized
