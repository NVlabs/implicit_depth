import numpy as np
import os.path as osp

''' DATA INFO '''
DATASET_NAME = {
    'cleargrasp': 'cleargrasp',
    'cleargrasp_synthetic': 'cleargrasp',
}

''' IMG_MEAN, IMG_NORM '''
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_NORM = [0.229, 0.224, 0.225]

# GRID RANGE
XMIN = [-1,-1,0]
XMAX = [1,1,2]

