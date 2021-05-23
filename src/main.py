import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import _init_paths
from opt import *
import importlib

def main():
    parser = argparse.ArgumentParser(description='LIDF Training')
    parser.add_argument('--default_cfg_path', default = None, help='default config file')
    parser.add_argument("--cfg_paths", type=str, nargs="+", default = None, help="List of updated config file")
    args = parser.parse_args()

    # setup opt
    if args.default_cfg_path is None:
        raise ValueError('default config path not found, should define one')
    opt = Params(args.default_cfg_path)
    if args.cfg_paths is not None:
        for cfg_path in args.cfg_paths:
            opt.update(cfg_path)
    
    # set up random or deterministic training
    if opt.seed is None:
        torch.backends.cudnn.benchmark = True 
    else:
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # set up CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.vis_gpu
    # set up trainer
    Trainer = importlib.import_module('trainers.train_' + opt.trainer_name)
    # single or multi gpu setup
    if opt.dist.ddp:
        print('Distributed Data Parallel')
        spawn_context = mp.spawn(Trainer.createAndRunTrainer, args=(opt,), nprocs=opt.dist.ngpus_per_node, join=False)
        while not spawn_context.join():
            pass

        for process in spawn_context.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    else:
        print('Single Process')
        # since we use CUDA_VISIBLE_DEVICES, gpu_id is always 0 for single GPU training
        Trainer.createAndRunTrainer(0, opt) 

if __name__ == '__main__':
    main()
