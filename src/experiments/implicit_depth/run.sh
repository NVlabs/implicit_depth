#!/usr/bin/env sh
default_cfg_path=./experiments/implicit_depth/default_config.yaml
cfg_paths=./experiments/implicit_depth/test_refine.yaml

python main.py \
    --default_cfg_path $default_cfg_path \
    --cfg_paths $cfg_paths
