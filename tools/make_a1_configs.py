#!/usr/bin/env python3
"""Axis A1 configs: trainable YOLO11-L keyframe stream on the v2ho lineage.

A1a: keyframe_freeze true -> false, keyframe backbone LR multiplier 0.1.
A1b: A1a with the keyframe backbone LR multiplier 0.1 -> 1.0.
Each for seeds 17 and 29, plus a two-step smoke of A1a for memory and speed.
BatchNorm stays in eval mode (model.keyframe_bn_eval, fixed a priori).
"""
import copy
import os
import sys

import yaml

PARENTS = {17: 'configs/yolost_v2ho_ho0_s17_30ep.yaml',
           29: 'configs/yolost_v2ho_ho0_s29_30ep.yaml'}


def build(parent, name, lr_mult, smoke=False):
    with open(parent) as handle:
        config = yaml.safe_load(handle)
    config = copy.deepcopy(config)
    changes = []

    def put(section, key, value):
        old = config[section].get(key, '<unset>')
        if old != value:
            config[section][key] = value
            changes.append(f'{section}.{key}: {old} -> {value}')

    put('model', 'keyframe_freeze', False)
    put('model', 'keyframe_bn_eval', True)
    put('train', 'keyframe_backbone_lr_multiplier', lr_mult)
    put('output', 'exp_dir', f'./experiments/{name[len("yolost_"):]}')
    if smoke:
        put('train', 'max_steps_per_epoch', 3)
        put('train', 'epochs', 1)
    path = f'configs/{name}.yaml'
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        handle.write(f'# Axis A1 ({name}); derived from {parent} by tools/make_a1_configs.py\n\n')
        yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)
    print(path)
    for change in changes:
        print('   ', change)


def main():
    build(PARENTS[17], 'yolost_v2ho_a1a_smoke', 0.1, smoke=True)
    for seed, parent in PARENTS.items():
        build(parent, f'yolost_v2ho_a1a_kftrain_lr01_s{seed}_30ep', 0.1)
        build(parent, f'yolost_v2ho_a1b_kftrain_lr1_s{seed}_30ep', 1.0)


if __name__ == '__main__':
    main()
