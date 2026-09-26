#!/usr/bin/env python3
"""JHMDB-21 J0 configs (plan 08, section 6.2).

WS2.0 architecture, K400 VideoMAE init, whole-video clips of 40 frames, 40
epochs, EMA warmup 100 (the run has only ~700 optimiser steps), trained on each
official split's train list minus its 15% dev videos and selected on dev.
Frame evaluation uses clip weighting 'none' (one clip per video).

Writes J0 for splits 1-3 with seed 17, and a two-step smoke on split 1.
"""
import copy
import os
import sys

import yaml

PARENT = 'configs/yolost_v2_ws2_0_tubequeries_30ep.yaml'


def build(split, seed, smoke=False):
    with open(PARENT) as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    name = (f'yolost_jhmdb_j0_smoke' if smoke
            else f'yolost_jhmdb_j0_split{split}_s{seed}_40ep')
    changes = []

    def put(section, key, value):
        old = config[section].get(key, '<unset>')
        if old != value:
            config[section][key] = value
            changes.append(f'{section}.{key}: {old} -> {value}')

    put('data', 'dataset', 'jhmdb')
    put('data', 'root', './data/ucf24/JHMDB/Frames')
    put('data', 'annot_file', f'./data/ucf24/JHMDB/dev_splits/split{split}/jhmdb_dev_GT.pkl')
    put('data', 'split_index', 0)
    put('data', 'clip_length', 40)
    put('model', 'num_classes', 21)
    put('train', 'epochs', 1 if smoke else 40)
    put('train', 'ema_warmup', 100)
    put('train', 'seed', seed)
    put('output', 'save_interval', 10)
    put('output', 'exp_dir', f'./experiments/{name[len("yolost_"):]}')
    if smoke:
        put('train', 'max_steps_per_epoch', 3)
    path = f'configs/{name}.yaml'
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        handle.write(f'# JHMDB-21 J0 ({name}); derived from {PARENT} by tools/make_j0_configs.py\n'
                     '# Evaluate frame mAP with --clip-weighting none (one clip per video).\n\n')
        yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)
    print(path)
    for change in changes:
        print('   ', change)


def main():
    build(1, 17, smoke=True)
    for split in (1, 2, 3):
        build(split, 17)


if __name__ == '__main__':
    main()
