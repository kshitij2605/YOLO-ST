#!/usr/bin/env python3
"""AVA v2.2 proxy configs (plan 08, sections 5C/5D and 6.3).

AVA-M0: WS2.0 architecture with tube queries off, 60 multi-label classes,
64-frame clips centred on the keyframe at stride 1, dense classification and
objectness supervised on the keyframe only, K400 VideoMAE init, 2 epochs on
30% of AVA-dev-train keyframes, selected on AVA-dev (34 held-out train videos).
Uses VideoMAE activation checkpointing (memory only).

AVA-M9 (supplement): M0 with keyframe-only supervision off.
Also a two-step smoke of M0.
"""
import copy
import os
import sys

import yaml

PARENT = 'configs/yolost_v2_ws2_0_tubequeries_30ep.yaml'
DEV = './data/ava/dev_splits'


def build(name, overrides, smoke=False):
    with open(PARENT) as handle:
        config = copy.deepcopy(yaml.safe_load(handle))
    changes = []

    def put(section, key, value):
        old = config[section].get(key, '<unset>')
        if old != value:
            config[section][key] = value
            changes.append(f'{section}.{key}: {old} -> {value}')

    put('data', 'dataset', 'ava')
    put('data', 'frames_root', './data/ava/frames')
    put('data', 'train_annot', f'{DEV}/ava_dev_train.csv')
    put('data', 'val_annot', f'{DEV}/ava_dev_val.csv')
    put('data', 'clip_length', 64)
    put('data', 'multi_label', True)
    put('data', 'keyframe_position', 'center')
    put('data', 'frame_stride', 1)
    put('data', 'supervised_frames', True)
    put('data', 'max_train_samples', 55000)
    put('data', 'max_val_samples', 256)
    put('model', 'num_classes', 60)
    put('model', 'apt_tube_queries', False)
    put('model', 'backbone_gradient_checkpointing', True)
    put('train', 'epochs', 2)
    put('output', 'save_interval', 1)
    for (section, key), value in overrides.items():
        put(section, key, value)
    if smoke:
        put('train', 'max_steps_per_epoch', 3)
        put('train', 'epochs', 1)
    put('output', 'exp_dir', f'./experiments/{name[len("yolost_"):]}')
    path = f'configs/{name}.yaml'
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        handle.write(f'# AVA proxy ({name}); derived from {PARENT} by tools/make_ava_configs.py\n'
                     '# Score with tools/export_ava_detections.py on data/ava/dev_splits/ava_dev_val.csv\n'
                     '# with --exclusions data/ava/dev_splits/ava_dev_excluded.csv\n\n')
        yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)
    print(path)
    for change in changes:
        print('   ', change)


def main():
    build('yolost_ava_m0_smoke', {}, smoke=True)
    build('yolost_ava_m0_proxy_s17', {})
    build('yolost_ava_m9_nokeymask_proxy_s17', {('data', 'supervised_frames'): False})


if __name__ == '__main__':
    main()
