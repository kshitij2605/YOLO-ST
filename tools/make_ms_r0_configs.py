#!/usr/bin/env python3
"""MultiSports MS-R0 configs (plan 08, sections 5C and 6.4).

The v2 architecture of V2HO-HO0-s17 (VideoMAE-L, DFL dense head, factorized
tube queries), 224x224 squash, 64-frame clips, K400 VideoMAE init, 6 epochs,
trained on the MultiSports trainval train list minus a 10% dev split and
selected on that dev split (data/multisports/dev_splits). 66 classes are
trained; the official evaluator scores 60. Weights-only periodic checkpoints.
Writes a two-step smoke and seed 17.
"""
import os
import sys

import yaml

SRC = 'configs/yolost_v2ho_ho0_s17_30ep.yaml'
DEV = './data/multisports/dev_splits/multisports_dev_GT.pkl'


def main():
    for name, smoke in (('yolost_ms_r0_224_smoke', True), ('yolost_ms_r0_224_s17_6ep', False)):
        path = f'configs/{name}.yaml'
        if os.path.exists(path):
            sys.exit(f'refusing to overwrite {path}')
        cfg = yaml.safe_load(open(SRC))
        cfg['data']['dataset'] = 'multisports'
        cfg['data']['root'] = './data/multisports/hf/data/trainval/rawframes'
        cfg['data']['annot_file'] = DEV
        cfg['data']['filter_empty_clips'] = True
        cfg['model']['num_classes'] = 66
        cfg['train']['epochs'] = 1 if smoke else 6
        cfg['output']['save_interval'] = 1
        cfg['output']['epoch_checkpoints'] = 'weights_only'
        cfg['output']['exp_dir'] = './experiments/' + name[len('yolost_'):]
        if smoke:
            cfg['train']['max_steps_per_epoch'] = 3
        with open(path, 'w') as handle:
            handle.write(f'# MultiSports MS-R0 ({name}); derived from {SRC} by tools/make_ms_r0_configs.py\n'
                         f'# dev split: {DEV} (1406 train / 168 dev videos)\n\n')
            yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)
        print('wrote', path)


if __name__ == '__main__':
    main()
