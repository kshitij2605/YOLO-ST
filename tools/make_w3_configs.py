#!/usr/bin/env python3
"""Next-round configs after the W2 results (plan 08).

The A axis is closed (A1a null over two seeds, A1b collapsed), so idle GPUs go
to single-variable rows that need no new code:

  MultiSports  MS-I1-s17   MS-R0 with init K400 -> UCF WS2.0 EMA (train.py --init)
               MS-R0-s29   seed replicate of MS-R0 (noise floor for MS gates)
  JHMDB-21     J0R-split1-s17  J0 split 1 replicate (same config and seed)
               JF1-split{1,2,3}-s17  J0 with VideoMAE fully frozen
                                     (unfreeze_last_n_blocks 8 -> 0)
               J0-split{1,2,3}-s29   seed replicate of J0
  AVA          AVA-M1-s17  M0 with init K400 -> UCF WS2.0 EMA
               AVA-C5-s17  M0 with frame_stride 1 -> 2 (64 frames span 5.1 s)
               Both set data.uint8_clips (memory only) so that both fit in RAM
               together.

All periodic checkpoints are weights-only. Nothing already on disk is overwritten.
"""
import os
import sys

import yaml

WROTE = []


def emit(src, name, changes, header):
    path = f'configs/{name}.yaml'
    if os.path.exists(path):
        sys.exit(f'refusing to overwrite {path}')
    cfg = yaml.safe_load(open(src))
    for (section, key), value in changes.items():
        cfg[section][key] = value
    cfg['output']['exp_dir'] = './experiments/' + name[len('yolost_'):]
    cfg['output']['epoch_checkpoints'] = 'weights_only'
    with open(path, 'w') as handle:
        handle.write(f'# {header}\n# Derived from {src} by tools/make_w3_configs.py\n\n')
        yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)
    WROTE.append(path)


def main():
    ms = 'configs/yolost_ms_r0_224_s17_6ep.yaml'
    emit(ms, 'yolost_ms_i1_ucfinit_224_s17_6ep', {},
         'MS-I1-s17: MS-R0 with --init experiments/v2_ws2_0_tubequeries_30ep/ema_final.pt')
    emit(ms, 'yolost_ms_r0_224_s29_6ep', {('train', 'seed'): 29}, 'MS-R0-s29: seed replicate of MS-R0')

    emit('configs/yolost_jhmdb_j0_split1_s17_40ep.yaml', 'yolost_jhmdb_j0r_split1_s17_40ep', {},
         'J0R-split1-s17: J0 split 1 replicate (same config and seed)')
    for k in (1, 2, 3):
        j0 = f'configs/yolost_jhmdb_j0_split{k}_s17_40ep.yaml'
        emit(j0, f'yolost_jhmdb_jf1_frozen_split{k}_s17_40ep', {('model', 'unfreeze_last_n_blocks'): 0},
             f'JF1-split{k}-s17: J0 with VideoMAE fully frozen (unfreeze_last_n_blocks 8 -> 0)')
        emit(j0, f'yolost_jhmdb_j0_split{k}_s29_40ep', {('train', 'seed'): 29},
             f'J0-split{k}-s29: seed replicate of J0 split {k}')

    ava = 'configs/yolost_ava_m0_proxy_s17.yaml'
    emit(ava, 'yolost_ava_m1_ucfinit_proxy_s17', {('data', 'uint8_clips'): True},
         'AVA-M1-s17: M0 with --init experiments/v2_ws2_0_tubequeries_30ep/ema_final.pt; uint8 clips (memory only)')
    emit(ava, 'yolost_ava_c5_stride2_proxy_s17', {('data', 'frame_stride'): 2, ('data', 'uint8_clips'): True},
         'AVA-C5-s17: M0 with frame_stride 1 -> 2; uint8 clips (memory only)')
    print('\n'.join(WROTE))


if __name__ == '__main__':
    main()
