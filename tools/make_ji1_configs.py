#!/usr/bin/env python3
"""JHMDB-21 J-I1 configs (plan 08, section 5D).

Single variable against J0: initialisation from Kinetics-400 VideoMAE only ->
the UCF101-24 V2-WS2.0 EMA checkpoint (experiments/v2_ws2_0_tubequeries_30ep/
ema_final.pt), loaded with `train.py --init`, which skips tensors whose shapes
differ (the 24-class heads), so class heads start fresh. Everything else,
including the dev split and 40 epochs, matches J0 exactly.
"""
import os
import sys

import yaml

INIT = 'experiments/v2_ws2_0_tubequeries_30ep/ema_final.pt'


def main():
    for split in (1, 2, 3):
        parent = f'configs/yolost_jhmdb_j0_split{split}_s17_40ep.yaml'
        name = f'yolost_jhmdb_ji1_ucfinit_split{split}_s17_40ep'
        with open(parent) as handle:
            config = yaml.safe_load(handle)
        old = config['output']['exp_dir']
        config['output']['exp_dir'] = f'./experiments/{name[len("yolost_"):]}'
        path = f'configs/{name}.yaml'
        if os.path.exists(path):
            sys.exit(f'refusing to overwrite {path}')
        with open(path, 'w') as handle:
            handle.write(f'# JHMDB J-I1 split {split}: {parent} with --init {INIT}\n'
                         f'# Launch: train.py --config {path} --init {INIT}\n\n')
            yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)
        print(path)
        print(f'    output.exp_dir: {old} -> {config["output"]["exp_dir"]}')
        print(f'    init: K400 VideoMAE -> {INIT}')


if __name__ == '__main__':
    main()
