#!/usr/bin/env python3
"""W1 configs for the v2 held-out (v2ho) UCF101-24 lineage.

HO0-s17 / HO0-s29: V2-WS2.0 recipe, trained on official train minus g22-g25,
evaluated on g24/g25 (tune). Two seeds measure sigma_HO, the noise floor every
later gate uses. E1: HO0-s17 with 20 epochs instead of 30.

Only annot_file, exp_dir, seed and epochs change; the script prints each change.
"""
import copy
import os
import sys

import yaml

PARENT = 'configs/yolost_v2_ws2_0_tubequeries_30ep.yaml'
TUNE = './data/ucf24/UCF101_v2/UCF101v2-GT-v2ho-tune-g24g25.pkl'

VARIANTS = {
    'yolost_v2ho_ho0_s17_30ep': {'seed': 17, 'epochs': 30,
        'header': 'V2HO-HO0-s17: WS2.0 recipe on v2ho train (minus g22-g25), eval g24/g25.'},
    'yolost_v2ho_ho0_s29_30ep': {'seed': 29, 'epochs': 30,
        'header': 'V2HO-HO0-s29: seed replicate of HO0 to measure sigma_HO.'},
    'yolost_v2ho_e1_20ep_s17': {'seed': 17, 'epochs': 20,
        'header': 'V2HO-E1: HO0-s17 with 30 -> 20 epochs (cosine rescaled).'},
}


def find_section(config, key):
    hits = [name for name, section in config.items()
            if isinstance(section, dict) and key in section]
    if len(hits) != 1:
        sys.exit(f'expected exactly one section with {key!r}, found {hits}')
    return hits[0]


def main():
    with open(PARENT) as handle:
        base = yaml.safe_load(handle)
    for name, spec in VARIANTS.items():
        config = copy.deepcopy(base)
        changes = []

        def put(key, value):
            section = find_section(config, key)
            old = config[section][key]
            if old != value:
                config[section][key] = value
                changes.append(f'{section}.{key}: {old} -> {value}')

        put('annot_file', TUNE)
        put('seed', spec['seed'])
        put('epochs', spec['epochs'])
        put('exp_dir', f'./experiments/{name[len("yolost_"):]}')
        path = f'configs/{name}.yaml'
        if os.path.exists(path):
            sys.exit(f'refusing to overwrite {path}')
        with open(path, 'w') as handle:
            handle.write(f'# {spec["header"]}\n# Derived from {PARENT} by tools/make_w1_configs.py\n\n')
            yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)
        print(path)
        for change in changes:
            print('   ', change)


if __name__ == '__main__':
    main()
