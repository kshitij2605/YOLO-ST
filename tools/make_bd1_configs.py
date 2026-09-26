#!/usr/bin/env python3
"""Plan 08 B-D1 configs: HO0 with actor-aligned tube-query transport.

Single variable: model.apt_tube_query_actor_aligned false -> true. Periodic
checkpoints are weights-only (disk only). Writes a two-step smoke and seed 17.
"""
import os
import sys

import yaml

SRC = 'configs/yolost_v2ho_ho0_s17_30ep.yaml'


def main():
    base = yaml.safe_load(open(SRC))
    if base['model'].get('apt_tube_query_actor_aligned') is not False:
        sys.exit('parent is not actor_aligned=false')
    for name, smoke in (('yolost_v2ho_bd1_actoraligned_smoke', True),
                        ('yolost_v2ho_bd1_actoraligned_s17_30ep', False)):
        path = f'configs/{name}.yaml'
        if os.path.exists(path):
            sys.exit(f'refusing to overwrite {path}')
        cfg = yaml.safe_load(open(SRC))
        cfg['model']['apt_tube_query_actor_aligned'] = True
        cfg['output']['exp_dir'] = './experiments/' + name[len('yolost_'):]
        cfg['output']['epoch_checkpoints'] = 'weights_only'
        if smoke:
            cfg['train']['epochs'] = 1
            cfg['train']['max_steps_per_epoch'] = 3
        with open(path, 'w') as handle:
            handle.write(f'# Plan 08 B-D1 ({name}): {SRC} with apt_tube_query_actor_aligned '
                         'false -> true; weights-only periodic checkpoints\n\n')
            yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)
        print('wrote', path)


if __name__ == '__main__':
    main()
