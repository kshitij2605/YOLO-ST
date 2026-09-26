#!/usr/bin/env python3
"""JHMDB-21 final-candidate configs (revised plan 10, workstream 5).

JFULL-split{k}-s17: the J0 recipe (WS2.0 architecture, K400 init, whole-video
clip 40, 40 epochs, EMA warmup 100) trained on the full official split-k train
list of JHMDB-GT.pkl and evaluated once on the split-k test list. Periodic
checkpoints are weights-only.
"""
import os
import pickle
import sys

import yaml

GT = './data/ucf24/JHMDB/JHMDB-GT.pkl'


def main():
    with open(GT, 'rb') as handle:
        gt = pickle.load(handle, encoding='latin1')
    for k in (1, 2, 3):
        src = 'configs/yolost_jhmdb_j0_split%d_s17_40ep.yaml' % k
        dst = 'configs/yolost_jhmdb_jfull_split%d_s17_40ep.yaml' % k
        if os.path.exists(dst):
            sys.exit('refusing to overwrite ' + dst)
        cfg = yaml.safe_load(open(src))
        cfg['data']['annot_file'] = GT
        cfg['data']['split_index'] = k - 1
        cfg['output']['exp_dir'] = './experiments/jhmdb_jfull_split%d_s17_40ep' % k
        cfg['output']['epoch_checkpoints'] = 'weights_only'
        n_train = len(gt['train_videos'][k - 1])
        n_test = len(gt['test_videos'][k - 1])
        header = ('# JFULL-split%d-s17: J0 recipe on the full official split-%d train list (%d videos); '
                  'evaluated once on split-%d test (%d videos). Derived from %s.\n\n'
                  % (k, k, n_train, k, n_test, src))
        with open(dst, 'w') as handle:
            handle.write(header)
            yaml.safe_dump(cfg, handle, sort_keys=True, default_flow_style=False)
        print(dst, 'train', n_train, 'test', n_test)


if __name__ == '__main__':
    main()
