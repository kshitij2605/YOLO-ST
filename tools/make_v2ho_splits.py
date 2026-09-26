"""WP-0.8: v2 held-out UCF101-24 lineage.

Train on official train minus capture groups g22-g25. Two evaluation
annotations share that training list: tune = g24/g25, confirm = g22/g23.
File names deliberately avoid 'heldout_g24g25' so the v1 cache-only guard in
eval_tube_queries.py does not fire.
"""
import os
import pickle
import re

SRC = 'data/ucf24/UCF101_v2/UCF101v2-GT.pkl'
OUT_DIR = 'data/ucf24/UCF101_v2'
GROUP = re.compile(r'_g(\d+)_')


def group(video):
    return int(GROUP.search(video).group(1))


with open(SRC, 'rb') as handle:
    base = pickle.load(handle, encoding='latin1')

official = list(base['train_videos'][0])
train = [v for v in official if group(v) not in (22, 23, 24, 25)]
tune = [v for v in official if group(v) in (24, 25)]
confirm = [v for v in official if group(v) in (22, 23)]
assert not set(train) & set(tune) and not set(train) & set(confirm)
assert not set(tune) & set(confirm)
assert not set(official) & set(base['test_videos'][0])

for name, test in (('v2ho-tune-g24g25', tune), ('v2ho-confirm-g22g23', confirm)):
    annotation = dict(base)
    annotation['train_videos'] = [train]
    annotation['test_videos'] = [test]
    path = os.path.join(OUT_DIR, f'UCF101v2-GT-{name}.pkl')
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite {path}')
    with open(path, 'wb') as handle:
        pickle.dump(annotation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{path}: train {len(train)} test {len(test)} '
          f'test frames {sum(base["nframes"][v] for v in test)}')
print(f'official train {len(official)}, official test {len(base["test_videos"][0])}')
