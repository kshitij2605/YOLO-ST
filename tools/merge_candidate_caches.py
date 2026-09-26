#!/usr/bin/env python3
"""Merge sharded tube-candidate caches from eval_tube_queries.py.

Every shard must come from the same checkpoint, config, annotation and
inference settings; the merged cache covers each video exactly once and is
scored with eval_cached_tube_protocol.py like an unsharded cache.

    python3 tools/merge_candidate_caches.py --out merged.pkl shard0.pkl shard1.pkl ...
"""

import argparse
import pickle
import sys

MUST_MATCH = ('version', 'config', 'checkpoint', 'annot_file', 'clip_length', 'num_classes',
              'visibility', 'min_length', 'clip_weighting', 'clip_overlap')


def merge(caches):
    if not caches:
        raise ValueError('no caches given')
    reference = caches[0]
    shard_counts = {tuple(cache.get('shard', (0, 1)))[1] for cache in caches}
    if len(shard_counts) != 1:
        raise ValueError(f'shards disagree on num_shards: {shard_counts}')
    expected = shard_counts.pop()
    indices = sorted(tuple(cache.get('shard', (0, 1)))[0] for cache in caches)
    if indices != list(range(expected)):
        raise ValueError(f'expected shards 0..{expected - 1}, got {indices}')
    merged = {key: value for key, value in reference.items() if key not in ('videos', 'shard')}
    merged['videos'] = {}
    merged['shard'] = [0, 1]
    merged['merged_from'] = expected
    for cache in caches:
        for key in MUST_MATCH:
            if cache.get(key) != reference.get(key):
                raise ValueError(f'{key} differs: {cache.get(key)!r} vs {reference.get(key)!r}')
        overlap = set(cache['videos']) & set(merged['videos'])
        if overlap:
            raise ValueError(f'{len(overlap)} videos appear in more than one shard')
        merged['videos'].update(cache['videos'])
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('caches', nargs='+')
    args = parser.parse_args(argv)
    caches = []
    for path in args.caches:
        with open(path, 'rb') as handle:
            caches.append(pickle.load(handle))
    try:
        merged = merge(caches)
    except ValueError as error:
        sys.exit(f'refusing to merge: {error}')
    with open(args.out, 'wb') as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"merged {len(caches)} shards, {len(merged['videos'])} videos -> {args.out}")


if __name__ == '__main__':
    main()
