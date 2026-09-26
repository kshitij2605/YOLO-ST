#!/usr/bin/env python3
"""Carve deterministic, class-stratified development splits from training data.

Why this exists
---------------
For UCF101-24 the reported benchmark is the official split-1 *test* set, and
selection happens on capture groups 24/25 carved out of *train*
(``tools/make_ucf24_calibration_split.py``). The other three benchmarks need the
same discipline and did not have it:

* **JHMDB-21** reports the three official test splits.
* **AVA v2.2** reports the official *validation* set.
* **MultiSports** reports the official *validation* set, because its test
  annotations are hidden behind CodaLab.

In all three cases the reported partition must never be used to choose a
setting. This tool carves a development partition out of the **training**
videos instead, deterministically and stratified by action class, so that
selection has somewhere legitimate to happen.

Determinism follows the existing convention in
``tools/make_ucf24_owner_calibration_subsplit.py``: videos are ordered by
``sha256(f"{seed}\\0{video}")`` and the first ``fraction`` of each class is taken.
Re-running with the same seed reproduces the same split exactly.

Outputs, per dataset, under ``--output-dir``:

``<name>_dev_videos.txt``    one video identifier per line
``<name>_train_videos.txt``  the remaining training videos
``<name>_dev_split.json``    counts, seed, fraction, source SHA-256
``<name>_dev_GT.pkl``        (UCF-style datasets only) a derived annotation
                             pickle whose ``test_videos`` is the dev list and
                             whose ``train_videos`` is the reduced train list,
                             so existing evaluation paths work unchanged

Usage::

    python tools/make_dev_splits.py jhmdb \\
        --annotations data/ucf24/JHMDB/JHMDB-GT.pkl \\
        --output-dir  data/ucf24/JHMDB/dev_splits

    python tools/make_dev_splits.py multisports \\
        --annotations data/multisports/hf/data/trainval/multisports_GT.pkl \\
        --output-dir  data/multisports/dev_splits

    python tools/make_dev_splits.py ava \\
        --train-csv data/ava/annotations/ava_train_v2.2.csv \\
        --output-dir data/ava/dev_splits
"""

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path


DEFAULT_SEED = 'yolost-v2-dev-split-v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rank(seed, video):
    return hashlib.sha256(f'{seed}\0{video}'.encode('utf-8')).hexdigest()


def stratified_dev_split(videos_by_class, fraction, seed):
    """Take ``fraction`` of each class into dev, leaving at least one behind.

    Returns ``(dev, train, per_class_counts)`` with both lists sorted.
    """
    if not 0.0 < fraction < 0.5:
        raise ValueError('fraction must be between 0 and 0.5')
    dev, train, counts = set(), set(), {}
    for class_id, class_videos in sorted(videos_by_class.items()):
        ordered = sorted(set(class_videos), key=lambda video: rank(seed, video))
        if len(ordered) < 2:
            # A single-video class cannot be split; keep it in train.
            train.update(ordered)
            counts[str(class_id)] = {
                'total': len(ordered), 'dev': 0, 'train': len(ordered)
            }
            continue
        count = max(1, round(len(ordered) * fraction))
        count = min(count, len(ordered) - 1)
        dev.update(ordered[:count])
        train.update(ordered[count:])
        counts[str(class_id)] = {
            'total': len(ordered), 'dev': count, 'train': len(ordered) - count
        }
    # A video may carry several classes; dev membership wins so the two
    # partitions stay disjoint.
    train -= dev
    return sorted(dev), sorted(train), counts


def split_ucf_style(annotations, output_dir, name, fraction, seed,
                    split_index=0, write_pickle=True):
    """Split a UCF-style tube annotation pickle (JHMDB, MultiSports)."""
    annotations = Path(annotations)
    with annotations.open('rb') as handle:
        annotation = pickle.load(handle, encoding='latin1')

    train_videos = list(annotation['train_videos'][split_index])
    videos_by_class = defaultdict(list)
    for video in train_videos:
        classes = sorted(
            int(value) for value in annotation['gttubes'].get(video, {})
        )
        if not classes:
            continue
        videos_by_class[classes[0]].append(video)

    dev, reduced_train, counts = stratified_dev_split(
        videos_by_class, fraction, seed
    )
    assert not set(dev) & set(reduced_train)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{name}_dev_videos.txt').write_text(
        '\n'.join(dev) + '\n', encoding='utf-8'
    )
    (output_dir / f'{name}_train_videos.txt').write_text(
        '\n'.join(reduced_train) + '\n', encoding='utf-8'
    )

    summary = {
        'dataset': name,
        'source_annotation': str(annotations),
        'source_sha256': file_sha256(annotations),
        'split_index': split_index,
        'seed': seed,
        'fraction': fraction,
        'num_source_train_videos': len(train_videos),
        'num_dev_videos': len(dev),
        'num_train_videos': len(reduced_train),
        'per_class': counts,
    }

    if write_pickle:
        derived = dict(annotation)
        derived['train_videos'] = [reduced_train]
        derived['test_videos'] = [dev]
        pickle_path = output_dir / f'{name}_dev_GT.pkl'
        with pickle_path.open('wb') as handle:
            pickle.dump(derived, handle, protocol=4)
        summary['derived_annotation'] = str(pickle_path)
        summary['derived_sha256'] = file_sha256(pickle_path)

    (output_dir / f'{name}_dev_split.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8'
    )
    return summary


def split_ava(train_csv, output_dir, name='ava', fraction=0.1,
              seed=DEFAULT_SEED):
    """Split AVA training *videos* (not keyframes) into train and dev."""
    train_csv = Path(train_csv)
    videos_by_class = defaultdict(set)
    videos = set()
    with train_csv.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 7:
                continue
            videos.add(row[0])
            videos_by_class[int(row[6])].add(row[0])
    if not videos:
        raise ValueError(f'no rows parsed from {train_csv}')

    # Stratify on the rarest class each video contains, so tail classes keep
    # representation on both sides.
    class_size = {
        class_id: len(members) for class_id, members in videos_by_class.items()
    }
    rarest_class = {}
    for class_id, members in videos_by_class.items():
        for video in members:
            current = rarest_class.get(video)
            if current is None or class_size[class_id] < class_size[current]:
                rarest_class[video] = class_id
    grouped = defaultdict(list)
    for video, class_id in rarest_class.items():
        grouped[class_id].append(video)

    dev, reduced_train, counts = stratified_dev_split(
        grouped, fraction, seed
    )
    assert not set(dev) & set(reduced_train)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{name}_dev_videos.txt').write_text(
        '\n'.join(dev) + '\n', encoding='utf-8'
    )
    (output_dir / f'{name}_train_videos.txt').write_text(
        '\n'.join(reduced_train) + '\n', encoding='utf-8'
    )
    summary = {
        'dataset': name,
        'source_annotation': str(train_csv),
        'source_sha256': file_sha256(train_csv),
        'seed': seed,
        'fraction': fraction,
        'num_source_train_videos': len(videos),
        'num_dev_videos': len(dev),
        'num_train_videos': len(reduced_train),
        'num_classes_seen': len(videos_by_class),
        'per_class': counts,
    }
    (output_dir / f'{name}_dev_split.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8'
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    subparsers = parser.add_subparsers(dest='dataset', required=True)

    for name, default_fraction in (('jhmdb', 0.15), ('multisports', 0.10)):
        sub = subparsers.add_parser(name)
        sub.add_argument('--annotations', required=True)
        sub.add_argument('--output-dir', required=True)
        sub.add_argument('--fraction', type=float, default=default_fraction)
        sub.add_argument('--seed', default=DEFAULT_SEED)
        sub.add_argument('--split-index', type=int, default=0)
        sub.add_argument('--no-pickle', action='store_true')

    ava = subparsers.add_parser('ava')
    ava.add_argument('--train-csv', required=True)
    ava.add_argument('--output-dir', required=True)
    ava.add_argument('--fraction', type=float, default=0.10)
    ava.add_argument('--seed', default=DEFAULT_SEED)

    args = parser.parse_args(argv)

    if args.dataset == 'ava':
        summary = split_ava(
            args.train_csv, args.output_dir, fraction=args.fraction,
            seed=args.seed,
        )
    else:
        summary = split_ucf_style(
            args.annotations, args.output_dir, args.dataset,
            fraction=args.fraction, seed=args.seed,
            split_index=args.split_index, write_pickle=not args.no_pickle,
        )

    print(json.dumps(
        {key: value for key, value in summary.items() if key != 'per_class'},
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
