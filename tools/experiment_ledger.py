#!/usr/bin/env python3
"""Append-only experiment ledger for the YOLO-ST v2 track.

Why this exists
---------------
The v1 record kept good per-decision JSON but its corpus-level view lived only
in prose: 380 KB of ``experiment_log.md`` and 164 KB of ``priority_plan.md``,
with 113 distinct four-metric vectors scattered through the narrative. That is
readable once and unqueryable thereafter. Extracting the list of closed
directions from it required reading 2,245 lines by hand.

This ledger stores the same information as structured events so the questions
that actually change decisions can be answered mechanically:

* which runs improved AP50 while regressing AP20, the single most repeated
  trade in v1 (global motion fusion, identity transport, the multirate
  contract);
* what a component has ever done, across every run that touched it;
* whether a proposed experiment repeats something already closed.

Design
------
**Event sourced.** ``ledger.jsonl`` is append-only; one JSON object per line,
each an event of type ``register``, ``result``, ``verdict`` or ``note``.
Current state is the fold of all events for an id. Nothing is ever rewritten or
deleted, so a result cannot be quietly revised, and the human-readable
``REGISTRY.md`` is regenerated from the events rather than typed, so it cannot
drift from the data.

**Metrics are a flat dict** with canonical keys so runs stay comparable:

===========  ==================================================
``frame``    frame mAP@0.5 (UCF101-24, JHMDB, MultiSports)
``ap20``     video mAP at spatio-temporal IoU 0.20
``ap50``     video mAP at 0.50
``strict``   video mAP averaged over 0.50:0.05:0.95
``ms_all``   MultiSports video mAP over 0.10:0.90, *not* strict
``ava_map``  AVA v2.2 mAP@0.5
``v02``      JHMDB video mAP@0.2
``v05``      JHMDB video mAP@0.5
===========  ==================================================

Every result also carries its ``partition`` and ``evaluator``, because a number
without those is not comparable to anything.

Usage
-----
::

    python tools/experiment_ledger.py register --freeze research_new/experiments/V2-WS1.1_FREEZE.json
    python tools/experiment_ledger.py result   --id V2-WS1.1 --tag final \\
        --partition ucf-test --evaluator moc-act --from-eval-log path/to/eval.log
    python tools/experiment_ledger.py verdict  --id V2-WS1.1 --verdict promoted --rationale "..."
    python tools/experiment_ledger.py compare  --id V2-WS1.1 --control V2-WS1.0
    python tools/experiment_ledger.py query    --trade
    python tools/experiment_ledger.py render
"""

import argparse
import datetime
import json
import os
import re
import sys
from collections import OrderedDict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(REPO_ROOT, 'research_new', 'experiments', 'ledger.jsonl')
REGISTRY = os.path.join(REPO_ROOT, 'research_new', 'experiments', 'REGISTRY.md')
CLOSED = os.path.join(REPO_ROOT, 'research_new', 'closed_directions.yaml')

#: Canonical metric keys, in the order they are displayed.
METRIC_ORDER = ['frame', 'ap20', 'ap50', 'strict', 'ms_all', 'v02', 'v05',
                'ava_map']

#: Higher is better for every metric we track, so a positive delta is a gain.
VERDICTS = ('registered', 'running', 'promoted', 'rejected', 'null',
            'infrastructure-invalid')

#: Patterns for the evaluators in this repository. Explicit --metric always
#: wins over anything parsed here.
EVAL_PATTERNS = [
    (r'Frame-mAP@0\.5:\s*([0-9.]+)\s*%', 'frame'),
    (r'AVA v2\.2 mAP@0\.5\s*=\s*([0-9.]+)', 'ava_map'),
    (r'mAP@0\.5IOU\s*[:=]\s*([0-9.]+)', 'ava_map'),
    (r'frameAP@0\.5\s+([0-9.]+)', 'frame'),
    (r'videoAP@0\.2\s+([0-9.]+)', 'ap20'),
    (r'videoAP@0\.5\s+([0-9.]+)', 'ap50'),
    (r'videoAP_all\s+([0-9.]+)', 'ms_all'),
    (r'\bAP20\b\D{0,12}([0-9]+\.[0-9]+)', 'ap20'),
    (r'\bAP50:95\b\D{0,12}([0-9]+\.[0-9]+)', 'strict'),
    (r'\bAP50\b\D{0,12}([0-9]+\.[0-9]+)', 'ap50'),
]


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


# ---------------------------------------------------------------------------
# Event store
# ---------------------------------------------------------------------------

def append_event(event, path=LEDGER):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(event, sort_keys=True) + '\n')


def load_events(path=LEDGER):
    if not os.path.isfile(path):
        return []
    events = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'{path}:{number} is not valid JSON: {error}'
                )
    return events


def fold(events):
    """Fold the event stream into the current state of every experiment."""
    state = OrderedDict()
    for event in events:
        run_id = event.get('id')
        if not run_id:
            continue
        kind = event.get('event')
        record = state.setdefault(run_id, {
            'id': run_id, 'results': [], 'notes': [],
            'verdict': 'registered', 'history': [],
        })
        record['history'].append(
            {'event': kind, 'ts': event.get('ts')}
        )
        if kind == 'register':
            payload = dict(event.get('payload', {}))
            payload.pop('results', None)
            record.update(payload)
            record.setdefault('registered_utc', event.get('ts'))
            record['verdict'] = payload.get('verdict', 'registered')
        elif kind == 'result':
            entry = dict(event.get('payload', {}))
            entry['recorded_utc'] = event.get('ts')
            record['results'].append(entry)
            if record.get('verdict') in ('registered', 'running'):
                record['verdict'] = 'complete'
        elif kind == 'verdict':
            record['verdict'] = event['payload']['verdict']
            if event['payload'].get('rationale'):
                record['verdict_rationale'] = event['payload']['rationale']
            record['completed_utc'] = event.get('ts')
        elif kind == 'note':
            record['notes'].append(
                {'ts': event.get('ts'), 'text': event['payload']['text']}
            )
    return state


def get_record(state, run_id):
    if run_id not in state:
        raise SystemExit(
            f'unknown experiment id: {run_id}\n'
            f'known ids: {", ".join(state) or "(none)"}'
        )
    return state[run_id]


def latest_result(record, partition=None, evaluator=None, tag=None):
    """Most recently recorded result matching the given filters."""
    matches = [
        entry for entry in record.get('results', [])
        if (partition is None or entry.get('partition') == partition)
        and (evaluator is None or entry.get('evaluator') == evaluator)
        and (tag is None or entry.get('tag') == tag)
    ]
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def parse_eval_log(path):
    """Extract canonical metrics from an evaluator log."""
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    metrics = {}
    for pattern, key in EVAL_PATTERNS:
        found = re.findall(pattern, text)
        if found:
            # Last occurrence wins: evaluators print per-class rows first and
            # the summary line last.
            metrics[key] = float(found[-1])
    return metrics


def parse_metric_args(pairs):
    metrics = {}
    for pair in pairs or []:
        if '=' not in pair:
            raise SystemExit(f'--metric expects key=value, got: {pair}')
        key, value = pair.split('=', 1)
        try:
            metrics[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f'--metric value must be numeric: {pair}')
    return metrics


# ---------------------------------------------------------------------------
# Closed directions
# ---------------------------------------------------------------------------

def load_closed(path=CLOSED):
    if not os.path.isfile(path):
        return []
    try:
        import yaml
    except ImportError:
        return []
    with open(path, encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    return data.get('closed_directions', [])


def match_closed(text, components, closed):
    """Return closed directions whose keywords appear in the description."""
    haystack = ' '.join([text or ''] + list(components or [])).lower()
    hits = []
    for entry in closed:
        for keyword in entry.get('keywords', []):
            if keyword.lower() in haystack:
                hits.append(entry)
                break
    return hits


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_register(args):
    payload = {}
    if args.freeze:
        with open(args.freeze, encoding='utf-8') as handle:
            freeze = json.load(handle)
        payload.update({
            'parent': freeze.get('parent'),
            'config': freeze.get('config'),
            'config_sha256': freeze.get('config_sha256'),
            'one_variable': freeze.get('one_variable'),
            'gate': freeze.get('gate'),
            'purpose': freeze.get('purpose'),
            'freeze': os.path.relpath(args.freeze, REPO_ROOT),
        })
        payload['id'] = args.id or freeze.get('id')
    else:
        payload['id'] = args.id

    if not payload.get('id'):
        raise SystemExit('--id is required when --freeze is not given')
    for key, value in (
        ('parent', args.parent),
        ('one_variable', args.one_variable), ('control_id', args.control),
        ('workstream', args.workstream), ('purpose', args.purpose),
    ):
        if value:
            payload[key] = value
    if args.component:
        payload['components'] = list(args.component)
    payload.setdefault('components', [])
    payload['verdict'] = 'running' if args.running else 'registered'

    state = fold(load_events(args.ledger))
    if payload['id'] in state and not args.force:
        raise SystemExit(
            f"{payload['id']} is already registered; pass --force to "
            'append a corrected registration event'
        )

    hits = match_closed(
        (payload.get('one_variable') or '') + ' '
        + (payload.get('purpose') or ''),
        payload.get('components'), load_closed(),
    )
    unacknowledged = [
        entry for entry in hits
        if entry.get('id') not in (args.acknowledge_closed or [])
    ]
    if unacknowledged:
        print('This experiment resembles directions already closed in v1:\n')
        for entry in unacknowledged:
            print(f"  [{entry.get('id')}] {entry.get('title')}")
            print(f"      evidence: {entry.get('evidence')}")
        print(
            '\nRe-run with --acknowledge-closed '
            + ' '.join(entry.get('id', '?') for entry in unacknowledged)
            + '\nif this run is deliberately different, and say how in '
              '--purpose.'
        )
        return 2
    if hits:
        payload['acknowledged_closed'] = [
            entry.get('id') for entry in hits
        ]

    append_event(
        {'ts': utc_now(), 'id': payload['id'], 'event': 'register',
         'payload': payload},
        args.ledger,
    )
    print(f"registered {payload['id']}")
    return 0


def cmd_result(args):
    metrics = {}
    if args.from_eval_log:
        metrics.update(parse_eval_log(args.from_eval_log))
        if not metrics:
            print(
                f'warning: no metrics parsed from {args.from_eval_log}',
                file=sys.stderr,
            )
    if args.from_json:
        with open(args.from_json, encoding='utf-8') as handle:
            data = json.load(handle)
        for key in METRIC_ORDER:
            if key in data:
                metrics[key] = float(data[key])
    metrics.update(parse_metric_args(args.metric))
    if not metrics:
        raise SystemExit('no metrics to record')

    state = fold(load_events(args.ledger))
    get_record(state, args.id)

    payload = {
        'tag': args.tag,
        'partition': args.partition,
        'evaluator': args.evaluator,
        'metrics': metrics,
    }
    if args.from_eval_log:
        payload['log'] = os.path.relpath(args.from_eval_log, REPO_ROOT)
    append_event(
        {'ts': utc_now(), 'id': args.id, 'event': 'result',
         'payload': payload},
        args.ledger,
    )
    print(f'recorded {args.id} [{args.tag}] ' + format_metrics(metrics))

    record = fold(load_events(args.ledger))[args.id]
    gate = record.get('gate') or {}
    if not isinstance(gate, dict):
        gate = {'text': str(gate)}
    if gate.get('metric') in metrics and gate.get('threshold') is not None:
        value = metrics[gate['metric']]
        passed = value >= float(gate['threshold'])
        print(
            f"gate {gate['metric']} >= {gate['threshold']}: "
            f"{value:.2f} -> {'PASS' if passed else 'FAIL'}"
        )
    return 0


def cmd_verdict(args):
    state = fold(load_events(args.ledger))
    get_record(state, args.id)
    if args.verdict not in VERDICTS:
        raise SystemExit(
            f'verdict must be one of: {", ".join(VERDICTS)}'
        )
    append_event(
        {'ts': utc_now(), 'id': args.id, 'event': 'verdict',
         'payload': {'verdict': args.verdict,
                     'rationale': args.rationale}},
        args.ledger,
    )
    print(f'{args.id} -> {args.verdict}')
    return 0


def cmd_note(args):
    state = fold(load_events(args.ledger))
    get_record(state, args.id)
    append_event(
        {'ts': utc_now(), 'id': args.id, 'event': 'note',
         'payload': {'text': args.text}},
        args.ledger,
    )
    print(f'noted on {args.id}')
    return 0


def format_metrics(metrics):
    parts = [
        f'{key}={metrics[key]:.2f}'
        for key in METRIC_ORDER if key in metrics
    ]
    parts += [
        f'{key}={value:.2f}' for key, value in sorted(metrics.items())
        if key not in METRIC_ORDER
    ]
    return ' '.join(parts) if parts else '(none)'


def delta(candidate, control):
    """Signed differences on the metric keys both records share."""
    return {
        key: candidate[key] - control[key]
        for key in candidate if key in control
    }


def cmd_compare(args):
    state = fold(load_events(args.ledger))
    record = get_record(state, args.id)
    control_id = args.control or record.get('control_id')
    if not control_id:
        raise SystemExit(
            f'{args.id} has no control_id; pass --control explicitly'
        )
    control = get_record(state, control_id)

    left = latest_result(record, args.partition, args.evaluator, args.tag)
    right = latest_result(control, args.partition, args.evaluator, args.tag)
    if left is None or right is None:
        raise SystemExit(
            'both runs need a result on the same partition and evaluator '
            f'(got {args.id}={left is not None}, '
            f'{control_id}={right is not None})'
        )
    if (left.get('partition'), left.get('evaluator')) != \
            (right.get('partition'), right.get('evaluator')):
        raise SystemExit(
            'refusing to compare across protocols: '
            f"{left.get('partition')}/{left.get('evaluator')} vs "
            f"{right.get('partition')}/{right.get('evaluator')}"
        )

    differences = delta(left['metrics'], right['metrics'])
    print(f"{args.id} vs {control_id}  "
          f"[{left.get('partition')}, {left.get('evaluator')}]")
    print(f'  candidate : {format_metrics(left["metrics"])}')
    print(f'  control   : {format_metrics(right["metrics"])}')
    print('  delta     : ' + ' '.join(
        f'{key}={differences[key]:+.2f}'
        for key in METRIC_ORDER if key in differences
    ))
    gains = [key for key, value in differences.items() if value > 0]
    losses = [key for key, value in differences.items() if value < 0]
    if gains and losses:
        print(
            f"  TRADE     : gains {','.join(sorted(gains))} "
            f"at the cost of {','.join(sorted(losses))}"
        )
    elif gains:
        print('  strictly better on every shared metric')
    elif losses:
        print('  strictly worse on every shared metric')
    return 0


def cmd_query(args):
    state = fold(load_events(args.ledger))
    rows = []
    for record in state.values():
        if args.workstream and record.get('workstream') != args.workstream:
            continue
        if args.verdict and record.get('verdict') != args.verdict:
            continue
        if args.component and not (
            set(args.component) & set(record.get('components', []))
        ):
            continue
        result = latest_result(record, args.partition, args.evaluator)
        metrics = result['metrics'] if result else {}
        if args.has_metric and args.has_metric not in metrics:
            continue
        rows.append((record, result, metrics))

    if args.trade:
        control_state = state
        filtered = []
        for record, result, metrics in rows:
            control_id = record.get('control_id')
            if not control_id or control_id not in control_state:
                continue
            control_result = latest_result(
                control_state[control_id],
                result.get('partition') if result else None,
                result.get('evaluator') if result else None,
            )
            if not control_result:
                continue
            differences = delta(metrics, control_result['metrics'])
            if any(v > 0 for v in differences.values()) and \
                    any(v < 0 for v in differences.values()):
                filtered.append((record, result, metrics, differences))
        if not filtered:
            print('no run trades one metric against another versus its control')
            return 0
        print('Runs that gain on one metric and lose on another '
              'versus their matched control:\n')
        for record, _, metrics, differences in filtered:
            print(f"{record['id']:<16} {format_metrics(metrics)}")
            print('                 delta ' + ' '.join(
                f'{key}={differences[key]:+.2f}'
                for key in METRIC_ORDER if key in differences
            ))
        return 0

    if args.sort and rows:
        rows.sort(
            key=lambda row: row[2].get(args.sort, float('-inf')),
            reverse=True,
        )

    if not rows:
        print('no experiments match')
        return 0
    print(f"{'id':<16} {'verdict':<14} {'partition':<14} metrics")
    for record, result, metrics in rows:
        partition = (result or {}).get('partition', '-')
        print(
            f"{record['id']:<16} {record.get('verdict', '?'):<14} "
            f"{partition:<14} {format_metrics(metrics)}"
        )
        if args.verbose and record.get('one_variable'):
            print(f"                 variable: {record['one_variable']}")
    return 0


def cmd_show(args):
    state = fold(load_events(args.ledger))
    record = get_record(state, args.id)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def render_markdown(state):
    lines = [
        '# YOLO-ST v2 experiment registry',
        '',
        '**Generated from `ledger.jsonl` by '
        '`tools/experiment_ledger.py render`. Do not edit by hand:'
        ' regenerating overwrites this file.**',
        '',
        'The ledger is append-only and event sourced, so a recorded result is'
        ' never rewritten. Every metric carries the partition and evaluator it'
        ' came from, because a number without those is not comparable.',
        '',
        f'Experiments: {len(state)}. '
        f"Generated {utc_now()}.",
        '',
        '## Runs',
        '',
        '| ID | Verdict | One variable | Partition | '
        'frame | AP20 | AP50 | strict | Gate |',
        '|---|---|---|---|---:|---:|---:|---:|---|',
    ]
    for record in state.values():
        result = latest_result(record)
        metrics = result['metrics'] if result else {}
        gate = record.get('gate') or {}
        if not isinstance(gate, dict):
            gate = {'text': str(gate)}
        if gate.get('threshold') is not None:
            gate_text = f"{gate.get('metric')} >= {gate.get('threshold')}"
        elif gate.get('text'):
            gate_text = gate['text'] if len(gate['text']) <= 40 else gate['text'][:37] + '...'
        else:
            gate_text = '-'

        def cell(key):
            return f"{metrics[key]:.2f}" if key in metrics else '-'

        variable = (record.get('one_variable') or '-')
        if len(variable) > 70:
            variable = variable[:67] + '...'
        lines.append(
            f"| {record['id']} | {record.get('verdict', '?')} | "
            f"{variable} | "
            f"{(result or {}).get('partition', '-')} | "
            f"{cell('frame')} | {cell('ap20')} | {cell('ap50')} | "
            f"{cell('strict')} | {gate_text} |"
        )

    lines += ['', '## Detail', '']
    for record in state.values():
        lines.append(f"### {record['id']}")
        lines.append('')
        if record.get('purpose'):
            lines.append(record['purpose'])
            lines.append('')
        meta = [
            ('parent', record.get('parent')),
            ('control', record.get('control_id')),
            ('config', record.get('config')),
            ('config sha256', record.get('config_sha256')),
            ('components', ', '.join(record.get('components', [])) or None),
            ('freeze', record.get('freeze')),
            ('verdict rationale', record.get('verdict_rationale')),
        ]
        for label, value in meta:
            if value:
                lines.append(f'- **{label}**: {value}')
        if record.get('results'):
            lines += ['', '| tag | partition | evaluator | metrics |',
                      '|---|---|---|---|']
            for entry in record['results']:
                lines.append(
                    f"| {entry.get('tag', '-')} | "
                    f"{entry.get('partition', '-')} | "
                    f"{entry.get('evaluator', '-')} | "
                    f"{format_metrics(entry.get('metrics', {}))} |"
                )
        for note in record.get('notes', []):
            lines.append(f"- note ({note['ts']}): {note['text']}")
        lines.append('')
    return '\n'.join(lines) + '\n'


def cmd_render(args):
    state = fold(load_events(args.ledger))
    text = render_markdown(state)
    target = args.output or REGISTRY
    with open(target, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print(f'wrote {target} ({len(state)} experiments)')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--ledger', default=LEDGER)
    sub = parser.add_subparsers(dest='command', required=True)

    register = sub.add_parser('register', help='pre-register an experiment')
    register.add_argument('--freeze')
    register.add_argument('--id')
    register.add_argument('--parent')
    register.add_argument('--control')
    register.add_argument('--workstream')
    register.add_argument('--purpose')
    register.add_argument('--one-variable', dest='one_variable')
    register.add_argument('--component', action='append')
    register.add_argument('--acknowledge-closed', nargs='*')
    register.add_argument('--running', action='store_true')
    register.add_argument('--force', action='store_true')
    register.set_defaults(func=cmd_register)

    result = sub.add_parser('result', help='attach a measured result')
    result.add_argument('--id', required=True)
    result.add_argument('--tag', default='final')
    result.add_argument('--partition', required=True)
    result.add_argument('--evaluator', required=True)
    result.add_argument('--from-eval-log', dest='from_eval_log')
    result.add_argument('--from-json', dest='from_json')
    result.add_argument('--metric', action='append')
    result.set_defaults(func=cmd_result)

    verdict = sub.add_parser('verdict', help='record a promotion decision')
    verdict.add_argument('--id', required=True)
    verdict.add_argument('--verdict', required=True)
    verdict.add_argument('--rationale')
    verdict.set_defaults(func=cmd_verdict)

    note = sub.add_parser('note', help='append a note')
    note.add_argument('--id', required=True)
    note.add_argument('--text', required=True)
    note.set_defaults(func=cmd_note)

    compare = sub.add_parser('compare', help='delta against a control')
    compare.add_argument('--id', required=True)
    compare.add_argument('--control')
    compare.add_argument('--partition')
    compare.add_argument('--evaluator')
    compare.add_argument('--tag')
    compare.set_defaults(func=cmd_compare)

    query = sub.add_parser('query', help='filter and rank experiments')
    query.add_argument('--workstream')
    query.add_argument('--verdict')
    query.add_argument('--component', action='append')
    query.add_argument('--partition')
    query.add_argument('--evaluator')
    query.add_argument('--has-metric', dest='has_metric')
    query.add_argument('--sort')
    query.add_argument(
        '--trade', action='store_true',
        help='only runs that gain one metric and lose another vs the control',
    )
    query.add_argument('--verbose', action='store_true')
    query.set_defaults(func=cmd_query)

    show = sub.add_parser('show', help='dump one experiment as JSON')
    show.add_argument('--id', required=True)
    show.set_defaults(func=cmd_show)

    render = sub.add_parser('render', help='regenerate REGISTRY.md')
    render.add_argument('--output')
    render.set_defaults(func=cmd_render)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
