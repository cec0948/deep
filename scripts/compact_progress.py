"""Predeclared conditional progress check; a deal remains one block across seeds."""
import argparse
import json
from pathlib import Path
import statistics

from fullhouse_v3_common import ROOT, atomic_json, digest
from summarize_compact import interval


def progress(rows, cfg):
    before, after = min(cfg['evaluation_iterations']), max(cfg['evaluation_iterations'])
    changes = {}
    for kind in ('current', 'mixture'):
        for field in cfg['evaluation_fields']:
            differences = []
            for seed in cfg['seeds']:
                a = rows[f'{field}-seed-{seed}-{kind}-{before}']['deal_blocks']
                b = rows[f'{field}-seed-{seed}-{kind}-{after}']['deal_blocks']
                if len(a) != cfg['evaluation_deals'] or len(b) != len(a):
                    raise ValueError('Incomplete paired evaluation')
                differences.append([y-x for x, y in zip(a, b)])
            # Card blocks are shared across training seeds. Average within a
            # block FIRST, rather than treating seed x deal as independent.
            pooled = [statistics.mean(v) for v in zip(*differences)]
            changes[f'{field}/{kind}'] = dict(**interval(pooled),
                per_seed=[dict(seed=seed, **interval(v)) for seed, v in zip(cfg['seeds'], differences)])
    public, mixed, weak = [changes[f'{f}/current'] for f in ('public', 'mixed', 'noregrets')]
    conditions = dict(
        public_lower_above_zero=public['mean']-public['ci95_half_width'] > 0,
        at_least_two_public_seeds_improve=sum(v['mean'] > 0 for v in public['per_seed']) >= 2,
        mixed_lower_above_minus_25=mixed['mean']-mixed['ci95_half_width'] >= -25,
        noregrets_lower_above_minus_50=weak['mean']-weak['ci95_half_width'] >= -50)
    return dict(before=before, after=after, changes=changes, conditions=conditions,
                recommendation='eligible_for_next_bounded_stage' if all(conditions.values()) else 'pause_and_analyze',
                limitations=['Nominal 95% paired deal-block intervals, conditional on these three trained seeds',
                             'Fixed budget rule, not proof of optimality, professional strength or future improvement',
                             'No model promotion or unbounded automatic training'])


def main():
    p = argparse.ArgumentParser();p.add_argument('folder', type=Path);a = p.parse_args()
    cfg = json.loads((a.folder/'config.json').read_text())
    summary = json.loads((ROOT/'reports/compact-v4'/a.folder.name/'summary.json').read_text())
    if not summary['complete']:
        raise ValueError('All declared evaluation cells must finish first')
    rows = {p.parent.name: json.loads(p.read_text()) for p in (a.folder/'evaluation').glob('*/results.json')}
    result = progress(rows, cfg);result['checker_sha256'] = digest(Path(__file__))
    atomic_json(a.folder/'progress-decision.json', result)
    atomic_json(ROOT/'reports/compact-v4'/a.folder.name/'progress-decision.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
