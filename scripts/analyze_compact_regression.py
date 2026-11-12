"""Read-only audit of a completed compact campaign; writes a separate report."""
import argparse
from collections import Counter
import datetime
from functools import lru_cache
import gzip
import json
from pathlib import Path
import statistics

from fullhouse_v3_common import ROOT, atomic_json, digest


def interval(values):
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** .5
    return dict(mean=mean, ci95_half_width=half, lower=mean-half, upper=mean+half)


def paired(rows, seeds, field, before_kind, before_iteration, after_kind, after_iteration):
    differences = []
    for seed in seeds:
        before = rows[f'{field}-seed-{seed}-{before_kind}-{before_iteration}']['deal_blocks']
        after = rows[f'{field}-seed-{seed}-{after_kind}-{after_iteration}']['deal_blocks']
        if len(before) != len(after):
            raise ValueError('Unpaired deal blocks')
        differences.append([b-a for a, b in zip(before, after)])
    # Seeds share cards: average within each deal before computing uncertainty.
    return dict(**interval([statistics.mean(v) for v in zip(*differences)]),
                per_seed=[dict(seed=s, **interval(v)) for s, v in zip(seeds, differences)])


def seed_range(cfg):
    start = cfg['evaluation_deal_seed']
    return dict(start=start, end=start+cfg['evaluation_deals']-1, deals=cfg['evaluation_deals'])


def overlap(a, b):
    return max(0, min(a['end'], b['end'])-max(a['start'], b['start'])+1)


def audit(folder, proposed_seed, proposed_deals):
    cfg = json.loads((folder/'config.json').read_text())
    rows = {p.parent.name: json.loads(p.read_text())
            for p in (folder/'evaluation').glob('*/results.json')}
    seeds = cfg['seeds']; fields = cfg['evaluation_fields']
    before, after = min(cfg['evaluation_iterations']), max(cfg['evaluation_iterations'])
    sha = lru_cache(None)(digest)
    checks = []
    raw_hands = 0
    for name, row in sorted(rows.items()):
        if row['deal_seed'] != cfg['evaluation_deal_seed'] or row['deals'] != cfg['evaluation_deals']:
            raise ValueError('Evaluation card configuration mismatch: '+name)
        if not row['complete'] or row['hands'] != row['deals'] * 6:
            raise ValueError('Incomplete result: '+name)
        blocks = [[None] * 6 for _ in range(row['deals'])]
        with gzip.open(folder/'evaluation'/name/'hands.jsonl.gz', 'rt') as stream:
            for line in stream:
                hand = json.loads(line); deal = hand['deal']; hero = hand['hero']
                if not 0 <= deal < len(blocks) or not 0 <= hero < 6 or blocks[deal][hero] is not None:
                    raise ValueError('Repeated or invalid raw hand')
                if hand['seed'] != cfg['evaluation_deal_seed'] + deal:
                    raise ValueError('Raw hand deck seed mismatch')
                if sum(hand['utilities']) != 0 or hand['chips'] != hand['utilities'][hero]:
                    raise ValueError('Raw payoff mismatch')
                blocks[deal][hero] = hand['chips']; raw_hands += 1
        if any(None in b for b in blocks):
            raise ValueError('Missing raw hand')
        observed = [statistics.mean(b) for b in blocks]
        if len(observed) != len(row['deal_blocks']) or any(abs(a-b)>1e-9 for a,b in zip(observed,row['deal_blocks'])):
            raise ValueError('Deal-block scores disagree with raw hands')
        metrics = interval(observed)
        if abs(metrics['mean']-row['bb_per_100'])>1e-9 or abs(metrics['ci95_half_width']-row['ci95_half_width'])>1e-9:
            raise ValueError('Reported score or interval mismatch')
        candidate = Path(row['candidate'])
        matched = sha(candidate) == row['candidate_sha256']
        if not matched:
            raise ValueError('Candidate changed: '+name)
        checks.append(dict(cell=name, candidate_sha256=row['candidate_sha256'], matches_local=True))
        if row['candidate_type'] == 'snapshot_mixture':
            manifest = json.loads(candidate.read_text())
            for policy in manifest['models']:
                if sha(candidate.parent/policy['file']) != policy['sha256']:
                    raise ValueError('Mixture member changed')
    expected = {f'{f}-seed-{s}-{k}-{n}' for f in fields for s in seeds
                for k in ('current', 'mixture') for n in cfg['evaluation_iterations']}
    expected |= {f'{f}-public' for f in fields}
    if set(rows) != expected:
        raise ValueError('Missing or unexpected evaluation cells')
    provenance = {key: sorted({str(row[key]) for row in rows.values()}) for key in
                  ('deal_seed','deals','engine_sha256','evaluator_sha256','runtime_sha256')}
    if any(len(values) != 1 for values in provenance.values()):
        raise ValueError('Evaluation implementation/provenance changed across cells')
    registry = json.loads((ROOT/'artifacts/selfplay/current.json').read_text())
    opponent_sha = sha(ROOT/registry['model'])
    if opponent_sha != registry['sha256'] or any(row['opponents']['noregrets'] != opponent_sha for row in rows.values()):
        raise ValueError('NoRegrets opponent mismatch')
    public_hashes = sorted({row['opponents']['public'] for row in rows.values() if row['opponents']['public']})
    if len(public_hashes) != 1:
        raise ValueError('Public opponent changed across cells')
    source_features = sha(folder/'source/bot/features.py')
    if any(row['source_features_sha256'] != source_features for name,row in rows.items() if '-seed-' in name):
        raise ValueError('Candidate source feature implementation mismatch')
    working_hashes = {key: dict(recorded=provenance[key][0], local=sha(path), matches=provenance[key][0]==sha(path))
                      for key,path in [('engine_sha256',ROOT/'.cache/noregrets/target/release/poker-bridge'),
                                       ('evaluator_sha256',ROOT/'scripts/evaluate_compact.py'),
                                       ('runtime_sha256',ROOT/'scripts/fullhouse_v4_runtime.py')]}
    changes = {}
    for field in fields:
        for kind in ('current','mixture'):
            changes[f'{field}/{kind}/{after}-minus-{before}'] = paired(rows,seeds,field,kind,before,kind,after)
        changes[f'{field}/mixture{after}-minus-current{after}'] = paired(rows,seeds,field,'current',after,'mixture',after)
        changes[f'{field}/mixture{after}-minus-current{before}'] = paired(rows,seeds,field,'current',before,'mixture',after)
    scores = {}; behavior = {}
    for field in fields:
        for kind in ('current','mixture'):
            for iteration in (before,after):
                records = [rows[f'{field}-seed-{s}-{kind}-{iteration}'] for s in seeds]
                counts = Counter()
                for row in records:
                    counts.update(row['behavior'])
                hands = sum(row['hands'] for row in records)
                key=f'{field}/{kind}/{iteration}'
                scores[key] = dict(**interval([statistics.mean(v) for v in zip(*(r['deal_blocks'] for r in records))]),
                                   per_seed=[dict(seed=s,bb_per_100=r['bb_per_100']) for s,r in zip(seeds,records)])
                behavior[key] = dict(hands=hands,counts=dict(counts),
                    preflop_folded_hands_fraction=counts['hero/preflop/fold']/hands,
                    preflop_raise_events_per_hand=counts['hero/preflop/raise']/hands,
                    hero_events_per_hand={k:v/hands for k,v in counts.items() if k.startswith('hero/')},
                    noregrets_fallback_fraction={street:counts[f'opponent/{street}/fallback']/counts[f'opponent/{street}/decisions']
                        for street in ('preflop','flop','turn','river') if counts[f'opponent/{street}/decisions']})
    ranges=[]
    for path in sorted((ROOT/'artifacts/compact-v4').glob('*/config.json')):
        config=json.loads(path.read_text())
        ranges.append(dict(run=path.parent.name,**seed_range(config)))
    overlaps=[]
    for i,a in enumerate(ranges):
        for b in ranges[i+1:]:
            n=overlap(a,b)
            if n:
                overlaps.append(dict(run_a=a['run'],run_b=b['run'],overlapping_decks=n,
                                     fraction_of_smaller=n/min(a['deals'],b['deals'])))
    proposed=dict(start=proposed_seed,end=proposed_seed+proposed_deals-1,deals=proposed_deals)
    proposed['overlaps_existing']=[dict(run=r['run'],overlapping_decks=overlap(proposed,r)) for r in ranges if overlap(proposed,r)]
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=folder.name,
        raw_verified_hands=raw_hands,verified_cells=len(rows),scores=scores,paired_comparisons=changes,behavior=behavior,
        provenance=dict(common_values=provenance,working_implementation_hashes=working_hashes,
            candidate_hash_checks=checks,noregrets_fixed_sha256=opponent_sha,public_fixed_sha256=public_hashes[0],
            source_features_sha256=source_features,all_mixture_members_hash_checked=True),
        deck_seed_audit=dict(existing_ranges=ranges,overlaps=overlaps,proposed=proposed,
            mechanism='Evaluator reset deck seed is evaluation_deal_seed + deal_index. Merely changing the start by 2 reuses 1998 of 2000 decks.',
            interpretation='Within-stage paired arithmetic is intact. Cross-stage fresh-card/holdout independence was overstated; action RNG also depends on deal index, so reused decks need not yield identical action trajectories.'),
        findings=[
            'Current policy at 120 folds preflop more often and raises less often than at 60; this is descriptive behavior, not proof of the training cause.',
            'Temporal mixture120 exceeds current120 on mixed and NoRegrets in these nominal paired intervals, while public difference is uncertain.',
            'The original four-condition current-policy stop rule remains failed; mixture preservation against its own60 baseline also does not certify the mixed margin.',
            'NoRegrets often uses fallback on later streets; its fixed score measures exploitation of that particular opponent rather than broad poker strength.',
            'Historical stages substantially reused card decks; next diagnostic needs an explicitly disjoint seed interval.'],
        limitations=[
            'Post hoc exploratory analysis, unadjusted nominal95% deal-block intervals and fixed trained seeds.',
            'Action counts are events; raises/calls/checks may occur several times in a hand. Preflop fold occurs at most once per hero hand.',
            'Fallback fractions count NoRegrets decisions, not hands; mixed means three public and two NoRegrets seats.',
            'Two fixed opponent models and one alternating mixed seat arrangement do not establish general or professional strength.',
            'No policy promotion, controller criterion change, old artifact mutation, or training was performed by this analyzer.'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--proposed-seed',type=int,default=2026120000)
    parser.add_argument('--proposed-deals',type=int,default=2000)
    args=parser.parse_args()
    if args.proposed_deals<2:
        parser.error('At least two proposed deals required')
    result=audit(args.folder.resolve(),args.proposed_seed,args.proposed_deals)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('run','raw_verified_hands','verified_cells','deck_seed_audit')}))
