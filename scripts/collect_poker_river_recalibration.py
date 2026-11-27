"""Collect fixed-budget river observations for a frozen parent and opponent panel."""
import argparse
from contextlib import closing
import gzip
import json
from pathlib import Path
import random

from collect_poker_river_call_continuation import Table, field
from collect_poker_river_calls import sample
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import PYTHON, ROOT, atomic_json, digest
from poker_reference_opponents import checked_action
from poker_river_call_diverse import DiverseRiverCallPolicy
from run_poker_preflop_confirmation import run_jobs
from train_poker_flop_repair import sources


REL = Path('preflop-followup/480-v1/river-call-recalibration-20261127')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
PRIOR = STUDY.parent/'river-call-continuation-20261127'
SCRIPT = Path(__file__).resolve()


class MixedPricePressure:
    """Use the pressure policy's bet probability at either 1/3 or 3/2 pot."""
    def __init__(self, policy):
        self.policy = policy

    def decide(self, state, seed):
        answer = self.policy.decide(state, seed)
        if not state['raise_bounds']:
            return answer
        low, high = state['raise_bounds']
        fraction = random.Random(f'river-recalibration-price:{seed}').choice((1/3, 1.5))
        price = state['your_bet_this_street'] + state['amount_owed']
        to = price + round(fraction * (state['pot'] + state['amount_owed']))
        raise_to = checked_action(state, dict(kind='raise', to=max(low, min(high, to))))
        action = raise_to if answer['action']['kind'] == 'raise' else answer['action']
        distribution = [dict(action=raise_to, probability=row['probability'])
                        if row['action']['kind'] == 'raise' else row for row in answer['distribution']]
        return dict(answer, action=action, distribution=distribution)


def design(sha):
    verify_files({STUDY/'design.json':sha, REPORT/'design.json':sha})
    spec = read(STUDY/'design.json')
    verify_files({ROOT/'scripts'/name: checksum for name, checksum in spec['collection_source_sha256'].items()})
    return spec


def collect(sha, split, group, seed):
    spec = design(sha)
    panel = spec['collection'][split]
    start, stop = panel['ranges'][group][str(seed)]
    ref = read(PRIOR/'selection.json')['policies'][str(seed)]
    verify_files({PRIOR/'selection.json':spec['parent_selection_sha256'],
                  ref['path']:spec['parent_policy_sha256'][str(seed)]})
    incumbent = DiverseRiverCallPolicy(ref['path'])
    key = 'pressure' if group == 'mixed-price-pressure' else group
    opponent, checksum = field(key, seed)
    expected = spec['opponents'][group]
    if checksum != (expected[str(seed)] if isinstance(expected, dict) else expected):
        raise ValueError('Opponent hash differs from design')
    if group == 'mixed-price-pressure':
        opponent = MixedPricePressure(opponent)
    _, _, _, _, historical = sources(seed)
    table = Table(split, incumbent, opponent, historical)
    folder = STUDY/split/group/f'seed-{seed}'
    folder.mkdir(parents=True)
    raw = folder/'positions.jsonl.gz'
    partial = folder/'positions.partial.gz'
    positions = 0
    with closing(Bridge()) as engine, gzip.open(partial, 'xt', compresslevel=1) as handle:
        for offset, deal in enumerate(range(start, stop)):
            hero = offset % 6
            table.start_hand(deal, hero, seed)
            row = sample(engine, table, incumbent, table, deal, hero)
            if row is not None:
                if row.pop('v1_distribution') != row['parent_distribution']:
                    raise ValueError('River protection is not idempotent')
                row.update(split=split,group=group,seed=seed)
                handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                positions += 1
                if positions == panel['positions_per_cell']:
                    break
            if (offset+1)%2000 == 0:
                atomic_json(folder/'progress.json',dict(positions=positions,attempted_deals=offset+1,utc=utc()))
    if positions != panel['positions_per_cell']:
        raise ValueError(f'Card range exhausted at {split}/{group}/{seed}: {positions}')
    design(sha)
    partial.rename(raw)
    atomic_json(folder/'collection-complete.json',dict(complete=True,design_sha256=sha,
                seed=seed,split=split,group=group,positions=positions,attempted_deals=offset+1,
                first_card=start,last_card=deal,native_settlements=2*positions,opponent_sha256=checksum,
                parent_policy_sha256=ref['sha256'],position_ledger_sha256=digest(raw)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design_sha')
    parser.add_argument('--cell', nargs=3)
    args = parser.parse_args()
    spec = design(args.design_sha)
    if args.cell:
        split, group, seed = args.cell
        collect(args.design_sha,split,group,int(seed))
        return
    cells = [(split,group,str(seed)) for split,panel in spec['collection'].items()
             for group in panel['ranges'] for seed in spec['seeds']]
    jobs = {'-'.join(cell):[str(PYTHON),str(SCRIPT),args.design_sha,'--cell',*cell] for cell in cells}
    with (STUDY/'collection.lock').open('x'):
        run_jobs(STUDY,jobs,3,'collection')
    records = [read(STUDY/split/group/f'seed-{seed}'/'collection-complete.json') for split,group,seed in cells]
    result = dict(complete=True, completed_utc=utc(),design_sha256=args.design_sha,cells=records)
    for folder in (STUDY,REPORT):
        atomic_json(folder/'collection-summary.json',result)


if __name__ == '__main__':
    main()
