"""Collect fixed-budget, naturally reached river call/fold labels against distinct policies."""
import argparse
from contextlib import closing
import gzip
import json
from pathlib import Path
import random

from collect_poker_river_calls import sample
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import PYTHON, ROOT, atomic_json, digest
from poker_flop_action_values import _nonce
from poker_reference_opponents import ReferenceOpponent
from poker_river_call_learning import RiverCallPolicy
from poker_solverlike_opponent import SolverLikeOpponent
from run_poker_preflop_confirmation import run_jobs
from train_poker_flop_repair import SEEDS, STAGES, sources


REL = Path('preflop-followup/480-v1/river-call-continuation-20261127')
STUDY = ROOT/'artifacts'/REL
PUBLIC = ROOT/'reports'/REL
PARENT = STUDY.parent/'river-call-learning-20261126'
SCRIPT = Path(__file__).resolve()
REFERENCE = SCRIPT.with_name('poker_reference_opponents.py')
PREVIOUS_COLLECTOR = SCRIPT.with_name('collect_poker_river_calls.py')


class RuleOpponent(ReferenceOpponent):
    def decide(self, state, seed):
        return dict(super().decide(state, seed), ok=True)


def field(group, seed):
    if group in ('tight-value', 'pressure'):
        return RuleOpponent(group), digest(REFERENCE)
    if group == 'public-fullhouse':
        policy = Fullhouse()
    elif group == 'solverlike':
        policy = SolverLikeOpponent()
    elif group == 'snapshot-420':
        design, _, _, _, historical = sources(seed)
        return historical[420], design['historical_selfplay_sha256']['420']
    else:
        raise ValueError(f'Unknown opponent {group}')
    return policy, policy.digest


class Table:
    """Route the current hero and the registered opponents on each street."""

    def __init__(self, split, hero_policy, opponent, historical):
        self.split = split
        self.hero_policy = hero_policy
        self.opponent = opponent
        self.historical = historical

    def start_hand(self, deal, hero, seed):
        self.hero = hero
        self.lineup = [480 if seat == hero else STAGES[random.Random(
            _nonce('opponent', seed, deal, seat)).randrange(len(STAGES))] for seat in range(6)
        ] if self.split == 'train' else None

    def decide(self, state, seed):
        actor = state['seat_to_act']
        if actor == self.hero:
            policy = self.hero_policy
        elif self.split == 'holdout' or state['street'] == 'river':
            policy = self.opponent
        elif state['street'] == 'preflop':
            policy = self.historical[480]
        else:
            policy = self.historical[self.lineup[actor]]
        return policy.decide(state, seed)


def load_design(sha):
    verify_files({STUDY/'design.json': sha, PUBLIC/'design.json': sha})
    return read(STUDY/'design.json')


def collect(sha, split, group, seed):
    spec = load_design(sha)
    plan = spec['collection'][split]
    count = plan['positions_per_seed_group']
    bounds = plan['by_group'][group][str(seed)]
    reference = read(PARENT/'selection.json')['policies'][str(seed)]
    if reference['sha256'] != spec['parent_policy_sha256'][str(seed)]:
        raise ValueError('Current selection differs from the registered seed parent')
    verify_files({reference['path']: reference['sha256'],
                  PARENT/'play-ready.json': spec['parent_play_ready_sha256'],
                  SCRIPT.with_name('poker_river_call_learning.py'): spec['parent_runtime_sha256'],
                  SCRIPT.with_name('poker_river_dominance.py'): spec['parent_guard_sha256']})
    incumbent = RiverCallPolicy(reference['path'])
    if incumbent.model is None:
        raise ValueError('Expected the selected learned river parent')
    _, base, _, _, historical = sources(seed)
    opponent, opponent_sha = field(group, seed)
    expected = spec['group_policy_sha256'][group]
    if opponent_sha != (expected[str(seed)] if isinstance(expected, dict) else expected):
        raise ValueError('Opponent source differs from the registered identity')
    table = Table(split, incumbent, opponent, historical)
    folder = STUDY/split/group/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    partial = folder/'positions.partial.gz'
    positions = 0
    generator_hash = digest(SCRIPT)
    with closing(Bridge()) as bridge, gzip.open(partial, 'xt', compresslevel=1) as stream:
        for offset, deal in enumerate(range(bounds['start'], bounds['limit'])):
            hero = offset % 6
            table.start_hand(deal, hero, seed)
            row = sample(bridge, table, incumbent, table, deal, hero)
            if row is not None:
                # The helper's raw hero response already includes the last learned river policy.
                row['incumbent_distribution'] = row.pop('v1_distribution')
                if row['incumbent_distribution'] != row['parent_distribution']:
                    raise ValueError('Parent changed on a second application of its protected call guard')
                row.update(seed=seed, split=split, group=group, opponents_by_seat=table.lineup,
                           river_policy_sha256=opponent_sha, parent_policy_sha256=reference['sha256'])
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                positions += 1
                if positions == count:
                    break
            if (offset+1)%2000 == 0:
                atomic_json(folder/'progress.json', dict(utc=utc(), positions=positions, attempted_deals=offset+1))
        engine_sha = digest(bridge.p.args[0])
    if positions != count:
        atomic_json(folder/'range-exhausted.json', dict(positions=positions, planned=count, attempted_deals=offset+1))
        raise ValueError(f'Registered range exhausted: {split}/{group}/{seed}, {positions}/{count}')
    load_design(sha)
    verify_files({SCRIPT: generator_hash, reference['path']: reference['sha256']})
    ledger = folder/'positions.jsonl.gz'
    partial.rename(ledger)
    row = dict(complete=True, completed_utc=utc(), design_sha256=sha, seed=seed, split=split, group=group,
               positions=positions, distinct_original_roots=positions, attempted_deals=offset+1,
               forced_terminal_actions=2*positions, first_card=bounds['start'], last_card=deal,
               parent_policy_sha256=reference['sha256'], river_policy_sha256=opponent_sha,
               preflop_parent_sha256=base['sha256'], native_engine_sha256=engine_sha,
               generator_sha256=generator_hash, settlement_source_sha256=digest(PREVIOUS_COLLECTOR),
               position_ledger_sha256=digest(ledger))
    atomic_json(folder/'collection-complete.json', row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-sha', required=True)
    parser.add_argument('--split', choices=['train', 'holdout'])
    parser.add_argument('--group')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    plan = load_design(args.design_sha)
    if args.seed is not None:
        collect(args.design_sha, args.split, args.group, args.seed)
        return
    jobs = {}
    for split in ('train', 'holdout'):
        for group in plan['collection'][split]['groups']:
            for seed in SEEDS:
                jobs[f'{split}-{group}-{seed}'] = [str(PYTHON), str(SCRIPT), '--design-sha', args.design_sha,
                                                   '--split', split, '--group', group, '--seed', str(seed)]
    with (STUDY/'collection.lock').open('x'):
        run_jobs(STUDY, jobs, 3, 'collection')
    rows = [read(STUDY/split/group/f'seed-{seed}/collection-complete.json') for split in plan['collection']
            for group in plan['collection'][split]['groups'] for seed in SEEDS]
    summary = dict(complete=True, completed_utc=utc(), design_sha256=args.design_sha, cells=rows,
                   original_hands=sum(r['attempted_deals'] for r in rows),
                   reached_roots=sum(r['positions'] for r in rows),
                   terminal_settlements=sum(r['forced_terminal_actions'] for r in rows))
    atomic_json(STUDY/'collection-summary.json', summary)
    atomic_json(PUBLIC/'collection-summary.json', summary)


if __name__ == '__main__':
    main()
