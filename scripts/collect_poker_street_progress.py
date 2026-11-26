"""Generate independent, matched action-value labels on a single reached street."""
import argparse
from collections import Counter
from contextlib import closing
import gzip
import json
from pathlib import Path
import random

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_flop_action_values import _nonce, _play, future_deck, source_cards
from poker_flop_repair import FlopRepairPolicy
from poker_multivalued_turn import Rules
from poker_public_state import CARDS
from poker_river_runtime import action_key, legal_menu
from run_poker_preflop_confirmation import run_jobs
from sample_poker_flop_diagnostic import abstract_indices
from train_poker_flop_repair import SEEDS, STAGES, OpponentsBySeat, sources

REL = Path('preflop-followup/480-v1/street-progress-20261126')
STUDY = ROOT / 'artifacts' / REL
REPORT = ROOT / 'reports' / REL
SCRIPT = Path(__file__).resolve()


def sample_root(bridge, street, parent, reach_policy, opponents, deal, hero):
    state = bridge.call(dict(op='reset', seed=deal, button=0))
    prefix = []
    while not state['terminal'] and state['street'] in ('preflop', 'flop', 'turn'):
        actor = state['seat_to_act']
        if state['street'] == street and actor == hero:
            actions = list(legal_menu(state).values())
            if len(actions) < 2:
                return None
            reply = _validate_reply(state, parent.decide(state, semantic_seed(deal, hero, state)))
            probabilities = {action_key(row['action']): row['probability'] for row in reply['distribution']}
            baseline = [probabilities.get(action_key(a), 0.) for a in actions]
            if abs(sum(baseline)-1.) > 1e-6:
                raise ValueError('Incumbent root policy includes an action outside the menu')
            return dict(deal_seed=deal, hero=hero, prefix=prefix, root_observation=state, legal_actions=actions,
                        abstract_to_concrete_index=abstract_indices(state, actions), parent_probabilities=baseline)
        if state['street'] == 'turn' and street == 'flop':
            return None
        if state['street'] == 'preflop':
            policy = parent
        elif actor == hero:
            policy = reach_policy
        else:
            policy = opponents
        action = _validate_reply(state, policy.decide(state, semantic_seed(deal, hero, state)))['action']
        prefix.append(dict(action, seat=actor))
        if len(prefix) > 512:
            raise ValueError('Source hand exceeded the action cap')
        state = bridge.call(dict(op='step', action=prefix[-1]))
    return None


def completion(fixed, board, deal, repeat):
    if tuple(board[:3]) != fixed[12:]:
        raise ValueError('Observed flop differed from its source deal')
    if len(board) == 3:
        return future_deck(fixed, deal, repeat)
    if len(board) != 4:
        raise ValueError('Expected a turn board')
    known = (*fixed, board[3])
    if len(set(known)) != 16 or not set(known).issubset(CARDS):
        raise ValueError('Invalid fixed private hands or turn board')
    remaining = [card for card in CARDS if card not in known]
    random.Random(_nonce('turn-deck', deal, repeat)).shuffle(remaining)
    return [*known, *remaining]


def reach_binding(street, seed, base_ref, parent):
    if street == 'flop':
        return parent, base_ref, None
    path = STUDY / 'flop/selected.json'
    selection = read(path)
    if not selection['complete'] or selection['selected'] not in ('candidate', 'parent'):
        raise ValueError('Flop selection is not frozen')
    ref = selection['policies'][str(seed)]
    verify_files({ref['path']:ref['sha256']})
    return FlopRepairPolicy(ref['path']), ref, digest(path)


def verify_design(key):
    verify_files({STUDY/'design.json':key, REPORT/'design.json':key})
    return read(STUDY/'design.json')


def collect(street, split, seed, design_sha):
    design = verify_design(design_sha)
    budget = design['collection'][street][split]
    bounds = budget['by_seed'][str(seed)]
    _, base_ref, _, parent, pool = sources(seed)
    reach, ref, choice_hash = reach_binding(street, seed, base_ref, parent)
    folder = STUDY/street/split/f'seed-{seed}'
    folder.mkdir()
    temp = folder/'positions.partial.gz'
    positions = branches = labels = zero_mass = 0
    assignments = Counter()
    target, repeats = budget['positions_per_seed'], budget['repeats']
    if not 1 <= target <= bounds['limit']-bounds['start'] or repeats < 1:
        raise ValueError('Invalid registered budget')
    with closing(Bridge()) as bridge, closing(Rules()) as rules, gzip.open(temp, 'xt', compresslevel=1) as out:
        for attempt, deal in enumerate(range(bounds['start'], bounds['limit'])):
            hero = attempt % 6
            lineup = [480 if seat == hero else STAGES[random.Random(_nonce('opponent', seed, deal, seat)).randrange(6)]
                      for seat in range(6)]
            opponents = OpponentsBySeat(pool, lineup)
            row = sample_root(bridge, street, parent, reach, opponents, deal, hero)
            if row is None:
                continue
            row.update(seed=seed, split=split, street=street, opponents_by_seat=lineup)
            fixed = source_cards(bridge, deal)
            payouts = [[] for _ in row['legal_actions']]
            for repeat in range(repeats):
                deck = completion(fixed, row['root_observation']['community_cards'], deal, repeat)
                for i, action in enumerate(row['legal_actions']):
                    utilities, _ = _play(rules, row, deck, action, parent, opponents, repeat, False)
                    payouts[i].append(utilities[hero])
            row['action_payoffs_chips'] = payouts
            out.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
            assignments.update(stage for seat, stage in enumerate(lineup) if seat != hero)
            positions += 1
            labels += len(payouts)
            branches += len(payouts)*repeats
            zero_mass += sum(value == 0 for value in row['parent_probabilities'])*repeats
            if positions % 128 == 0:
                atomic_json(folder/'progress.json', dict(utc=utc(), positions=positions, terminal_rollouts=branches))
            if positions == target:
                break
    if positions != target:
        raise ValueError('Registered scan cap reached before collecting enough roots')
    verify_design(design_sha)
    verify_files({ref['path']:ref['sha256'], base_ref['path']:base_ref['sha256']})
    if choice_hash:
        verify_files({STUDY/'flop/selected.json':choice_hash})
    raw = folder/'positions.jsonl.gz'
    temp.rename(raw)
    atomic_json(folder/'collection-complete.json', dict(complete=True, completed_utc=utc(), seed=seed,
                street=street, split=split, positions=positions, repeats=repeats, terminal_rollouts=branches,
                action_labels=labels, zero_parent_mass_rollouts=zero_mass, opponents_by_seat_assignment=assignments,
                first_card=bounds['start'], last_card=deal, attempted_deals=attempt+1,
                parent_sha256=base_ref['sha256'], reach_policy_sha256=ref['sha256'],
                reach_flop_policy_sha256=ref['sha256'], selected_flop_receipt_sha256=choice_hash,
                training_design_sha256=design_sha, position_ledger_sha256=digest(raw),
                generator_sha256=digest(SCRIPT), simulator_sha256=digest(rules.p.args[0]),
                bridge_sha256=digest(bridge.p.args[0])))


def launch(street, split, design_sha):
    verify_design(design_sha)
    folder = STUDY/street/split
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'collection.lock').open('x'):
        pass
    jobs = {f'seed-{seed}':[str(PYTHON), str(SCRIPT), street, split, '--design-sha', design_sha, '--seed', str(seed)]
            for seed in SEEDS}
    run_jobs(folder, jobs, len(jobs), 'collection')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('street', choices=('flop','turn'))
    parser.add_argument('split', choices=('train','dev','confirm'))
    parser.add_argument('--design-sha', required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.seed is None:
        launch(args.street, args.split, args.design_sha)
    else:
        collect(args.street, args.split, args.seed, args.design_sha)
