"""Recompute a frozen flop diagnostic and replay its recorded actions in the training rules engine."""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

from fullhouse_v3_common import ROOT, atomic_json, digest

SOURCE = ROOT / '.cache/fullhouse-v3'
GAME_HASH = 'df405d6d9efbeca3ecb778cfa70dd770fee5c34d04eb3f65ed1ab11261696381'
FEATURES_HASH = '84bcbceeb95d853f74b860693abe8ca4d26ffe7578fd639633aa99e6f033def1'
CARDS = tuple(rank + suit for suit in 'cdhs' for rank in '23456789TJQKA')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_training_engine():
    module = SOURCE / 'training/game.py'
    require(digest(module) == GAME_HASH and digest(SOURCE/'bot/features.py') == FEATURES_HASH,
            'Changed independent game or card evaluator adapter')
    sys.path.insert(0, str(SOURCE))
    # Load private module state for this standalone audit. Explicit external bet
    # sizes can be expressed with a sixth fraction in the existing chip calculator.
    spec = importlib.util.spec_from_file_location('independent_flop_audit_game', module)
    game = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(game)
    return game


def action_index(game, state, move):
    seat = int(state[game.S_TOACT])
    require(not state[game.S_TERMINAL] and move['seat'] == seat,
            'Action at terminal state or from wrong seat')
    committed, bet = int(state[game.S_BETS+seat]), int(state[game.S_CURBET])
    owed = bet - committed
    if move['kind'] == 'fold':
        require(owed > 0, 'Fold without a bet to face')
        return game.FOLD
    if move['kind'] in ('check','call'):
        require((move['kind'] == 'check') == (owed == 0), 'Check/call mismatch')
        return game.CHECK_CALL
    require(move['kind'] == 'raise' and type(move['to']) is int, 'Unknown action')
    target = move['to']
    stack, pot = int(state[game.S_STACKS+seat]), int(state[game.S_POT])
    minimum = min(committed+stack, bet+int(state[game.S_LAST_FULL_RAISE]))
    require(target > bet and minimum <= target <= committed+stack, 'Raise outside chip bounds')
    for action in (game.BET_50, game.BET_POT, game.ALL_IN):
        if committed + game._bet_chip_amount(action, pot, owed, stack) == target:
            return action
    # Check exact translation of any reference-opponent bet outside the neural menu.
    game.BET_FRACS = (*game.BET_FRACS[:5], (target-bet)/(pot+owed))
    require(committed + game._bet_chip_amount(5, pot, owed, stack) == target, 'Off-menu raise translation')
    return 5


def replay(game, trace):
    raw = trace['deck']
    require(len(raw) == 52 and set(raw) == set(CARDS), 'Duplicate cards in a forced branch')
    arr = np.asarray([CARDS.index(card) for card in raw], dtype=np.int8)
    state, deck, holes = game.new_hand(6, trace['button'], np.random.default_rng(0))
    deck[:] = arr
    holes[:] = arr[:12].reshape(6, 2)
    hero = trace['hero']
    for move in trace['prefix']:
        game.step(state, deck, holes, action_index(game, state, move))
    require(state[game.S_STREET] == 1 and state[game.S_TOACT] == hero and not state[game.S_TERMINAL],
            'Replayed prefix did not end at a hero flop decision')
    for move in trace['actions']:
        game.step(state, deck, holes, action_index(game, state, move))
    utilities = state[game.S_PAYOFFS:game.S_PAYOFFS+6].tolist()
    require(state[game.S_TERMINAL] and utilities == trace['utilities'] and sum(utilities) == 0,
            'Independent training rules produce a different settlement')
    return utilities[hero]


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


from contextlib import closing
from datetime import datetime, timezone
import hashlib
import random

from fullhouse_runtime import Bridge, Fullhouse
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import ReferenceOpponent
from poker_river_runtime import legal_menu

STUDY = ROOT/'artifacts/preflop-followup/480-v1/flop-action-diagnostic-20261123'
CONFIG = ROOT/'configs/poker-flop-action-diagnostic-20261123.json'
OLD = STUDY.parent/'postflop-averaging-20261120'
FIELDS = ('self480', 'public', 'tight-value')
SLOTS = ('480', '486', '492', '498', '504')


def close(actual, expected):
    require(np.allclose(actual, expected, atol=1e-9, rtol=1e-10), 'Different recomputed measurement')


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def locked_inputs():
    registration = read_json(STUDY/'registration.json')
    require(digest(CONFIG) == registration['config_sha256'], 'Unregistered design')
    plan = OLD/'population/evaluation-plan.json'
    cfg = read_json(CONFIG)
    require(digest(plan) == cfg['parent_evaluation_plan_sha256'], 'Wrong source evaluation')
    for path, sha in {**registration['frozen_files'], **read_json(plan)['frozen_files']}.items():
        require(digest(path) == sha, f'Changed frozen source: {Path(path).name}')
    for stage in ('sample','rollout'):
        inputs = STUDY/(stage+'-inputs.json')
        if inputs.exists():
            for path, sha in read_json(inputs)['code'].items():
                require(digest(path) == sha, 'Study implementation changed')
    return cfg, read_json(plan)


def frozen_policies(plan, seed):
    refs = {label: plan['policies'][label][str(seed)] for label in ('specialist480','mixture504')}
    for ref in refs.values():
        require(digest(ref['path']) == ref['sha256'], 'Wrong frozen policy spec')
    base = evaluator_policy(refs['specialist480']['path'], plan['source'])
    mix = Path(refs['mixture504']['path'])
    weights = read_json(mix)['postflop']
    require([str(w['iteration']) for w in weights] == list(SLOTS[1:]), 'Wrong four snapshots')
    policies = {'480':base}
    for weight in weights:
        path = mix.parent / weight['file']
        require(digest(path) == weight['sha256'], 'Wrong snapshot weights')
        policies[str(weight['iteration'])] = evaluator_policy(path, plan['source'])
    return policies


def joined_hash(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(',',':')).encode()
    return hashlib.sha256(blob).digest()


def prefix_seed(deal, hero, state):
    return int.from_bytes(joined_hash(dict(protocol='rich-common-card-evaluation-v1',deal_seed=deal,
                                          hero=hero,actor=state['seat_to_act'],action_log=state['action_log']))[:8],'big')


def check_root(policies, state, row, seed, deal, hero, prefix, draw):
    require((row['seed'],row['hero'],row['deal_seed']) == (seed,hero,deal) and state == row['root_observation']
            and prefix == row['prefix'] and joined_hash([seed,deal,hero,prefix,state]).hex() == row['state_id'],
            'Original decision identity, actor observation, or natural prefix differs')
    menu = row['legal_actions']
    require(menu == list(legal_menu(state).values()) and set(row['policies']) == set(SLOTS),
            'Root action menu or policy set differs')
    for label, policy in policies.items():
        reply = policy.decide(state, draw)
        probabilities = [0.0]*len(menu)
        for outcome in reply['distribution']:
            probabilities[menu.index(outcome['action'])] += outcome['probability']
        saved = row['policies'][label]
        close(probabilities,saved['probabilities'])
        require(menu[saved['chosen_index']] == reply['action'], 'Different frozen policy draw')


def scan_prefixes(seed, cfg, plan):
    folder = STUDY/f'seed-{seed}'
    raw = folder/'states.jsonl'
    rows = iter(json_lines(raw))
    receipt = read_json(folder/'summary.json')
    first = cfg['first_deal_by_seed'][str(seed)]
    count = cfg['eligible_original_deals_per_seed']
    scanned = receipt['attempts']
    require(receipt['first_card'] == first and receipt['last_scanned'] == first+scanned-1
            and count <= scanned <= cfg['maximum_deals_scanned_per_seed'] and digest(raw) == receipt['states_sha256'],
            'Incomplete or unexpected source deals')
    policies = frozen_policies(plan, seed)
    verified = steps = 0
    with closing(Bridge()) as bridge:
        for offset in range(scanned):
            deal, hero = first+offset, offset%6
            state = bridge.call(dict(op='reset',seed=deal,button=0))
            prefix = []
            while not state['terminal'] and state['street'] in ('preflop','flop'):
                actor = state['seat_to_act']
                draw = prefix_seed(deal,hero,state)
                if state['street'] == 'flop' and actor == hero and len(legal_menu(state)) >= 2:
                    row = next(rows, None)
                    require(row is not None, 'Missing eligible natural flop decision')
                    check_root(policies,state,row,seed,deal,hero,prefix,draw)
                    verified += 1
                    break
                move = dict(policies['480'].decide(state,draw)['action'],seat=actor)
                prefix.append(move)
                steps += 1
                require(len(prefix)<512, 'Unbounded hand')
                state = bridge.call(dict(op='step',action=move))
    require(verified == count and next(rows, None) is None, 'Extra sampled decisions')
    return dict(seed=seed, scanned=scanned, eligible=verified, natural_prefix_actions=steps,
                root_distributions=verified*len(SLOTS), source_sha256=digest(raw))


def phase_prefix():
    cfg, plan = locked_inputs()
    sample = {str(seed):scan_prefixes(seed,cfg,plan) for seed in cfg['training_seeds']}
    return dict(verified_utc=timestamp(),seeds=sample,config_sha256=digest(CONFIG))


def random_key(*parts):
    text = 'flop-action-values-v1:' + ':'.join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode('ascii')).digest()[:8], 'big')


def original_cards(bridge, deal):
    state = bridge.call(dict(op='reset',button=0,seed=deal))
    holes = [None]*6
    while state['street'] == 'preflop':
        seat = state['seat_to_act']
        require(holes[seat] is None, 'Seat dealt two private hands')
        holes[seat] = state['your_cards']
        state = bridge.call(dict(op='step',action=dict(seat=seat,kind='check' if state['can_check'] else 'call')))
    require(all(holes) and len(state['community_cards']) == 3, 'Incorrect private-card prefix')
    return [card for hand in holes for card in hand]+state['community_cards']


def replicate_deck(fixed, deal, repeat):
    unseen = [c for c in CARDS if c not in fixed]
    require(len(fixed) == 15 and len(unseen) == 37, 'Invalid original private cards or flop')
    random.Random(random_key('deck',deal,repeat)).shuffle(unseen)
    return fixed+unseen


def replay_policy_branch(oracle, deck, row, move, field, bots, repeat, want_actions):
    deal, hero = row['deal_seed'], row['hero']
    state = oracle.call(dict(op='reset',button=0,deck=deck))
    for action in row['prefix']:
        state = oracle.call(dict(op='step',action=action))
    forced = dict(move,seat=hero)
    require(state['your_cards'] == row['root_observation']['your_cards']
            and state['community_cards'] == row['root_observation']['community_cards'] and state['seat_to_act'] == hero,
            'Replayed root cards or actor differ')
    state = oracle.call(dict(op='step',action=forced))
    count = Counter()
    history = [forced] if want_actions else None
    decisions = 1
    while not state['terminal']:
        actor, street = state['seat_to_act'], state['street']
        require(state['your_cards'] == deck[2*actor:2*actor+2] and
                state['community_cards'] == deck[12:12+{'flop':3,'turn':4,'river':5}[street]],
                'More than the acting hand and current public cards reached the policy')
        seed = random_key('action', deal, hero, repeat, street, actor, count[street,actor])
        count[street,actor] += 1
        for player in state['players']:
            player.pop('hand_commit')
        bot = bots['self480'] if actor == hero else bots[field]
        dist = bot.decide(state,seed)['distribution']
        ranked = sorted(dist, key=lambda d:(('fold','check','call','raise').index(d['action']['kind']),
                                            d['action'].get('to',0)))
        threshold = (seed//2048)*math.fsum(d['probability'] for d in ranked)/2**53
        cumulative = 0.
        picked = None
        for d in ranked:
            cumulative += d['probability']
            if cumulative > threshold:
                picked = d['action']
                break
        require(picked is not None, 'Invalid cumulative probability')
        action = dict(picked,seat=actor)
        if want_actions:
            history.append(action)
        decisions += 1
        require(decisions+len(row['prefix']) <= 512, 'Overlong policy replay')
        state = oracle.call(dict(op='step',action=action))
    require(sum(state['utilities']) == 0, 'Policy replay did not conserve chips')
    return state['utilities'], history, decisions


def samples_and_returns(folder, cfg, seed):
    states = {row['state_id']:row for row in json_lines(folder/'states.jsonl')}
    file = folder/'values.jsonl.gz'
    repeated = cfg['paired_repetitions']
    expected_states = cfg['eligible_original_deals_per_seed']
    require(len(states) == expected_states and len({r['deal_seed'] for r in states.values()}) == expected_states,
            'Incomplete or repeated original deals')
    cells = {}
    gains = {label:dict() for label in ('mixture','486','504')}
    baseline = {}
    for row in json_lines(file):
        ident, field = row['state_id'],row['lineup']
        key=ident,field
        require(field in FIELDS and key not in cells and ident in states, 'Unexpected or duplicate raw valuation')
        s = states[ident]
        require(s['seed'] == row['seed'] == seed and row['hero'] == s['hero'] and row['deal_seed'] == s['deal_seed']
                and row['legal_actions'] == s['legal_actions'] and row['repeats'] == repeated, 'Mispaired action results')
        n = len(s['legal_actions'])
        probs = np.array([s['policies'][label]['probabilities'] for label in SLOTS])
        close(probs.sum(axis=1), np.ones(5))
        probs = probs / probs.sum(axis=1, keepdims=True)
        support = np.any(probs>0,axis=0)
        q = row['action_payoffs_chips']
        require(len(q) == n and all((outcomes is None) == (not active) for active,outcomes in zip(support,q)),
                'Unexpected support in forced-action outcomes')
        for i in np.flatnonzero(support):
            require(len(q[i]) == repeated and all(type(chips) is int and -10000<=chips<=50000 for chips in q[i]),
                    'Incomplete or out-of-bounds terminal chips')
        # Work directly from all terminal utility draws and each normalized
        # policy, avoiding the centered-delta implementation in the scorer.
        expected = [math.fsum(p * chips for i,p in enumerate(ps) if q[i] is not None for chips in q[i])
                    /(100*repeated) for ps in probs]
        baseline[key] = expected[0]
        for label, candidate in [('mixture',sum(expected[1:])/4),('486',expected[1]),('504',expected[4])]:
            gains[label][key] = candidate-expected[0]
        cells[key] = q
    require(len(cells) == len(states)*len(FIELDS), 'Missing matched opponent returns')
    return states,cells,gains,baseline


def audit_traces(folder, states, cells, cfg, plan, seed):
    from poker_multivalued_turn import Rules
    game = load_training_engine()
    parent = frozen_policies(plan, seed)['480']
    bots = {'self480':parent,'public':Fullhouse(),'tight-value':ReferenceOpponent('tight-value')}
    trace_file = folder/'traces.jsonl.gz'
    wanted = {(ident,field,i) for (ident,field),payoffs in cells.items() for i,p in enumerate(payoffs) if p is not None}
    source = {}
    # Verify complete decisions and all paired random repeats for a systematic
    # subset of the frozen sample; rerun the identical observations, not new games.
    picked = set(list(states)[::64])
    checked = actions = policy_checked = policy_decisions = 0
    with closing(Bridge()) as bridge, closing(Rules()) as oracle:
        for trace in json_lines(trace_file):
            ident, field, i = trace['state_id'],trace['lineup'],trace['action_index']
            k = ident,field,i
            require(k in wanted, 'Extra trace or action outside support')
            wanted.remove(k)
            row = states[ident]
            deal, hero = row['deal_seed'],row['hero']
            if deal not in source:
                source[deal]=original_cards(bridge,deal)
            require(trace['seed'] == seed and trace['deal_seed'] == deal and trace['hero'] == hero and trace['button'] == 0
                    and trace['prefix'] == row['prefix'] and trace['replicate'] == 0
                    and trace['deck'] == replicate_deck(source[deal],deal,0)
                    and trace['actions'][0] == dict(row['legal_actions'][i],seat=hero), 'Wrong forced branch or cards')
            require(replay(game,trace) == cells[ident,field][i][0], 'First paired payoff disagrees with separate rules')
            checked += 1
            actions += len(trace['actions'])+len(row['prefix'])
            if ident in picked:
                for r in range(cfg['paired_repetitions']):
                    chips, history, decisions = replay_policy_branch(oracle,replicate_deck(source[deal],deal,r),row,
                                row['legal_actions'][i],field,bots,r,r==0)
                    require(chips[hero] == cells[ident,field][i][r] and (r>0 or (chips == trace['utilities']
                            and history == trace['actions'])), 'Recorded policy/RNG or settlement mismatch')
                    policy_checked += 1
                    policy_decisions += decisions
    require(not wanted, 'Missing first-repeat traces')
    return dict(original_decks_verified=len(source), python_engine_replayed_terminals=checked,
                python_engine_replayed_actions=actions, policy_replayed_states=len(picked),
                policy_replayed_terminal_branches=policy_checked, policy_replayed_actions=policy_decisions,
                states_sha256=digest(folder/'states.jsonl'), values_sha256=digest(folder/'values.jsonl.gz'),
                traces_sha256=digest(trace_file))


def audit_scores(scores, gain_seeds, baseline_seeds, cfg):
    z = NormalDist().inv_cdf(.975)
    simultaneous = NormalDist().inv_cdf(.995)
    checked = {}
    for field in FIELDS:
        values = [np.mean([v for (_, f),v in seed.items() if f == field]) for seed in baseline_seeds]
        close(np.mean(values),scores['absolute_480_bb'][field])
    for policy in ('mixture','486','504'):
        reported = (scores['primary_mixture_minus_480'] if policy == 'mixture' else scores['secondary_minus_480'][policy])
        fields = (*FIELDS,'public_minus_self480','tight-value_minus_self480')
        require(set(reported['contrasts']) == set(fields),'Missing planned paired contrasts')
        for field in fields:
            seed_means, variances = [], []
            reference = reported['contrasts'][field]
            for seed in cfg['training_seeds']:
                gains = gain_seeds[seed][policy]
                ids = sorted({name for name,_ in gains})
                if '_minus_' in field:
                    lhs,rhs = field.split('_minus_')
                    by_deal = [gains[key,lhs]-gains[key,rhs] for key in ids]
                else:
                    by_deal = [gains[key,field] for key in ids]
                avg = math.fsum(by_deal)/len(by_deal)
                # These preregistered model seeds own disjoint original deals.
                v = math.fsum((val-avg)**2 for val in by_deal)/(len(by_deal)-1)
                seed_means.append(avg)
                variances.append(v/len(by_deal))
                close(avg,reference['mean_bb_by_seed'][str(seed)])
            mean = math.fsum(seed_means)/len(seed_means)
            se = math.sqrt(math.fsum(variances))/len(seed_means)
            close([mean,se],[reference['mean_bb'],reference['cluster_se_bb']])
            close([mean-z*se,mean+z*se],reference['paired_95_bb'])
            close([mean-simultaneous*se,mean+simultaneous*se],reference['simultaneous_95_bb'])
            checked.setdefault(policy,{})[field] = dict(mean_bb=mean,cluster_se_bb=se)
    return checked


def phase_results(prefix_audit):
    cfg,plan = locked_inputs()
    bound = read_json(STUDY/'rollout-inputs.json')
    require(prefix_audit['config_sha256'] == digest(CONFIG) == read_json(STUDY/'registration.json')['config_sha256'],
            'Mismatched prefix audit design')
    gain_seeds, baseline_seeds, details = {}, [], {}
    cards = set()
    for seed in cfg['training_seeds']:
        folder = STUDY/f'seed-{seed}'
        source = prefix_audit['seeds'][str(seed)]['source_sha256']
        done = read_json(folder/'rollout-complete.json')
        require(digest(folder/'states.jsonl') == source == done['source_sha256'] == bound['states'][str(seed)]
                and done['values_sha256'] == digest(folder/'values.jsonl.gz')
                and done['traces_sha256'] == digest(folder/'traces.jsonl.gz'), 'Unbound raw output')
        states,cells,gains,baseline = samples_and_returns(folder,cfg,seed)
        deals = {s['deal_seed'] for s in states.values()}
        require(cards.isdisjoint(deals), 'Shared deal in separate training seed range')
        cards.update(deals)
        details[str(seed)] = audit_traces(folder,states,cells,cfg,plan,seed)
        require(details[str(seed)]['python_engine_replayed_terminals'] == done['trace_count']
                and done['terminal_branches'] == done['trace_count']*cfg['paired_repetitions'], 'Branch coverage mismatch')
        gain_seeds[seed] = gains
        baseline_seeds.append(baseline)
    summary = read_json(STUDY/'summary.json')
    require(summary['distinct_original_deals'] == len(cards) and summary['states'] == len(cards), 'Incomplete scores')
    terminals = sum(v['python_engine_replayed_terminals'] for v in details.values())
    require(summary['evaluated_terminal_branches'] == terminals*cfg['paired_repetitions'], 'Incomplete terminal outcomes')
    score = audit_scores(summary,gain_seeds,baseline_seeds,cfg)
    return dict(verified_utc=timestamp(), config_sha256=digest(CONFIG), registration_sha256=digest(STUDY/'registration.json'),
                audit_script_sha256=digest(__file__), summary_sha256=digest(STUDY/'summary.json'), original_deals=len(cards),
                primary_and_secondary_contrasts=score, all_reaggregated_terminal_branches=summary['evaluated_terminal_branches'],
                python_game_sha256=GAME_HASH, python_engine_replayed_terminals=terminals,
                python_engine_replayed_actions=sum(v['python_engine_replayed_actions'] for v in details.values()),
                reproduced_policy_branches=sum(v['policy_replayed_terminal_branches'] for v in details.values()),
                complete_prefix_scan=prefix_audit['seeds'], by_seed=details)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('prefix','all'))
    args = ap.parse_args()
    prefix = STUDY/'prefix-audit.json'
    if args.phase == 'prefix' or not prefix.exists():
        atomic_json(prefix,phase_prefix())
    if args.phase == 'all':
        atomic_json(STUDY/'final-audit.json',phase_results(read_json(prefix)))
