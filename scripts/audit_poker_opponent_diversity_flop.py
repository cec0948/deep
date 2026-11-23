"""Recalculate the conditional first-flop intervention from the original decisions and terminal chips."""
from collections import defaultdict
from contextlib import closing
import hashlib
import math
from pathlib import Path
from statistics import NormalDist, mean, variance

from audit_poker_flop_diagnostic import (
    close, frozen_policies, json_lines, load_training_engine, original_cards, replay,
    replay_policy_branch, replicate_deck, require,
)
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import atomic_json, digest
from poker_multivalued_turn import Rules
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import ReferenceOpponent
from register_poker_opponent_diversity import CONTROL, REPORT, STUDY

LOCAL = STUDY/'flop-diagnostic'
ORIGINAL = STUDY.parent/'flop-action-diagnostic-20261123'
SEEDS = (342, 343, 344)
FIELDS = ('self480', 'public', 'tight-value')


def concrete_probabilities(policy, row):
    menu = row['legal_actions']
    weights = [0.] * len(menu)
    for d in policy.decide(row['root_observation'], 0)['distribution']:
        weights[menu.index(d['action'])] += d['probability']
    require(all(math.isfinite(x) and x >= 0 for x in weights) and abs(sum(weights)-1) < 1e-6,
            'Invalid concrete flop action probabilities')
    return [p / math.fsum(weights) for p in weights]


def verify_replays(seed, extra, rows, bots):
    game = load_training_engine()
    checked = actions = 0
    with closing(Bridge()) as bridge, closing(Rules()) as oracle:
        for field in FIELDS:
            picks = [k for k in extra if k[0] in rows and k[1] == field]
            # Two predetermined keys per field/seed plus one representative with varying terminal payoffs.
            rank = lambda k: hashlib.sha256(repr(k).encode()).digest()
            different_returns = [k for k in picks if len(set(extra[k])) > 1]
            require(different_returns and len(picks) >= 3, 'Incomplete newly valued action coverage')
            selected = set(sorted(picks, key=rank)[:2]) | {min(different_returns, key=rank)}
            for ident, _, action in sorted(selected):
                row = rows[ident]
                hero, deal = row['hero'], row['deal_seed']
                forced = row['legal_actions'][action]
                fixed = original_cards(bridge, deal)
                for repetition, payoff in enumerate(extra[ident,field,action]):
                    deck = replicate_deck(fixed, deal, repetition)
                    utility, history, steps = replay_policy_branch(
                        oracle, deck, row, forced, field, bots, repetition, True)
                    require(history[0] == dict(forced,seat=hero) and utility[hero] == payoff,
                            'Replayed forced action or terminal chips differ from missing-action tape')
                    trace = dict(button=0,deck=deck,hero=hero,prefix=row['prefix'],actions=history,utilities=utility)
                    require(replay(game,trace) == payoff, 'Independent six-seat engine disagrees with chip settlement')
                    checked += 1
                    actions += steps
    return dict(seed=seed, terminals=checked, policy_decisions=actions)


def seed_effect(seed, plan, stage, extra, consumed, cards):
    spec = stage['policies']['poolMixture504'][str(seed)]
    model = Path(spec['path'])
    verify_files({model:spec['sha256']})
    captures = read(model)['postflop']
    require([c['iteration'] for c in captures] == [486,492,498,504], 'Incorrect four weights')
    verify_files({model.parent/c['file']:c['sha256'] for c in captures})
    treatment = [evaluator_policy(model.parent/c['file'],plan['source']) for c in captures]
    old = frozen_policies(plan,seed)
    original = ORIGINAL/f'seed-{seed}'
    states = {s['state_id']:s for s in json_lines(original/'states.jsonl')}
    require(len(states)==1024, 'Wrong number of original deals')
    probabilities = {}
    for ident, row in states.items():
        require(row['seed'] == seed and row['deal_seed'] not in cards, 'Unexpected or duplicated original deal')
        cards.add(row['deal_seed'])
        old_probs = {}
        for label, policy in old.items():
            old_probs[label] = concrete_probabilities(policy,row)
            saved = row['policies'][label]['probabilities']
            close(old_probs[label], [p / math.fsum(saved) for p in saved])
        ctrl = list(map(mean, zip(*(old_probs[str(i)] for i in (486,492,498,504)),strict=True)))
        new = list(map(mean, zip(*(concrete_probabilities(p,row) for p in treatment),strict=True)))
        probabilities[ident] = old_probs['480'], ctrl, new
    gain = defaultdict(dict)
    for cell in json_lines(original/'values.jsonl.gz'):
        ident,field = cell['state_id'],cell['lineup']
        s = states[ident]
        require(field in FIELDS and field not in gain[ident] and cell['legal_actions']==s['legal_actions']
                and cell['seed']==seed and cell['hero']==s['hero'] and cell['deal_seed']==s['deal_seed']
                and cell['repeats']==8 and len(cell['action_payoffs_chips'])==len(s['legal_actions']),
                'Mispaired duplicate or missing field')
        baseline,ctrl,new = probabilities[ident]
        terminal = []
        for i,payoffs in enumerate(cell['action_payoffs_chips']):
            key = (ident,field,i)
            if payoffs is None and new[i] > 0:
                require(key in extra and baseline[i]==ctrl[i]==0,'Missing action outside the registered intervention')
                payoffs = extra[key]
                consumed.add(key)
            if payoffs is None:
                require(baseline[i]==ctrl[i]==new[i]==0,'Unvalued action with positive probability')
                payoffs = [0]*8
            require(len(payoffs)==8 and all(type(v) is int and -10000<=v<=50000 for v in payoffs),
                    'Invalid terminal chips')
            terminal.append(payoffs)
        # Weight each forced terminal in chips before differencing the three policy EVs.
        ev = [math.fsum(p*v for p,values in zip(ps,terminal,strict=True) for v in values)/800
              for ps in (baseline,ctrl,new)]
        gain[ident][field] = (ev[2]-ev[1],ev[2]-ev[0])
    require(len(gain)==len(states) and all(len(fields)==3 for fields in gain.values()),'Incomplete card-by-field table')
    bots = {'self480':old['480'],'public':Fullhouse(),'tight-value':ReferenceOpponent('tight-value')}
    proof = verify_replays(seed,extra,states,bots)
    return {f:[v[f] for v in gain.values()] for f in FIELDS},proof


def audit():
    design = read(LOCAL/'design.json')
    verify_files(design['source_hashes'])
    out = read(LOCAL/'effect.json')
    extra_file = LOCAL/'missing-actions.jsonl.gz'
    verify_files({STUDY/'stage-ready.json':out['stage_sha256'],LOCAL/'design.json':out['design_sha256'],
                  extra_file:out['missing_actions_sha256']})
    extras = {}
    for item in json_lines(extra_file):
        k = item['state_id'],item['lineup'],item['action_index']
        require(k not in extras, 'Duplicate new tape')
        extras[k] = item['chips']
    used,card_ids = set(),set()
    plan = read(CONTROL/'population/evaluation-plan.json')
    stage = read(STUDY/'stage-ready.json')
    gains,proof = {},{}
    for seed in SEEDS:
        gains[seed],proof[str(seed)] = seed_effect(seed,plan,stage,extras,used,card_ids)
    require(used == set(extras) and len(card_ids)==out['original_deals']==3072
            and out['additional_original_deals']==0 and sum(map(len,extras.values()))==out['added_terminal_branches'],
            'Extra conditional outcomes counted as new deals or left unmatched')
    simultaneous = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
    nominal = NormalDist().inv_cdf(.975)
    contrasts = {}
    for f in FIELDS:
        # Deal IDs are disjoint across the three fixed strata. Count each original deal once.
        differences = [[d[0] for d in gains[s][f]] for s in SEEDS]
        estimate = mean(map(mean,differences))
        sem = math.sqrt(sum(variance(group)/len(group) for group in differences))/len(SEEDS)
        critical = [estimate-simultaneous*sem, estimate+simultaneous*sem]
        alt = mean(mean(d[1] for d in gains[s][f]) for s in SEEDS)
        expected = out['diverse_minus_control'][f]
        close([estimate,sem], [expected['mean_bb'],expected['cluster_se_bb']])
        close([estimate-nominal*sem,estimate+nominal*sem],expected['paired_95_bb'])
        close(critical,expected['simultaneous_95_bb'])
        close([mean(d) for d in differences],[expected['mean_bb_by_seed'][str(s)] for s in SEEDS])
        close(alt,out['diverse_minus_480_point_bb'][f])
        contrasts[f] = dict(diversity_minus_control_bb=estimate,simultaneous_95_bb=critical,diversity_minus_480_bb=alt)
    receipt = dict(verified_utc=utc(),effect_sha256=digest(LOCAL/'effect.json'),added_tape_sha256=digest(extra_file),
                   original_cards=len(card_ids),new_independent_cards=0,rows_by_lineup=len(card_ids),
                   added_state_lineup_actions=len(extras),added_conditional_terminal_branches=out['added_terminal_branches'],
                   independently_replayed_terminals=sum(v['terminals'] for v in proof.values()),
                   independently_replayed_policy_decisions=sum(v['policy_decisions'] for v in proof.values()),
                   fixed_seed_replay_checks=proof,checked_policy_values=contrasts,
                   units='BB per first hero flop decision conditional on archived 480-reached states')
    for dest in (LOCAL,REPORT/'flop-diagnostic'):
        atomic_json(dest/'effect-audit.json',receipt)


if __name__ == '__main__':
    audit()
