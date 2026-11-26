"""Audit independent deals, matched branches, fitted weights, and conditional decision value."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from audit_poker_flop_diagnostic import CARDS, GAME_HASH, action_index, load_training_engine
from audit_poker_flop_repair import nonce, repair_menus, request_at_root, require
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, semantic_seed
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_action_values import _actor_view, continuation_action, source_cards
from poker_multivalued_turn import RULES, Rules
from poker_river_runtime import action_key
from poker_street_progress import StreetProgressPolicy
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges
from train_poker_flop_repair import SEEDS, sources


REL = Path('preflop-followup/480-v1/street-progress-20261126')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
V1 = STUDY.parent/'flop-repair-20261126'
DESIGN = '5ce6f2b5ffc7f8a3ea82019845c5923ea3af874ba0de409d8e59a331bdbc5700'
W = ('w1', 'b1', 'w2', 'b2')


def preflight():
    verify_files({STUDY/'design.json': DESIGN, REPORT/'design.json': DESIGN})
    design = read(STUDY/'design.json')
    verify_files({V1/'fit-complete.json': design['source_fit_sha256']})
    prior = {k: v['range'] for k, v in prior_card_ranges(STUDY).items()}
    require(len(prior) == design['previous_distinct_card_ranges_checked'], 'Baseline reservations changed')
    for street, splits in design['collection'].items():
        for split, budget in splits.items():
            for sid, r in budget['by_seed'].items():
                check_card_range(r['start'], r['limit']-r['start'], prior)
                prior[f'{street}/{split}/{sid}'] = (r['start'], r['limit'])
    for sid, ref in design['source_policies'].items():
        old = V1/f'seed-{sid}'
        verify_files({old/'policy.json': ref['manifest_sha256'], old/'critic.npz': ref['weights_sha256'],
                      old/'positions.jsonl.gz': ref['original_flop_ledger_sha256']})
    return design


def verify_original_reach(bridge, row, street, parent, reach, models):
    hero, deal, lineup = row['hero'], row['deal_seed'], row['opponents_by_seat']
    state = bridge.call(dict(op='reset', seed=deal, button=0))
    different_hero_flop_action = False
    for recorded in row['prefix']:
        seat = state['seat_to_act']
        if state['street'] == 'preflop':
            policy = parent
        elif seat == hero:
            policy = reach
        else:
            policy = models[lineup[seat]]
        draw = semantic_seed(deal,hero,state)
        predicted = policy.decide(state,draw)['action']
        require(dict(predicted, seat=seat) == recorded, 'Selected reach or opponent prefix action differs')
        if street=='turn' and seat==hero and state['street']=='flop':
            different_hero_flop_action |= predicted!=parent.decide(state,draw)['action']
        state = bridge.call(dict(op='step', action=recorded))
    require(state == row['root_observation'] and state['street'] == street and state['seat_to_act'] == hero,
            'Original deal did not reach saved decision')
    rebuilt = bridge.call(dict(op='replay', **request_at_root(row)))
    require(rebuilt == state, 'Saved observation cannot be built from the actor-visible request')
    count = 1 if street == 'flop' else 2
    boundaries = [i for i,e in enumerate(state['action_log']) if e['action']=='deal']
    require(len(boundaries) == count and all(e.get('seat') != hero for e in state['action_log'][boundaries[-1]+1:]),
            'Saved decision is not first on its street')
    return different_hero_flop_action


def independent_utilities(game, row, deck, tail):
    raw = np.asarray([CARDS.index(c) for c in deck], dtype=np.int8)
    state, cards, hands = game.new_hand(6, 0, np.random.default_rng(0))
    cards[:] = raw
    hands[:] = raw[:12].reshape(6, 2)
    for action in row['prefix']:
        game.step(state, cards, hands, action_index(game, state, action))
    street_number = {'flop':1, 'turn':2}[row['street']]
    require(state[game.S_STREET] == street_number and state[game.S_TOACT] == row['hero']
            and not state[game.S_TERMINAL], 'Independent engine root mismatch')
    for action in tail:
        game.step(state, cards, hands, action_index(game, state, action))
    util = state[game.S_PAYOFFS:game.S_PAYOFFS+6].tolist()
    require(state[game.S_TERMINAL] and sum(util)==0, 'Incomplete independent settlement')
    return util


def audit_sample(row, street, repeats, engine, rules, game, parent, reach, models):
    changed_hero = verify_original_reach(engine,row,street,parent,reach,models)
    obs, hero, deal = row['root_observation'], row['hero'], row['deal_seed']
    private = list(source_cards(engine,deal)[:12])
    known = private + obs['community_cards']
    require(private[2*hero:2*hero+2] == obs['your_cards'] and len(set(known)) == len(known),
            'Actual hero cards or public board mismatch')
    count, rivers = 0, set()
    for rep in range(repeats):
        unseen = [c for c in CARDS if c not in known]
        random.Random(nonce('deck' if street=='flop' else 'turn-deck', deal, rep)).shuffle(unseen)
        deck = known + unseen
        rivers.add(deck[16])
        for k, forced in enumerate(row['legal_actions']):
            state = rules.call(dict(op='reset', deck=deck, button=0))
            for action in row['prefix']:
                state = rules.call(dict(op='step', action=action))
            require(_actor_view(state) == obs, 'Unseen suffix changed the known public/private hero root')
            tail = [dict(forced,seat=hero)]
            state = rules.call(dict(op='step',action=tail[0]))
            ordinals = Counter()
            while not state['terminal']:
                seat = state['seat_to_act']
                key = state['street'], seat
                draw = nonce('action',deal,hero,rep,*key,ordinals[key])
                ordinals[key] += 1
                policy = parent if seat == hero else models[row['opponents_by_seat'][seat]]
                action = continuation_action(policy, _actor_view(state), draw)
                tail.append(dict(action,seat=seat))
                require(len(tail) < 512, 'Excessive branch length')
                state = rules.call(dict(op='step',action=tail[-1]))
            util = independent_utilities(game,row,deck,tail)
            require(util == state['utilities'] and util[hero] == row['action_payoffs_chips'][k][rep],
                    'Independent branch replay differs from recorded chip label')
            count += 1
    return count, len(rivers), changed_hero


def probabilities(policy, state, keys):
    result = _validate_reply(state,policy.decide(state,0))
    prob = {action_key(item['action']): item['probability'] for item in result['distribution']}
    require(len(prob) == len(result['distribution']) and set(prob).issubset(keys), 'Off-menu or repeated action')
    values = np.array([prob.get(key,0.) for key in keys])
    require(np.all(np.isfinite(values)) and min(values)>=0 and abs(values.sum()-1)<1e-6,'Invalid action mass')
    return values/values.sum()


def reference_distribution(policy, state, keys, mapping, base_mass):
    """Derive the deployed masked critic decision directly from its exported arrays."""
    street = state['street']
    w = policy.flop.weights if street == 'flop' else policy.turn_weights
    if w is None:
        return base_mass
    post = policy.flop.base.policies['postflop']
    encoded, _ = post.features.encode_state_dict(state, post.lookup._equity_tables)
    heads = [mapping.index(i) for i in range(len(keys))]
    base_heads = np.zeros(5)
    base_heads[heads] = base_mass
    x = np.clip((np.r_[encoded,base_heads]-w['center'])/w['scale'], -6., 6.)
    values = []
    for n in range(2):
        hidden = np.maximum(w['w1'][n] @ x + w['b1'][n], 0.)
        values.append((w['w2'][n] @ hidden + w['b2'][n])[heads])
    estimates = np.stack(values)
    order = range(len(keys))
    if street == 'flop':
        order = sorted(order,key=lambda i:(('fold','check','call','raise').index(keys[i][0]),keys[i][1] or 0))
    best = max(order,key=lambda i: estimates[:,i].mean())
    gain = min(estimates[:,best] - estimates @ base_mass)
    mass = min(.25, max(0., gain-.05))
    expected = (1.-mass)*base_mass
    expected[best] += mass
    require(abs(expected.sum()-1) < 1e-6 and np.abs(expected-base_mass).sum()/2 <= .25+1e-7,'Bad masked repair')
    return expected


def check_seed(street, split, seed, design, engine, rules, game):
    sid = str(seed)
    folder = STUDY/street/split/f'seed-{seed}'
    budget = design['collection'][street][split]
    bounds = budget['by_seed'][sid]
    receipt = read(folder/'collection-complete.json')
    raw = folder/'positions.jsonl.gz'
    verify_files({raw:receipt['position_ledger_sha256']})
    require(receipt['complete'] and receipt['training_design_sha256']==DESIGN, 'Wrong collector design')
    _, ref, _, base, models = sources(seed)
    baseline_policy = StreetProgressPolicy(V1/f'seed-{seed}'/'policy.json')
    reach = base
    if street=='turn':
        chosen = read(STUDY/'flop/selected.json')['policies'][sid]
        require(chosen['sha256']==receipt['reach_policy_sha256']
                and receipt['selected_flop_receipt_sha256']==digest(STUDY/'flop/selected.json'),
                'Turn branches refer to a different flop selection')
        verify_files({chosen['path']:chosen['sha256']})
        baseline_policy = reach = StreetProgressPolicy(chosen['path'])
    else:
        require(receipt['reach_policy_sha256'] == ref['sha256'], 'Wrong flop reach baseline')
    candidate = None
    if split != 'train':
        cr = read(STUDY/street/'candidate.json')['policies'][sid]
        verify_files({cr['path']:cr['sha256']})
        candidate = StreetProgressPolicy(cr['path'])
    n, reps = budget['positions_per_seed'], budget['repeats']
    previous = bounds['start']-1
    counts, gains = Counter(), []
    within_root_mc_variance = 0.
    river_diversity = []
    with gzip.open(raw,'rt') as stream:
        for index,line in enumerate(stream):
            row = json.loads(line)
            deal,hero = row['deal_seed'],row['hero']
            require(previous < deal < bounds['limit'] and hero==(deal-bounds['start'])%6 and row['seed']==seed
                    and (row['street'],row['split'])==(street,split), 'Duplicate, misrouted or invalid root')
            previous = deal
            state = row['root_observation']
            require(len(state['community_cards'])==(3 if street=='flop' else 4) and len(state['your_cards'])==2,
                    'Wrong observable card count')
            keys,mapping = repair_menus(state)
            require(row['abstract_to_concrete_index']==mapping and keys==[action_key(a) for a in row['legal_actions']],
                    'Clamped legal menu differs')
            payouts = np.array(row['action_payoffs_chips'])
            require(payouts.shape==(len(keys),reps) and payouts.dtype.kind in 'iu' and payouts.min()>=-10000
                    and payouts.max()<=50000,'Invalid action payoff array')
            if ('fold',None) in keys:
                require((payouts[keys.index(('fold',None))] == state['your_stack']-10000).all(),'Invalid fold value')
            if index in (0, n//2, n-1):
                sample, distinct_rivers, different_hero = audit_sample(row,street,reps,engine,rules,game,base,reach,models)
                counts['paid_branches_independently_replayed'] += sample
                counts['original_reaches_reconstructed'] += 1
                river_diversity.append(distinct_rivers)
                if street=='turn':
                    counts['reaches_with_flop_action_different_from_480'] += int(different_hero)
            elif street=='turn' and index<200 and not counts['reaches_with_flop_action_different_from_480']:
                different_hero = verify_original_reach(engine,row,street,base,reach,models)
                counts['reaches_with_flop_action_different_from_480'] += int(different_hero)
                counts['original_reaches_reconstructed'] += 1
            if candidate is not None:
                ps = np.array(row['parent_probabilities'])
                require(np.allclose(ps,probabilities(base,state,keys),atol=1e-7,rtol=0),'Stored 480 differs')
                p = probabilities(baseline_policy,state,keys)
                q = probabilities(candidate,state,keys)
                for runtime,dist in ((baseline_policy,p),(candidate,q)):
                    expected = reference_distribution(runtime,state,keys,mapping,ps)
                    require(np.allclose(expected,dist,atol=1e-6,rtol=0),'Runtime differs from masked tensor scores')
                gains.append(float(np.dot(q-p,payouts.mean(axis=1))/100.))
                if split=='dev':
                    paired_runouts = (q-p) @ payouts/100.
                    within_root_mc_variance += float(paired_runouts.var(ddof=1))/reps
            counts['positions'] += 1
            counts['payoffs'] += payouts.size
    require(counts['positions']==receipt['positions']==n and counts['payoffs']==receipt['terminal_rollouts']
            and previous==receipt['last_card'] and max(river_diversity)>1, 'Wrong totals or fixed river runouts')
    out = dict(counts,ledger_sha256=digest(raw),distinct_rivers_on_sampled_roots=river_diversity)
    if gains:
        out.update(mean_bb=statistics.mean(gains),variance_bb2=statistics.variance(gains))
    if split=='dev':
        out['average_within_root_mc_variance_bb2'] = within_root_mc_variance/n
    return out


def check_split(street, split, design):
    if split!='train':
        require((STUDY/street/f'{split}-score.json').is_file(),'Wait for frozen scored split')
    if split=='confirm':
        require(read(STUDY/street/'dev-score.json')['eligible_for_confirmation'], 'Confirmation was not qualified')
    game = load_training_engine()
    with closing(Bridge()) as engine, closing(Rules()) as rules:
        seeds = {str(s): check_seed(street,split,s,design,engine,rules,game) for s in SEEDS}
    out = dict(verified_utc=utc(),street=street,split=split,seeds=seeds)
    if split!='train':
        n = design['collection'][street][split]['positions_per_seed']
        mean = statistics.mean(s['mean_bb'] for s in seeds.values())
        se = math.sqrt(sum(s['variance_bb2']/n for s in seeds.values()))/len(seeds)
        out.update(mean_bb=mean, card_se_bb=se, lower_bb=mean-1.96*se, upper_bb=mean+1.96*se)
        if split=='dev':
            mc_var = sum(s['average_within_root_mc_variance_bb2']/n for s in seeds.values())/len(seeds)**2
            out['within_root_mc_share_of_score_variance'] = mc_var/se**2 if se else None
    atomic_json(STUDY/street/f'{split}-independent.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='seeds'}),flush=True)


def tensor_hash(arrays):
    return hashlib.sha256(b''.join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()


def verify_fit(street, design):
    """Compare the receipt to the actual frozen pre-update and exported tensors."""
    folder = STUDY/street
    candidate = read(folder/'candidate.json')
    training = design['training']
    require(candidate['training_complete'] and candidate['design_sha256']==DESIGN and
            candidate['optimizer_updates']==len(SEEDS)*2*training['steps_per_critic'],'Unexpected training count')
    if street=='turn':
        reach = read(STUDY/'flop/selected.json')
        require(candidate['source_flop_choice_sha256']==digest(STUDY/'flop/selected.json'),'Turn parent changed')
    for seed in SEEDS:
        sid = str(seed)
        source = design['source_policies'][sid]
        rec = candidate['seeds'][sid]
        ref = candidate['policies'][sid]
        path = Path(ref['path'])
        expected_parent = source['manifest_sha256'] if street=='flop' else reach['policies'][sid]['sha256']
        verify_files({path:ref['sha256'],folder/'train'/f'seed-{seed}'/'positions.jsonl.gz':rec['train_ledger_sha256']})
        require(rec['initial_flop_sha256']==expected_parent and rec['policy_sha256']==ref['sha256']
                and rec['fresh_train_positions']==3072 and rec['positions']==(4096 if street=='flop' else 3072),
                'Wrong parent, new fit set, or prior source')
        spec = read(path)
        weight_ref = spec['critic' if street=='flop' else 'turn_critic']
        require(rec['critic_sha256']==weight_ref['sha256'] and spec['seed']==seed,'Wrong seed or exported hash')
        if street=='turn':
            require(spec['flop_policy']['sha256']==expected_parent,'Turn export references different selected flop')
        target = path.parent/weight_ref['path']
        verify_files({target:weight_ref['sha256']})
        with np.load(target,allow_pickle=False) as dest, np.load(V1/f'seed-{seed}'/'critic.npz',allow_pickle=False) as v1:
            norm = ('center','scale')
            require(tensor_hash(dest[k] for k in norm)==rec['normalization_sha256'],'Wrong exported normalization')
            if street=='flop':
                require(rec['warm_source_v1_weights_sha256']==source['weights_sha256'] and
                        all(np.array_equal(dest[k],v1[k]) for k in norm),'Warm-start normalizer changed')
            require([h['head'] for h in rec['fit_heads']]==[0,1], 'Critic count differs')
            for h in rec['fit_heads']:
                i = h['head']
                initial, final = h['initial_weights_sha256'],h['final_weights_sha256']
                if street=='flop':
                    require(tensor_hash(v1[k][i] for k in W)==initial, 'Warm-start network tensor differs')
                require(tensor_hash(dest[k][i] for k in W)==final and initial!=final and
                        h['steps']==training['steps_per_critic'] and
                        h['position_draws']==h['steps']*training['batch_size'],'Fit updates or final tensors differ')
    return candidate


def verify_score(street, split, independent, candidate):
    official = read(STUDY/street/f'{split}-score.json')
    require(official['complete'] and official['design_sha256']==DESIGN and
            official['candidate_receipt_sha256']==digest(STUDY/street/'candidate.json') and
            official['candidate_policy_sha256']=={s:r['sha256'] for s,r in candidate['policies'].items()},
            'Scoring did not use frozen candidate')
    require(datetime.fromisoformat(candidate['frozen_utc']) <= datetime.fromisoformat(official['completed_utc']),
            'Impossible scoring chronology')
    for sid, r in independent['seeds'].items():
        proof = official['proofs'][sid]
        other = official['paired_gain']['by_seed'][sid]
        require(r['ledger_sha256']==proof['ledger_sha256'] and r['payoffs']==proof['terminal_rollouts']
                and other['deals']==r['positions']==proof['root_deals'],'Score provenance differs')
        require(math.isclose(r['mean_bb'],other['mean'],rel_tol=0,abs_tol=1e-9) and
                abs(math.sqrt(r['variance_bb2']/r['positions'])-other['standard_error']) < 1e-9,'Seed scores differ')
    keys = {'mean_bb':'mean','card_se_bb':'standard_error','lower_bb':'lower','upper_bb':'upper'}
    for own, theirs in keys.items():
        require(abs(independent[own]-official['paired_gain'][theirs])<1e-9,'Card-level bounds or units differ')
    gate = independent['mean_bb']>0 if split=='dev' else independent['lower_bb']>0
    require(gate==official['eligible_for_confirmation' if split=='dev' else 'local_confirmation_passed'],
            'Inconsistent registered gate')
    return gate


def final_audit(design):
    out = dict(complete=True, verified_utc=utc(), design_sha256=DESIGN, auditor_sha256=digest(__file__),
               rust_rules_sha256=digest(RULES), rust_bridge_sha256=digest(BRIDGE), python_game_sha256=GAME_HASH,
               registered_disjoint_ranges=18, prior_ranges_checked=design['previous_distinct_card_ranges_checked'],
               conditional_unit='bb per reached first decision; one unit per unique root deal, equal fixed seed weights.',
               whole_game_performance_measured=False, streets={})
    for street in ('flop','turn'):
        folder = STUDY/street
        candidate = verify_fit(street,design)
        selected = read(folder/'selected.json')
        require(selected['complete'] and selected['design_sha256']==DESIGN and
                selected['candidate_receipt_sha256']==digest(folder/'candidate.json'),'Incomplete selection')
        incumbent = ({s:r['manifest_sha256'] for s,r in design['source_policies'].items()} if street=='flop' else
                     {s:r['sha256'] for s,r in read(STUDY/'flop/selected.json')['policies'].items()})
        dev = read(folder/'dev-independent.json')
        qualified = verify_score(street,'dev',dev,candidate)
        candidate_won = False
        checks = {}
        splits = ('train','dev','confirm') if qualified else ('train','dev')
        for split in splits:
            result = read(folder/f'{split}-independent.json')
            seeds = result['seeds'].values()
            checks[split] = dict(positions=sum(r['positions'] for r in seeds), terminal_branch_labels=sum(r['payoffs'] for r in seeds),
                                 roots_fully_reconstructed=sum(r['original_reaches_reconstructed'] for r in seeds),
                                 terminal_branches_reproduced_by_python_and_rust=sum(r['paid_branches_independently_replayed'] for r in seeds),
                                 sampled_distinct_river_range=[f(n for r in seeds for n in r['distinct_rivers_on_sampled_roots'])
                                                              for f in (min,max)])
            if street=='turn':
                checks[split]['turn_reaches_with_flop_action_different_from_480'] = sum(
                    r['reaches_with_flop_action_different_from_480'] for r in seeds)
            if split!='train':
                checks[split].update({k:result[k] for k in ('mean_bb','card_se_bb','lower_bb','upper_bb')})
                score = read(folder/f'{split}-score.json')
                require(score['prior_policy_sha256']==incumbent and selected[split+'_sha256']==digest(folder/f'{split}-score.json'),
                        'Wrong scored incumbent or selection input')
            if split=='dev':
                checks[split]['estimated_fraction_of_mean_variance_from_finite_runouts'] = result[
                    'within_root_mc_share_of_score_variance']
            if split=='confirm':
                candidate_won = verify_score(street,split,result,candidate)
                require(datetime.fromisoformat(read(folder/'dev-score.json')['completed_utc'])
                        <= datetime.fromisoformat(score['completed_utc']),'Confirmation predates development gate')
        if not qualified:
            require(not (folder/'confirm-score.json').exists() and not (folder/'confirm').exists() and
                    selected['confirm_sha256'] is None,'Unexpected confirmation activity')
        winner = {s:r['sha256'] for s,r in candidate['policies'].items()} if candidate_won else incumbent
        require(selected['selected']==('candidate' if candidate_won else 'parent') and
                {s:r['sha256'] for s,r in selected['policies'].items()}==winner,'Wrong exported winning policy')
        out['streets'][street] = dict(candidate_sha256=digest(folder/'candidate.json'),selected=selected['selected_stage'],
                                      held_out_dev_qualified=qualified,confirmed_local_gain=candidate_won,
                                      verified_exported_networks=2*len(SEEDS),splits=checks)
    atomic_json(STUDY/'independent-audit.json',out)
    atomic_json(REPORT/'independent-audit.json',out)
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('preflight','flop','turn','final'))
    ap.add_argument('split',nargs='?',choices=('train','dev','confirm'))
    args=ap.parse_args()
    d = preflight()
    if args.mode=='final':
        final_audit(d)
    elif args.mode!='preflight':
        check_split(args.mode,args.split,d)
