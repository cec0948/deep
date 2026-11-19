"""Audit both ends of the 480–504 online window and independently match raw evaluation hands."""
from contextlib import closing
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

from audit_poker_warm_preflop import DEALS, FIELDS, SEATS, SEEDS, FirstActionDifferences, check, check_estimate
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import load_checkpoint


ROOT_STUDIES = ROOT/'artifacts/preflop-followup/480-v1'
STUDY = ROOT_STUDIES/'fixed-preflop-504'
PARENT, ARM = 'specialist480', 'fixed504'
START, INITIAL, FINAL = 2037800000, 480, 504
REPLAY_STRIDE = 250
SOURCE_HASHES = {
    ROOT_STUDIES/'preflop-dose'/'summary.json': '6298d5e5a89f3212bed1f706376e41b298ec06d60f70e0f32d474358946e9610',
    ROOT_STUDIES/'preflop-dose'/'final-audit.json': '5f502fe9459f7f985166ba5b138ce35c9460161042e49c2847d4004170aa50c3',
    ROOT_STUDIES/'confirmation'/'summary.json': '52d1a10ff493680189afd12fdef4408b4bd8dc980ce56786d7ad7dcba4dfdff1',
    ROOT_STUDIES/'confirmation'/'final-audit.json': 'b847f113a1965302619517d2395706f0d2070915c5447de417b453b0ef720930',
}


def deployed_weights(ref, mapping):
    """Reconstruct the worker's tensor-order fingerprint directly from the deployed arrays."""
    spec = read(ref['path'])
    check(spec['format'] == 'preflop-street-policy-v1', 'Unexpected deployment route')
    arrays = {'main_scales': 'main_scales', 'opening_scale': 'opening_scale'}
    for head, prefix in (('main',''),('opening','opening_')):
        arrays.update({f'{head}.{k}':prefix+v for k,v in mapping.items()})
    hashes = {}
    for street, tag in (('preflop','preflop'),('postflop','general')):
        asset = spec[street]
        verify_files({asset['path']:asset['sha256'], ref['path']:ref['sha256']})
        h = hashlib.sha256()
        with np.load(asset['path'], allow_pickle=False) as raw:
            for key, name in arrays.items():
                data = raw[name]
                check(np.isfinite(data).all(), 'Nonfinite network')
                h.update(key.encode())
                h.update(data.tobytes())
        hashes[tag] = h.hexdigest()
    return hashes


def audit_seed(seed, design, stage, policies, mapping):
    folder = STUDY/ARM/f'seed-{seed}'
    ready = stage['arms'][ARM][str(seed)]
    check(ready == read(folder/'ready.json'), 'Ready record mismatch')
    verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256'],
                  folder/'start-verification.json':ready['start_sha256']})
    source_ref, output_ref = (policies[k][str(seed)] for k in (PARENT, ARM))
    check(source_ref == design['parent_policies'][str(seed)] and output_ref == ready['policy'], 'Wrong assigned policy')
    base = read(source_ref['path'])
    out = read(output_ref['path'])
    check(out['preflop']['sha256'] == base['preflop']['sha256'] and out['preflop'] == ready['preflop_export']
          and out['postflop']['sha256'] != base['postflop']['sha256'], 'Street identity differs from design')
    before, entry = load_checkpoint(Path(design['campaign'])/f'seed-{seed}')
    after, end = load_checkpoint(folder)
    source_ptr = read(Path(design['campaign'])/f'seed-{seed}'/'current.json')
    start = read(folder/'start-verification.json')
    check(start['deployed_policy_sha256'] == source_ref['sha256'] and start['checkpoint_files'] == source_ptr['files']
          and start['replay_root_reference_and_rng_verified'] and before['iteration'] == INITIAL
          and after['iteration'] == FINAL and after['seed'] == before['seed'] == seed
          and end == Path(ready['checkpoint']) and source_ptr['iteration'] == INITIAL
          and base['postflop']['path'] == str(entry/'current.npz') and out['postflop']['path'] == str(end/'current.npz')
          and ready['files'] == read(folder/'current.json')['files'], 'Wrong checkpoint identity')
    verify_export(out['preflop']['path'], after['specialist_net'], mapping)
    verify_export(out['postflop']['path'], after['net'], mapping)
    weight_chain = deployed_weights(source_ref, mapping)
    check(weight_chain == start['weights'], 'Incorrect initial network fingerprints')
    rows = [json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
    check([r['iteration'] for r in rows] == list(range(INITIAL+1, FINAL+1)), 'Gap in window')
    by_seat = [0]*SEATS
    unique_weights = {weight_chain['general']}
    for r in rows:
        actor = (r['iteration']-1)%SEATS
        output = r['output_weights']
        check(r['seed'] == seed and r['traverser'] == actor and r['traversals'] == 256
              and r['input_weights'] == weight_chain and output['general'] not in unique_weights
              and output['preflop'] == weight_chain['preflop'] and r['general_steps'] == 768
              and r['opening_steps'] == r['preflop_steps'] == 0 and r['fit_draws'] == 768*512
              and np.isfinite(r['main_loss']) and min(r['fit_rows_by_street']) > 0
              and len(r['fit_rows_by_street']) == len(r['fit_draws_by_street']) == 3
              and min(r['fit_draws_by_street']) > 0 and sum(r['fit_draws_by_street']) == r['fit_draws']
              and r['frozen_preflop_max_probability_error'] == 0 and r['frozen_preflop_probes'] == 265
              and r['new_regret_rows'] > 0 and r['root_rows'] == 169 * 4
              and r['checkpoint_iteration'] == INITIAL + 6 * ((r['iteration'] - INITIAL) // 6)
              and r['buffer_rows'] == [100000]*SEATS and r['reference_collection']['rows'] == 100000,
              'Unexpected round, weight chain, replay or fitting budget')
        by_seat[actor] += r['new_regret_rows']
        unique_weights.add(output['general'])
        weight_chain = output
    check(weight_chain == deployed_weights(output_ref, mapping), 'Final checkpoint never reached evaluation')
    added = [b['total_seen']-a['total_seen'] for a,b in zip(before['buffers'], after['buffers'], strict=True)]
    check(added == by_seat and min(added) > 0, 'Diagnostics disagree with reservoir insertion counters')
    for seat, b in enumerate(after['buffers']):
        recent = np.unique(b['iterations'][b['iterations']>INITIAL])
        check(np.array_equal(recent, [i for i in range(INITIAL+1, FINAL+1) if (i-1)%SEATS==seat]),
              'Final reservoir lacks expected new selfplay rounds')
    selected = sum((b['features'][:, 18:21].sum(axis=0, dtype=np.int64) for b in after['buffers']), np.zeros(3, dtype=np.int64))
    check(selected.tolist() == rows[-1]['fit_rows_by_street'], 'Final flop/turn/river sampling pool differs')
    for key in before['net']:
        if key == 'opening_scale' or key.startswith('opening.'):
            check(np.array_equal(before['net'][key].numpy(), after['net'][key].numpy()), 'Frozen opening tensor changed')
    root_delta = int(np.sum(after['root_replay']['count']) - np.sum(before['root_replay']['count']))
    check(root_delta == sum(r['root_rows'] for r in rows) and root_delta > 0
          and after['strategy_replay']['size'] == 100000
          and after['strategy_replay']['total_seen'] == rows[-1]['reference_collection']['total_seen'],
          'Opening observations or strategy reference counter mismatch')
    return dict(seed=seed, initial_iteration=INITIAL, final_iteration=FINAL, distinct_trained_rounds=len(rows),
                fresh_traversals=sum(r['traversals'] for r in rows), new_regret_rows_by_traverser=added,
                opening_observations=root_delta, final_sample_pool_by_street=selected.tolist(),
                fit_draws_by_street=np.sum([r['fit_draws_by_street'] for r in rows], axis=0).tolist(),
                preflop_probability_probes_per_round=265, postflop_steps=sum(r['general_steps'] for r in rows),
                source_preflop_sha256=base['preflop']['sha256'], source_postflop_sha256=base['postflop']['sha256'],
                endpoint_preflop_sha256=out['preflop']['sha256'], endpoint_postflop_sha256=out['postflop']['sha256'])


def verified_cell(plan, hashes, label, seed, field):
    name = f'{field}-seed-{seed}-{label}'
    directory = STUDY/'evaluation'/name
    result = read(directory/'results.json')
    raw = directory/'hands.jsonl.gz'
    ref = plan['policies'][label][str(seed)]
    provenance = read(directory/'provenance.json')
    verify_files({directory/'results.json': hashes[name], raw: result['raw_hands_sha256'],
                  directory/'provenance.json': result['provenance_sha256'], BRIDGE: result['engine_sha256']})
    for meta in (result, provenance):
        check((meta['deals'],meta['hands'],meta['deal_seed'],meta['field']) == (DEALS, DEALS*SEATS, START, field)
              and meta['candidate_type'] == 'preflop-street-policy-v1' and meta['candidate'] == ref['path']
              and meta['candidate_sha256'] == ref['sha256'] and meta['opponents'] == plan['opponents'][field],
              'Unexpected evaluation assignment')
    v = result['verification']
    check(result['complete'] and result['big_blind_chips']==100 and result['source_provenance_verified']
          and result['raw_hands_verified'] and result['terminal_payoffs_replayed'] and v['terminal_payoffs_replayed']
          and v['aggregate_recomputed'] and v['chip_conservation_verified'] and v['hands']==DEALS*SEATS
          and v['card_blocks']==DEALS, 'Incomplete or unverified cell')
    deps = dict(result['dependency_hashes'])
    check(deps == provenance['dependency_hashes'], 'Changed provenance')
    verify_files(deps)
    spec = read(ref['path'])
    for asset in (ref,spec['preflop'],spec['postflop']):
        check(deps.pop(asset['path']) == asset['sha256'], 'Wrong deployed policy')
    check(all(plan['frozen_files'].get(p) == sha for p,sha in deps.items()), 'Changed common dependency')
    return raw, result, (deps, provenance['semantic_rng'])


def check_terminal(engine, actions, card, utilities):
    state = engine.call(dict(op='reset', seed=card, button=0))
    for action in actions:
        check(not state['terminal'] and state['seat_to_act'] == action['seat'], 'Illegal action order')
        state = engine.call(dict(op='step', action=action))
    check(state['terminal'] and state['utilities'] == utilities, 'Spot replay payoff mismatch')


def match(cells, traces, engine):
    (p, left, rules), (q, right, other_rules) = cells
    check(rules == other_rules and rules[1]=='Shared SHA256 of card seed, hero, actor and complete public action log',
          'Different dependencies or action RNG protocol')
    chips = np.empty((2, DEALS, SEATS), dtype=np.int64)
    count = 0
    with gzip.open(p,'rb') as first, gzip.open(q,'rb') as second:
        for index, lines in enumerate(zip_longest(first,second)):
            deal, hero = divmod(index, SEATS)
            check(deal < DEALS and None not in lines, 'Missing matched hands')
            card = START+deal
            hand_seed = int.from_bytes(hashlib.sha256(f'rich-evaluation-hand:{card}:{hero}'.encode()).digest()[:8], 'big')
            rows = [json.loads(line) for line in lines]
            for i,row in enumerate(rows):
                check((row['deal'],row['hero'],row['seed'],row['hand_seed']) == (deal,hero,card,hand_seed),
                      'Seat, card or seed mismatch')
                utilities = row['utilities']
                check(len(utilities)==SEATS and all(type(n) is int for n in utilities) and sum(utilities)==0
                      and type(row['chips']) is int and row['chips']==utilities[hero]
                      and 0 < row['steps']==len(row['actions']) <= 512, 'Invalid terminal record')
                chips[i,deal,hero] = row['chips']
                if deal % REPLAY_STRIDE == 0:
                    check_terminal(engine, row['actions'], card, utilities)
            traces.add(*rows)
            count += 1
    check(count==DEALS*SEATS, 'Incomplete card and seat coverage')
    means = chips.mean(axis=2)
    for blocks, result in zip(means, (left,right), strict=True):
        np.testing.assert_allclose(blocks,result['deal_blocks'],rtol=0,atol=1e-8)
        np.testing.assert_allclose(blocks.mean(),result['bb_per_100'],rtol=0,atol=1e-8)
    return means


def audit():
    design, plan, stage, outcome = (read(STUDY/k) for k in
        ('plan.json','evaluation-plan.json','stage-ready.json','summary.json'))
    verify_files(SOURCE_HASHES)
    verified_sources = read(ROOT_STUDIES/'confirmation'/'summary.json')
    check(verified_sources['independent_confirmation_pass']
          and plan['policies'][PARENT] == read(ROOT_STUDIES/'preflop-dose'/'evaluation-plan.json')['policies']['steps0'],
          'Expected confirmed 480 specialist')
    verify_files({STUDY/'evaluation-plan.json':outcome['evaluation_plan_sha256'], **plan['frozen_files']})
    verify_files(design['frozen_files'])
    check(all(design['frozen_files'].get(str(p))==h for p,h in SOURCE_HASHES.items()), 'Unpinned prior studies')
    check(plan['frozen_files'][str(STUDY/'stage-ready.json')] == digest(STUDY/'stage-ready.json')
          and plan['frozen_files'][str(STUDY/'plan.json')] == digest(STUDY/'plan.json')
          and plan['opponents'] == read(ROOT_STUDIES/'confirmation'/'evaluation-plan.json')['opponents'],
          'Training assets, design or opponents changed')
    expected = {f'{f}-seed-{s}-{a}':(a,s,f) for a in (PARENT,ARM) for s in SEEDS for f in FIELDS}
    check(len(plan['jobs'])==len(expected)
          and {j['id']:(j['label'],j['seed'],j['field']) for j in plan['jobs']}==expected
          and set(outcome['result_sha256']) == set(expected)
          and (plan['card_start'],plan['deals'],plan['planned_hands']) == (START,DEALS,2160000)
          and (design['initial_iteration'],design['final_iteration']) == (INITIAL,FINAL)
          and (design['general_steps'],design['preflop_steps'],design['opening_steps'],design['main_batch']) == (768,0,0,512)
          and design['checkpoint_every']==6 and design['seeds']==list(SEEDS) and set(design['fields'])==set(FIELDS)
          and (design['evaluation_deal_seed'],design['evaluation_deals']) == (START,DEALS)
          and all(b <= START or a >= START+DEALS for a,b in read(STUDY/'config.json')['prior_card_ranges'])
          and stage['training_complete'] and stage['iteration']==FINAL and outcome['completed']
          and (outcome['completed_cells'],outcome['verified_hands'],outcome['independent_card_blocks']) == (18,2160000,DEALS)
          and outcome['new_selfplay_iterations_per_seed']==FINAL-INITIAL and outcome['model_promoted'] is False,
          'Incomplete study, unexpected design or overlapping cards')
    sys.path[:0] = [plan['source'], str(Path(design['campaign'])/'tools')]
    from compact_regret import ARRAY_MAP
    training = [audit_seed(seed, design, stage, plan['policies'], ARRAY_MAP) for seed in SEEDS]
    z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
    estimates, groups, fallback = {}, {}, {}
    with closing(Bridge()) as engine:
        for field in FIELDS:
            traces = FirstActionDifferences(engine)
            scores = []
            for seed in SEEDS:
                cells = [verified_cell(plan,outcome['result_sha256'],label,seed,field) for label in (PARENT,ARM)]
                scores.append(match(cells,traces,engine))
                for label, (_,result,_) in zip((PARENT,ARM),cells,strict=True):
                    name = f'{field}-seed-{seed}-{label}'
                    counters = result['behavior']
                    fallback[name] = [sum(v for k,v in counters.items() if k.startswith('opponent/') and k.endswith('/'+s))
                                      for s in ('fallback','decisions')]
                    check(fallback[name] == outcome['noregrets_fallback_and_decisions'][name], 'Wrong fallback totals')
            data = np.asarray(scores)  # training seed, endpoint, original card block
            for i,label in enumerate((PARENT,ARM)):
                check_estimate(data[:,i].mean(axis=0),outcome['absolute_scores'][label][field])
            delta_by_seed = data[:,1]-data[:,0]
            delta = delta_by_seed.mean(axis=0)
            per_seed = delta_by_seed.mean(axis=1).tolist()
            estimate = outcome['trained_minus_frozen_parent'][field]
            check_estimate(delta, estimate)
            check([x['seed'] for x in estimate['per_seed']] == list(SEEDS),'Seed order changed')
            np.testing.assert_allclose(per_seed,[x['mean'] for x in estimate['per_seed']],rtol=0,atol=1e-8)
            mean = float(delta.mean())
            se = float(delta.std(ddof=1)/np.sqrt(DEALS))
            nominal = [mean-1.96*se,mean+1.96*se]
            simultaneous = [mean-z*se, mean+z*se]
            sim = outcome['simultaneous_95_bonferroni'][field]
            np.testing.assert_allclose(simultaneous,[sim['lower'],sim['upper']],rtol=0,atol=1e-8)
            groups[field], moves = traces.results(delta)
            check(not moves and all(groups[field][k]['paired_hands']==0 for k in ('opening','other_preflop')),
                  'First hero divergence during frozen preflop policy')
            estimates[field] = dict(mean=mean, per_seed_mean=per_seed, nominal_95=nominal, simultaneous_95=simultaneous)
    conditions = dict(public_lower_above_zero=estimates['public']['nominal_95'][0]>0,
                      at_least_two_public_seeds_improve=sum(v>0 for v in estimates['public']['per_seed_mean'])>=2,
                      mixed_lower_above_minus_25=estimates['mixed']['nominal_95'][0]>=-25,
                      noregrets_lower_above_minus_50=estimates['noregrets']['nominal_95'][0]>=-50)
    check(outcome['selection_conditions']==conditions and outcome['candidate_recommended']==all(conditions.values()),
          'Recommendation inconsistent with predeclared benchmarks')
    return dict(checked_utc=utc(), summary_sha256=digest(STUDY/'summary.json'),
                plan_sha256=digest(STUDY/'plan.json'), stage_sha256=digest(STUDY/'stage-ready.json'),
                evaluation_plan_sha256=digest(STUDY/'evaluation-plan.json'), training=training,
                new_selfplay_iterations_per_seed=24,new_cells=18,reused_cells=0,
                raw_hand_records_reaggregated=2160000,matched_card_seat_seed_field_pairs=1080000,
                independent_card_blocks=DEALS,first_card=START,last_card=START+DEALS-1,
                fixed_training_seeds=list(SEEDS),seats_per_card=SEATS,paired_estimates=estimates,
                nominal_95_by_field_and_simultaneous_95_bonferroni_3_verified=True,
                raw_payoffs_and_chip_conservation_recomputed=True, terminal_payoffs_replayed_by_all_source_cells=True,
                terminal_hands_independently_replayed=18*SEATS*len(range(0,DEALS,REPLAY_STRIDE)),
                terminal_spot_replay_card_stride=REPLAY_STRIDE,first_action_divergence=groups,
                noregrets_lookup_fallback_hits_and_decisions=fallback, selection_conditions=conditions,
                candidate_recommended=all(conditions.values()), model_promoted=False)


if __name__ == '__main__':
    atomic_json(STUDY/'final-audit.json', audit())
