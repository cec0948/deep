"""Independent readback of a completed frozen conditional-river pilot.

Replays recorded actions; it never runs a new search, trains, retries the study,
or mutates source/registries. Card blocks, not selected cases, are statistical
units. Raw cases and hand traces remain in private artifacts.
"""
import argparse
from collections import defaultdict, Counter
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v4_runtime import FullhouseV4

ARMS = ('baseline', 'search-1s', 'search-5s', 'search-10s')
T95 = (None, 12.7062047364, 4.3026527297, 3.1824463053,
       2.7764451052, 2.5705818356, 2.4469118511, 2.3646242510,
       2.3060041352, 2.2621571629, 2.2281388520, 2.2009851601)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def same_number(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def semantic_rng(case, state):
    key = dict(protocol='river-pilot-public-decision-rng-v1',
               deal_seed=case['deal_seed'], hero=case['hero'],
               actor=state['seat_to_act'], public_action_history=state['action_log'])
    encoded = json.dumps(key, sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big')


def legal(state, action):
    assert action['seat'] == state['seat_to_act']
    if action['kind'] == 'raise':
        assert type(action.get('to')) is int and state['raise_bounds'] is not None
        assert state['raise_bounds'][0] <= action['to'] <= state['raise_bounds'][1]
    else:
        assert action['kind'] == ('check' if state['can_check'] else 'call') or (
            action['kind'] == 'fold' and not state['can_check'])


def audit(folder):
    folder = Path(folder).resolve()
    status, result = load(folder/'status.json'), load(folder/'results.json')
    config, provenance = load(folder/'config.json'), load(folder/'provenance.json')
    assert status['status'] == 'complete' and result['complete']
    assert sha(folder/'results.json') == status['results_sha256']
    assert sha(folder/'config.json') == result['config_sha256']
    assert sha(folder/'cases.json') == result['cases_sha256']
    assert sha(folder/'private-hands.jsonl.gz') == result['private_hands_sha256']
    assert sha(provenance['config_source']) == provenance['config_source_sha256']
    assert load(provenance['config_source']) == config
    assert provenance['frozen_hashes'] == config['expected_hashes']
    for path, expected in config['expected_hashes'].items():
        assert sha(path) == expected, path
    assert result['elapsed_seconds'] <= config['max_seconds']
    cases = load(folder/'cases.json')['cases']
    assert len(cases) == 12 and Counter(c['model_seed'] for c in cases) == {342:4,343:4,344:4}
    assert len({(c['model_seed'],c['deal_block']) for c in cases}) == 12
    keyed = {}
    for case in cases:
        assert case['deal_seed'] == config['deal_seed_start']+case['deal_block']
        assert 'your_cards' not in case['root_public_state']
        for player in case['root_observation']['players']:
            assert set(player) <= {'seat','stack','bet_this_street','is_folded','is_all_in'}
        for arm in ARMS:
            row = load(folder/'continuations'/f"{case['id']}-{arm}.json")
            assert row['case'] == case['id'] and row['arm'] == arm
            assert row['model_seed'] == case['model_seed'] and row['deal_block'] == case['deal_block']
            assert row['hero'] == case['hero'] and row['big_blind_chips'] == 100
            keyed[case['id'],arm] = row
    with gzip.open(folder/'private-hands.jsonl.gz', 'rt') as stream:
        traces = [json.loads(line) for line in stream]
    by_arm = defaultdict(list)
    for trace in traces:
        by_arm[trace['case'],trace['arm']].append(trace)
    assert set(by_arm) == set(keyed)
    opponent = Fullhouse(config.get('public_opponent_model'))
    models = {m['seed']: FullhouseV4(m['model'],m['source']) for m in config['models']}
    decisions = repeated_search_history = 0
    engine = Bridge()
    try:
        for case in cases:
            for arm in ARMS:
                state = engine.call(dict(op='reset',seed=case['deal_seed'],button=case['button']))
                for action in case['prefix']:
                    legal(state,action)
                    policy = models[case['model_seed']] if action['seat'] == case['hero'] else opponent
                    expected = policy.decide(state,semantic_rng(case,state))['action']
                    assert {k:v for k,v in action.items() if k!='seat'} == expected
                    state = engine.call(dict(op='step',action=action))
                assert state == case['root_observation']
                row, raw = keyed[case['id'],arm], by_arm[case['id'],arm]
                assert len(raw) == row['continuation_steps']
                hero_decisions, hero_diagnostics = [], []
                earlier_searches = 0
                for index, trace in enumerate(raw):
                    assert not state['terminal']
                    assert trace['continuation_step'] == index
                    assert trace['model_seed'] == case['model_seed'] and trace['deal_block'] == case['deal_block']
                    assert trace['actor'] == state['seat_to_act']
                    assert trace['public_action_history'] == state['action_log']
                    assert trace['seed'] == semantic_rng(case,state)
                    legal(state,trace['action'])
                    if trace['actor'] == case['hero']:
                        hero_decisions.append(dict(seed=trace['seed'], public_action_history=state['action_log'],action=trace['action']))
                        diag = trace['river_search']; hero_diagnostics.append(diag)
                        same_number(diag['measured_total_seconds'],trace['elapsed_seconds'])
                        if arm == 'baseline':
                            expected = models[case['model_seed']].decide(state,trace['seed'])['action']
                            assert {k:v for k,v in trace['action'].items() if k!='seat'} == expected
                        else:
                            assert trace['elapsed_seconds'] < int(arm.split('-')[1][:-1])+.25
                            if diag.get('used'):
                                assert diag['solver_iterations'] >= 2
                                own = next(p for p in diag['ranges'] if p['seat']==case['hero'])
                                assert own['previous_search_decisions_replayed'] == earlier_searches
                                assert all(p['conditioned_on_actual_hero_hole'] is False for p in diag['ranges'])
                                repeated_search_history += int(earlier_searches > 0)
                                earlier_searches += 1
                    else:
                        expected = opponent.decide(state,trace['seed'])['action']
                        assert {k:v for k,v in trace['action'].items() if k!='seat'} == expected
                    state = engine.call(dict(op='step',action=trace['action']));decisions += 1
                assert state['terminal'] and sum(state['utilities']) == 0
                assert state['utilities'] == row['utilities']
                assert state['utilities'][case['hero']] == row['chips']
                assert hero_decisions == row['hero_decisions'] and hero_diagnostics == row['diagnostics']
    finally:
        engine.close()
    for arm in ARMS:
        selected = [keyed[c['id'],arm] for c in cases]
        diags = [d for row in selected for d in row['diagnostics']]
        report = result['arms'][arm]
        assert report['hero_decisions'] == len(diags)
        assert report['search_used'] == sum(bool(d.get('used')) for d in diags)
        assert report['search_attempted'] == sum(bool(d.get('attempted')) for d in diags)
        assert report['timeouts'] == sum('deadline' in str(d.get('reason','')) for d in diags)
        assert report['outside_scope'] == sum(d.get('reason')=='outside_two_active_player_river' for d in diags)
        assert report['reasons'] == dict(Counter(str(d.get('reason','baseline')) for d in diags))
        assert report['cfr_iterations'] == sum(int(d.get('solver_iterations',0)) for d in diags)
        assert report['fallback_decisions'] == (sum(not bool(d.get('used')) for d in diags) if arm!='baseline' else 0)
        elapsed = [d['measured_total_seconds'] for d in diags]
        quantiles = statistics.quantiles(elapsed,n=100,method='inclusive')
        same_number(report['elapsed_p50_seconds'],statistics.median(elapsed))
        same_number(report['elapsed_p95_seconds'],quantiles[94])
        same_number(report['elapsed_p99_seconds'],quantiles[98])
        if arm == 'baseline': continue
        blocks = defaultdict(list)
        for case in cases:
            blocks[case['deal_block']].append((keyed[case['id'],arm]['chips']-keyed[case['id'],'baseline']['chips'])/100)
        values = [statistics.mean(v) for _,v in sorted(blocks.items())]
        n, mean = len(values), statistics.mean(values)
        half = T95[n-1]*statistics.stdev(values)/math.sqrt(n) if n>1 else None
        ref = result['comparisons'][arm+'-minus-baseline']
        assert ref['independent_deal_blocks'] == n
        same_number(ref['mean_bb'],mean)
        same_number(ref['lower95_bb'],mean-half if half is not None else None)
        same_number(ref['upper95_bb'],mean+half if half is not None else None)
        assert ref['block_mean_deltas_bb'] == {str(k):statistics.mean(v) for k,v in blocks.items()}
        same_number(ref['case_weighted_mean_bb'],statistics.mean(v for values in blocks.values() for v in values))
    return dict(passed=True,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        audit_scope='Independent recorded-trajectory/terminal-chip/RNG/grouped-metric readback only; no new search',
        source_provenance_verified=True,frozen_inputs_verified=len(config['expected_hashes']),
        selected_cases=len(cases),independent_deal_blocks=len({c['deal_block'] for c in cases}),
        replayed_arm_continuations=len(keyed),replayed_continuation_decisions=decisions,
        repeated_own_search_history_checks=repeated_search_history,root_observations_reproduced=True,
        terminal_chips_and_conservation_verified=True,semantic_rng_and_frozen_opponent_actions_verified=True,
        private_hands_and_cases_hashes_verified=True,grouped_conditional_metrics_verified=True,
        deadline_fallback_counts_and_latency_quantiles_verified=True,
        no_actual_hero_conditioning_in_recorded_range_provenance=True,
        no_new_search_or_training=True,full_game_strength_proven=False,results_sha256=sha(folder/'results.json'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();result=audit(args.study)
    destination=args.study/'independent-readback-audit.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
