"""Checks for study isolation, paired uncertainty and separate learning gates."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fullhouse_v3_common import digest
from run_poker_foundation_study import (
    OPPONENTS, VARIANTS, summarize_benchmarks, summarize_reference,
    terminate_children, validate_config, verify_reference,
)


def configuration():
    return json.loads((Path(__file__).resolve().parents[1] /
                       'configs/poker-foundation-study-20261112.json').read_text())


def test_study_config_rejects_alias_disagreement_and_unsafe_concurrency():
    cfg = configuration(); validate_config(cfg)
    for key, value in [('evaluation_deal_seed', cfg['evaluation_deal_seed'] + 1),
                       ('parallelism', 4), ('promotion', 'automatic')]:
        changed = copy.deepcopy(cfg); changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_reference_comparisons_average_training_seeds_within_shared_card_blocks():
    cfg = configuration()['reference']
    values = [0., 100., 0., 100.]
    rows = {f'{opponent}-seed-{seed}-{variant}': dict(
                deal_blocks=[0.] * 4 if variant == 'current60' else values,
                hands=24)
            for opponent in OPPONENTS for seed in cfg['seeds'] for variant in VARIANTS}
    result = summarize_reference(rows, cfg)
    change = result['paired_changes']['pressure/current120-minus-current60']
    assert change['mean'] == 50
    assert change['ci95_half_width'] == pytest.approx(1.96 * statistics.stdev(values) / 2)
    assert result['diagnostic_only'] is True
    assert result['deployment_authorized'] is False
    assert result['paired_changes']['pressure/mixture120-minus-current120']['mean'] == 0


def test_benchmark_keeps_strategy_fitting_failure_separate_from_realized_average():
    cfg = configuration()['leduc']
    rows = {'tabular': dict(initial_uniform=dict(exploitability=2.),
                           curve=[dict(metrics=dict(average=dict(exploitability=.1)))])}
    for seed in cfg['seeds']:
        rows[f'deep-{seed}'] = dict(curve=[dict(metrics={
            name:dict(exploitability=value) for name,value in
            [('current',1.9), ('realized_average',.9), ('fitted_average',1.1)]})])
    result = summarize_benchmarks(rows, cfg)
    assert result['checks']['tabular_final_at_most_tenth_uniform']
    assert result['checks']['every_realized_average_at_most_half_uniform']
    assert not result['checks']['every_fitted_average_at_most_half_uniform']
    assert not result['all_reference_checks_pass']
    assert result['six_player_adoption_authorized'] is False


def raw_fixture(tmp_path):
    model = tmp_path/'model.npz'; model.write_bytes(b'fixed-candidate')
    code = tmp_path/'code.py'; code.write_text('source')
    output = tmp_path/'evaluation'; output.mkdir()
    plan = dict(frozen_files={str(code):digest(code)})
    (output/'plan.json').write_text(json.dumps(plan))
    (output/'traces.json').write_text('[]')
    hands=[]
    for deal in range(2):
        for hero in range(6):
            hand_seed=int.from_bytes(hashlib.sha256(f'compact-mixture:100:{deal}:{hero}'.encode()).digest()[:8], 'big')
            hands.append(dict(deal=deal,hero=hero,seed=100+deal,chips=0,utilities=[0]*6,
                              hand_seed=hand_seed,policy_iteration=None))
    with gzip.open(output/'hands.jsonl.gz','wt') as stream:
        for hand in hands:
            stream.write(json.dumps(hand)+'\n')
    expected=dict(candidate=digest(model), **{key:digest(code) for key in (
        'evaluator_sha256','runtime_sha256','engine_sha256',
        'source_features_sha256','opponent_source_sha256')})
    result=dict(complete=True,opponent='pressure',deals=2,hands=12,deal_seed=100,
        candidate=str(model),candidate_sha256=digest(model),candidate_type='compact_advantage',
        plan_sha256=digest(output/'plan.json'),raw_hands_sha256=digest(output/'hands.jsonl.gz'),
        traces_sha256=digest(output/'traces.json'),deal_blocks=[0,0],bb_per_100=0,
        interval=dict(mean=0,ci95_half_width=0,lower=0,upper=0),
        **{key:value for key,value in expected.items() if key!='candidate'})
    (output/'results.json').write_text(json.dumps(result))
    return output, dict(model=str(model),opponent='pressure',variant='current120'), dict(evaluation_deals=2,evaluation_deal_seed=100), expected, hands


def test_raw_verifier_rejects_reordered_hands_even_when_archive_hash_is_updated(tmp_path):
    output,job,cfg,expected,hands=raw_fixture(tmp_path)
    assert verify_reference(output,job,cfg,expected)['hands']==12
    hands[0],hands[1]=hands[1],hands[0]
    with gzip.open(output/'hands.jsonl.gz','wt') as stream:
        for hand in hands: stream.write(json.dumps(hand)+'\n')
    result=json.loads((output/'results.json').read_text())
    result['raw_hands_sha256']=digest(output/'hands.jsonl.gz')
    (output/'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='reordered'):
        verify_reference(output,job,cfg,expected)


def test_raw_verifier_rejects_changed_frozen_source(tmp_path):
    output,job,cfg,expected,_=raw_fixture(tmp_path)
    (tmp_path/'code.py').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):
        verify_reference(output,job,cfg,expected)


def test_cleanup_kills_descendant_after_failed_leader_has_exited(tmp_path):
    pidfile=tmp_path/'descendant.pid'
    script=('import subprocess,sys,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
            f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); sys.exit(1)')
    leader=subprocess.Popen([sys.executable,'-c',script],start_new_session=True)
    leader.wait(timeout=5)
    child=int(pidfile.read_text())
    try:
        terminate_children({'failed':dict(process=leader,log=io.StringIO())})
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            stat=Path(f'/proc/{child}/stat')
            if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('Descendant survived cleanup of an exited leader')
    finally:
        try: os.kill(child,9)
        except ProcessLookupError: pass
