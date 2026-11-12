"""Pre-fit objective gate and unchanged performance-decision checks."""
import copy
import json
from pathlib import Path
import statistics
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fullhouse_v3_common import digest
from run_raw_objective_probe import (GATE_CHECKS, summarize_all, validate_config, verify_gate,
                                     verify_prior_refit_record)


def configuration():
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/raw-objective-probe-20261112.json').read_text())
    parent=dict(seeds=[342,343,344],iterations=120,batch_size=512,learning_rate=.001)
    return cfg,parent


def rows_for(cfg):
    fields=cfg['evaluation_fields']+cfg['reference_opponents']
    return {f'{field}-seed-{seed}-{variant}':dict(deal_blocks=[0.,0.,0.,0.],hands=24)
            for field in fields for seed in cfg['seeds'] for variant in cfg['variants']}


def test_probe_rejects_changed_objective_budget_or_rng():
    cfg,parent=configuration();validate_config(cfg,parent)
    for key,value in [('parallelism',4),('gate_seconds',121),('promotion','auto'),
                       ('fit_seed_base',123),('extra_steps',4096)]:
        changed=copy.deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError):validate_config(changed,parent)


def test_only_original_fields_control_raw_candidate_gate():
    cfg,_=configuration();rows=rows_for(cfg)
    for seed in cfg['seeds']:
        rows[f'public-seed-{seed}-raw-objective']['deal_blocks']=[10.]*4
        for field in cfg['reference_opponents']:
            rows[f'{field}-seed-{seed}-raw-objective']['deal_blocks']=[-10000.]*4
        rows[f'public-seed-{seed}-refit']['deal_blocks']=[500.]*4
    result=summarize_all(rows,cfg)
    assert list(result['decisions'])==['raw-objective']
    assert result['decisions']['raw-objective']['original_four_conditions_pass'] is True
    assert result['decisions']['raw-objective']['deployment_authorized'] is False
    assert result['paired_changes']['public/raw-objective-minus-refit']['mean']==-490
    assert result['paired_changes']['pressure/raw-objective-minus-current']['mean']==-10000
    # Favorable diagnostic fields cannot rescue a failed original mixed gate.
    for seed in cfg['seeds']:
        rows[f'mixed-seed-{seed}-raw-objective']['deal_blocks']=[-26.]*4
        rows[f'pressure-seed-{seed}-raw-objective']['deal_blocks']=[10000.]*4
    assert not summarize_all(rows,cfg)['decisions']['raw-objective']['original_four_conditions_pass']


def test_shared_cards_are_one_block_across_training_seeds():
    cfg,_=configuration();rows=rows_for(cfg);change=[0.,100.,0.,100.]
    for seed in cfg['seeds']:
        rows[f'public-seed-{seed}-raw-objective']['deal_blocks']=change
    result=summarize_all(rows,cfg)['paired_changes']['public/raw-objective-minus-current']
    assert result['ci95_half_width']==pytest.approx(1.96*statistics.stdev(change)/2)


def gate_fixture(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cfg,_=configuration();cfg=cfg['gate_config']
    replay=tmp_path/'retained-replay.npz';replay.write_bytes(b'gate-evidence')
    report=dict(complete=True,config=cfg,method=cfg['method'],passed=False,
        gate=dict(passed=False,checks={key:key!='all_exact_gradient_identities' for key in GATE_CHECKS}),
        source_hashes={str(root/'scripts'/name):digest(root/'scripts'/name) for name in
                       ('validate_sampling_objective.py','raw_objective_fitting.py')},
        replay_file=dict(name=replay.name,bytes=replay.stat().st_size,sha256=digest(replay)))
    (tmp_path/'results.json').write_text(json.dumps(report))
    return cfg,report


def test_failed_objective_gate_is_not_accepted_as_a_completed_gate(tmp_path):
    cfg,_=gate_fixture(tmp_path)
    with pytest.raises(ValueError,match='no poker fitting'):
        verify_gate(tmp_path,cfg)


def test_partial_check_list_cannot_claim_objective_gate_pass(tmp_path):
    cfg,report=gate_fixture(tmp_path)
    report['passed']=True;report['gate']=dict(passed=True,checks={'all_exact_gradient_identities':True})
    (tmp_path/'results.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='boolean checks'):
        verify_gate(tmp_path,cfg)


@pytest.mark.parametrize('key,changed', [
    ('parent_checkpoint_sha256','wrong-parent'), ('fit_seed',123), ('batch_size',1024),
    ('learning_rate',.0001), ('artifact_kind','derived-offline-anchored-refit'),
    ('format','different-fit-format'), ('extra_steps',4096), ('new_selfplay_samples',1),
])
def test_prior_comparator_requires_matched_parent_rng_objective_and_budget(key,changed):
    cfg,_=configuration();parent=Path('/example/verified-parent120');sha='a'*64;seed=342
    row=dict(format='compact-offline-refit-v1',artifact_kind='derived-offline-refit',seed=seed,
        parent=str(parent),parent_iteration=120,parent_checkpoint_sha256=sha,
        extra_steps=3072,fit_seed=cfg['fit_seed_base']+seed,batch_size=512,learning_rate=.001,
        new_selfplay_samples=0,resumable_selfplay=False,complete=True,
        opening_unchanged=True,scales_unchanged=True,parent_checkpoint_unchanged=True)
    verify_prior_refit_record(row,seed,cfg,parent,sha)
    row[key]=changed
    with pytest.raises(ValueError,match='matched comparison'):
        verify_prior_refit_record(row,seed,cfg,parent,sha)
