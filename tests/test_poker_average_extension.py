import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_poker_average_extension import validate_config,summarize,jobs,LABELS,SEEDS,FIELDS,verify_inherited_source
from evaluate_average_baseline import interval
from plan_average_extension_backup_retention import replacement_candidates,MODE
from fullhouse_v3_common import digest


def test_inherited_manifest_verifies_policies_before_their_new_names_are_copied(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child'
    for folder in (parent,child):
        (folder/'source').mkdir(parents=True);(folder/'source/code.py').write_text('frozen')
    (parent/'seed-342').mkdir();(parent/'seed-342/original120.npz').write_bytes(b'policy')
    files={str(p.relative_to(parent)):digest(p) for p in parent.rglob('*') if p.is_file()}
    verify_inherited_source(parent,child,dict(files=files))
    (parent/'seed-342/original120.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):verify_inherited_source(parent,child,dict(files=files))


def test_new_authorized_budget_preserves_the_learning_kernel_and_normalizer():
    cfg=json.loads((ROOT/'configs/poker-average-extension-20261117.json').read_text())
    validate_config(cfg)
    old=json.loads((ROOT/'configs/poker-average-continuation-20261117.json').read_text())
    for k in ('traversals','buffer_capacity','main_steps','batch_size','learning_rate','root_repetitions',
              'opening_steps','opening_max_steps','strategy_capacity','reference_hands','average_fit_steps',
              'average_batch','average_width','average_normalize'):
        assert cfg[k]==old[k]
    assert cfg['resume_iterations']==dict.fromkeys(map(str,SEEDS),420)
    assert cfg['iterations']==480 and cfg['weight_horizon']==420
    for k,v in [('iterations',540),('weight_horizon',480),('buffer_capacity',50000),
                ('primary_candidate','average361_480'),('average_normalize',True),
                ('auxiliary_quality_error_stops_main',True),('automatic_retry',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{k:v}))


def test_paired_improvement_uses_parent420_and_card_blocks_not_repeated_seeds():
    rows={};blocks=[-100.,-20.,40.,100.]
    for label in LABELS:
        for seed in SEEDS:
            for field in FIELDS:
                values=blocks if label=='current480' else [-50.]*4 if label=='average361_480' else [0.]*4
                rows[f'{field}-seed-{seed}-{label}']=dict(deal_blocks=values,opponents={'fixed':True})
    result=summarize(rows,deals=4)
    actual=result['diagnostic_comparisons']['public/current480-minus-parent420']
    expected=interval(blocks)
    assert actual['mean']==expected['mean']
    assert actual['ci95_half_width']==pytest.approx(expected['ci95_half_width'])
    assert result['independent_card_blocks']==4
    assert actual['simultaneous_lower']<0<actual['simultaneous_upper']
    assert not result['model_promoted']
    with pytest.raises(ValueError):summarize(dict(list(rows.items())[:-1]),deals=4)


def record(stage,sha):
    return dict(kind='training-checkpoint',status='complete',checkpoint_mode=MODE,
        training_family='legacy-window-average',iterations=dict.fromkeys(map(str,SEEDS),stage),
        averaging_window=[361,stage],restoration_verified=True,warm_start_state_verified=True,
        resumable_selfplay=False,exact_legacy_resume=False,average_policy_available=True,
        historical_mixture_available=False,verified_remote_size=True,verified_owner_only=True,
        sha256=sha,file_id='private-'+sha[:4])


def test_retention_requires_fully_verified_self_contained_successor_and_dependency_closure():
    parent=record(420,'a'*64);child=record(480,'b'*64)
    child.update(replaces_archive_sha256=parent['sha256'],parent_comparison_policies_in_archive=True)
    active={0:parent,1:child}
    assert replacement_candidates(active)=={0}
    for key in ('parent_comparison_policies_in_archive','verified_remote_size','verified_owner_only'):
        changed=copy.deepcopy(active);changed[1][key]=False
        assert not replacement_candidates(changed)
    partial=copy.deepcopy(active);partial[1]['restoration_verified']=False
    with pytest.raises(ValueError):replacement_candidates(partial)
    dependent=dict(kind='research-validation',restore_dependencies=[dict(parent_checkpoint_sha256='a'*64)])
    assert not replacement_candidates(dict(active,**{})|{2:dependent})
    assert not replacement_candidates({0:parent})


def test_unsafe_or_unapproved_backup_stage_blocks_retention():
    bad=record(540,'c'*64)
    with pytest.raises(ValueError):replacement_candidates({0:bad})
