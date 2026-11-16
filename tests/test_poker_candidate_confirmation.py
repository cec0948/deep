import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_poker_candidate_confirmation as study


def config():
    return json.loads((ROOT/'configs/poker-candidate-confirmation-20261116.json').read_text())


def results():
    rows={}
    for job in study.jobs():
        delta={'original120':0,'parent360':2,'legacy_control':3,study.CANDIDATE:10}[job['label']]
        rows[job['id']]=dict(deal_blocks=[float(delta),float(delta+1),float(delta+4)],
            opponents={'fixed':job['field']})
    return rows


@pytest.mark.parametrize('key,value',[
    ('candidate','rich_full'),('evaluation_deal_seed',2037000000),
    ('new_fitting_jobs',1),('new_selfplay_iterations',1),
    ('automatic_retry',True),('model_promotion',True),('max_seconds',28800)])
def test_confirmation_cannot_change_candidate_cards_or_authorized_scope(key,value):
    cfg=config();study.validate_config(cfg);cfg[key]=value
    with pytest.raises(ValueError,match='frozen'):study.validate_config(cfg)


def test_all_cells_and_card_blocks_required_without_inflating_sample_size():
    rows=results();final=study.summarize(rows,deals=3)
    assert len(rows)==36 and final['total_hands']==36*3*6
    assert final['independent_card_blocks']==3
    assert final['candidate_minus_original120']['public']['mean']==10
    assert final['confirmation_pass'] and not final['model_promoted']
    assert not final['adoption_against_original120']['deployment_authorized']
    assert final['new_selfplay_iterations']==final['new_fitting_jobs']==0
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError,match='All36'):study.summarize(rows,deals=3)


def test_seed_repetitions_are_averaged_within_the_same_card_block():
    rows=results()
    for seed,values in zip(study.SEEDS,[[0,0,0],[30,0,0],[0,30,0]]):
        rows[f'public-seed-{seed}-{study.CANDIDATE}']['deal_blocks']=values
        rows[f'public-seed-{seed}-original120']['deal_blocks']=[0,0,0]
    actual=study.summarize(rows,deals=3)['candidate_minus_original120']['public']
    expected=study.interval([10,10,0])
    assert all(actual[k]==v for k,v in expected.items())


@pytest.mark.parametrize('change',['nonfinite','missing_cards','opponent_change'])
def test_invalid_coverage_or_provenance_stops_final_claim(change):
    rows=results();r=rows[f'public-seed-342-{study.CANDIDATE}']
    if change=='nonfinite':r['deal_blocks'][0]=float('nan')
    elif change=='missing_cards':r['deal_blocks'].pop()
    else:r['opponents']={'changed':True}
    with pytest.raises(ValueError,match='coverage|finite|Opponent'):study.summarize(rows,deals=3)


def test_primary_conditions_unchanged_and_secondary_intervals_corrected_separately():
    rows=results()
    for seed in study.SEEDS:
        rows[f'public-seed-{seed}-{study.CANDIDATE}']['deal_blocks']=[0,5,20]
    final=study.summarize(rows,deals=3)
    assert final['adoption_against_original120']==study.adoption_conditions(final['candidate_minus_original120'])
    row=final['diagnostic_comparisons'][f'public/{study.CANDIDATE}-minus-parent360']
    assert row['family_comparisons']==6
    assert row['simultaneous_lower']<row['lower']<row['upper']<row['simultaneous_upper']
    for seed in study.SEEDS:
        rows[f'noregrets-seed-{seed}-{study.CANDIDATE}']['deal_blocks']=[-100,-99,-98]
    assert not study.summarize(rows,deals=3)['confirmation_pass']


def test_orphaned_cell_card_ranges_are_checked_even_without_a_config(tmp_path):
    path=tmp_path/'orphan/results.json';path.parent.mkdir()
    path.write_text(json.dumps(dict(deal_seed=2037100000,deals=20000)))
    prior=study.prior_card_ranges(tmp_path/'new-run',[tmp_path])
    assert list(prior.values())[0]['records']==1
    with pytest.raises(ValueError,match='overlap'):
        study.check_card_range(2037100000,20000,{k:v['range'] for k,v in prior.items()})
    study.check_card_range(2037120000,20000,{k:v['range'] for k,v in prior.items()})


def test_actual_selected_policies_are_copied_without_touching_parents(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'FOLDER',tmp_path)
    monkeypatch.setattr(study,'live_poker_jobs',lambda:[])
    registries=[ROOT/'artifacts'/name/'current.json' for name in
        ['controlled-refit','bounded-continuation','rich-selfplay']]
    before={str(p):p.read_bytes() for p in registries}
    folder=tmp_path/'20261116T052000000001Z'
    plan=study.prepare(folder,ROOT/'configs/poker-candidate-confirmation-20261116.json')
    assert plan['planned_evaluations']==36 and plan['new_fitting_jobs']==0
    assert plan['existing_candidate_backup_sha256']==study.BACKUP_SHA
    assert set(plan['policies'])==set(study.LABELS)
    assert len(list((folder/'policies').glob('*.npz')))==12
    assert all(p.read_bytes()==before[str(p)] for p in registries)
    study.verify_files(plan['frozen_files'])
    target=Path(plan['policies'][study.CANDIDATE]['342']['path'])
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):study.verify_files(plan['frozen_files'])
    with pytest.raises(FileExistsError):study.prepare(folder,ROOT/'configs/poker-candidate-confirmation-20261116.json')


def test_previously_started_confirmation_is_not_automatically_resumed(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'FOLDER',tmp_path)
    monkeypatch.setattr(study,'live_poker_jobs',lambda:[])
    folder=tmp_path/'20261116T052000000002Z';folder.mkdir()
    (folder/'status.json').write_text('{}')
    with pytest.raises(ValueError,match='Never restart'):
        study.run(folder,ROOT/'configs/poker-candidate-confirmation-20261116.json',prepared=True)
