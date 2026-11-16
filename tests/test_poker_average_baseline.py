import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_poker_average_baseline import validate_config,jobs,summarize,SEEDS,LABELS,FIELDS
from train_poker_average_baseline import load_checkpoint
from poker_strategy_average import AverageNet,export_average
from fullhouse_v3_common import digest


def config():return json.loads((ROOT/'configs/poker-average-baseline-20261116.json').read_text())


def test_fixed_budget_preserves_data_and_compute_and_stops_at420():
    cfg=config();validate_config(cfg)
    assert cfg['buffer_capacity']==100000 and cfg['main_steps']*cfg['batch_size']==524288
    assert cfg['strategy_capacity']==100000 and cfg['model_promotion'] is False
    assert len(jobs())==36 and len({j['id'] for j in jobs()})==36
    assert len(jobs())*cfg['evaluation_deals']*6==4320000


@pytest.mark.parametrize('key,value',[('buffer_capacity',5000),('main_steps',256),('batch_size',128),
    ('iterations',480),('stages',[372,420,480]),('model_promotion',True),('evaluation_deals',2000),
    ('automatic_retry',True),('average_max_replay_tv',.3)])
def test_sample_reductions_and_unplanned_extensions_rejected(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_summary_uses_one_card_block_after_averaging_all_seeds():
    rows={}
    for label in LABELS:
        for seed in SEEDS:
            for field in FIELDS:
                values=[-3.,-1.,1.,3.]
                if label=='average361_420':values=[x+10 for x in values]
                if seed==343:values=[x+100 for x in values]
                if seed==344:values=[x-100 for x in values]
                rows[f'{field}-seed-{seed}-{label}']=dict(deal_blocks=values,opponents={'frozen':'same'})
    result=summarize(rows,deals=4)
    for field in FIELDS:
        row=result['average_minus_original120'][field]
        assert row['mean']==10 and row['ci95_half_width']==0 and len(row['per_seed'])==3
    # Absolute interval is based on4cards, not12seed/card pairs or72seats.
    assert result['absolute_scores']['public/original120']['ci95_half_width']==pytest.approx(2.530349119522179)
    assert result['independent_card_blocks']==4 and len(result['diagnostic_comparisons'])==9
    assert result['model_promoted'] is False


def test_partial_cells_and_changed_opponents_rejected():
    with pytest.raises(ValueError):summarize({},deals=4)
    rows={j['id']:dict(deal_blocks=[1.,2.,3.,4.],opponents={'same':True}) for j in jobs()}
    rows[jobs()[0]['id']]['opponents']={'same':False}
    with pytest.raises(ValueError):summarize(rows,deals=4)


def test_numpy_inference_does_not_import_torch():
    code="import sys;sys.path.insert(0,'scripts');import poker_average_inference;assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_missing_or_tampered_checkpoint_is_rejected(tmp_path):
    folder=tmp_path/'seed-342';folder.mkdir();(folder/'checkpoints').mkdir()
    (folder/'current.json').write_text(json.dumps(dict(iteration=420,directory='../escape',files={})))
    with pytest.raises(ValueError):load_checkpoint(folder)
    (folder/'current.json').write_text(json.dumps(dict(iteration=420,directory='iter-00420',files={})))
    with pytest.raises(ValueError):load_checkpoint(folder)
