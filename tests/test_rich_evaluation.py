import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_rich_selfplay import (semantic_seed,hand_seed,interval,_verify_rows,run,verify_results)
from rich_regret_model import RichRegretModel
from rich_poker_observation import RICH_DIM,SCHEMA

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign'


def test_action_rng_is_shared_across_policy_versions_and_ignores_private_cards():
    state=dict(seat_to_act=2,action_log=[dict(seat=3,action='call',amount=100)],your_cards=['As','Kd'])
    other=dict(state,your_cards=['2c','3d'])
    assert semantic_seed(2035000000,2,state)==semantic_seed(2035000000,2,other)
    assert semantic_seed(2035000000,2,state)!=semantic_seed(2035000001,2,state)
    changed=dict(state,action_log=state['action_log']+[dict(seat=4,action='fold',amount=0)])
    assert semantic_seed(2035000000,2,state)!=semantic_seed(2035000000,2,changed)


def raw_fixture(tmp_path):
    path=tmp_path/'hands.jsonl.gz';blocks=[]
    with gzip.open(path,'wt') as stream:
        for deal in range(2):
            chips=[]
            for hero in range(6):
                values=[0]*6;values[hero]=100*(deal+hero);values[(hero+1)%6]=-values[hero]
                chips.append(values[hero]);stream.write(json.dumps(dict(deal=deal,hero=hero,
                    seed=700+deal,chips=values[hero],utilities=values,hand_seed=hand_seed(700+deal,hero),
                    actions=[],steps=0))+'\n')
            blocks.append(statistics.mean(chips))
    stats=interval(blocks)
    return path,dict(deals=2,hands=12,deal_seed=700,deal_blocks=blocks,
                     bb_per_100=stats['mean'],ci95_half_width=stats['ci95_half_width'])


def test_every_raw_card_and_seat_is_counted_once_and_aggregate_is_recomputed(tmp_path):
    path,result=raw_fixture(tmp_path)
    checked=_verify_rows(result,path,replay=False)
    assert checked['hands']==12 and checked['card_blocks']==2
    assert checked['terminal_payoffs_replayed'] is False
    wrong=dict(result,deal_blocks=[0,0])
    with pytest.raises(ValueError,match='aggregate'):_verify_rows(wrong,path,replay=False)
    with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
    rows[1]['hero']=0
    with gzip.open(path,'wt') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='out-of-order'):_verify_rows(result,path,replay=False)


def test_small_real_legacy_cell_records_and_replays_all_concrete_actions(tmp_path):
    model=PARENT/'seed-342/current_00360.npz';source=PARENT/'source'
    if not model.exists():pytest.skip('Requires retained policies and trusted rules engine')
    out=tmp_path/'real-cell'
    #Separate card range from the newly declared2035000000holdout.
    result=run(model,source,'public',2,2034999900,60,out)
    assert result['complete'] and result['raw_hands_verified'] and result['terminal_payoffs_replayed']
    assert result['verification']['concrete_decisions_replayed']>0
    assert result['hands']==12 and result['statistical_unit'].startswith('Original card block')
    assert verify_results(out,replay=True)['deal_blocks']==result['deal_blocks']
    with pytest.raises(FileExistsError):run(model,source,'public',2,2034999900,60,out)
    changed=json.loads((out/'results.json').read_text());changed['bb_per_100']+=1
    (out/'results.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='aggregate'):verify_results(out)


def test_zero_residual_cell_has_identical_paired_payoffs_to_parent(tmp_path):
    teacher=PARENT/'seed-342/current_00360.npz';source=PARENT/'source'
    if not teacher.exists():pytest.skip('Requires retained policies and trusted rules engine')
    torch.set_num_threads(1);torch.manual_seed(41)
    model=RichRegretModel(teacher,RICH_DIM,source=source)
    weights=tmp_path/'current.npz';model.export(weights)
    descriptor=tmp_path/'rich.json'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    descriptor.write_text(json.dumps(dict(format='rich-policy-export-v1',
        model=str(weights),model_sha256=sha(weights),teacher=str(teacher),teacher_sha256=sha(teacher),
        source=str(source),iteration=0,seed=342,feature_schema=SCHEMA)))
    parent=run(teacher,source,'public',2,2034999910,60,tmp_path/'parent')
    candidate=run(descriptor,source,'public',2,2034999910,60,tmp_path/'rich')
    assert candidate['candidate_type']=='rich_regret_residual'
    assert candidate['candidate_sha256']==sha(descriptor)
    assert candidate['terminal_payoffs_replayed'] and candidate['verification']['hands']==12
    assert candidate['deal_blocks']==parent['deal_blocks']
    with gzip.open(tmp_path/'parent/hands.jsonl.gz','rt') as stream:a=[json.loads(x) for x in stream]
    with gzip.open(tmp_path/'rich/hands.jsonl.gz','rt') as stream:b=[json.loads(x) for x in stream]
    assert a==b
