import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fullhouse_runtime import Bridge
from fullhouse_v4_runtime import FullhouseV4
from rich_poker_observation import RICH_DIM,SCHEMA,encode_bridge
from rich_poker_runtime import RichPolicy,load_descriptor,load_residual
from rich_regret_model import RichRegretModel

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign'
TEACHER=PARENT/'seed-342/current_00360.npz'
SOURCE=PARENT/'source'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def exported(tmp_path):
    if not TEACHER.exists():pytest.skip('Requires the retained360teacher and engine assets')
    torch.set_num_threads(1);torch.manual_seed(11)
    model=RichRegretModel(TEACHER,RICH_DIM,source=SOURCE)
    weights=tmp_path/'current.npz';model.export(weights)
    descriptor=tmp_path/'policy.json'
    spec=dict(format='rich-policy-export-v1',model=str(weights),model_sha256=sha(weights),
        teacher=str(TEACHER),teacher_sha256=sha(TEACHER),source=str(SOURCE),iteration=0,
        seed=342,feature_schema=SCHEMA,parent_iteration=360,average_policy_available=False)
    descriptor.write_text(json.dumps(spec))
    return model,weights,descriptor,spec


def observations():
    engine=Bridge()
    try:
        yield engine.call(dict(op='replay',hero=3,button=0,hole=['As','Kd'],board=[],actions=[]))
        actions=[dict(seat=s,kind='call') for s in (3,4,5,0,1)]+[dict(seat=2,kind='check')]
        for count in (3,4,5):
            yield engine.call(dict(op='replay',hero=1,button=0,hole=['As','Kd'],
                board=['2c','7d','Jh','9c','Qs'][:count],actions=actions))
            actions += [dict(seat=s,kind='check') for s in (1,2,3,4,5,0)]
    finally:engine.close()


def test_zero_residual_runtime_is_exact_teacher_on_all_streets(exported):
    _,_,descriptor,_=exported
    candidate=RichPolicy(descriptor,SOURCE);teacher=FullhouseV4(TEACHER,SOURCE)
    for state in observations():
        np.testing.assert_array_equal(candidate._strategy(state),teacher._strategy(state))
        a=candidate.decide(state,19);b=teacher.decide(state,19)
        assert a['action']==b['action'] and a['distribution']==b['distribution']
        assert a['strength_verified'] is False and a['new_family_iteration']==0
        assert a['feature_dimension']==2885 and a['average_policy_available'] is False
        assert a['parent_iteration']==360


def test_nonzero_residual_runtime_matches_training_numpy_policy(exported):
    model,weights,descriptor,spec=exported
    with torch.no_grad():model.residual.layers[-1].bias.copy_(torch.tensor([.5,-.1,.2,-.3,.15]))
    model._sync_numpy();model.export(weights);spec['model_sha256']=sha(weights);spec['iteration']=6
    descriptor.write_text(json.dumps(spec));candidate=RichPolicy(descriptor,SOURCE)
    for state in observations():
        old,legal=candidate.features.encode_state_dict(state,candidate.lookup._equity_tables)
        rich=encode_bridge(state,old)
        np.testing.assert_array_equal(candidate._strategy(state),model.strategy(rich,legal))


def test_export_checksum_and_schema_changes_are_rejected(exported):
    _,weights,descriptor,spec=exported
    changed=dict(spec,model_sha256='0'*64);descriptor.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='dependency changed'):load_descriptor(descriptor)
    changed=dict(spec,feature_schema='legacy51');descriptor.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='descriptor'):load_descriptor(descriptor)
    changed=dict(spec,average_policy_available=True);descriptor.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='capability'):load_descriptor(descriptor)
    with np.load(weights,allow_pickle=False) as data:arrays={k:data[k].copy() for k in data.files}
    arrays['residual_w0'][0,0]=np.nan;np.savez_compressed(weights,**arrays)
    with pytest.raises(ValueError,match='finite'):load_residual(weights,spec['teacher_sha256'])


def test_hidden_opponent_cards_are_not_accepted_by_rich_runtime(exported):
    candidate=RichPolicy(exported[2],SOURCE)
    state=next(observations())
    state['opponent_holes']=[['Ac','Ad']]
    with pytest.raises(ValueError,match='hidden fields'):candidate.decide(state,1)
