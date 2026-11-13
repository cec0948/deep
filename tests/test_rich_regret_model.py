from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rich_regret_model import RichRegretModel

SOURCE=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/source'
TEACHER=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/seed-342/current_00360.npz'


def features(opening=False,dimension=64):
    x=np.zeros(dimension,np.float32);x[17]=1;x[24 if opening else 21]=1
    x[27]=.015 if opening else .2;x[45]=1;x[46:51]=1
    return x


@pytest.fixture
def model():
    if not TEACHER.exists():pytest.skip('Requires existing360teacher asset')
    torch.set_num_threads(1);torch.manual_seed(17)
    return RichRegretModel(TEACHER,64,SOURCE,width=16)


def test_zero_residual_teacher_policy_is_byte_identical_and_opening_stays_frozen(model):
    for opening in [False,True]:
        x=features(opening)
        expected=model.teacher_strategy(x[:51],x[46:51])
        actual=model.strategy(x,x[46:51])
        assert actual.dtype==expected.dtype and np.array_equal(actual,expected)
    with torch.no_grad():model.residual.layers[-1].bias.fill_(.7)
    model._sync_numpy()
    x=features(True)
    assert np.array_equal(model.strategy(x,x[46:51]),model.teacher_strategy(x[:51],x[46:51]))


def test_scalar_numpy_residual_matches_torch_and_restore(model,tmp_path):
    with torch.no_grad():
        model.residual.layers[-1].weight.normal_(0,.01)
        model.residual.layers[-1].bias.normal_(0,.01)
    model._sync_numpy();x=features()
    expected=model.residual(torch.from_numpy(x)[None])[0].detach().numpy()
    assert np.allclose(model.residual_numpy(x),expected,rtol=1e-5,atol=1e-5)
    restored=RichRegretModel(TEACHER,64,SOURCE,width=16);restored.load_state_dict(model.state_dict())
    assert np.array_equal(model.strategy(x,x[46:51]),restored.strategy(x,x[46:51]))
    model.export(tmp_path/'current.npz')
    with np.load(tmp_path/'current.npz',allow_pickle=False) as archive:
        assert str(archive['teacher_sha256'])==model.teacher_sha256
        assert not bool(archive['average_policy_available'])
        assert archive['residual_w0'].shape==(16,64)


def test_fresh_weighted_legal_regret_fitting_reduces_fixed_target_error(model):
    x=np.stack([features() for _ in range(8)])
    x[:,51]=np.linspace(0,1,8)
    target=np.stack([model.base_regrets(row[:51],row[46:51]) for row in x]).astype(np.float32)
    target[:,1]+=25
    class Buffer:
        features=x;advantages=target;iterations=np.arange(1,9,dtype=np.float32)
        def __len__(self):return len(x)
    np.random.seed(9);torch.manual_seed(9)
    report=model.fit([Buffer()],steps=96,batch_size=16)
    assert report['replay_mse_after_bb2'] < report['teacher_prior_mse_bb2']*.1
    assert report['objective']=='uniform_replay_sampling_iteration_weighted_raw_BB_MSE'
    assert 0<=report['relative_policy_tv']<=1
    assert not model.zero_residual


def test_invalid_observation_and_mismatched_teacher_restore_rejected(model):
    x=features();x[0]=np.nan
    with pytest.raises(ValueError):model.strategy(x,x[46:51])
    state=model.state_dict();state['teacher_sha256']='0'*64
    with pytest.raises(ValueError):model.load_state_dict(state)
