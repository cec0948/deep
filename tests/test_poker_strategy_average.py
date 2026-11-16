import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from poker_strategy_average import (StrategyReplay,AverageNet,collect_reference,fit_average,
    export_average,load_average,average_strategy)
from validate_poker_strategy_average import math_check
from compact_lightweight_backup import save_state,load_state,equal_state


def feature(index=0):
    x=np.zeros(51,np.float32);x[index%40]=1;x[46:48]=1
    return x


@pytest.mark.parametrize('players',[2,3])
def test_exact_multiplayer_estimator_and_biased_negative_control(players):
    report=math_check(players)
    assert report['passed'] and report['max_error']<1e-12
    assert report['wrong_on_policy_frequency_error']>1e-3


def test_uniform_reference_keeps_actions_that_target_policy_never_takes():
    replay=StrategyReplay(100,seed=3)
    # Same target player acts twice. Its zero-reach branch must have no mass.
    for seed in range(100):
        collect_reference((),2,replay,2,np.random.default_rng(seed),
            terminal=lambda s:len(s)==2,chance=lambda s:False,outcomes=lambda s:[],
            actor=lambda s:0,observe=lambda s:(feature(len(s)),feature()[46:51]),
            strategy=lambda s,x,l:np.array([1,0,0,0,0]),child=lambda s,a:s+(a,))
    assert replay.total_seen>100 and replay.total_seen<200
    assert np.all(replay.mass[:replay.size]==2)


def test_replay_cap_serialization_and_next_reservoir_draw_parity(tmp_path):
    a=StrategyReplay(17,seed=93)
    for i in range(51):a.add(feature(i),[.2,.8,0,0,0],i+1)
    save_state(a.state_dict(),tmp_path/'replay')
    b=StrategyReplay.from_state(load_state(tmp_path/'replay'))
    for i in range(51,200):
        for replay in (a,b):replay.add(feature(i),[.7,.3,0,0,0],i+1)
    assert a.size==17 and a.total_seen==200
    assert equal_state(a.state_dict(),b.state_dict())


@pytest.mark.parametrize('bad',[float('nan'),-1,float('inf')])
def test_invalid_mass_rejected(bad):
    with pytest.raises(ValueError):StrategyReplay().add(feature(),[.5,.5,0,0,0],bad)


def test_invalid_or_illegal_targets_rejected():
    for y in ([.5,.4,0,0,0],[-.1,1.1,0,0,0],[.4,.5,.1,0,0]):
        with pytest.raises(ValueError):StrategyReplay().add(feature(),y,1)


def test_average_fit_does_not_change_legacy_training_rng_or_replay():
    torch.set_num_threads(1)
    a=StrategyReplay(31,seed=3)
    for i in range(20):a.add(feature(i),[.2,.8,0,0,0],i+1)
    original=copy.deepcopy(a.state_dict())
    torch.manual_seed(10);np.random.seed(10)
    ts=torch.get_rng_state();ns=np.random.get_state()
    fit_average(a,seed=99,steps=8,batch_size=16,width=16)
    assert torch.equal(ts,torch.get_rng_state())
    assert equal_state(ns,np.random.get_state())
    assert equal_state(original,a.state_dict())


def test_export_runtime_parity_and_no_history_required(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(8)
    net=AverageNet(32);path=tmp_path/'average.npz'
    export_average(net,path,window_start=361,window_end=420)
    layers,meta=load_average(path)
    x=np.stack([feature(i) for i in range(10)])
    with torch.no_grad():expected=net(torch.from_numpy(x)).exp().numpy()
    actual=np.stack([average_strategy(layers,row,row[46:51]) for row in x])
    assert np.max(np.abs(expected-actual))<1e-6
    assert meta==dict(window_start=361,window_end=420)
    assert list(tmp_path.iterdir())==[path]


def test_uniform_zero_logits_and_illegal_large_logit_no_overflow(tmp_path):
    net=AverageNet(8)
    with torch.no_grad():
        for p in net.parameters():p.zero_()
        net.body[-1].bias[4]=1e20
    p=tmp_path/'a.npz';export_average(net,p)
    layers,_=load_average(p)
    assert np.array_equal(average_strategy(layers,feature(),feature()[46:51]),[.5,.5,0,0,0])


def test_weighted_objective_recovers_time_weighted_not_unweighted_average():
    torch.set_num_threads(1);a=StrategyReplay(16)
    a.add(feature(),[1,0,0,0,0],1)
    a.add(feature(),[0,1,0,0,0],9)
    net,report=fit_average(a,seed=11,steps=256,batch_size=32,width=16)
    with torch.no_grad():p=net(torch.from_numpy(feature()[None])).exp().numpy()[0]
    assert abs(p[0]-.1)<.025 and abs(p[1]-.9)<.025
    assert report['weighted_replay_tv']<.025


def test_corrupt_replay_rejected():
    a=StrategyReplay(10);a.add(feature(),[1,0,0,0,0],1)
    state=a.state_dict();state['y'][0,4]=.1
    with pytest.raises(ValueError):StrategyReplay.from_state(state)
