"""Independent cross-module checks before fresh six-player neural learning."""
import importlib
from pathlib import Path
import random
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from compact_lightweight_backup import equal_state
from rich_poker_observation import RICH_DIM
from rich_regret_model import RichRegretModel
import rich_poker_traversal as traversal
from train_rich_selfplay import checkpoint_state,commit_checkpoint,load_current


def test_external_sampling_regret_uses_raw_bb_without_extra_opponent_probability(monkeypatch):
    """One sampled toy tree: p(opponentselectedact)=.25 is already sampled."""
    game = ModuleType('training.game');game.S_PAYOFFS=46
    features = ModuleType('bot.features');features.load_preflop_equity=lambda:None
    def new_hand(n,dealer,rng):
        s=np.zeros(59,dtype=np.int64);s[:6]=10000;s[36]=150;s[37]=100
        s[43]=dealer;s[44]=6;s[52]=100
        for seat,amount in [((dealer+1)%6,50),((dealer+2)%6,100)]:
            s[seat]-=amount;s[6+seat]=amount;s[12+seat]=amount
        return s,np.arange(52),np.arange(12).reshape(6,2)
    def legal(state,seat):return np.array([1,1,0,0,0],bool)
    def step(state,deck,holes,action):
        actor=state[40]
        if actor==0 and action==1:
            state[40]=1
        else:
            value=-100 if actor==0 else 400
            state[45]=1;state[46]=value;state[47]=-value
    def encode(state,holes,deck,seat,mask,equity):
        x=np.zeros(51,np.float32);x[0]=seat;x[46:51]=mask;return x
    game.new_hand=new_hand;game.legal_actions=legal;game.step=step;features.encode_state=encode
    monkeypatch.setitem(sys.modules,'training.game',game)
    monkeypatch.setitem(sys.modules,'bot.features',features)
    monkeypatch.setattr(traversal,'encode_game',lambda s,d,h,p,l,hist,old:np.r_[old,np.zeros(RICH_DIM-51,np.float32)])
    class Rng:
        def integers(self,*args):return 0
        def choice(self,*args,**kwargs):
            assert kwargs['p'][1]==pytest.approx(.25)
            return 1
    class Policy:
        def strategy(self,x,legal):
            return np.array([.4,.6,0,0,0]) if x[0]==0 else np.array([.75,.25,0,0,0])
    class Buffer:
        def __init__(self):self.rows=[]
        def add(self,x,y,iteration):self.rows.append((x.copy(),y.copy(),iteration))
    buffers=[Buffer() for _ in range(6)]
    result=traversal.run_traversals(1,0,Policy(),buffers,7,Rng())
    assert result['mean_return_bb']==pytest.approx(2.)
    assert len(buffers[0].rows)==1 and all(not b.rows for b in buffers[1:])
    # Branch returns−1BB,+4BB and currentσ=(.4,.6): V=2BB.
    assert buffers[0].rows[0][1]==pytest.approx([-3.,2.,0,0,0])
    assert buffers[0].rows[0][2]==7


@pytest.fixture
def existing_teacher(monkeypatch):
    current=ROOT/'artifacts/bounded-continuation/current.json'
    if not current.exists():pytest.skip('Requires verified360 teacher assets')
    import json
    campaign=Path(json.loads(current.read_text())['folder'])/'campaign'
    source=campaign/'source';model=campaign/'seed-342/current_00360.npz'
    monkeypatch.syspath_prepend(str(source));monkeypatch.setenv('BOT_DATA_DIR',str(ROOT/'.cache/fullhouse-bot/data'))
    return model,source


def test_zero_residual_teacher_matches_independent_runtime_experts_and_scales(existing_teacher):
    from fullhouse_runtime import Bridge
    from fullhouse_v4_runtime import FullhouseV4
    from regret_partition import is_opening,base_strata
    teacher,source=existing_teacher
    baseline=FullhouseV4(teacher,source);model=RichRegretModel(teacher,RICH_DIM,source,width=8)
    engine=Bridge();seen=opening=0
    try:
        state=engine.call(dict(op='reset',seed=2039111300,button=0))
        while not state['terminal']:
            old,mask=baseline.features.encode_state_dict(state,baseline.lookup._equity_tables)
            x=np.r_[old,np.zeros(RICH_DIM-51,np.float32)].astype(np.float32)
            expert=baseline.opening_model if bool(is_opening(old)) else baseline.lookup._models[0]
            expected=baseline.lookup._model_strategy(old,mask.astype(np.float32),expert)
            assert np.array_equal(model.strategy(x,mask),expected)
            scale=float(model.teacher['opening_scale']) if bool(is_opening(old)) else float(model.teacher['main_scales'][int(base_strata(old[None])[0])])
            assert np.array_equal(model.base_regrets(old,mask),baseline.lookup._forward_single(old,mask.astype(np.float32),expert).astype(np.float64)*scale)
            opening+=int(bool(is_opening(old)));seen+=1
            state=engine.call(dict(op='step',action=dict(seat=state['seat_to_act'],kind='check' if state['can_check'] else 'call')))
    finally:engine.close()
    assert seen==24 and opening==1


def test_checkpoint_restores_model_reservoir_and_all_four_rng_streams(existing_teacher,tmp_path):
    teacher,source=existing_teacher
    from training.buffer import ReservoirBuffer
    config=dict(iterations=6,buffer_capacity=3,stages=[6],source=str(source))
    random.seed(61);np.random.seed(62);torch.manual_seed(63);rng=np.random.default_rng(64)
    model=RichRegretModel(teacher,RICH_DIM,source,width=8)
    with torch.no_grad():model.residual.layers[-1].bias.fill_(.03)
    model._sync_numpy()
    buffers=[ReservoirBuffer(3,RICH_DIM) for _ in range(6)]
    x=np.zeros(RICH_DIM,np.float32);x[18]=1;x[21]=1;x[27]=.2;x[46:48]=1
    for i in range(5):
        for buffer in buffers:buffer.add(x+0,np.array([-20.,15.,0,0,0],np.float32),1+i%2)
    folder=tmp_path/'seed-342';folder.mkdir()
    snapshot=checkpoint_state(model,buffers,config,342,2,rng)
    commit_checkpoint(folder,snapshot,model,config)
    def next_update(m,bs,g):
        draw=g.integers(0,100000,size=5)
        py=random.random()
        for i,b in enumerate(bs):b.add(x+0,np.array([-10.+i,12.,0,0,0],np.float32),3)
        m.fit(bs,steps=3,batch_size=8,learning_rate=.001)
        return draw,py,checkpoint_state(m,bs,config,342,3,g)
    expected=next_update(model,buffers,rng)
    restored,_=load_current(folder,config,342,RICH_DIM)
    other=RichRegretModel(teacher,RICH_DIM,source,width=8);other.load_state_dict(restored['model'])
    other_buffers=[ReservoirBuffer(3,RICH_DIM) for _ in range(6)]
    for b,state in zip(other_buffers,restored['buffers']):b.load_state_dict(state)
    random.setstate(restored['rng']['python']);np.random.set_state(restored['rng']['numpy'])
    next_rng=np.random.default_rng();next_rng.bit_generator.state=restored['rng']['generator']
    torch.set_rng_state(restored['rng']['torch'])
    actual=next_update(other,other_buffers,next_rng)
    assert np.array_equal(expected[0],actual[0]) and expected[1]==actual[1]
    assert equal_state(expected[2],actual[2])
