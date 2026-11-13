from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/source'
sys.path[:0]=[str(ROOT/'scripts'),str(SOURCE)]
from rich_poker_observation import RICH_DIM, HISTORY_OFFSET
from rich_poker_traversal import run_traversals
from training import game


class MemoryBuffer:
    def __init__(self):self.rows=[]
    def add(self,x,y,iteration,**kw):self.rows.append((x.copy(),y.copy(),iteration,kw))


class CheckingPolicy:
    def strategy(self,x,legal):
        assert x.shape==(RICH_DIM,)
        assert np.array_equal(x[46:51],legal)
        result=np.zeros(5);result[1]=1.
        return result


def test_actual_frozen_engine_external_traversal_collects_raw_bb_regrets():
    buffers=[MemoryBuffer() for _ in range(6)]
    stats=run_traversals(2,3,CheckingPolicy(),buffers,7,np.random.default_rng(41),max_nodes=20000)
    assert stats['terminals']>2 and stats['new_rows_by_seat']['3']>0
    assert all(not b.rows for i,b in enumerate(buffers) if i!=3)
    assert stats['new_rows_by_seat']['3']==len(buffers[3].rows)
    assert any(np.any(np.abs(row[1])>1) for row in buffers[3].rows)
    for x,y,iteration,_ in buffers[3].rows:
        assert x.shape==(RICH_DIM,) and iteration==7 and np.isfinite(y).all()
        assert np.all(y[x[46:51]==0]==0)
        assert x[HISTORY_OFFSET:].any()


def test_one_traverser_decision_matches_independent_five_branch_payoffs(monkeypatch):
    # Controlled one-decision payoff fixture isolates CFR weighting from the
    # real-engine integration exercised in the preceding test.
    state,deck,holes=game.new_hand(6,0,np.random.default_rng(5))
    # Test a controlled terminal child contract separately from real-engine
    # traversal above, isolating the counterfactual weighting algebra.
    legal=np.array([1,1,1,1,1],bool)
    monkeypatch.setattr(game,'new_hand',lambda *args:(state.copy(),deck.copy(),holes.copy()))
    monkeypatch.setattr(game,'legal_actions',lambda *args:legal.copy())
    def terminal(after,deck,holes,action):
        after[45]=1;after[46:52]=0;after[46+3]=(action-2)*100;after[46+4]=-(action-2)*100
    monkeypatch.setattr(game,'step',terminal)
    import bot.features
    monkeypatch.setattr(bot.features,'encode_state',lambda *args:np.r_[np.zeros(46),legal].astype(np.float32))
    class Mix:
        def strategy(self,x,mask):return np.array([.1,.2,.3,.2,.2])
    buffers=[MemoryBuffer() for _ in range(6)]
    stats=run_traversals(1,3,Mix(),buffers,2,np.random.default_rng(5))
    target=np.arange(-2,3)-np.dot(np.array([.1,.2,.3,.2,.2]),np.arange(-2,3))
    assert len(buffers[3].rows)==1
    assert buffers[3].rows[0][1]==pytest.approx(target)


def test_invalid_policy_or_excessive_node_work_stops_without_retry():
    buffers=[MemoryBuffer() for _ in range(6)]
    class Bad:
        def strategy(self,x,legal):return np.full(5,np.nan)
    with pytest.raises(ValueError,match='invalid legal strategy'):
        run_traversals(1,3,Bad(),buffers,1,np.random.default_rng(4))
    with pytest.raises(RuntimeError,match='node budget'):
        run_traversals(1,3,CheckingPolicy(),buffers,1,np.random.default_rng(4),max_nodes=1)
    assert not any(b.rows for b in buffers)
