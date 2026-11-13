"""Causal randomization, panel eligibility and clustered paired reporting."""
import copy
import io
from pathlib import Path
import sys
import time

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_river_strength_pilot import (ARM_NAMES,clustered_interval,eligible,percentile,
    semantic_seed,summarize,validate_reply)
import run_river_strength_pilot as pilot


def state():
    return dict(terminal=False,street='river',seat_to_act=1,can_check=True,raise_bounds=[100,10000],
        your_cards=['Ac','Ad'],community_cards=['2s','7d','Th','3c','4c'],action_log=[
        dict(seat=1,action='small_blind',amount=50),dict(seat=2,action='big_blind',amount=100)],
        players=[dict(seat=i,stack=1000,is_folded=i not in [1,2],is_all_in=False) for i in range(6)])


def test_semantic_seed_shared_across_arms_and_ignores_hidden_cards():
    a=state();b=copy.deepcopy(a);b['your_cards']=['Kh','Qh'];b['community_cards']=['2c','7h','Td','3s','4s']
    assert semantic_seed(2032111300,1,a)==semantic_seed(2032111300,1,b)
    c=copy.deepcopy(a);c['action_log'].append(dict(seat=1,action='check',amount=0))
    assert semantic_seed(2032111300,1,c)!=semantic_seed(2032111300,1,a)
    assert semantic_seed(2032111301,1,a)!=semantic_seed(2032111300,1,a)
    assert semantic_seed(2032111300,2,a)!=semantic_seed(2032111300,1,a)


def test_only_hero_two_active_live_stack_river_is_eligible():
    s=state();assert eligible(s,1)
    for mutate in [lambda s:s.update(street='turn'),lambda s:s.update(seat_to_act=2),
                   lambda s:s.update(terminal=True),lambda s:s['players'][0].update(is_folded=False),
                   lambda s:s['players'][2].update(is_all_in=True),lambda s:s['players'][2].update(stack=0)]:
        v=copy.deepcopy(s);mutate(v);assert not eligible(v,1)


def reply(reason='completed',used=True):
    return dict(ok=True,action={'kind':'check'},distribution=[{'action':{'kind':'check'},'probability':1.}],
                river_search=dict(reason=reason,attempted=True,used=used))


def test_supported_deadline_fallback_is_recordable_but_unexpected_errors_stop():
    assert validate_reply(state(),reply('hard_deadline',False))['ok']
    assert validate_reply(state(),reply('off_tree_history_not_modeled',False))['ok']
    for reason in ['AssertionError: mismatch','TypeError: bad worker','zero_or_nonfinite_range_mass']:
        with pytest.raises(RuntimeError):validate_reply(state(),reply(reason,False))
    invalid=reply();invalid['distribution'][0]['probability']=float('nan')
    with pytest.raises(ValueError):validate_reply(state(),invalid)
    illegal=reply();illegal['action']={'kind':'raise','to':50000}
    with pytest.raises(ValueError):validate_reply(state(),illegal)


def records_for(cases):
    rows=[]
    for i,c in enumerate(cases):
        for arm in ARM_NAMES:
            change=0 if arm=='baseline' else (i+1)*100
            diagnostic=dict(measured_total_seconds=float(i+1),attempted=True,used=arm!='baseline',
                            reason='completed',solver_iterations=2)
            rows.append(dict(case=c['id'],arm=arm,chips=100+change,diagnostics=[diagnostic]))
    return rows


def test_pairing_clusters_models_and_seats_within_same_dealblock():
    cases=[dict(id='a',model_seed=342,deal_block=0),dict(id='b',model_seed=343,deal_block=0),
           dict(id='c',model_seed=344,deal_block=1)]
    result=summarize(cases,records_for(cases));d=result['comparisons']['search-5s-minus-baseline']
    assert result['selected_cases']==3 and result['independent_deal_blocks']==2
    assert d['mean_bb']==pytest.approx((1.5+3)/2)
    assert d['case_weighted_mean_bb']==2
    assert d['independent_deal_blocks']==2
    assert d['lower95_bb']<0<d['upper95_bb']
    assert result['full_game_bb_per_100_reported'] is False
    assert result['full_game_strength_proven'] is False
    assert result['arms']['search-5s']['cfr_iterations']==6


def test_incomplete_or_repeated_arms_never_report_complete():
    cases=[dict(id='a',model_seed=342,deal_block=0)];records=records_for(cases)
    with pytest.raises(ValueError):summarize(cases,records[:-1])
    with pytest.raises(ValueError):summarize(cases,records+[records[0]])


def test_small_sample_intervals_and_latency_quantiles_are_explicit():
    assert clustered_interval({0:[1.,3.]})==dict(mean_bb=2.,lower95_bb=None,upper95_bb=None,independent_deal_blocks=1)
    assert percentile([.1,.2,.3,.4],.5)==pytest.approx(.25)
    assert percentile([.1,.2,.3,.4],.95)==pytest.approx(.385)
    assert percentile([], .95) is None


class FakePublic:
    def __init__(self,raw):self.raw=copy.deepcopy(raw)
    @classmethod
    def from_bridge(cls,raw):return cls(raw)
    def to_observation(self):return copy.deepcopy(self.raw)
    def public_only(self):return self
    def to_json(self):
        value=self.to_observation();value.pop('your_cards',None);return value


class FakeEngine:
    def call(self,request):
        if request['op']=='reset':return state()
        if request['op']=='step':return dict(terminal=True,utilities=[0,100,-100,0,0,0])
        raise AssertionError('Unexpected engine call')


class FakePolicy:
    def __init__(self):self.seeds=[];self.hand_seeds=[]
    def decide(self,observation,seed):self.seeds.append(seed);return reply()
    def begin_hand(self,seed):self.hand_seeds.append(seed)


def test_selection_is_first_four_per_model_one_per_block_and_stops_fixed_scan(monkeypatch):
    monkeypatch.setattr(pilot,'PublicState',FakePublic)
    config=dict(max_deal_blocks=128,deal_seed_start=2032111300,button=0,model_seeds=[342,343,344],
                cases_per_model=4,hero_seats=list(range(6)))
    models={s:FakePolicy() for s in config['model_seeds']};saved=[];progress=[]
    cases,coverage=pilot.select_cases(config,FakeEngine(),models,FakePolicy(),time.monotonic()+5,
        lambda phase,**fields:progress.append(fields),on_case=lambda rows:saved.append(copy.deepcopy(rows)))
    assert len(cases)==12 and coverage['scanned_deal_blocks']==4
    assert len({(c['model_seed'],c['deal_block']) for c in cases})==12
    assert all(c['hero']==1 for c in cases)
    assert [c['deal_block'] for c in cases if c['model_seed']==342]==[0,1,2,3]
    assert [len(rows) for rows in saved]==list(range(1,13))
    assert any(p.get('selected_per_model')=={342:4,343:4,344:4} for p in progress)
    short=dict(config,max_deal_blocks=3);partial=[]
    with pytest.raises(ValueError,match='Insufficient'):
        pilot.select_cases(short,FakeEngine(),models,FakePolicy(),time.monotonic()+5,
            lambda *a,**kw:None,on_case=lambda rows:partial.append(copy.deepcopy(rows)))
    assert len(partial[-1])==9


def test_arms_restart_identical_prefix_clear_cache_and_share_semantic_rng(monkeypatch):
    monkeypatch.setattr(pilot,'PublicState',FakePublic)
    case=dict(id='example',model_seed=342,deal_block=0,deal_seed=2032111300,hero=1,button=0,
              prefix=[],root_observation=state(),hand_seed=123)
    base=FakePolicy();search=FakePolicy();opponent=FakePolicy()
    a=pilot.replay_case(case,FakeEngine(),base,opponent,'baseline',time.monotonic()+5,io.StringIO(),lambda *a,**kw:None)
    b=pilot.replay_case(case,FakeEngine(),search,opponent,'search-5s',time.monotonic()+5,io.StringIO(),lambda *a,**kw:None)
    assert a['chips']==b['chips']==100
    assert base.seeds==search.seeds and base.hand_seeds==search.hand_seeds==[123]
    bad=dict(case,root_observation=dict(state(),street='turn'))
    with pytest.raises(ValueError,match='did not reproduce'):
        pilot.replay_case(bad,FakeEngine(),search,opponent,'search-5s',time.monotonic()+5,io.StringIO(),lambda *a,**kw:None)
