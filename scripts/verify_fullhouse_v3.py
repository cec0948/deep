"""Independent rules, conditional-average, and exact-resume gates for V3."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.cache/fullhouse-v3'
sys.path.insert(0,str(SOURCE))
from training import game as g,r1_support as r1
from training.buffer import ReservoirBuffer
from bot.features import encode_state,encode_state_dict,load_preflop_equity


def card(c):
    return '23456789TJQKA'[int(c)%13]+'cdhs'[int(c)//13]


class Oracle:
    def __init__(self):
        self.p=subprocess.Popen([str(ROOT/'.cache/noregrets/target/release/poker-rules')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def call(self,r):
        self.p.stdin.write(json.dumps(r)+'\n');self.p.stdin.flush()
        if not select.select([self.p.stdout],[],[],30)[0]:raise TimeoutError('Rules oracle')
        value=json.loads(self.p.stdout.readline())
        if not value['ok']:raise ValueError(value['error'])
        return value['state']
    def close(self):
        self.p.stdin.close()
        try:self.p.wait(timeout=3)
        except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
        self.p.stdout.close()


def action(s,a):
    seat=int(s[g.S_TOACT]);owed=max(0,int(s[g.S_CURBET]-s[g.S_BETS+seat]))
    if a==0:return dict(seat=seat,kind='fold')
    if a==1:return dict(seat=seat,kind='call' if owed else 'check')
    chips=g._bet_chip_amount(a,int(s[g.S_POT]),owed,int(s[g.S_STACKS+seat]))
    return dict(seat=seat,kind='raise',to=int(s[g.S_BETS+seat]+chips))


def compare(s,deck,hole,observation,eq,markers=True):
    assert bool(s[g.S_TERMINAL])==observation['terminal'],(s.tolist(),observation)
    if observation['terminal']:
        np.testing.assert_array_equal(s[g.S_PAYOFFS:g.S_PAYOFFS+6],observation['utilities'])
        assert sum(observation['utilities'])==0
        return
    seat=int(s[g.S_TOACT]);assert seat==observation['seat_to_act']
    assert ['preflop','flop','turn','river'][s[g.S_STREET]]==observation['street']
    assert s[g.S_POT]==observation['pot'] and s[g.S_CURBET]==observation['current_bet']
    for i,p in enumerate(observation['players']):
        for offset,key in [(g.S_STACKS,'stack'),(g.S_BETS,'bet_this_street'),(g.S_INVESTED,'hand_commit'),(g.S_FOLDED,'is_folded'),(g.S_ALLIN,'is_all_in')]:
            assert s[offset+i]==p[key],(offset,i,s.tolist(),observation)
    legal=g.legal_actions(s,seat)
    x=encode_state(s,hole,deck,seat,legal,eq)
    y,mask=encode_state_dict(observation,eq)
    np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-6,err_msg=json.dumps(observation))
    np.testing.assert_array_equal(legal,mask)
    if markers:
        plain=copy.deepcopy(observation);plain['action_log']=[e for e in plain['action_log'] if e['action']!='deal']
        np.testing.assert_allclose(x,encode_state_dict(plain,eq)[0],rtol=1e-6,atol=1e-6,err_msg=json.dumps(plain))


def rules(hands):
    rng=np.random.default_rng(61971);eq=load_preflop_equity();oracle=Oracle();states=0;fixtures=0
    try:
        folder=ROOT/'reports/fullhouse-regression/20261109'
        examples=[]
        for name in ('parity-120.json','parity-300.json','parity-600.json','parity-pending-repair-600.json','parity.json'):
            examples += json.loads((folder/name).read_text())['examples']
        for ex in examples:
            request=ex['request'];s,deck,hole=g.new_hand(6,request['button'],rng)
            deck[:]=ex['training_deck'];hole[:]=deck[:12].reshape(6,2)
            obs=oracle.call(dict(op='reset',button=request['button'],deck=[card(c) for c in deck]))
            for concrete in request['actions']:
                compare(s,deck,hole,obs,eq);states+=1
                legal=g.legal_actions(s,int(s[g.S_TOACT]))
                candidates=[int(a) for a in np.flatnonzero(legal) if action(s,int(a))==concrete]
                assert candidates,('Counterexample history became illegal',concrete,s.tolist())
                obs=oracle.call(dict(op='step',action=concrete));g.step(s,deck,hole,candidates[0])
            compare(s,deck,hole,obs,eq);states+=1;fixtures+=1
        for hand in range(hands):
            s,deck,hole=g.new_hand(6,int(rng.integers(6)),rng)
            obs=oracle.call(dict(op='reset',button=int(s[g.S_DEALER]),deck=[card(c) for c in deck]));steps=0
            while True:
                compare(s,deck,hole,obs,eq);states+=1
                if s[g.S_TERMINAL]:break
                legal=g.legal_actions(s,int(s[g.S_TOACT]))
                p=legal.astype(float)/legal.sum()*.5;p[1]+=.5
                a=int(rng.choice(5,p=p));concrete=action(s,a)
                obs=oracle.call(dict(op='step',action=concrete));g.step(s,deck,hole,a)
                steps+=1;assert steps<512
        # A board royal flush splits a synthetic 305-chip layered pot three ways.
        s,deck,hole=g.new_hand(6,0,rng);deck[12:17]=[8,9,10,11,12]
        hole[:]=np.array([[13,14],[15,16],[17,18],[19,20],[21,22],[23,24]],dtype=np.int8)
        invested=np.array([101,101,101,1,1,0]);s[g.S_INVESTED:g.S_INVESTED+6]=invested
        s[g.S_STACKS:g.S_STACKS+6]=10000-invested;s[g.S_FOLDED:g.S_FOLDED+6]=[0,0,0,1,1,1]
        s[g.S_POT]=305;g._resolve_showdown(s,deck,hole)
        np.testing.assert_array_equal(s[g.S_PAYOFFS:g.S_PAYOFFS+6],[1,1,0,-1,-1,0])
        with_unsupported=False
        try:g.new_hand(6,0,rng,np.array([5000,10000,10000,10000,10000,10000]))
        except ValueError:with_unsupported=True
        assert with_unsupported
    finally:oracle.close()
    return dict(hands=hands,compared_states=states,historical_counterexamples=fixtures,terminal_payoffs_checked=True,
                markers_and_inferred_streets_checked=True,three_way_odd_chips_pass=True,contract='six equal 100BB starting stacks')


def sampling():
    r1.WEIGHT_HORIZON=10
    b=ReservoirBuffer(8,51,store_sample_weights=True)
    inputs=[]
    for street,position,pot in [(0,3,.015),(0,3,.015),(1,1,1.),(1,1,1.),(2,2,2.),(2,2,2.)]:
        x=np.zeros(51,np.float32);x[17+street]=1;x[21+position]=1;x[27]=pot;x[46:48]=1
        inputs.append(x)
    for i,(x,w,t) in enumerate(zip(inputs,[1,9,1e7,3e7,0,0],[1,2,3,4,1,2])):
        y=np.zeros(5,np.float32);y[i%2]=1;b.add(x,y,t,sample_weight=w)
    sampler=r1.StrategySampler(b);p=sampler.probabilities
    assert np.isfinite(p).all() and np.isclose(p.sum(),1) and not p[4:].any()
    expected=np.array([1.,18.]);np.testing.assert_allclose(p[:2]/p[:2].sum(),expected/expected.sum())
    expected=np.array([3.,12.]);np.testing.assert_allclose(p[2:4]/p[2:4].sum(),expected/expected.sum())
    np.testing.assert_allclose([p[:2].sum(),p[2:4].sum()],[.5,.5])
    np.random.seed(61);rng=np.random.default_rng(61);state=r1.pack_rng(rng)
    one=sampler.sample(128);r1.restore_rng(state,rng);two=sampler.sample(128)
    for a,c in zip(one,two):np.testing.assert_array_equal(a,c)
    assert np.all(one[2]==1)
    return dict(conditional_average_oracle=True,strata_mass=[.5,.5],zero_weight_support=True,exact_rng_replay=True)


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def resume():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='fullhouse-v3-',dir=ROOT/'artifacts') as directory:
        root=Path(directory)
        base=[sys.executable,'-m','training.train','--iterations','12','--traversals','8','--train-steps','8',
              '--strategy-train-steps','8','--strategy-batch-size','64','--batch-size','64','--buffer-capacity','3000',
              '--strategy-buffer-capacity','3000','--workers','1','--device','cpu','--seed','42','--checkpoint-every','6',
              '--save-resume-buffers','--no-stack-randomization','--opponent-exploration-start','0','--opponent-exploration-end','0']
        def run(name,extra=(),success=True):
            result=subprocess.run(base+['--checkpoint-dir',str(root/name)]+list(extra),cwd=SOURCE,env=env,capture_output=True,text=True,timeout=120)
            if success:assert result.returncode==0,result.stderr[-4000:]
            else:assert result.returncode and 'config/revision mismatch' in result.stderr,result.stderr
        run('continuous');run('split',['--stop-after','6']);run('split',['--resume']);run('repeat')
        reference=torch.load(root/'continuous/latest.pt',weights_only=False,map_location='cpu')
        for name in ('split','repeat'):equal(reference,torch.load(root/name/'latest.pt',weights_only=False,map_location='cpu'))
        smoke=ROOT/'artifacts/fullhouse-v3-verification';smoke.mkdir(parents=True,exist_ok=True)
        (smoke/'smoke-iter12.npz').write_bytes((root/'continuous/iter_00012.npz').read_bytes())
        (smoke/'smoke-iter12.pt').write_bytes((root/'continuous/iter_00012.pt').read_bytes())
        (root/'legacy').mkdir();torch.save({'r1_config':{'revision':'r1-cpu-v2'}},root/'legacy/latest.pt')
        run('legacy',['--resume'],False)
    return dict(continuous_equals_resumed=True,continuous_equals_repeat=True,includes=['models','reservoirs','RNG','config'],legacy_resume_rejected=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--hands',type=int,default=1000);p.add_argument('--skip-resume',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=dict(revision=r1.REVISION,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,call in [('sampling',sampling),('rules',lambda:rules(args.hands)),('resume',resume)]:
        if name=='resume' and args.skip_resume:continue
        result[name]=call();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result[name]),flush=True)
    result['passed']=True;args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
