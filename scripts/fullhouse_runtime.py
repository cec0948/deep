"""Shared public-observation policy and trusted engine client."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import select

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.cache/fullhouse-bot'
BRIDGE=ROOT/'.cache/noregrets/target/release/poker-bridge'

def validate_source():
    cfg=json.loads((ROOT/'configs/fullhouse-model.json').read_text())
    head=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip()
    if head!=cfg['commit'] or dirty:raise ValueError('Fullhouse source changed')
    for name,digest in cfg['files'].items():
        if hashlib.sha256((SOURCE/'data'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Model/table integrity mismatch: '+name)
    return cfg

class Bridge:
    def __init__(self,model=None):
        self.p=subprocess.Popen([str(BRIDGE),*([str(model)] if model else [])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    def call(self,request):
        self.p.stdin.write(json.dumps(request)+'\n');self.p.stdin.flush()
        if not select.select([self.p.stdout],[],[],30)[0]:
            self.p.kill();self.p.wait();raise RuntimeError('Engine response timeout')
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('Engine stopped')
        result=json.loads(line)
        if not result['ok']:raise ValueError(result['error'])
        return result['state']
    def close(self):
        self.p.stdin.close()
        try:self.p.wait(timeout=3)
        except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
        self.p.stdout.close()

class Fullhouse:
    def __init__(self,model=None):
        self.cfg=validate_source()
        os.environ['BOT_DATA_DIR']=str(SOURCE/'data')
        sys.path.insert(0,str(SOURCE))
        import numpy as np
        from bot.deep_cfr_lookup import DeepCFRLookup
        self.np=np
        self.model=Path(model or SOURCE/'data/deep_cfr_model.npz').resolve()
        self.digest=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.lookup=DeepCFRLookup(self.model)
        self.model_type=self.lookup._models[0]['_model_type']
    def decide(self,state,seed):
        np=self.np
        probs=self.lookup.get_strategy(state).astype(float)
        if probs.shape!=(5,) or not np.isfinite(probs).all() or (probs<0).any() or not np.isclose(probs.sum(),1):raise ValueError('Invalid policy probabilities')
        bounds=state['raise_bounds'];call={'kind':'check' if state['can_check'] else 'call'}
        actions=[{'kind':'fold'},call,call,call,call]
        if state['can_check']:actions[0]=call
        if bounds:
            lo,hi=bounds
            for i,f in [(2,.5),(3,1.)]:
                to=max(lo,min(hi,state['your_bet_this_street']+state['amount_owed']+round(f*(state['pot']+state['amount_owed']))))
                actions[i]={'kind':'raise','to':to}
            actions[4]={'kind':'raise','to':hi}
        # Mask incompatible actions rather than silently changing a learned bet.
        if not bounds:probs[2:]=0
        if state['can_check']:probs[0]=0
        if probs.sum()<=0:raise ValueError('No probability on legal engine actions')
        probs/=probs.sum()
        merged={}
        for a,p in zip(actions,probs):
            if p>0:
                key=json.dumps(a,sort_keys=True);merged[key]=merged.get(key,0)+float(p)
        dist=[{'action':json.loads(k),'probability':p} for k,p in merged.items()]
        action=random.Random(seed).choices([d['action'] for d in dist],[d['probability'] for d in dist])[0]
        return dict(ok=True,action=action,distribution=dist,policy_source='neural_policy',backend='fullhouse',model_type=self.model_type,
                    model_sha256=self.digest,model_iterations=None,strength_verified=False,players=6,starting_stack_bb=100,
                    pot=state['pot'],to_call=state['amount_owed'],raise_bounds=bounds,street=state['street'].capitalize())

def validate_request(r):
    if not isinstance(r,dict) or set(r)-{'hero','button','hole','board','actions','seed'}:raise ValueError('Unknown request fields')
    if any(type(r.get(k)) is not int or not 0<=r[k]<6 for k in ('hero','button')):raise ValueError('Invalid hero/button')
    if type(r.get('seed',0)) is not int or not 0<=r.get('seed',0)<2**64:raise ValueError('Invalid seed')
    if not isinstance(r.get('hole'),list) or len(r['hole'])!=2:raise ValueError('Two hole cards required')
    if not isinstance(r.get('board',[]),list) or len(r.get('board',[])) not in (0,3,4,5):raise ValueError('Invalid board')
    actions=r.get('actions',[])
    if not isinstance(actions,list) or len(actions)>512:raise ValueError('Invalid actions')
    for a in actions:
        if not isinstance(a,dict) or set(a)-{'seat','kind','to'}:raise ValueError('Invalid action fields')
        if type(a.get('seat')) is not int or not 0<=a['seat']<6:raise ValueError('Invalid action seat')
        if a.get('kind')=='raise':
            if type(a.get('to')) is not int or not 0<=a['to']<2**32:raise ValueError('Invalid raise')
        elif a.get('kind') not in ('check','call','fold') or 'to' in a:raise ValueError('Invalid action')

def main():
    policy=Fullhouse();bridge=Bridge()
    try:
        for line in sys.stdin:
            try:
                if len(line)>1048576:raise ValueError('Request exceeds 1 MiB')
                r=json.loads(line);validate_request(r)
                state=bridge.call(dict(r,op='replay'))
                result=policy.decide(state,r.get('seed',0))
            except (ValueError,KeyError,TypeError,FloatingPointError) as e:result={'ok':False,'error':str(e)}
            print(json.dumps(result),flush=True)
    finally:bridge.close()

if __name__=='__main__':main()
