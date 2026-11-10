"""Compact expert deployment: NumPy/eval7 only, no Torch or live search."""
import argparse
import importlib.util
import json
import random
from pathlib import Path
import sys

import numpy as np
from fullhouse_runtime import Fullhouse,Bridge,validate_request,ROOT
from regret_partition import is_opening


class FullhouseV4(Fullhouse):
    def __init__(self,model,source=None):
        super().__init__(model)
        m=self.lookup._models[0]
        if str(m.get('training_revision'))!='r1-cpu-v4':raise ValueError('V4 model required')
        self.opening_model=None
        if m['_model_type']=='advantage':
            self.opening_model={k[len('opening_'):]:v for k,v in m.items() if k.startswith('opening_')}
            if 'trunk_w0' not in self.opening_model:raise ValueError('Missing opening expert')
            self.opening_model['_model_type']='advantage';self.model_type='compact_advantage'
        source=Path(source or ROOT/'.cache/fullhouse-v3')
        spec=importlib.util.spec_from_file_location('fullhouse_v4_features',source/'bot/features.py')
        self.features=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.features)
        self.lookup.get_strategy=self._strategy

    def _strategy(self,state):
        x,mask=self.features.encode_state_dict(state,self.lookup._equity_tables)
        if not np.isfinite(x).all() or not mask.any():raise ValueError('Invalid V4 observation')
        model=self.opening_model if self.opening_model is not None and bool(is_opening(x)) else self.lookup._models[0]
        return self.lookup._model_strategy(x,mask.astype(np.float32),model)


class SnapshotMixture:
    """Choose a complete past policy once per hand, independent of its cards."""
    def __init__(self,manifest,source=None):
        import hashlib
        self.model=Path(manifest).resolve();self.digest=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.spec=json.loads(self.model.read_text());self.source=source;self.model_type='snapshot_mixture'
        if self.spec.get('model_type')!='compact_snapshot_mixture':raise ValueError('Invalid snapshot mixture')
        self.models=self.spec['models'];self.cache={};self.active=None;self.active_iteration=None
        if [m['iteration'] for m in self.models]!=list(range(1,len(self.models)+1)):
            raise ValueError('Mixture requires every iteration, including the initial policy')
        if not self.models or any(m['weight']!=m['iteration'] for m in self.models):raise ValueError('Invalid time weights')

    def begin_hand(self,hand_seed):
        import hashlib
        if type(hand_seed) is not int or not 0<=hand_seed<2**64:raise ValueError('Invalid independent hand seed')
        draw=random.Random(hand_seed).randrange(sum(m['weight'] for m in self.models));selected=None
        for row in self.models:
            draw-=row['weight']
            if draw<0:selected=row;break
        index=selected['iteration'];path=(self.model.parent/selected['file']).resolve()
        if path.parent!=self.model.parent:raise ValueError('Snapshot must be inside its manifest directory')
        if index not in self.cache:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=selected['sha256']:raise ValueError('Snapshot changed')
            self.cache[index]=FullhouseV4(path,self.source)
        self.active=self.cache[index];self.active_iteration=index

    def decide(self,state,seed):
        if self.active is None:raise ValueError('An independent hand_seed is required for a policy mixture')
        reply=self.active.decide(state,seed)
        reply.update(policy_source='snapshot_mixture',mixture_iteration=self.active_iteration,mixture_sha256=self.digest)
        return reply


def policy_for(model,source=None):
    if model is not None and Path(model).suffix=='.json':return SnapshotMixture(model,source)
    if model is not None:
        with np.load(model) as data:
            if 'training_revision' in data and str(data['training_revision'])=='r1-cpu-v4':return FullhouseV4(model,source)
    from fullhouse_v3_runtime import policy_for as previous
    return previous(model,source)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path);a=p.parse_args()
    policy=policy_for(a.model,a.source);engine=Bridge()
    try:
        for line in sys.stdin:
            try:
                if len(line)>1048576:raise ValueError('Request exceeds 1 MiB')
                r=json.loads(line)
                if isinstance(policy,SnapshotMixture):policy.begin_hand(r.pop('hand_seed',None))
                validate_request(r)
                reply=policy.decide(engine.call(dict(r,op='replay')),r.get('seed',0))
                reply['training_revision']='r1-cpu-v4'
            except (ValueError,TypeError,KeyError,FloatingPointError) as exc:reply=dict(ok=False,error=str(exc))
            print(json.dumps(reply),flush=True)
    finally:engine.close()


if __name__=='__main__':main()
