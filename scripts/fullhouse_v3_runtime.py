"""Explicit V3 inference with the matching feature encoder; legacy stays intact."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from fullhouse_runtime import Fullhouse,Bridge,validate_request,ROOT


class FullhouseV3(Fullhouse):
    def __init__(self,model,source=None):
        super().__init__(model)
        with np.load(self.model) as data:
            if str(data['training_revision'])!='r1-cpu-v3':raise ValueError('V3 checkpoint required')
        source=Path(source or ROOT/'.cache/fullhouse-v3')
        path=source/'bot/features.py'
        spec=importlib.util.spec_from_file_location('fullhouse_v3_features',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.features=module
        self.lookup.get_strategy=self._strategy

    def _strategy(self,state):
        x,mask=self.features.encode_state_dict(state,self.lookup._equity_tables)
        if not np.isfinite(x).all() or not mask.any():raise ValueError('Invalid V3 observation')
        return self.lookup._model_strategy(x,mask.astype(np.float32),self.lookup._models[0])


def policy_for(model,source=None):
    if model is None:return Fullhouse()
    with np.load(model) as data:
        version=str(data['training_revision']) if 'training_revision' in data else None
    if version is None:return Fullhouse(model)
    if version=='r1-cpu-v3':return FullhouseV3(model,source)
    raise ValueError('Unsupported policy revision: '+version)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path);a=p.parse_args()
    policy=FullhouseV3(a.model,a.source);engine=Bridge()
    try:
        for line in sys.stdin:
            try:
                if len(line)>1048576:raise ValueError('Request exceeds 1 MiB')
                r=json.loads(line);validate_request(r)
                result=policy.decide(engine.call(dict(r,op='replay')),r.get('seed',0))
                result['training_revision']='r1-cpu-v3'
            except (ValueError,TypeError,KeyError,FloatingPointError) as exc:result={'ok':False,'error':str(exc)}
            print(json.dumps(result),flush=True)
    finally:engine.close()


if __name__=='__main__':main()
