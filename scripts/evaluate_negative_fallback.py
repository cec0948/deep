"""Exploratory current-policy ablation of Deep CFR's all-negative fallback.

Keeps the original fixed-weight evaluation separate; no deployed default change.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from fullhouse_v3_common import atomic_json,digest


def regret_strategy(scores,legal,fallback):
    positive=np.maximum(scores,0)*legal;total=positive.sum()
    if total>0:return (positive/total).astype(np.float32)
    if fallback=='uniform':return (legal/legal.sum()).astype(np.float32)
    best=legal*(scores==np.max(np.where(legal>0,scores,-np.inf)))
    return (best/best.sum()).astype(np.float32)


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--negative-fallback',choices=['uniform','argmax'],required=True)
    options,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
    output=Path(rest[rest.index('--output')+1]);counts=Counter()
    import evaluate_compact as evaluator
    original=evaluator.policy_for
    def factory(model,source):
        policy=original(model,source)
        if policy.model_type!='compact_advantage':raise ValueError('Probe a current compact policy only')
        def strategy(x,legal,parameters):
            raw=policy.lookup._forward_single(x,legal,parameters);counts['decisions']+=1
            counts['all_negative']+=int(not np.any((raw>0)&(legal>0)))
            return regret_strategy(raw,legal,options.negative_fallback)
        policy.lookup._model_strategy=strategy
        return policy
    evaluator.policy_for=factory;evaluator.main()
    path=output/'results.json';result=json.loads(path.read_text())
    result.update(policy_variant='negative-'+options.negative_fallback,probe_sha256=digest(Path(__file__)),negative_score_counts=dict(counts),
                  probe_limitation='Exploratory post-training strategy conversion; not the original self-play policy or its exact average')
    atomic_json(path,result)


if __name__=='__main__':main()
