"""Measure CPU deployment latency/RSS without importing the training stack."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from fullhouse_runtime import Bridge
from fullhouse_v4_runtime import policy_for
from fullhouse_v3_common import atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    policy=policy_for(a.model,a.source);engine=Bridge();states=[]
    try:
        state=engine.call(dict(op='replay',hero=3,button=0,hole=['Ac','Ad'],board=[],actions=[]));states.append(state)
        state=engine.call(dict(op='reset',seed=1771,button=0))
        while state['street']=='preflop' and not state['terminal']:
            state=engine.call(dict(op='step',action=dict(kind='check' if state['can_check'] else 'call',seat=state['seat_to_act'])))
        states.append(state)
        if hasattr(policy,'begin_hand'):
            for seed in range(1000):policy.begin_hand(seed)
        measured=[]
        for state in states:
            for seed in range(20):policy.decide(state,seed)
            times=[]
            for seed in range(500):
                start=time.perf_counter_ns();policy.decide(state,seed);times.append((time.perf_counter_ns()-start)/1e6)
            measured.append(dict(street=state['street'],median_ms=statistics.median(times),p95_ms=sorted(times)[474],calls=len(times)))
        status=Path('/proc/self/status').read_text().splitlines()
        rss=next(int(s.split()[1])*1024 for s in status if s.startswith('VmRSS:'))
        result=dict(model=str(a.model),model_sha256=digest(a.model),model_bytes=a.model.stat().st_size,model_type=policy.model_type,
                    resident_bytes=rss,torch_imported='torch' in sys.modules,measurements=measured,
                    cached_snapshots=len(getattr(policy,'cache',{})),
                    limitations=['Current cloud CPU, single BLAS thread, with concurrent training/evaluation','Warm policy calls; engine replay and model loading excluded'])
        if result['torch_imported']:raise AssertionError('Deployment imported Torch')
        atomic_json(a.output,result);print(json.dumps(result,indent=2))
    finally:engine.close()


if __name__=='__main__':main()
