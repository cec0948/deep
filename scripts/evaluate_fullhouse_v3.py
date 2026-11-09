"""Common-deal six-seat evaluation against two explicit opponent fields."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import statistics
import time

from fullhouse_runtime import Bridge,Fullhouse
from fullhouse_v3_runtime import policy_for
from fullhouse_v3_common import ROOT,digest,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--field',choices=['noregrets','public'],required=True);p.add_argument('--deals',type=int,default=2000)
    p.add_argument('--deal-seed',type=int,default=160261109);p.add_argument('--timeout',type=float,default=1200)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.deals<2 or a.timeout<=0:p.error('At least two deals and positive timeout required')
    a.output.mkdir(parents=True,exist_ok=False)
    policy=policy_for(a.model,a.source);public=Fullhouse() if a.field=='public' else None
    registry=json.loads((ROOT/'artifacts/selfplay/current.json').read_text())
    opponent_model=ROOT/registry['model']
    opponent_hash=public.digest if public else digest(opponent_model)
    if not public and opponent_hash!=registry['sha256']:raise ValueError('Opponent hash changed')
    engine=Bridge(None if public else opponent_model)
    start=time.monotonic();blocks=[];counts=Counter();peak=0
    try:
        with gzip.open(a.output/'hands.jsonl.gz','wt',compresslevel=1) as stream:
            for deal in range(a.deals):
                chips=[]
                for hero in range(6):
                    state=engine.call({'op':'reset','seed':a.deal_seed+deal,'button':0});steps=0
                    while not state['terminal']:
                        if time.monotonic()-start>a.timeout:raise TimeoutError('Evaluation limit')
                        seat=state['seat_to_act'];seed=deal*100000+hero*1000+steps
                        if seat==hero:
                            reply=policy.decide(state,seed)
                            counts[f"hero/{state['street']}/{reply['action']['kind']}"]+=1
                        elif public:reply=public.decide(state,seed)
                        else:
                            reply=engine.call({'op':'blueprint','seed':seed})
                            counts[f"opponent/{state['street']}/decisions"]+=1
                            counts[f"opponent/{state['street']}/fallback"]+=int(reply.get('fallback',False))
                        state=engine.call({'op':'step','action':dict(reply['action'],seat=seat)});steps+=1
                        if steps>512:raise RuntimeError('Nonterminating hand')
                    assert sum(state['utilities'])==0
                    row=dict(deal=deal,hero=hero,seed=a.deal_seed+deal,chips=state['utilities'][hero],utilities=state['utilities'],steps=steps)
                    chips.append(row['chips']);stream.write(json.dumps(row)+'\n')
                blocks.append(statistics.mean(chips))
                if (deal+1)%100==0:
                    rss=0
                    for pid in (os.getpid(),engine.p.pid):
                        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
                        rss+=next((int(v.split()[1])*1024 for v in lines if v.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                    if rss>8*1024**3:raise MemoryError('Evaluation RSS limit')
                    atomic_json(a.output/'progress.json',dict(completed_deals=deal+1,planned_deals=a.deals,elapsed_seconds=time.monotonic()-start))
    finally:engine.close()
    result=dict(complete=True,field=a.field,deals=a.deals,hands=a.deals*6,deal_seed=a.deal_seed,
                candidate=str(policy.model),candidate_sha256=policy.digest,candidate_type=policy.model_type,
                opponent_sha256=opponent_hash,bb_per_100=statistics.mean(blocks),
                ci95_half_width=1.96*statistics.stdev(blocks)/len(blocks)**.5,deal_blocks=blocks,
                behavior=dict(counts),elapsed_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,
                engine_sha256=digest(ROOT/'.cache/noregrets/target/release/poker-bridge'),
                evaluator_sha256=digest(Path(__file__)),v3_runtime_sha256=digest(ROOT/'scripts/fullhouse_v3_runtime.py'),
                legacy_runtime_sha256=digest(ROOT/'scripts/fullhouse_runtime.py'),
                source_features_sha256=digest(a.source/'bot/features.py') if a.source else None,
                limitations=['Nominal deal-block interval','Fixed opponent field','No automatic promotion'])
    atomic_json(a.output/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('deal_blocks','behavior')}),flush=True)


if __name__=='__main__':main()
