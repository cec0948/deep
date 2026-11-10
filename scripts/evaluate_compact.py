"""Common-card six-seat evaluation, with one snapshot fixed for each hand."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from fullhouse_runtime import Bridge,Fullhouse
from fullhouse_v4_runtime import policy_for
from fullhouse_v3_common import ROOT,digest,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--field',choices=['noregrets','public','mixed'],required=True);p.add_argument('--deals',type=int,default=2000)
    p.add_argument('--deal-seed',type=int,default=170261110);p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.deals<2:p.error('At least two deals required')
    a.output.mkdir(parents=True,exist_ok=False)
    policy=policy_for(a.model,a.source);public=Fullhouse() if a.field in ('public','mixed') else None
    registry=json.loads((ROOT/'artifacts/selfplay/current.json').read_text());opponent_model=ROOT/registry['model']
    if digest(opponent_model)!=registry['sha256']:raise ValueError('NoRegrets opponent changed')
    engine=Bridge(opponent_model if a.field in ('noregrets','mixed') else None)
    start=time.monotonic();blocks=[];counts=Counter();peak=0
    try:
        with gzip.open(a.output/'hands.jsonl.gz','wt',compresslevel=1) as stream:
            for deal in range(a.deals):
                chips=[]
                for hero in range(6):
                    state=engine.call(dict(op='reset',seed=a.deal_seed+deal,button=0));steps=0
                    hand_seed=int.from_bytes(hashlib.sha256(f'compact-mixture:{a.deal_seed}:{deal}:{hero}'.encode()).digest()[:8],'big')
                    if hasattr(policy,'begin_hand'):policy.begin_hand(hand_seed)
                    while not state['terminal']:
                        if time.monotonic()-start>a.timeout:raise TimeoutError('Evaluation time bound')
                        seat=state['seat_to_act'];seed=deal*100000+hero*1000+steps
                        if seat==hero:
                            reply=policy.decide(state,seed);counts[f"hero/{state['street']}/{reply['action']['kind']}"]+=1
                        elif a.field=='public' or (a.field=='mixed' and (seat-hero)%6%2==1):
                            reply=public.decide(state,seed)
                        else:
                            reply=engine.call(dict(op='blueprint',seed=seed))
                            counts[f"opponent/{state['street']}/fallback"]+=int(reply.get('fallback',False))
                            counts[f"opponent/{state['street']}/decisions"]+=1
                        state=engine.call(dict(op='step',action=dict(reply['action'],seat=seat)));steps+=1
                        if steps>512:raise RuntimeError('Nonterminating hand')
                    if sum(state['utilities'])!=0:raise AssertionError('Chip conservation')
                    row=dict(deal=deal,hero=hero,seed=a.deal_seed+deal,chips=state['utilities'][hero],utilities=state['utilities'],
                             steps=steps,hand_seed=hand_seed,policy_iteration=getattr(policy,'active_iteration',None))
                    chips.append(row['chips']);stream.write(json.dumps(row)+'\n')
                blocks.append(statistics.mean(chips))
                if (deal+1)%100==0:
                    rss=0
                    for pid in (os.getpid(),engine.p.pid):
                        lines=Path(f'/proc/{pid}/status').read_text().splitlines()
                        rss+=next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                    if rss>8*1024**3:raise MemoryError('Evaluation memory bound')
                    atomic_json(a.output/'progress.json',dict(completed_deals=deal+1,planned_deals=a.deals,elapsed_seconds=time.monotonic()-start))
    finally:engine.close()
    opponents=dict(noregrets=registry['sha256'],public=public.digest if public else None)
    result=dict(complete=True,field=a.field,deals=a.deals,hands=a.deals*6,deal_seed=a.deal_seed,candidate=str(policy.model),
                candidate_sha256=policy.digest,candidate_type=policy.model_type,opponents=opponents,
                bb_per_100=statistics.mean(blocks),ci95_half_width=1.96*statistics.stdev(blocks)/len(blocks)**.5,deal_blocks=blocks,
                behavior=dict(counts),elapsed_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak,
                evaluator_sha256=digest(Path(__file__)),runtime_sha256=digest(ROOT/'scripts/fullhouse_v4_runtime.py'),
                engine_sha256=digest(ROOT/'.cache/noregrets/target/release/poker-bridge'),
                source_features_sha256=digest(a.source/'bot/features.py') if a.source else None,
                limitations=['Nominal 95% deal-block interval','Fixed two-model opponent pool; mixed field contains three public and two NoRegrets seats',
                             'Mixture policy selected once per hand, independent of cards','No professional-strength claim or automatic promotion'])
    atomic_json(a.output/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('deal_blocks','behavior')}),flush=True)


if __name__=='__main__':main()
