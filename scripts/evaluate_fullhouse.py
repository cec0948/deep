"""Paired-deal six-seat pilot in one trusted rules engine; not a strength claim."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import time
from fullhouse_runtime import ROOT, Bridge, Fullhouse

def main():
    p=argparse.ArgumentParser();p.add_argument('--deals',type=int,default=100);p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--model',type=Path,help='Explicit candidate npz; default is the public advantage model')
    p.add_argument('--deal-seed',type=int,default=20261106)
    p.add_argument('--arms',nargs='+',choices=['fullhouse','noregrets','same-noregrets'],default=['fullhouse','noregrets','same-noregrets'])
    args=p.parse_args()
    if args.deals<2 or args.timeout<=0:p.error('deals >=2 and positive timeout required')
    entry=json.loads((ROOT/'artifacts/selfplay/current.json').read_text())
    import hashlib
    model=ROOT/entry['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('NoRegrets model hash mismatch')
    policy=Fullhouse(args.model);engine=Bridge(model)
    out=ROOT/'artifacts/fullhouse-evaluation'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    import os
    start=time.monotonic();rows=[];latency={'fullhouse':[],'noregrets':[]};fall={};complete=True;peak_rss=0
    try:
        with (out/'hands.jsonl').open('w') as stream:
            for arm in args.arms:
                for deal in range(args.deals):
                    for hero in range(6):
                        if time.monotonic()-start>args.timeout:complete=False;break
                        state=engine.call({'op':'reset','seed':args.deal_seed+deal,'button':0});steps=0
                        while not state['terminal']:
                            if time.monotonic()-start>args.timeout:raise TimeoutError('evaluation time limit')
                            rss=0
                            for pid in (os.getpid(),engine.p.pid):
                                status=Path(f'/proc/{pid}/status').read_text()
                                rss+=next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0)
                            peak_rss=max(peak_rss,rss)
                            if rss>20*1024**3:raise TimeoutError('evaluation RSS limit')
                            seat=state['seat_to_act']
                            backend=('fullhouse' if seat==hero else 'noregrets') if arm=='fullhouse' else ('noregrets' if seat==hero or arm=='same-noregrets' else 'fullhouse')
                            rng_seed=deal*100000+hero*1000+steps
                            before=time.monotonic()
                            if backend=='fullhouse':answer=policy.decide(state,rng_seed)
                            else:answer=engine.call({'op':'blueprint','seed':rng_seed})
                            latency[backend].append(time.monotonic()-before)
                            key=f'{backend}/{state["street"]}/{sum(not x["is_folded"] for x in state["players"])}'
                            counter=fall.setdefault(key,{'decisions':0,'fallback':0});counter['decisions']+=1;counter['fallback']+=int(answer.get('fallback',False))
                            action=dict(answer['action'],seat=seat)
                            state=engine.call({'op':'step','action':action});steps+=1
                            if steps>512:raise RuntimeError('Hand failed to terminate')
                        assert sum(state['utilities'])==0
                        row=dict(arm=arm,deal=deal,hero=hero,seed=args.deal_seed+deal,utilities=state['utilities'],chips=state['utilities'][hero],steps=steps)
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    if not complete:break
                if not complete:break
    except TimeoutError:
        complete=False
    finally:engine.close()
    summary={}
    for arm in args.arms:
        block={}
        for r in rows:
            if r['arm']==arm:block.setdefault(r['deal'],[]).append(r['chips'])
        means=[statistics.mean(x) for x in block.values() if len(x)==6]
        summary[arm]={'hands':sum(len(x) for x in block.values()),'complete_deals':len(means),'bb_per_100':statistics.mean(means) if means else None,
                      'deal_ci95_half_width':1.96*statistics.stdev(means)/len(means)**.5 if len(means)>1 else None}
    timing={k:{'count':len(v),'p50_ms':statistics.median(v)*1000 if v else None,'p95_ms':sorted(v)[min(len(v)-1,int(len(v)*.95))]*1000 if v else None} for k,v in latency.items()}
    result=dict(complete=complete,deals_requested=args.deals,deal_seed=args.deal_seed,candidate_path=str(policy.model),candidate_model_type=policy.model_type,
                engine_binary_sha256=hashlib.sha256((ROOT/'.cache/noregrets/target/release/poker-bridge').read_bytes()).hexdigest(),
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_sha256=hashlib.sha256((ROOT/'scripts/fullhouse_runtime.py').read_bytes()).hexdigest(),
                summary=summary,latency=timing,decisions=fall,elapsed_seconds=time.monotonic()-start,peak_sampled_rss_bytes=peak_rss,
                model_hashes={'fullhouse':policy.digest,'noregrets':entry['sha256']},strength_verified=False,limitations=['pilot only','same-noregrets sanity is statistical, not exactly zero','nominal deal-block CI','RSS sampled before decisions'])
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print(out)

if __name__=='__main__':main()
