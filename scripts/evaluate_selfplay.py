#!/usr/bin/env python3
"""Evaluate saved milestones using upstream duplicate and crossplay runners."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'.cache/noregrets/target/release/pluribus'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path)
    p.add_argument('--hands',type=int,default=30000)
    p.add_argument('--crossplay-hands',type=int,default=60000)
    a=p.parse_args()
    if a.hands<=0 or a.hands%6 or a.crossplay_hands<=0:p.error('hands must be positive and divisible by six; crossplay-hands must be positive')
    campaign=a.campaign.resolve();manifest=json.loads((campaign/'campaign.json').read_text())
    if sha(BIN)!=manifest['binary_sha256']:raise SystemExit('Binary integrity mismatch')
    for stage in manifest['stages']:
        if sha(campaign/stage['blueprint'])!=stage['blueprint_sha256']:raise SystemExit('Model integrity mismatch')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=campaign/'evaluations'/stamp;out.mkdir(parents=True)
    env=os.environ.copy();env['RAYON_NUM_THREADS']='4'
    rows=[]
    def run(name,args,hands,context):
        command=[str(BIN),*args];log=out/(name+'.log');start=time.monotonic()
        print('Evaluating '+name,flush=True)
        with log.open('w') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        content=log.read_text();match=re.search(r'(?:focal )?winrate: ([+-]?[\d.]+) mbb/hand \(95% CI ±([\d.]+)\) over (\d+) hands',content)
        if result.returncode or not match or int(match[3])!=hands:
            raise SystemExit(f'Incomplete or failed evaluation: {name}; see {log}')
        mean=float(match[1])*0.1;ci=float(match[2])*0.1
        row=dict(name=name,command=command,context=context,hands=hands,bb_per_100=round(mean,3),
                 ci95_half_width=round(ci,3),elapsed_seconds=round(time.monotonic()-start,3),
                 returncode=result.returncode,log=log.name,sha256=sha(log))
        fallback=re.search(r'unseen-infoset fallbacks to check/call: (\d+)/(\d+)',content)
        if fallback:row['fallback']={'count':int(fallback[1]),'lookups':int(fallback[2])}
        rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(campaign=str(campaign),
            source_commit=manifest['config']['upstream_commit'],models=manifest['stages'],results=rows,
            limitations=['one training seed','weak scripted baselines','crossplay only within same lineage',
                         'nominal confidence intervals; not a professional-strength benchmark']),indent=2)+'\n')
        print(json.dumps(row),flush=True)
    for s in manifest['stages']:
        for opponent in ('caller','random'):
            run(f'{s["iterations"]}-{opponent}', ['eval','--players','6','--blueprint',str(campaign/s['blueprint']),
                '--hands',str(a.hands),'--baseline',opponent,'--duplicate','--seed','271828'],a.hands,
                {'type':'scripted-baseline','iterations':s['iterations'],'opponent':opponent,'duplicate_deals':a.hands//6})
    first=manifest['stages'][0];last=manifest['stages'][-1]
    if first['iterations']!=last['iterations']:
        for focal,field in ((last,first),(first,last)):
            run(f'cross-{focal["iterations"]}-vs-{field["iterations"]}', ['crossplay','--focal',str(campaign/focal['blueprint']),
                '--field',str(campaign/field['blueprint']),'--hands',str(a.crossplay_hands),'--seed','314159'],a.crossplay_hands,
                {'type':'crossplay','focal':focal['iterations'],'field':field['iterations']})
    print('Results: '+str(out/'results.json'),flush=True)

if __name__=='__main__':main()
