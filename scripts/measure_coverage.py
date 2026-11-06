#!/usr/bin/env python3
"""Synthetic, fixed observation panel; measures lookup coverage, not strength."""
import collections
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BIN=ROOT/'.cache/noregrets/target/release/poker-json'

def panel():
    for button in range(6):
        pre=[(button+x)%6 for x in (3,4,5,0,1,2)]
        post=[(button+x)%6 for x in (1,2,3,4,5,0)]
        for raised in (False,True):
            for hole in (['As','Kd'],['Ah','Ad'],['7c','6c'],['2d','3d']):
                actions=[]
                for i,hero in enumerate(pre):
                    yield 'Preflop',dict(hero=hero,button=button,hole=hole,board=[],actions=list(actions),seed=424242)
                    act=dict(seat=hero,kind='call')
                    if raised and i==0:act=dict(seat=hero,kind='raise',to=250)
                    elif not raised and i==5:act=dict(seat=hero,kind='check')
                    actions.append(act)
                for n,street in ((3,'Flop'),(4,'Turn'),(5,'River')):
                    for hero in post:
                        yield street,dict(hero=hero,button=button,hole=hole,board=['2c','7d','Jh','9c','Qs'][:n],actions=list(actions),seed=424242)
                        actions.append(dict(seat=hero,kind='check'))
        # Include the common case where a six-seat hand reaches a heads-up flop.
        for hole in (['As','Kd'],['Ah','Ad'],['7c','6c'],['2d','3d']):
            actions=[dict(seat=(button+x)%6,kind='fold') for x in (3,4,5)]
            actions += [dict(seat=button,kind='raise',to=250),
                        dict(seat=(button+1)%6,kind='fold'),dict(seat=(button+2)%6,kind='call')]
            for n,street in ((3,'Flop'),(4,'Turn'),(5,'River')):
                for hero in ((button+2)%6,button):
                    yield 'HU-flop-line-'+street,dict(hero=hero,button=button,hole=hole,board=['2c','7d','Jh','9c','Qs'][:n],actions=list(actions),seed=424242)
                    actions.append(dict(seat=hero,kind='check'))

def main():
    results=[]
    for filename in sys.argv[1:]:
        counts=collections.defaultdict(collections.Counter)
        with subprocess.Popen([str(BIN),filename],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as proc:
            for street,request in panel():
                proc.stdin.write(json.dumps(request)+'\n');proc.stdin.flush()
                response=json.loads(proc.stdout.readline())
                if not response.get('ok'):raise RuntimeError(response)
                counts[street][response['policy_source']]+=1
            proc.stdin.close();proc.wait(timeout=10)
            if proc.returncode:raise RuntimeError('Engine exited unsuccessfully')
        results.append(dict(model=filename,streets=dict(counts)))
    print(json.dumps(dict(panel='six-way passive/single-raised and heads-up single-raised check-downs; synthetic, not representative play',results=results),indent=2))

if __name__=='__main__':main()
