"""Publish small factual progress records; never publish model weights or Drive IDs."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

from fullhouse_v3_common import ROOT,alive,atomic_json


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default


def collect():
    registry=read(ROOT/'artifacts/compact-v4/current.json',{})
    folder=Path(registry['folder'])
    cfg=read(folder/'config.json',{});campaign=read(folder/'campaign-status.json',{})
    seeds={}
    for seed in cfg.get('seeds',[]):
        saved=read(folder/f'seed-{seed}/status.json',{})
        latest={};path=folder/f'seed-{seed}/diagnostics.jsonl'
        if path.exists():
            for line in reversed(path.read_text().splitlines()[-3:]):
                try:latest=json.loads(line);break
                except json.JSONDecodeError:pass
        seeds[str(seed)]=dict(iteration=latest.get('iteration',0),checkpoint=saved.get('iteration',0),
                             quality=latest.get('quality'),iteration_seconds=latest.get('seconds'))
    evaluation=read(folder/'evaluation-status.json',{})
    backups=read(ROOT/'artifacts/offsite-backups.json',{}).get('backups',[])
    checkpoints=[b for b in backups if b.get('kind')=='training-checkpoint' and not b.get('deleted_remote',False)]
    external=checkpoints[-1] if checkpoints else None
    loop=read(ROOT/'artifacts/retraining-loop.json',{})
    decision=read(folder/'progress-decision.json')
    return dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                run=folder.name,phase=campaign.get('status','starting'),target_iterations=cfg.get('iterations'),
                supervisor_alive=alive(campaign.get('supervisor_pid',0),campaign.get('supervisor_start')),
                seeds=seeds,evaluation=dict(completed=len(evaluation.get('finished',[])),
                    failed=len(evaluation.get('failed',[])),pending=len(evaluation.get('pending',[])),
                    running=len(evaluation.get('running',{}))),
                external_checkpoint=dict(run=external.get('run'),iterations=external.get('iterations'),
                    recorded_at=external.get('recorded_at'),verified_size=external.get('verified_remote_size'),
                    verified_private=external.get('verified_owner_only')) if external else None,
                loop_status=loop.get('status'),max_iterations=loop.get('max_iterations'),
                progress_decision=decision,errors={k:v for k,v in [('campaign',campaign.get('error')),('controller',loop.get('error'))] if v},
                limitations=['Observed at the timestamp, not a guarantee the cloud process remains alive',
                             'Checkpoint iteration is saved progress; latest iteration may be newer',
                             'Training fit metrics do not establish poker strength'])


def main(branch,once=False):
    lock=(ROOT/'artifacts/progress-publisher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None;last=0
    while True:
        actual=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if actual!=branch:raise ValueError('Repository branch changed; refusing publication')
        result=collect()
        signature=json.dumps([result['run'],result['phase'],[s['checkpoint'] for s in result['seeds'].values()],
                              result['evaluation']['completed']//3,result['external_checkpoint'],result['loop_status'],result['errors']],sort_keys=True)
        if signature!=previous or time.monotonic()-last>900:
            path=ROOT/'reports/retraining-status.json';atomic_json(path,result)
            subprocess.run(['git','add','--','reports/retraining-status.json'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            subprocess.run(['git','commit','--only','-m','Record verified poker training progress','--','reports/retraining-status.json'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
            pushed=subprocess.run(['git','push','origin',f'HEAD:refs/heads/{branch}'],cwd=ROOT,timeout=60)
            if pushed.returncode==0:previous=signature;last=time.monotonic()
            print(json.dumps({k:result[k] for k in ('observed_utc','run','phase','seeds','evaluation','loop_status')}),flush=True)
        if once or result['loop_status'] in ('paused_for_failure','paused_for_analysis','bounded_plan_complete','controller_failed'):return
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--branch',required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    main(args.branch,args.once)
