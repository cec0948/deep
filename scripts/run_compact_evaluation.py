"""Evaluate fixed baselines and the declared checkpoints on common cards."""
import argparse
import gzip
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time

from fullhouse_v3_common import ROOT,atomic_json,digest


def evaluation_jobs(folder,cfg):
    folder=Path(folder)
    jobs=[]
    baselines = cfg.get('evaluation_baselines', ['public', 'old-v3-adv42'])
    if len(baselines) != len(set(baselines)) or set(baselines)-{'public','old-v3-adv42'}:
        raise ValueError('Invalid evaluation baselines')
    available = [('public',None,None),('old-v3-adv42',ROOT/cfg['source_run']/'seed-42_adv/iter_00120.npz',ROOT/cfg['source_run']/'source')]
    for label,model,source in available:
        if label not in baselines:continue
        for field in cfg['evaluation_fields']:jobs.append(dict(id=f'{field}-{label}',model=str(model) if model else None,source=str(source) if source else None,field=field))
    for iteration in cfg['evaluation_iterations']:
        fields=cfg['evaluation_fields'] if cfg.get('evaluation_all_stages_fields') or iteration==max(cfg['evaluation_iterations']) else ['noregrets','public']
        for seed in cfg['seeds']:
            for kind in ('current','mixture'):
                model=folder/f'seed-{seed}'/f'{kind}_{iteration:05d}.{ "json" if kind=="mixture" else "npz"}'
                for field in fields:jobs.append(dict(id=f'{field}-seed-{seed}-{kind}-{iteration}',model=str(model),source=str(folder/'source'),field=field,iteration=iteration,seed=seed,kind=kind))
    return jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    cfg=json.loads((folder/'config.json').read_text());evaluation=folder/'evaluation';evaluation.mkdir(exist_ok=True)
    jobs=evaluation_jobs(folder,cfg)
    running=[];finished=[];failed=[];retries=[];started=time.monotonic()
    while jobs or running:
        launch=json.loads((folder/'launch.json').read_text());training=launch['status']=='running'
        limit=1 if training else 3
        if any(j.get('startup_attempt') for j in jobs) or any(j.get('startup_attempt') for j,_,_ in running):limit=1
        for job in jobs[:]:
            result=evaluation/job['id']/'results.json'
            if result.exists():
                r=json.loads(result.read_text())
                if r['deal_seed']!=cfg['evaluation_deal_seed'] or r['deals']!=cfg['evaluation_deals']:raise ValueError('Evaluation config changed')
                finished.append(job);jobs.remove(job)
        for job in jobs[:]:
            if len(running)>=limit:break
            ready=job['model'] is None or Path(job['model']).exists()
            if not ready:
                if not training:failed.append(dict(**job,error='Model unavailable after training'));jobs.remove(job)
                continue
            output=evaluation/job['id']
            if output.exists():raise ValueError('Incomplete evaluation exists: '+str(output))
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_compact.py'),'--field',job['field'],'--deals',str(cfg['evaluation_deals']),
                 '--deal-seed',str(cfg['evaluation_deal_seed']),'--output',str(output)]
            if job['model']:cmd+=['--model',job['model']]
            if job['source']:cmd+=['--source',job['source']]
            log=(evaluation/(job['id']+'.log')).open('w');child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            running.append((job,child,log));jobs.remove(job);print('started',job['id'],child.pid,flush=True)
        for job,child,log in running[:]:
            if child.poll() is None:continue
            log.close();running.remove((job,child,log))
            if child.returncode==0:
                r=json.loads((evaluation/job['id']/'results.json').read_text());finished.append(job)
                print('complete',job['id'],r['bb_per_100'],r['ci95_half_width'],flush=True)
            else:
                output=evaluation/job['id'];logpath=evaluation/(job['id']+'.log')
                empty=False;raw=output/'hands.jsonl.gz'
                if raw.exists():
                    with gzip.open(raw,'rt') as stream:empty=next(stream,None) is None
                if empty and 'Engine response timeout' in logpath.read_text() and not job.get('startup_attempt'):
                    saved=evaluation/'failed-startup';saved.mkdir(exist_ok=True)
                    shutil.move(str(output),saved/(job['id']+'-attempt-1'))
                    shutil.move(str(logpath),saved/(job['id']+'-attempt-1.log'))
                    retries.append(dict(id=job['id'],reason='Engine startup timeout, zero hands',preserved=str(saved)))
                    jobs.append(dict(**job,startup_attempt=1));print('retry startup',job['id'],flush=True)
                else:failed.append(dict(**job,error=f'exit {child.returncode}'));print('failed',job['id'],child.returncode,flush=True)
        atomic_json(folder/'evaluation-status.json',dict(status='running' if jobs or running else 'complete',finished=finished,failed=failed,
                    pending=[j['id'] for j in jobs],running={j['id']:p.pid for j,p,_ in running},seconds=time.monotonic()-started,
                    startup_retries=retries,evaluator_sha256=digest(ROOT/'scripts/evaluate_compact.py')))
        if jobs or running:time.sleep(3)
    if failed:raise SystemExit(2)


if __name__=='__main__':main()
