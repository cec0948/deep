"""Bounded multi-seed V3 campaign, fresh buffers, frozen source, resumable stages."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

from fullhouse_v3_common import ROOT,PYTHON,digest,atomic_json,process_start,alive

BASE=ROOT/'artifacts/fullhouse-v3'
STOP=False


def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def save(folder,state):
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(folder/'status.json',state)


def verify(folder,state):
    for name,sha in state['source_hashes'].items():
        if digest(folder/name)!=sha:raise RuntimeError('Frozen source changed: '+name)
    for name,sha in state['external_hashes'].items():
        if digest(ROOT/name)!=sha:raise RuntimeError('External dependency changed: '+name)


def checkpoint(path,cfg,seed):
    if not (path/'latest.pt').exists():return 0
    import torch
    c=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
    expected=dict(revision=cfg['revision'],strategy_optimizer=cfg['strategy_optimizer'],strategy_batch_size=cfg['strategy_batch_size'],
                  average_sampler=cfg['average_sampler'],horizon=cfg['horizon'],traversals=cfg['traversals'],
                  buffer_capacity=cfg['buffer_capacity'],strategy_buffer_capacity=cfg['strategy_buffer_capacity'],
                  train_steps=cfg['train_steps'],strategy_train_steps=cfg['strategy_train_steps'],batch_size=cfg['batch_size'],
                  lr=.001,seed=seed,checkpoint_every=cfg['checkpoint_every'])
    if c.get('r1_config')!=expected:raise ValueError('Checkpoint/config mismatch')
    if len(c['buffers'])!=6 or not c['strategy_buffer']['count']:raise ValueError('Missing fresh replay')
    if not all(torch.isfinite(v).all() for key in ('net','strategy_net') for v in c[key].values()):raise ValueError('Nonfinite network')
    return c['iteration']


def trim(path,iteration):
    for name in ('diagnostics.jsonl','strategy-audit.jsonl'):
        p=path/name
        if not p.exists():continue
        kept=[]
        for line in p.read_text().splitlines():
            try:
                if json.loads(line)['iteration']<=iteration:kept.append(line)
            except json.JSONDecodeError:pass
        p.write_text('\n'.join(kept)+'\n' if kept else '')


def run_jobs(folder,state,jobs):
    """Three independent processes; do not parallelize one learner's RNG."""
    waiting=list(jobs);active={};cfg=state['config'];last=0
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        while waiting or active:
            if STOP:raise InterruptedError('Stop requested')
            while waiting and len(active)<cfg['parallel_jobs']:
                job=waiting.pop(0);verify(folder,state)
                log=(folder/job['log']).open('w')
                p=subprocess.Popen(list(map(str,job['command'])),cwd=job['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job['id']]={'process':p,'log':log,'job':job,'start':time.monotonic()}
                state['active'][job['id']]=dict(pid=p.pid,process_start=process_start(p.pid),log=job['log'],command=list(map(str,job['command'])))
                save(folder,state)
            rss=0
            for key,entry in list(active.items()):
                p=entry['process'];job=entry['job'];elapsed=time.monotonic()-entry['start']
                if p.poll() is None:
                    try:
                        lines=Path(f'/proc/{p.pid}/status').read_text().splitlines()
                        rss+=next((int(x.split()[1])*1024 for x in lines if x.startswith('VmRSS:')),0)
                    except FileNotFoundError:pass
                    if elapsed>job['timeout']:raise TimeoutError(key)
                else:
                    entry['log'].close();del active[key];state['active'].pop(key,None)
                    if p.returncode:raise RuntimeError(f'{key} exited {p.returncode}; {job["log"]}')
                    if 'seed' in job:
                        n=checkpoint(folder/f'seed-{job["seed"]}',cfg,job['seed'])
                        if n!=job['target']:raise RuntimeError('Incomplete training stage')
                        state['seeds'][str(job['seed'])]['iteration']=n
                        state['seeds'][str(job['seed'])]['training_seconds']+=elapsed
                    else:
                        r=json.loads((Path(job['output'])/'results.json').read_text())
                        if not r['complete'] or r['deals']!=cfg['evaluation_deals']:raise RuntimeError('Incomplete evaluation')
                        state['evaluations'][key]=dict(path=str(Path(job['output']).relative_to(folder)),**r)
                    state['completed_jobs'].append(dict(id=key,elapsed_seconds=elapsed,log=job['log']))
                    save(folder,state)
            if rss>cfg['max_total_children_rss_bytes']:raise MemoryError('Combined child RSS cap')
            if time.monotonic()-last>10:
                state['children_rss_bytes']=rss
                for seed in cfg['seeds']:
                    p=folder/f'seed-{seed}'/'diagnostics.jsonl'
                    if p.exists():
                        for line in reversed(p.read_text().splitlines()[-2:]):
                            try:state['seeds'][str(seed)]['progress']=json.loads(line);break
                            except json.JSONDecodeError:pass
                save(folder,state);last=time.monotonic()
            if active:time.sleep(1)
    finally:
        for entry in active.values():
            p=entry['process']
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            entry['log'].close()
        state['active']={};save(folder,state)


def worker(folder):
    global STOP
    def stop(*unused):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((folder/'status.json').read_text());cfg=state['config']
        state.update(status='running',supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),error=None,active={});save(folder,state)
        try:
            verify(folder,state)
            state['phase']='training'
            for target in cfg['stages']:
                jobs=[]
                for seed in cfg['seeds']:
                    path=folder/f'seed-{seed}';path.mkdir(exist_ok=True);n=checkpoint(path,cfg,seed)
                    state['seeds'][str(seed)]['iteration']=n
                    if n>=target:continue
                    trim(path,n)
                    cmd=[PYTHON,'-m','training.train','--iterations',cfg['horizon'],'--stop-after',target,
                         '--traversals',cfg['traversals'],'--train-steps',cfg['train_steps'],'--strategy-train-steps',cfg['strategy_train_steps'],
                         '--batch-size',cfg['batch_size'],'--strategy-batch-size',cfg['strategy_batch_size'],
                         '--buffer-capacity',cfg['buffer_capacity'],'--strategy-buffer-capacity',cfg['strategy_buffer_capacity'],
                         '--seed',seed,'--checkpoint-dir',path,'--checkpoint-every',cfg['checkpoint_every'],
                         '--workers',1,'--device','cpu','--average-sampler',cfg['average_sampler'],
                         '--opponent-exploration-start',0,'--opponent-exploration-end',0,'--no-stack-randomization',
                         '--save-resume-buffers','--export-advantage-npz']
                    if n:cmd+=['--resume']
                    jobs.append(dict(id=f'train-{seed}-{target}',seed=seed,target=target,command=cmd,cwd=folder/'source',log=f'train-{seed}-{n}-{target}-{stamp()}.log',timeout=cfg['max_training_seconds_each_stage']))
                run_jobs(folder,state,jobs)
            state['phase']='evaluation';save(folder,state)
            candidates={'public':None,**{k:ROOT/v for k,v in cfg['baseline_models'].items()}}
            for seed in cfg['seeds']:
                for iteration in cfg['evaluation_iterations']:candidates[f'seed-{seed}-average-{iteration}']=folder/f'seed-{seed}'/f'iter_{iteration:05d}.npz'
                candidates[f'seed-{seed}-advantage-{cfg["through"]}']=folder/f'seed-{seed}_adv'/f'iter_{cfg["through"]:05d}.npz'
            jobs=[]
            for field in cfg['fields']:
                for name,model in candidates.items():
                    key=f'{field}-{name}'
                    if key in state['evaluations']:continue
                    out=folder/'evaluation'/key;out.parent.mkdir(exist_ok=True)
                    if out.exists():out.rename(out.with_name(out.name+'-interrupted-'+stamp()))
                    cmd=[PYTHON,ROOT/'scripts/evaluate_fullhouse_v3.py','--source',folder/'source','--field',field,
                         '--deals',cfg['evaluation_deals'],'--deal-seed',cfg['evaluation_deal_seed'],
                         '--timeout',cfg['max_evaluation_seconds_each'],'--output',out]
                    if model:cmd+=['--model',model]
                    jobs.append(dict(id=key,command=cmd,cwd=ROOT,log=f'evaluate-{key}-{stamp()}.log',timeout=cfg['max_evaluation_seconds_each']+60,output=out))
            run_jobs(folder,state,jobs)
            state['paired']={}
            for field in cfg['fields']:
                for seed in cfg['seeds']:
                    a=state['evaluations'][f'{field}-seed-{seed}-average-{cfg["through"]}']['deal_blocks']
                    for other in ('public','old-120','old-600',f'seed-{seed}-average-60'):
                        b=state['evaluations'][f'{field}-{other}']['deal_blocks'];delta=[x-y for x,y in zip(a,b)]
                        state['paired'][f'{field}/seed-{seed}-final minus {other}']=dict(bb_per_100=statistics.mean(delta),ci95_half_width=1.96*statistics.stdev(delta)/len(delta)**.5,deals=len(delta))
            state.update(status='complete',phase='complete',promoted_model=None);save(folder,state);atomic_json(folder/'results.json',state)
        except BaseException as exc:
            state.update(status='interrupted' if STOP else 'failed',error=f'{type(exc).__name__}: {exc}');save(folder,state);raise


def launch(folder):
    state=json.loads((folder/'status.json').read_text())
    if state['status']=='complete':raise RuntimeError('Campaign already complete')
    if alive(state.get('supervisor_pid',0),state.get('supervisor_start')):raise RuntimeError('Already running')
    if (folder/'launch.json').exists():
        prior=json.loads((folder/'launch.json').read_text())
        if alive(prior['pid'],prior['process_start']):raise RuntimeError('Supervisor is starting or running')
    for item in state.get('active',{}).values():
        if alive(item['pid'],item['process_start']):raise RuntimeError('Existing child still running')
    state['status']='starting';save(folder,state)
    with (folder/'supervisor.log').open('a') as log:
        p=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'worker',str(folder)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'launch.json',dict(pid=p.pid,process_start=process_start(p.pid)));print(folder,flush=True)


def prepare():
    if (BASE/'current.json').exists():
        prior=Path(json.loads((BASE/'current.json').read_text())['folder'])
        old=json.loads((prior/'status.json').read_text())
        if alive(old.get('supervisor_pid',0),old.get('supervisor_start')):raise RuntimeError('Campaign running: '+str(prior))
        if (prior/'launch.json').exists():
            running=json.loads((prior/'launch.json').read_text())
            if alive(running['pid'],running['process_start']):raise RuntimeError('Campaign starting: '+str(prior))
    gates=[ROOT/'artifacts/fullhouse-v3-verification/final-verification.json',ROOT/'artifacts/fullhouse-v3-verification/stabilized-exact/strategy-fitting.json']
    if not all(json.loads(p.read_text())['passed'] for p in gates):raise RuntimeError('Verification gates not passed')
    from build_fullhouse_v3 import build
    source=build();cfg=json.loads((ROOT/'configs/fullhouse-v3.json').read_text())
    folder=BASE/stamp();folder.mkdir(parents=True);(folder/'source').mkdir()
    for name in ('training','bot'):shutil.copytree(source/name,folder/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    (folder/'source/data').symlink_to((source/'data').resolve(),target_is_directory=True)
    shutil.copyfile(source/'LICENSE.upstream',folder/'source/LICENSE.upstream')
    shutil.copyfile(ROOT/'patches/fullhouse-v3.patch',folder/'source/fullhouse-v3.patch')
    atomic_json(folder/'config.json',cfg)
    for i,p in enumerate(gates):shutil.copyfile(p,folder/f'verification-{i}.json')
    external=[Path(__file__).resolve(),ROOT/'scripts/fullhouse_v3_common.py',ROOT/'scripts/fullhouse_v3_runtime.py',ROOT/'scripts/evaluate_fullhouse_v3.py',ROOT/'scripts/fullhouse_runtime.py',
              ROOT/'.cache/noregrets/target/release/poker-bridge',ROOT/'.cache/fullhouse-bot/data/deep_cfr_model.npz',ROOT/'.cache/fullhouse-bot/data/preflop_equity.npz',ROOT/'artifacts/selfplay/current.json']
    registry=json.loads((ROOT/'artifacts/selfplay/current.json').read_text());external+=[ROOT/registry['model'],*[ROOT/v for v in cfg['baseline_models'].values()]]
    files=[p for p in (folder/'source').rglob('*') if p.is_file() and not p.is_symlink()]
    state=dict(status='prepared',phase='prepared',config=cfg,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               source_hashes={str(p.relative_to(folder)):digest(p) for p in files},external_hashes={str(p.relative_to(ROOT)):digest(p) for p in external},
               seeds={str(seed):dict(iteration=0,training_seconds=0.,initialization='fresh empty replay') for seed in cfg['seeds']},active={},completed_jobs=[],evaluations={},error=None)
    save(folder,state);atomic_json(BASE/'current.json',dict(folder=str(folder)));launch(folder)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','worker','resume','status']);p.add_argument('folder',nargs='?',type=Path);a=p.parse_args()
    if a.action=='start':prepare();return
    folder=(a.folder or Path(json.loads((BASE/'current.json').read_text())['folder'])).resolve()
    if a.action=='worker':worker(folder)
    elif a.action=='resume':launch(folder)
    else:
        state=json.loads((folder/'status.json').read_text())
        print(json.dumps(dict(folder=str(folder),status=state['status'],phase=state['phase'],alive=alive(state.get('supervisor_pid',0),state.get('supervisor_start')),
                             seeds={k:dict(committed=v['iteration'],latest=v.get('progress',{}).get('iteration'),training_seconds=v['training_seconds']) for k,v in state['seeds'].items()},
                             evaluation_count=len(state['evaluations']),error=state['error']),indent=2))


if __name__=='__main__':main()
