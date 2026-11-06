"""Durable, bounded 600-iteration CPU training and predeclared evaluation.

Start a detached supervisor, inspect status, or resume an interrupted campaign.
No model promotion and no paid resources are provisioned.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=ROOT.parent/'.venvs/fullhouse/bin/python'
BASE=ROOT/'artifacts/fullhouse-medium'
STOP=False
CHILD=None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None


def alive(pid,start):
    return start is not None and process_start(pid)==start


def diagnostics(folder):
    p=folder/'training/diagnostics.jsonl'
    if not p.exists():return None
    # The final row may still be being written by the trainer.
    rows=p.read_text().splitlines()
    for line in reversed(rows):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return None


def verify_sources(folder,state):
    for path,expected in state['source_hashes'].items():
        if digest(folder/path)!=expected:raise RuntimeError('Frozen source changed: '+path)
    for path,expected in state['external_hashes'].items():
        if digest(ROOT/path)!=expected:raise RuntimeError('Runtime/model changed: '+path)


def save_state(folder,state):
    state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(folder/'status.json',state)


def request_stop(signum,frame):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)


def run_command(folder,state,cmd,cwd,log,timeout,training=False):
    global CHILD
    if STOP:raise InterruptedError('Stop requested')
    start=time.monotonic();peak=0
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('w') as f:
        CHILD=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                               stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['child_pid']=CHILD.pid;state['child_process_start']=process_start(CHILD.pid)
        state['active_command']=list(map(str,cmd));save_state(folder,state)
        try:
            last_save=0.
            while CHILD.poll() is None:
                try:
                    text=Path(f'/proc/{CHILD.pid}/status').read_text()
                    peak=max(peak,next((int(x.split()[1])*1024 for x in text.splitlines() if x.startswith('VmRSS:')),0))
                except FileNotFoundError:pass
                elapsed=time.monotonic()-start
                if STOP:raise InterruptedError('Stop requested')
                if elapsed>timeout or peak>state['config']['max_rss_bytes']:
                    raise TimeoutError('Resource limit: '+str(log))
                if time.monotonic()-last_save>=10:
                    state['active_elapsed_seconds']=elapsed;state['active_peak_parent_rss_bytes']=peak
                    if training:
                        row=diagnostics(folder)
                        if row:
                            state['progress']=row
                            n=row['iteration']-state['stage_start_iteration']
                            if n>=3:
                                state['estimated_training_remaining_seconds']=(elapsed/n)*(state['config']['iterations']-row['iteration'])
                    save_state(folder,state);last_save=time.monotonic()
                time.sleep(1)
            rc=CHILD.returncode
        except BaseException:
            if training:state['training_elapsed_seconds']=state.get('training_elapsed_seconds',0.)+time.monotonic()-start
            raise
        finally:
            if CHILD.poll() is None:
                os.killpg(CHILD.pid,signal.SIGTERM)
                try:CHILD.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
    result=dict(command=list(map(str,cmd)),returncode=rc,elapsed_seconds=time.monotonic()-start,peak_parent_rss_bytes=peak,log=str(log))
    state['child_pid']=None;state['child_process_start']=None
    if rc:raise RuntimeError(f'Command exited {rc}; see {log}')
    return result


def inspect_checkpoint(folder,expected):
    # Local, trusted checkpoint. A separate process bounds memory in supervisor.
    code="""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
p=Path(sys.argv[1]);expected=int(sys.argv[2]);c=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
assert c['iteration']==expected
assert c['r1_config']['horizon']==600 and c['r1_config']['traversals']==256
assert c['r1_config']['train_steps']==512 and c['r1_config']['strategy_train_steps']==128
assert c['r1_config']['average_sampler']=='importance'
assert len(c['buffers'])==6 and c['strategy_buffer']['count']>0
assert all(torch.isfinite(v).all() for key in ('net','strategy_net') for v in c[key].values())
assert all(k in c['r1_rng'] for k in ('python','numpy','generator','torch'))
model=p/f'iter_{expected:05d}.npz'
with np.load(model) as a:assert all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind in 'fc')
print(json.dumps(dict(iteration=expected,model=str(model),sha256=hashlib.sha256(model.read_bytes()).hexdigest(),regret_buffer_counts=c['buffer_sizes'],strategy_buffer_count=c['strategy_buffer_size'],resume_state_present=True)))
"""
    result=subprocess.run([str(PYTHON),'-c',code,str(folder/'training'),str(expected)],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=120)
    if result.returncode:raise RuntimeError('Checkpoint validation failed: '+result.stderr[-2000:])
    return json.loads(result.stdout)


def trim_uncommitted_diagnostics(folder,iteration):
    p=folder/'training/diagnostics.jsonl'
    if p.exists():
        good=[]
        for line in p.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row['iteration']<=iteration:good.append(line)
        p.write_text('\n'.join(good)+('\n' if good else ''))


def worker(folder):
    global STOP
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    state=json.loads((folder/'status.json').read_text());cfg=state['config']
    with (BASE/'campaign.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another medium campaign is running')
        state.update(status='running',supervisor_pid=os.getpid(),supervisor_process_start=process_start(os.getpid()),error=None)
        save_state(folder,state)
        try:
            verify_sources(folder,state)
            checkpoint=folder/'training/latest.pt'
            last=state.get('completed_training_iteration',0)
            # A prior supervisor can stop after checkpoint save but before manifest commit.
            if checkpoint.exists():
                code="import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['iteration'])"
                r=subprocess.run([str(PYTHON),'-c',code,str(checkpoint)],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),capture_output=True,text=True,timeout=120,check=True)
                last=int(r.stdout);inspect_checkpoint(folder,last);trim_uncommitted_diagnostics(folder,last)
            state['completed_training_iteration']=last
            for end in cfg['stages']:
                if end<=last:continue
                verify_sources(folder,state)
                state.update(phase='training',stage_start_iteration=last,stage_end_iteration=end)
                used=state.get('training_elapsed_seconds',0.)
                remaining=cfg['max_training_seconds']-used
                if remaining<=0:raise TimeoutError('Campaign training budget exhausted')
                cmd=[PYTHON,'-m','training.train','--iterations',cfg['iterations'],'--stop-after',end,
                    '--traversals',cfg['traversals'],'--train-steps',cfg['train_steps'],
                    '--strategy-train-steps',cfg['strategy_train_steps'],'--batch-size',cfg['batch_size'],
                    '--buffer-capacity',cfg['buffer_capacity'],'--strategy-buffer-capacity',cfg['strategy_buffer_capacity'],
                    '--checkpoint-every',cfg['checkpoint_every'],'--checkpoint-dir',folder/'training',
                    '--seed',cfg['seed'],'--workers',1,'--device','cpu','--average-sampler',cfg['average_sampler'],
                    '--save-resume-buffers','--no-stack-randomization','--opponent-exploration-start',0,'--opponent-exploration-end',0]
                if last:cmd+=['--resume']
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f')
                result=run_command(folder,state,cmd,folder/'source',folder/f'train-{last:05d}-{end:05d}-{stamp}.log',remaining,training=True)
                state['training_elapsed_seconds']=used+result['elapsed_seconds']
                check=inspect_checkpoint(folder,end)
                state['training_stages'].append(dict(**result,checkpoint=check))
                state['completed_training_iteration']=end;state['progress']=diagnostics(folder);last=end;save_state(folder,state)
            state.update(phase='evaluation',estimated_training_remaining_seconds=0);save_state(folder,state)
            candidates=[('public',None),('small-100',ROOT/cfg['baseline_model'])]
            candidates += [(f'medium-{n}',folder/'training'/f'iter_{n:05d}.npz') for n in cfg['evaluation_iterations']]
            for name,model in candidates:
                if any(r['name']==name for r in state['evaluations']):continue
                verify_sources(folder,state);state['evaluation_candidate']=name
                timeout=cfg['max_evaluation_seconds_each']
                cmd=[PYTHON,ROOT/'scripts/evaluate_fullhouse.py','--deals',cfg['evaluation_deals'],
                    '--deal-seed',cfg['evaluation_deal_seed'],'--arms','fullhouse','--timeout',timeout]
                if model:cmd+=['--model',model]
                log=folder/f'evaluate-{name}.log'
                result=run_command(folder,state,cmd,ROOT,log,timeout+60)
                path=Path(log.read_text().strip().splitlines()[-1]);ev=json.loads((path/'results.json').read_text())
                if not ev['complete']:raise RuntimeError('Incomplete evaluation: '+str(path))
                rows=[json.loads(line) for line in (path/'hands.jsonl').read_text().splitlines()]
                n=cfg['evaluation_deals'];assert len(rows)==6*n and len({(r['deal'],r['hero']) for r in rows})==6*n
                assert all(sum(r['utilities'])==0 and r['chips']==r['utilities'][r['hero']] for r in rows)
                blocks=[statistics.mean(r['chips'] for r in rows if r['deal']==d) for d in range(n)]
                state['evaluations'].append(dict(name=name,**result,directory=str(path),summary=ev['summary']['fullhouse'],deal_blocks=blocks,model_hashes=ev['model_hashes']))
                save_state(folder,state)
            by_name={x['name']:x['deal_blocks'] for x in state['evaluations']}
            state['paired_differences']={}
            for name,block in by_name.items():
                if not name.startswith('medium'):continue
                state['paired_differences'][name]={}
                for other in ('public','small-100','medium-120'):
                    if name==other:continue
                    diffs=[a-b for a,b in zip(block,by_name[other])]
                    state['paired_differences'][name][other]=dict(bb_per_100=statistics.mean(diffs),nominal_ci95_half_width=1.96*statistics.stdev(diffs)/len(diffs)**.5)
            state.update(status='complete',phase='complete',strength_verified=False,child_pid=None)
            atomic_json(folder/'results.json',state);save_state(folder,state)
        except BaseException as e:
            state.update(status='interrupted' if STOP else 'failed',error=f'{type(e).__name__}: {e}',child_pid=None)
            save_state(folder,state);raise


def launch(folder):
    state=json.loads((folder/'status.json').read_text())
    if alive(state.get('supervisor_pid',0),state.get('supervisor_process_start')):raise RuntimeError('Supervisor already running')
    if alive(state.get('child_pid',0),state.get('child_process_start')):raise RuntimeError('Existing child still running')
    if state['status']=='complete':raise RuntimeError('Campaign already complete')
    with (folder/'supervisor.log').open('a') as log:
        p=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'worker',str(folder)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(folder/'launch.json',dict(supervisor_pid=p.pid,process_start=process_start(p.pid),folder=str(folder)))
    print(folder,flush=True)


def prepare():
    current=BASE/'current.json'
    if current.exists():
        prior=Path(json.loads(current.read_text())['folder'])/'status.json'
        if prior.exists():
            state=json.loads(prior.read_text())
            if alive(state.get('supervisor_pid',0),state.get('supervisor_process_start')):
                raise RuntimeError('A medium campaign is already running: '+str(prior.parent))
    from build_fullhouse_r1 import build
    cfg=json.loads((ROOT/'configs/fullhouse-medium.json').read_text())
    source=build();BASE.mkdir(parents=True,exist_ok=True)
    folder=BASE/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir()
    (folder/'source').mkdir();(folder/'training').mkdir()
    for name in ('training','bot'):
        shutil.copytree(source/name,folder/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    (folder/'source/data').symlink_to((source/'data').resolve(),target_is_directory=True)
    shutil.copyfile(source/'LICENSE.upstream',folder/'source/LICENSE.upstream')
    shutil.copyfile(ROOT/'configs/fullhouse-medium.json',folder/'config.json')
    shutil.copyfile(ROOT/'patches/fullhouse-r1.patch',folder/'source/fullhouse-r1.patch')
    files=[p for p in (folder/'source').rglob('*') if p.is_file() and not p.is_symlink()]
    external=[ROOT/'scripts/evaluate_fullhouse.py',ROOT/'scripts/fullhouse_runtime.py',Path(__file__).resolve(),ROOT/cfg['baseline_model'],ROOT/'artifacts/selfplay/current.json',ROOT/'.cache/noregrets/target/release/poker-bridge',ROOT/'.cache/fullhouse-bot/data/preflop_equity.npz']
    state=dict(status='prepared',phase='prepared',config=cfg,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p.relative_to(folder)):digest(p) for p in files},
        external_hashes={str(p.relative_to(ROOT)):digest(p) for p in external},
        training_stages=[],evaluations=[],completed_training_iteration=0,training_elapsed_seconds=0.,strength_verified=False)
    save_state(folder,state);atomic_json(BASE/'current.json',dict(folder=str(folder)))
    launch(folder)


def show_status(folder):
    state=json.loads((folder/'status.json').read_text());live=alive(state.get('supervisor_pid',0),state.get('supervisor_process_start'))
    row=diagnostics(folder);iteration=row['iteration'] if row else 0
    print(json.dumps(dict(folder=str(folder),status=state['status'],phase=state['phase'],supervisor_alive=live,
        child_alive=alive(state.get('child_pid',0),state.get('child_process_start')),iteration=iteration,
        committed_checkpoint_iteration=state['completed_training_iteration'],iterations_planned=state['config']['iterations'],
        regret_root_traversals_completed=iteration*state['config']['traversals'],
        estimated_training_remaining_seconds=state.get('estimated_training_remaining_seconds'),
        evaluations_completed=len(state['evaluations']),error=state.get('error'),updated_utc=state['updated_utc']),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','worker','resume','status']);p.add_argument('folder',nargs='?',type=Path);args=p.parse_args()
    if args.action=='start':prepare();return
    folder=args.folder or Path(json.loads((BASE/'current.json').read_text())['folder']);folder=folder.resolve()
    if args.action=='worker':worker(folder)
    elif args.action=='resume':launch(folder)
    else:show_status(folder)

if __name__=='__main__':main()
