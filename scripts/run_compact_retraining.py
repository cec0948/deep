"""Run one bounded fresh/restored campaign and checkpoint during training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, alive, atomic_json, digest, process_start


def run(folder, resume=False):
    folder=Path(folder).resolve();cfg=json.loads((folder/'config.json').read_text())
    lock=(folder/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((folder/'launch.json').read_text()) if (folder/'launch.json').exists() else {}
    if previous.get('status')=='complete' and not resume:raise ValueError('Training already complete')
    if previous.get('status')=='running':
        for seed,pid in previous.get('pids',{}).items():
            if alive(pid,previous.get('process_starts',{}).get(seed)):
                raise ValueError('Training worker is still alive')
    if (folder/'campaign-status.json').exists():
        prior=json.loads((folder/'campaign-status.json').read_text())
        if alive(prior.get('evaluator_pid',0),prior.get('evaluator_start')):
            raise ValueError('Evaluator is still alive')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic();workers={};logs=[];evaluator=None;backup=None;last_backup=0
    helpers=['run_compact_retraining.py','run_compact_evaluation.py','evaluate_compact.py',
             'summarize_compact.py','compact_progress.py','verify_compact_campaign.py','export_compact.py','snapshot_compact.py']
    hashes={name:digest(ROOT/'scripts'/name) for name in helpers}
    atomic_json(folder/'supervisor-manifest.json',hashes)
    def status(phase,**extra):
        atomic_json(ROOT/'artifacts/compact-v4/current.json',dict(folder=str(folder),status=phase,iterations=cfg['iterations']))
        atomic_json(folder/'campaign-status.json',dict(status=phase,seconds=time.monotonic()-started,
                    supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),
                    evaluator_pid=evaluator.pid if evaluator else None,
                    evaluator_start=process_start(evaluator.pid) if evaluator else None,**extra))
    def checked(name):
        if digest(ROOT/'scripts'/name)!=hashes[name]:raise ValueError('Campaign helper changed: '+name)
        with (folder/(name+'.log')).open('w') as stream:
            subprocess.run([str(PYTHON),str(ROOT/'scripts'/name),str(folder)],cwd=ROOT,env=env,
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    def stop(*unused):raise InterruptedError('Campaign stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if resume:
            evaluation=folder/'evaluation'
            for output in sorted(evaluation.iterdir()) if evaluation.exists() else []:
                if output.is_dir() and output.name!='failed-startup' and not (output/'results.json').exists():
                    preserved=folder/'interrupted-evaluations';preserved.mkdir(exist_ok=True)
                    output.rename(preserved/f'{output.name}-{time.time_ns()}')
        status('training')
        for seed in cfg['seeds']:
            saved=folder/f'seed-{seed}/status.json'
            if resume and saved.exists() and json.loads(saved.read_text())['iteration']==cfg['iterations']:continue
            command=[str(PYTHON),str(ROOT/'scripts/train_compact_selfplay.py'),'--run-dir',str(folder),'--seed',str(seed)]
            if resume:command.append('--resume')
            log=(folder/f'train-{seed}.log').open('a' if resume else 'w');logs.append(log)
            workers[seed]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(folder/'launch.json',dict(status='running' if workers else 'complete',
                    pids={str(s):w.pid for s,w in workers.items()},process_starts={str(s):process_start(w.pid) for s,w in workers.items()}))
        log=(folder/'evaluation-supervisor.log').open('a' if resume else 'w');logs.append(log)
        evaluator=subprocess.Popen([str(PYTHON),str(ROOT/'scripts/run_compact_evaluation.py'),str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while any(w.poll() is None for w in workers.values()):
            if any(w.poll() not in (None,0) for w in workers.values()):raise RuntimeError('Training worker failed; preserved logs and checkpoints')
            if evaluator.poll() not in (None,0):raise RuntimeError('Evaluation failed; preserved evidence')
            if backup and backup.poll() is not None:
                if backup.returncode:raise RuntimeError('Checkpoint snapshot verification failed')
                backup=None
            checkpoints={}
            for seed in cfg['seeds']:
                path=folder/f'seed-{seed}/status.json'
                if path.exists():checkpoints[str(seed)]=json.loads(path.read_text())['iteration']
            common=min(checkpoints.values()) if len(checkpoints)==len(cfg['seeds']) else 0
            # Save one verified archive whenever all workers cross another 6-round boundary.
            if common>last_backup and backup is None:
                log=(folder/f'backup-through-{common}.log').open('w');logs.append(log)
                backup=subprocess.Popen([str(PYTHON),str(ROOT/'scripts/snapshot_compact.py'),str(folder)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                last_backup=common
            status('training',checkpoint_iterations=checkpoints)
            time.sleep(3)
        if any(w.returncode for w in workers.values()):raise RuntimeError('Training failed')
        atomic_json(folder/'launch.json',dict(status='complete',exits={str(s):w.returncode for s,w in workers.items()},seconds=time.monotonic()-started))
        status('evaluating')
        if backup and backup.wait()!=0:raise RuntimeError('Snapshot failed')
        backup=None
        checked('snapshot_compact.py')
        if evaluator.wait()!=0:raise RuntimeError('Evaluation failed')
        for name in ('summarize_compact.py','verify_compact_campaign.py','compact_progress.py','export_compact.py'):checked(name)
        report=ROOT/'reports/compact-v4'/folder.name
        for name in ('launch.json','campaign-status.json','evaluation-status.json','verification.json','supervisor-manifest.json'):
            shutil.copyfile(folder/name,report/name)
        for seed in cfg['seeds']:
            target=report/f'seed-{seed}';target.mkdir(exist_ok=True)
            for name in ('diagnostics.jsonl','status.json'):shutil.copyfile(folder/f'seed-{seed}'/name,target/name)
        status('complete')
        print('complete',folder,flush=True)
    except BaseException as exc:
        status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        raise
    finally:
        for process in [*workers.values(),evaluator,backup]:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    run(args.folder,args.resume)
