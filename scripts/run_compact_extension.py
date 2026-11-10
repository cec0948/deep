"""Supervise one bounded continuation, its common-card evaluation and backup."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start


def main():
    p = argparse.ArgumentParser();p.add_argument('folder', type=Path);a = p.parse_args()
    folder = a.folder.resolve();cfg = json.loads((folder/'config.json').read_text())
    lock = (folder/'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if json.loads((folder/'launch.json').read_text())['status'] != 'prepared':
        raise ValueError('Start a newly prepared extension only')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic();workers = {};logs = [];exits = {}
    tools = ['run_compact_extension.py', 'run_compact_evaluation.py', 'evaluate_compact.py',
             'summarize_compact.py', 'compact_progress.py', 'verify_compact_campaign.py', 'export_compact.py']
    hashes = {name: digest(ROOT/'scripts'/name) for name in tools}
    atomic_json(folder/'supervisor-manifest.json', hashes)
    def registry(status):
        atomic_json(ROOT/'artifacts/compact-v4/current.json', dict(folder=str(folder), status=status,
                    training_seeds=cfg['seeds'], iterations=cfg['iterations'], promotion='none'))
        atomic_json(folder/'campaign-status.json', dict(status=status, seconds=time.monotonic()-started,
                    supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid())))
    def checked_script(name):
        if digest(ROOT/'scripts'/name) != hashes[name]:
            raise ValueError('Supervisor helper changed: '+name)
        subprocess.run([str(PYTHON), str(ROOT/'scripts'/name), str(folder)], cwd=ROOT, env=env, check=True)
    try:
        registry('training')
        for seed in cfg['seeds']:
            log = (folder/f'train-{seed}.log').open('w');logs.append(log)
            workers[seed] = subprocess.Popen([str(PYTHON),str(ROOT/'scripts/train_compact_selfplay.py'),
                '--run-dir',str(folder),'--seed',str(seed),'--resume'], cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(folder/'launch.json', dict(status='running',pids={str(s):w.pid for s,w in workers.items()}))
        log = (folder/'evaluation-supervisor.log').open('w');logs.append(log)
        evaluator = subprocess.Popen([str(PYTHON),str(ROOT/'scripts/run_compact_evaluation.py'),str(folder)],
                                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while len(exits) < len(workers):
            for seed, worker in workers.items():
                if seed not in exits and worker.poll() is not None:
                    exits[seed] = worker.returncode
                    print('training exited',seed,worker.returncode,flush=True)
            if len(exits) < len(workers):
                time.sleep(3)
        atomic_json(folder/'launch.json',dict(status='complete' if not any(exits.values()) else 'failed',
                    exits=exits,seconds=time.monotonic()-started))
        registry('evaluating')
        if evaluator.wait() != 0 or any(exits.values()):
            raise RuntimeError('Training or evaluation failed; inspect preserved logs')
        for name in ('summarize_compact.py','verify_compact_campaign.py','compact_progress.py'):
            checked_script(name)
        report = ROOT/'reports/compact-v4'/folder.name
        for name in ('extension.json','supervisor-manifest.json','launch.json','evaluation-status.json','verification.json'):
            shutil.copyfile(folder/name,report/name)
        for seed in cfg['seeds']:
            target=report/f'seed-{seed}';target.mkdir(exist_ok=True)
            for name in ('diagnostics.jsonl','status.json'):
                shutil.copyfile(folder/f'seed-{seed}'/name,target/name)
        registry('complete')
        checked_script('export_compact.py')
        print('complete',folder,flush=True)
    except Exception:
        registry('failed')
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
