"""Continue only completed, evaluated campaigns under predeclared progress rules."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from extend_compact_run import extend
from fullhouse_v3_common import ROOT,PYTHON,atomic_json,process_start


def extension_allowed(iteration, decision):
    checks=decision['conditions']
    required=['public_lower_above_zero','at_least_two_public_seeds_improve']
    if iteration>=120:
        required+=['mixed_lower_above_minus_25','noregrets_lower_above_minus_50']
    return all(checks.get(key) is True for key in required)


def watch(folder, maximum):
    lock=(ROOT/'artifacts/retraining-loop.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    folder=Path(folder).resolve();runs=[str(folder)];child=None
    def record(phase,**extra):
        atomic_json(ROOT/'artifacts/retraining-loop.json',dict(status=phase,current=str(folder),runs=runs,
                    max_iterations=maximum,controller_pid=os.getpid(),controller_start=process_start(os.getpid()),
                    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    try:
        while True:
            record('watching')
            while True:
                path=folder/'campaign-status.json'
                state=json.loads(path.read_text()) if path.exists() else {'status':'starting'}
                if state['status']=='complete':break
                if state['status'] in ('failed','interrupted'):
                    record('paused_for_failure',campaign=state);return
                if child and child.poll() is not None:
                    record('paused_for_failure',error='Campaign process exited before completion');return
                time.sleep(15)
            cfg=json.loads((folder/'config.json').read_text());iteration=cfg['iterations']
            decision=json.loads((folder/'progress-decision.json').read_text())
            if not extension_allowed(iteration,decision):
                record('paused_for_analysis',decision=decision);return
            if iteration>=maximum:
                record('bounded_plan_complete',decision=decision);return
            through=min(iteration+60,maximum)
            next_folder=ROOT/'artifacts/compact-v4'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            # Distinct, fixed card seeds per stage; all candidates within one stage share cards.
            extend(folder,next_folder,through,2026111201+through//60)
            folder=next_folder;runs.append(str(folder))
            with (folder/'supervisor.log').open('w') as log:
                child=subprocess.Popen([str(PYTHON),str(ROOT/'scripts/run_compact_retraining.py'),str(folder),'--resume'],
                    cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(folder/'launcher.json',dict(pid=child.pid,process_start=process_start(child.pid),folder=str(folder)))
            record('starting_extension',through=through)
    except BaseException as exc:
        record('controller_failed',error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--max-iterations',type=int,default=240);args=parser.parse_args()
    if args.max_iterations%6:parser.error('Use a checkpoint boundary')
    watch(args.folder,args.max_iterations)
