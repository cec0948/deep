"""Supervise the newly authorized240→300→360 current-only learning study."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
import time

from bounded_poker_study_support import (check_frozen,check_learning_config,read,snapshot,utc,validate_config,verify_stage)
from compact_lightweight_backup import equal_state,load_state,verify_archive
from fullhouse_v3_common import ROOT,PYTHON,atomic_json,digest,process_start
from run_compact_learning_curve import DiagnosticGuard,terminate_owned
from run_compact_refit_probe import check_card_range,previous_ranges
from train_compact_bounded import commit_checkpoint,load_current

HELPERS=('run_bounded_poker_study.py','train_compact_bounded.py','evaluate_bounded_poker_study.py',
    'publish_bounded_poker_progress.py','bounded_poker_study_support.py','compact_lightweight_backup.py',
    'fullhouse_v3_common.py','run_compact_learning_curve.py','run_compact_refit_probe.py',
    'evaluate_compact_learning_curve.py','evaluate_compact.py','fullhouse_v4_runtime.py','fullhouse_v3_runtime.py',
    'fullhouse_runtime.py','regret_partition.py','publish_continuation_progress.py','extend_compact_run.py',
    'prepare_compact_run.py')
PROTECTED=('artifacts/compact-v4/current.json','artifacts/retraining-loop.json',
           'artifacts/continuation-study/current.json','artifacts/refit-probe/current.json','artifacts/foundation-study/current.json')


def archive_record(index,iteration):
    rows=[x for x in index['backups'] if not x.get('deleted_remote') and x.get('kind')=='training-checkpoint'
          and set(x.get('iterations',{}).values())=={iteration}]
    if not rows:raise ValueError('Verified source backup not indexed')
    row=max(rows,key=lambda x:x['recorded_at'])
    if not row.get('verified_remote_size') or not row.get('verified_owner_only'):raise ValueError('Source backup verification incomplete')
    path=Path(row['local_archive']).resolve()
    if path.parent!=ROOT/'artifacts/exports' or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
        raise ValueError('Actual source archive missing or changed')
    return row


def extract_baseline120(record,campaign):
    names={};manifest=None
    with tarfile.open(record['local_archive'],'r|gz') as tar:
        for m in tar:
            if m.name=='checkpoint-backup-manifest.json':manifest=json.load(tar.extractfile(m))
            for seed in [342,343,344]:
                if m.name.endswith(f'/seed-{seed}/current_00120.npz'):
                    target=campaign/f'seed-{seed}/current_00120.npz'
                    with tar.extractfile(m) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
                    names[seed]=(m.name,target)
    if set(names)!={342,343,344} or manifest is None:raise ValueError('Original120 baseline missing')
    for name,path in names.values():
        if digest(path)!=manifest['files'][name]['sha256']:raise ValueError('Original120 baseline hash mismatch')
    return {str(seed):digest(path) for seed,(_,path) in names.items()}


def prepare(output,config,audit):
    cfg=read(config);validate_config(cfg);audit=read(audit)
    if not audit.get('passed') or not audit.get('continuation_equivalence_passed'):
        raise ValueError('The bounded-continuation integration audit must pass first')
    check_frozen(audit['source_hashes'])
    prior=previous_ranges(output);check_card_range(cfg['evaluation_deal_seed'],cfg['evaluation_deals'],prior)
    index=read(ROOT/'artifacts/offsite-backups.json');parent=archive_record(index,240);baseline=archive_record(index,120)
    if parent.get('checkpoint_mode')!='compact-bounded-replay-recovery-v1' or not parent.get('warm_start_state_verified'):
        raise ValueError('The verified lightweight240 recovery is required')
    protected={str(ROOT/n):digest(ROOT/n) for n in PROTECTED}
    output.mkdir(parents=True,exist_ok=False);recovery=output/'parent-recovery';verify_archive(parent['local_archive'],recovery)
    original=load_state(recovery/'seed-342');original_manifest=original['source_manifest'];parent_cfg=original['config']
    candidates=[p.parent for p in (recovery/'frozen').rglob('source-manifest.json')
                if read(p)==original_manifest and (p.parent/'source').is_dir() and (p.parent/'tools').is_dir()]
    if len(candidates)!=1:raise ValueError('Self-contained frozen training source is ambiguous')
    parent_source=candidates[0];campaign=output/'campaign';campaign.mkdir();(output/'logs').mkdir()
    for name in ('source','tools'):
        shutil.copytree(parent_source/name,campaign/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
    for name,sha in original_manifest['files'].items():
        if digest(campaign/name)!=sha:raise ValueError('Restored algorithm source changed')
    worker_cfg=dict(parent_cfg,iterations=360,initial_iteration=240,stages=cfg['stages'],
        evaluation_iterations=cfg['evaluation_iterations'],evaluation_deals=cfg['evaluation_deals'],
        evaluation_deal_seed=cfg['evaluation_deal_seed'],evaluation_fields=cfg['evaluation_fields'],
        deployment_average='current_only_no_historical_mixture',seconds_per_training_worker=cfg['training_seconds_per_stage'])
    check_learning_config(parent_cfg,worker_cfg)
    for name in ('compact_lightweight_backup.py','train_compact_bounded.py','bounded_poker_study_support.py','fullhouse_v3_common.py'):
        shutil.copyfile(ROOT/'scripts'/name,campaign/'tools'/name)
    root_files={str(ROOT/'scripts'/name):digest(ROOT/'scripts'/name) for name in HELPERS}
    files={str(p.relative_to(campaign)):digest(p) for n in ('source','tools') for p in sorted((campaign/n).rglob('*')) if p.is_file()}
    manifest=dict(files=files,driver_sha256=digest(ROOT/'scripts/train_compact_bounded.py'),root_files=root_files,
        source_recovery_sha256=parent['sha256'],historical_mixture_available=False)
    atomic_json(campaign/'config.json',worker_cfg);atomic_json(campaign/'source-manifest.json',manifest)
    preserved={};models240={}
    for seed in cfg['seeds']:
        before=load_state(recovery/f'seed-{seed}')
        if before['iteration']!=240 or before['config']!=parent_cfg or before['source_manifest']!=original_manifest:
            raise ValueError('Recovered seed provenance differs')
        state=dict(before,config=worker_cfg,source_manifest=manifest)
        target=campaign/f'seed-{seed}';target.mkdir();(target/'worker.lock').touch();(target/'diagnostics.jsonl').touch()
        commit_checkpoint(campaign,seed,state,lambda p,seed=seed:shutil.copyfile(recovery/f'seed-{seed}/current.npz',p),{120,240,300,360})
        restored,_=load_current(campaign,seed)
        if not equal_state({k:v for k,v in before.items() if k not in ('config','source_manifest')},
                           {k:v for k,v in restored.items() if k not in ('config','source_manifest')}):
            raise ValueError('Imported network/replay/root/RNG changed')
        models240[str(seed)]=digest(target/'current_00240.npz')
        if models240[str(seed)]!=digest(recovery/f'seed-{seed}/current.npz'):raise ValueError('Parent policy bytes changed')
        preserved[str(seed)]=dict(iteration=240,numeric_state_identical=True,current_policy_identical=True,
            replay_rows=[b['count'] for b in restored['buffers']])
        atomic_json(target/'status.json',dict(status='complete',iteration=240,through=240))
        del state,before,restored
    models120=extract_baseline120(baseline,campaign)
    ready=dict(updated_utc=utc(),completed_stages=[],checkpoints={
        '120':dict(iteration=120,models=models120,original_baseline_verified=True),
        '240':dict(iteration=240,models=models240,parent_recovery_verified=True)})
    atomic_json(output/'stage-ready.json',ready);atomic_json(output/'config.json',cfg)
    atomic_json(output/'connection-audit.json',audit)
    preparation=dict(complete=True,prepared_utc=utc(),parent_iteration=240,source_recovery_sha256=parent['sha256'],
        parent_config=parent_cfg,preserved=preserved,learning_hyperparameters_unchanged=True,
        historical_policy_files_created=False,original_registries_unchanged=True)
    atomic_json(output/'preparation.json',preparation)
    frozen=dict(root_files);frozen.update(protected)
    for n in ('config.json','connection-audit.json','preparation.json'):frozen[str(output/n)]=digest(output/n)
    for n in ('config.json','source-manifest.json'):frozen[str(campaign/n)]=digest(campaign/n)
    frozen.update({str(campaign/n):sha for n,sha in files.items()})
    for seed in cfg['seeds']:
        for iteration in (120,240):
            p=campaign/f'seed-{seed}'/f'current_{iteration:05d}.npz';frozen[str(p)]=digest(p)
    plan=dict(created_utc=utc(),config=cfg,campaign=str(campaign),parent_iteration=240,
        source_recovery_archive=parent['local_archive'],source_recovery_sha256=parent['sha256'],
        baseline120_archive=baseline['local_archive'],baseline120_sha256=baseline['sha256'],
        frozen_files=frozen,protected_registry_files=protected,prior_card_ranges=prior,
        planned_evaluations=36,planned_hands=4320000,no_model_promotion=True,no_automatic_retry=True,
        method_audit_sha256=digest(output/'connection-audit.json'),preparation_sha256=digest(output/'preparation.json'))
    atomic_json(output/'plan.json',plan)
    return cfg,worker_cfg,plan


def run(output,config,audit,prepared=False):
    output=Path(output).resolve()
    if output.parent!=ROOT/'artifacts/bounded-continuation' or not re.fullmatch(r'\d{8}T\d{6,12}Z',output.name):
        raise ValueError('A new isolated timestamp directory is required')
    output.parent.mkdir(parents=True,exist_ok=True);lock=(output.parent/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=output.parent/'current.json'
    if registry.exists():
        old=read(registry)
        if process_start(old.get('pid'))==old.get('process_start') and old.get('process_start'):
            raise RuntimeError('A bounded study is already running')
    if prepared:
        cfg=read(output/'config.json');validate_config(cfg);plan=read(output/'plan.json')
        if (cfg!=read(config) or digest(output/'connection-audit.json')!=digest(audit)
                or plan['preparation_sha256']!=digest(output/'preparation.json') or plan['config']!=cfg):
            raise ValueError('Prepared study inputs changed')
        check_frozen(plan['frozen_files']);worker_cfg=read(output/'campaign/config.json')
    else:cfg,worker_cfg,plan=prepare(output,Path(config),Path(audit))
    campaign=output/'campaign';plan_sha=digest(output/'plan.json')
    atomic_json(registry,dict(folder=str(output),pid=os.getpid(),process_start=process_start(os.getpid()),
        started_utc=utc(),purpose='Authorized240-to360 online self-play from compact recovery; current policies only'))
    began=time.monotonic();active={};phase='prepared';stage=240;completed=[];backups={}
    guard=DiagnosticGuard(campaign,cfg['seeds'],initial=240)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def status(state,**extra):
        atomic_json(output/'status.json',dict(status=state,phase=phase,stage=stage,updated_utc=utc(),
            supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),seconds=time.monotonic()-began,
            plan_sha256=plan_sha,parent_iteration=240,planned_iterations=360,completed_stages=completed[:],
            running={n:dict(pid=x['process'].pid,process_start=x['start']) for n,x in active.items() if x['process'].poll() is None},
            checkpoint_iterations={str(s):read(campaign/f'seed-{s}/current.json')['iteration'] for s in cfg['seeds']},
            latest_completed_rounds={str(s):guard.state[s]['iteration'] for s in cfg['seeds']},
            original_registries_unchanged=True,backups=backups.copy(),**extra))
    def launch(name,command):
        log=(output/'logs'/(name.replace('/','-')+'.log')).open('w')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[name]=dict(process=child,start=process_start(child.pid),log=log,started=time.monotonic());return child
    def consume(name):
        item=active.pop(name);item['log'].close()
        if item['process'].returncode:raise RuntimeError('Subprocess failed: '+name)
    def monitor():
        if time.monotonic()-began>cfg['max_seconds']:raise TimeoutError('Fixed study time budget exhausted')
        check_frozen(plan['frozen_files'])
        if digest(output/'plan.json')!=plan_sha:raise ValueError('Study plan changed')
        guard.check()
        for name,item in active.items():
            if item['process'].poll() not in (None,0):raise RuntimeError('Subprocess failed; no retry: '+name)
        status('running')
    def stop(*unused):raise InterruptedError('Bounded study interrupted')
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        atomic_json(campaign/'launch.json',dict(status='running',through=300));status('running')
        evaluator=launch('evaluation',[str(PYTHON),str(ROOT/'scripts/evaluate_bounded_poker_study.py'),'--study',str(output)])
        for stage in cfg['stages']:
            phase='training';started=time.monotonic()
            workers={seed:launch(f'train/{seed}',[str(PYTHON),str(ROOT/'scripts/train_compact_bounded.py'),
                '--run-dir',str(campaign),'--seed',str(seed),'--through',str(stage)]) for seed in cfg['seeds']}
            atomic_json(campaign/'launch.json',dict(status='running',through=stage,
                pids={str(s):p.pid for s,p in workers.items()},process_starts={str(s):process_start(p.pid) for s,p in workers.items()}))
            while any(p.poll() is None for p in workers.values()):
                if time.monotonic()-started>cfg['training_seconds_per_stage']+120:raise TimeoutError('Training stage timed out')
                monitor();time.sleep(3)
            for seed in cfg['seeds']:consume(f'train/{seed}')
            guard.finished(stage);phase='verifying_stage';status('running');check_frozen(plan['frozen_files'])
            verification=verify_stage(campaign,stage,worker_cfg);atomic_json(output/f'stage-{stage}-verification.json',verification)
            ready=read(output/'stage-ready.json');completed.append(stage);ready['completed_stages']=completed[:]
            ready['checkpoints'][str(stage)]=dict(iteration=stage,verification_sha256=digest(output/f'stage-{stage}-verification.json'),
                models={str(x['seed']):x['current_sha256'] for x in verification['seeds']})
            ready['updated_utc']=utc();atomic_json(output/'stage-ready.json',ready)
            phase='snapshotting';status('running')
            exporter=launch('snapshot',[str(PYTHON),str(ROOT/'scripts/bounded_poker_study_support.py'),
                '--study',str(output),'--stage',str(stage)])
            while exporter.poll() is None:monitor();time.sleep(3)
            consume('snapshot');backups[str(stage)]=read(output/f'stage-{stage}-backup.json');monitor()
        atomic_json(campaign/'launch.json',dict(status='complete',through=360))
        phase='evaluating';status('running')
        while evaluator.poll() is None:monitor();time.sleep(3)
        consume('evaluation');phase='verifying';status('running');check_frozen(plan['frozen_files'])
        summary=read(output/'evaluation-summary.json')
        if (not summary.get('complete') or summary['completed_evaluations']!=36 or summary['total_hands']!=4320000
                or not summary.get('raw_hands_verified') or not summary.get('source_provenance_verified')):
            raise ValueError('Full evaluation verification is incomplete')
        for item in backups.values():
            if digest(item['archive'])!=item['sha256']:raise ValueError('Verified stage archive changed')
        atomic_json(output/'summary.json',dict(complete=True,completed_utc=utc(),seconds=time.monotonic()-began,
            parent_iteration=240,selfplay_iteration=360,additional_selfplay_iterations=120,completed_stages=completed,
            plan_sha256=plan_sha,completed_evaluations=36,total_hands=4320000,
            evaluation_summary_sha256=digest(output/'evaluation-summary.json'),raw_hands_verified=True,
            source_provenance_verified=True,original_registries_unchanged=True,no_model_promotion=True,
            no_automatic_retry=True,backups=backups))
        phase='complete';status('complete')
    except BaseException as error:
        previous=phase;phase='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed'
        status(phase,error=str(error),failure_stage=previous)
        atomic_json(campaign/'launch.json',dict(status=phase,error=str(error)));raise
    finally:
        terminate_owned(active)
        for item in active.values():item['log'].close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run-prepared',action='store_true')
    a=p.parse_args()
    if a.prepare_only:
        output=a.output.resolve()
        if output.parent!=ROOT/'artifacts/bounded-continuation' or not re.fullmatch(r'\d{8}T\d{6,12}Z',output.name):
            raise ValueError('Preparation requires an isolated timestamp directory')
        prepare(output,a.config,a.audit);print(json.dumps(dict(prepared=True,training_started=False,output=str(output))))
    else:run(a.output,a.config,a.audit,a.run_prepared)
