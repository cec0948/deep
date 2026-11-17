"""Supervise the newly authorized saved-state continuation through480."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import tarfile
import time

import numpy as np

from fullhouse_v3_common import ROOT,PYTHON,atomic_json,digest,process_start,alive
from controlled_poker_refit import utc,read,verify_files
from compact_lightweight_backup import equal_state,load_state
from snapshot_poker_average_continuation import verify_archive as verify_parent_archive
from train_compact_bounded import load_current as load_parent
from train_poker_average_extension import CHECKPOINT_FORMAT,commit_checkpoint,load_checkpoint
from poker_strategy_average import StrategyReplay
from run_poker_candidate_confirmation import prior_card_ranges,live_poker_jobs as prior_live_poker_jobs
from run_compact_refit_probe import check_card_range
from evaluate_compact_learning_curve import adoption_conditions
from evaluate_average_baseline import interval,verify_results

FOLDER=ROOT/'artifacts/average-extension'
RECOVERY=ROOT/'artifacts/average-continuation/20261117T032100Z/campaign'
PARENT=RECOVERY.parent
SEEDS=[342,343,344]
LABELS=['original120','parent420','current480','average361_480']
FIELDS=['public','mixed','noregrets']


def validate_config(cfg):
    fixed=dict(seeds=SEEDS,initial_iteration=420,iterations=480,stages=[480],
        traversals=256,buffer_capacity=100000,main_steps=1024,batch_size=512,
        opening_steps=2048,opening_max_steps=8192,max_opening_tv=.1,
        root_repetitions=4,learning_rate=.001,strategy_capacity=100000,reference_hands=512,
        average_fit_steps=8192,average_batch=512,average_max_replay_tv=.15,
        evaluation_deals=20000,evaluation_deal_seed=2037400000,evaluation_labels=LABELS,
        evaluation_fields=FIELDS,max_training_workers=3,max_evaluation_workers=3,
        training_seconds_per_worker=5400,evaluation_seconds_per_cell=2400,max_seconds=21600,
        automatic_retry=False,model_promotion=False,extend_after_480=False,
        resume_iterations={'342':420,'343':420,'344':420},weight_horizon=420,
        auxiliary_quality_error_stops_main=False,primary_candidate='current480',
        source_recovery_sha256='13eaca0fbc9d23815005ef6ea993ed503991221ab141ec0aafa1f5aab1fbd0bd',
        original360_recovery_sha256='65ec65f90cc1c0f73700ec4683d421baaaeb4bb82bb10c14fa2c9249fc504be8')
    if any(cfg.get(k)!=v or type(cfg.get(k))!=type(v) for k,v in fixed.items()):
        raise ValueError('The predeclared continuation budget changed')
    if cfg.get('average_width') != 128 or cfg.get('average_normalize') is not False:
        raise ValueError('Unsupported diagnosed average fitter')


def jobs():
    return [dict(id=f'{field}-seed-{seed}-{label}',label=label,seed=seed,field=field)
        for label in LABELS for seed in SEEDS for field in FIELDS]


def live_poker_jobs():
    found=prior_live_poker_jobs()
    names={'run_poker_average_baseline.py','train_poker_average_baseline.py',
        'run_poker_average_extension.py','train_poker_average_extension.py',
        'run_poker_average_continuation.py','train_poker_average_continuation.py',
        'evaluate_average_baseline.py','poker_average_diagnostics.py'}
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name)==os.getpid():continue
            args=(proc/'cmdline').read_bytes().decode().split('\0')
            if len(args)>1 and Path(args[1]).name in names:
                found.append(dict(pid=int(proc.name),process_start=process_start(int(proc.name)),command=args))
        except (OSError,UnicodeError):pass
    return found


def verify_inherited_source(recovery,campaign,manifest):
    # A completed campaign manifest also names comparison policies. Those are
    # verified at the parent and copied later under the new comparison labels.
    for name,sha in manifest['files'].items():
        if digest(recovery/name)!=sha:raise ValueError('Inherited parent dependency differs')
        if Path(name).parts[0] in ('source','tools') and digest(campaign/name)!=sha:
            raise ValueError('Inherited frozen source differs')


def prepare(study,config_path,gate_path):
    cfg=read(config_path);validate_config(cfg);gate=read(gate_path)
    if study.parent!=FOLDER or study.exists():raise ValueError('Use a fresh continuation folder')
    if live_poker_jobs():raise ValueError('Existing poker job; no duplicate launch')
    if gate.get('status')!='complete' or gate.get('passed') is not True:
        raise ValueError('Original mathematical/collector validation must remain valid')
    verify_files(gate['source_sha256'])
    diagnostic=Path(cfg['diagnostic_summary']);report=read(diagnostic)
    if not report.get('complete') or digest(diagnostic)!=cfg['diagnostic_summary_sha256']:
        raise ValueError('Frozen diagnostic results required')
    decision_path=Path(cfg['fitter_decision']);decision=read(decision_path)
    if digest(decision_path)!=cfg['fitter_decision_sha256'] or decision['diagnostic_summary_sha256']!=digest(diagnostic):
        raise ValueError('Fitter decision provenance changed')
    selected=decision['chosen']
    if any(cfg[k]!=selected[v] for k,v in [('average_width','width'),('average_normalize','normalize'),('average_fit_steps','steps')]):
        raise ValueError('Fitter differs from the documented unchanged-baseline decision')
    verify_files(read(diagnostic.parent/'plan.json')['frozen_files'])
    parent_summary=read(PARENT/'summary.json')
    if not parent_summary.get('complete') or parent_summary.get('completed_evaluations')!=36:
        raise ValueError('The completed420 parent is required')
    parent_ready=read(PARENT/'stage-ready.json')
    if not parent_ready.get('complete') or parent_ready.get('stage')!=420:
        raise ValueError('The completed420 checkpoint stage is required')
    if read(PARENT/'backup-ready.json')['sha256']!=cfg['source_recovery_sha256']:
        raise ValueError('The safe backup is not this420 parent')
    verify_files(read(PARENT/'plan.json')['frozen_files'])
    prior=prior_card_ranges(study)
    check_card_range(cfg['evaluation_deal_seed'],cfg['evaluation_deals'],{k:v['range'] for k,v in prior.items()})
    check_card_range(2037380000,20,{k:v['range'] for k,v in prior.items()})
    index=read(ROOT/'artifacts/offsite-backups.json')
    parents=[r for r in index['backups'] if not r.get('deleted_remote') and r.get('sha256')==cfg['source_recovery_sha256']]
    if len(parents)!=1:raise ValueError('Unique verified420 window-average backup required')
    backup=parents[0];archive=Path(backup['local_archive'])
    if (archive.stat().st_size!=backup['bytes'] or digest(archive)!=backup['sha256']
            or not backup.get('verified_owner_only') or not backup.get('verified_remote_size')
            or backup.get('status')!='complete' or not backup.get('restoration_verified')):
        raise ValueError('Safe420 archive does not match its record')
    if not verify_parent_archive(archive).get('passed'):raise ValueError('420 parent restore failed')
    study.mkdir(parents=True);campaign=study/'campaign';campaign.mkdir();(study/'logs').mkdir()
    for name in ('source','tools'):
        shutil.copytree(RECOVERY/name,campaign/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi','.git'))
    old_manifest=read(RECOVERY/'source-manifest.json')
    verify_inherited_source(RECOVERY,campaign,old_manifest)
    helpers=['train_poker_average_extension.py']
    for name in helpers:shutil.copyfile(ROOT/'scripts'/name,campaign/'tools'/name)
    frozen={str(p):digest(p) for name in ('source','tools') for p in (campaign/name).rglob('*') if p.is_file()}
    support=helpers+['run_poker_average_extension.py','preflight_poker_average_extension.py',
        'poker_strategy_average.py','poker_average_runtime.py','poker_average_inference.py',
        'evaluate_average_baseline.py','snapshot_poker_average_extension.py','plan_average_backup_retention.py',
        'train_poker_average_baseline.py','train_poker_average_continuation.py',
        'poker_average_diagnostics.py','snapshot_poker_average_continuation.py',
        'plan_average_extension_backup_retention.py','controlled_poker_refit.py','fullhouse_v3_common.py','compact_lightweight_backup.py']
    for name in support:
        p=ROOT/'scripts'/name;frozen[str(p)]=digest(p)
    for p in [Path(config_path).resolve(),Path(gate_path).resolve(),diagnostic,diagnostic.parent/'plan.json',decision_path,
              PARENT/'plan.json',PARENT/'summary.json',PARENT/'stage-ready.json',PARENT/'backup-ready.json']:
        frozen[str(p)]=digest(p)
    frozen.update(gate['source_sha256'])
    for name in ['compact-v4','bounded-continuation','rich-selfplay','controlled-refit','candidate-confirmation','average-baseline','average-continuation']:
        p=ROOT/'artifacts'/name/'current.json';frozen[str(p)]=digest(p)
    from evaluate_average_baseline import policy_dependency_hashes
    from poker_average_runtime import policy_for
    from poker_strategy_average import AverageNet
    policies={'original120':{},'parent420':{}};parent_paths={};proof=[]
    for seed in SEEDS:
        before,path=load_checkpoint(RECOVERY/f'seed-{seed}')
        if before['iteration']!=cfg['resume_iterations'][str(seed)]:raise ValueError('Recovery stage changed')
        if (str(path)!=parent_ready['seeds'][str(seed)]['checkpoint'] or
                any(digest(path/name)!=sha for name,sha in parent_ready['seeds'][str(seed)]['files'].items())):
            raise ValueError('The parent checkpoint differs from its verified completed stage')
        for key in ('traversals','buffer_capacity','main_steps','batch_size','opening_steps','opening_max_steps',
                    'max_opening_tv','root_repetitions','learning_rate','strategy_capacity','reference_hands'):
            if before['config'][key]!=cfg[key]:raise ValueError('Original learning budget changed:'+key)
        folder=campaign/f'seed-{seed}';folder.mkdir();(folder/'diagnostics.jsonl').touch()
        parent_paths[str(seed)]=dict(checkpoint=str(path),iteration=before['iteration'],
            files={p.name:digest(p) for p in path.iterdir() if p.is_file()})
        for p in list(path.iterdir())+[RECOVERY/f'seed-{seed}/current.json']:
            if p.is_file():frozen[str(p)]=digest(p)
        for label in policies:
            original=(path/'current.npz') if label=='parent420' else RECOVERY/f'seed-{seed}'/f'{label}.npz'
            target=folder/f'{label}.npz'
            shutil.copyfile(original,target);frozen[str(original)]=digest(original)
            policy=policy_for(target,campaign/'source');frozen.update(policy_dependency_hashes(policy,campaign/'source'))
            policies[label][str(seed)]=dict(path=str(target),sha256=digest(target))
        del before
    config=dict(cfg,frozen_files=frozen)
    manifest=dict(files={str(Path(k).relative_to(campaign)):v for k,v in frozen.items() if Path(k).is_relative_to(campaign)},
        driver_sha256=digest(ROOT/'scripts/train_poker_average_extension.py'),
        parent_source_recovery_sha256=backup['sha256'])
    atomic_json(campaign/'config.json',config);atomic_json(campaign/'source-manifest.json',manifest)
    for seed in SEEDS:
        before,path=load_checkpoint(RECOVERY/f'seed-{seed}')
        state=dict(before,config=config,source_manifest=manifest)
        avg=None
        if state['average_state'] is not None:
            avg=AverageNet(int(state['average_state']['body.0.weight'].shape[0]));avg.load_state_dict(state['average_state'])
        folder=campaign/f'seed-{seed}'
        commit_checkpoint(folder,state,lambda p,source=path:shutil.copyfile(source/'current.npz',p),avg)
        restored,_=load_checkpoint(folder)
        keys=('net','buffers','root_replay','rng','strategy_replay','reference_rng','average_state','averaging_window')
        checks={k:equal_state(before[k],restored[k]) for k in keys}
        if not all(checks.values()):raise ValueError('Authorized recovery changed learned state')
        proof.append(dict(seed=seed,iteration=before['iteration'],checks=checks,
            remaining_new_iterations=480-before['iteration'],current_policy_sha256=digest(path/'current.npz')))
        del before,state,restored,avg
    plan=dict(created_utc=utc(),study_kind=cfg['study_kind'],config=cfg,campaign=str(campaign),
        campaign_config_sha256=digest(campaign/'config.json'),frozen_files=frozen,parent_paths=parent_paths,
        policies=policies,connection_verification=proof,gate_report_sha256=digest(gate_path),gate_report=str(gate_path),
        diagnostic_summary_sha256=digest(diagnostic),original_regret_learner_unchanged=True,
        new_selfplay_iterations_per_seed={str(s):480-cfg['resume_iterations'][str(s)] for s in SEEDS},
        total_continuation_window_iterations=120,new_run_iterations=60,averaging_window=[361,480],average_before361_available=False,
        candidate_preselected='current480',average_is_secondary_diagnostic=True,
        auxiliary_quality_threshold_reference=.15,auxiliary_quality_error_stops_main=False,
        planned_evaluations=36,planned_hands=4320000,independent_card_blocks=20000,prior_card_ranges=prior,
        model_promoted=False,automatic_retry=False,full_paper_reproduction=False,
        legacy_information_loss_unresolved=True,authorization='User2026-11-17 explicitly asked to continue training after the completed420 results',
        rationale='Observe the next fixed60 rounds without retuning; preserve all420 state and separate adoption from finite auxiliary approximation error')
    atomic_json(study/'plan.json',plan)
    return plan


def summarize(rows,deals=20000):
    if set(rows)!={j['id'] for j in jobs()}:raise ValueError('All36 verified cells required')
    for field in FIELDS:
        if len({json.dumps(v.get('opponents'),sort_keys=True) for k,v in rows.items() if k.startswith(field+'-')})!=1:
            raise ValueError('Opponent composition changed')
    def arrays(label,field):
        a=[rows[f'{field}-seed-{s}-{label}']['deal_blocks'] for s in SEEDS]
        if any(len(v)!=deals or not all(math.isfinite(x) for x in v) for v in a):raise ValueError('Incomplete cards')
        return a
    def change(left,right,field):
        per=[[b-a for a,b in zip(x,y)] for x,y in zip(arrays(left,field),arrays(right,field))]
        return dict(**interval([statistics.mean(block) for block in zip(*per)]),
            per_seed=[dict(seed=s,**interval(v)) for s,v in zip(SEEDS,per)])
    primary={f:change('original120','current480',f) for f in FIELDS}
    average={f:change('original120','average361_480',f) for f in FIELDS}
    z=statistics.NormalDist().inv_cdf(1-.05/(2*9));diagnostics={}
    for left,right in [('parent420','current480'),('parent420','average361_480'),('current480','average361_480')]:
        for field in FIELDS:
            r=change(left,right,field);h=r['ci95_half_width']/1.96*z
            r.update(simultaneous_lower=r['mean']-h,simultaneous_upper=r['mean']+h,family_comparisons=9)
            diagnostics[f'{field}/{right}-minus-{left}']=r
    return dict(complete=True,completed_evaluations=36,total_hands=36*deals*6,independent_card_blocks=deals,
        absolute_scores={f'{f}/{l}':interval([statistics.mean(b) for b in zip(*arrays(l,f))]) for l in LABELS for f in FIELDS},
        current_minus_original120=primary,adoption_against_original120=adoption_conditions(primary),
        average_minus_original120=average,average_diagnostic_conditions=adoption_conditions(average),
        diagnostic_comparisons=diagnostics,new_selfplay_iterations_per_seed={'342':60,'343':60,'344':60},averaging_window=[361,480],
        model_promoted=False,full_paper_reproduction=False,
        statistical_unit='One fresh shared card block; seats and fixed training seeds averaged inside block',
        limitations=['Legacy51 information loss remains','Fixed-window average is a neural approximation',
            'Finite-seed fixed-opponent evidence is not professional or general6player strength'])


def run(study,config_path,gate_path,prepared=False):
    study=Path(study).resolve();FOLDER.mkdir(parents=True,exist_ok=True)
    lock=(FOLDER/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};rows={};workers={};began=time.monotonic();handlers={};plan=None
    try:
        if (FOLDER/'current.json').exists():
            old=read(FOLDER/'current.json')
            if alive(old['pid'],old['process_start']):raise ValueError('Average baseline supervisor is already alive')
        if live_poker_jobs():raise ValueError('Other live poker learning/evaluation jobs exist')
        if prepared and any((study/n).exists() for n in ('status.json','summary.json','failure.json')):
            raise ValueError('Do not automatically restart an attempted baseline')
        plan=read(study/'plan.json') if prepared else prepare(study,config_path,gate_path)
        preflight=read(study/'preflight.json')
        if preflight.get('passed') is not True or preflight.get('plan_sha256')!=digest(study/'plan.json'):
            raise ValueError('Actual-parent full-update and trusted-engine preflight must pass before launch')
        verify_files(plan['frozen_files']);campaign=Path(plan['campaign']);cfg=plan['config']
        if digest(campaign/'config.json')!=plan['campaign_config_sha256']:raise ValueError('Campaign config changed')
        atomic_json(FOLDER/'current.json',dict(folder=str(study),pid=os.getpid(),process_start=process_start(os.getpid()),
            started_utc=utc(),purpose=cfg['study_kind']))
        public=ROOT/'reports/average-extension'/study.name;public.mkdir(parents=True,exist_ok=True)
        atomic_json(public/'plan.json',plan)
        atomic_json(public/'preflight.json',preflight)
        last_write=0
        def publish(phase,state='running',error=None,force=False):
            nonlocal last_write
            live={k:dict(pid=v['p'].pid,process_start=v['token'],command=v['command']) for k,v in active.items()
                  if v['p'].poll() is None and alive(v['p'].pid,v['token'])}
            progress={}
            for seed in SEEDS:
                path=campaign/f'seed-{seed}/status.json'
                progress[str(seed)]=read(path) if path.exists() else dict(iteration=cfg['resume_iterations'][str(seed)],status='prepared')
            record=dict(status=state,phase=phase,observed_utc=utc(),run=study.name,
                study_kind=cfg['study_kind'],supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),
                training_progress=progress,completed_training_workers=len(workers),planned_training_workers=3,
                new_selfplay_iterations_per_seed={s:max(0,r.get('iteration',r.get('last_committed_iteration',cfg['resume_iterations'][s]))-cfg['resume_iterations'][s]) for s,r in progress.items()},
                planned_new_selfplay_iterations_per_seed={s:480-i for s,i in cfg['resume_iterations'].items()},
                training_complete=len(workers)==3,
                averaging_window=[361,480],historical_average_before361_available=False,
                completed_evaluations=len(rows),planned_evaluations=36,verified_hands=len(rows)*20000*6,
                planned_hands=4320000,independent_card_blocks=20000,running=live,error=error,
                small_game_average_gate_passed=True,full_paper_reproduction=False,
                model_promoted=False,automatic_retry=False,study_complete=state=='complete',
                evaluation_complete=len(rows)==36,elapsed_seconds=time.monotonic()-began,max_seconds=cfg['max_seconds'],
                backup_ready=(study/'backup-ready.json').exists(),
                next_action='Inspect failure; no automatic retry' if state=='failed' else
                    'Analyze declared comparisons; no automatic extension or model replacement' if state=='complete'
                    else 'Finish saved-state continuation through480 and36cell evaluation; average TV is diagnostic')
            atomic_json(study/'status.json',record)
            if force or time.monotonic()-last_write>=15 or state!='running':
                atomic_json(ROOT/'reports/average-extension-status.json',record)
                atomic_json(ROOT/'reports/training-neural-status.json',record);last_write=time.monotonic()
        def stop(*args):raise InterruptedError('Baseline interrupted; no automatic retry')
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop)
        def launch(key,command,kind,**fields):
            log=(study/'logs'/(key+'.log')).open('w')
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            token=process_start(p.pid)
            if token is None:raise RuntimeError('Owned process exited before registration')
            active[key]=dict(p=p,token=token,command=command,kind=kind,log=log,started=time.monotonic(),**fields)
        publish('preparing',force=True)
        for seed in SEEDS:
            launch(f'train-{seed}',[str(PYTHON),str(ROOT/'scripts/train_poker_average_extension.py'),
                '--campaign',str(campaign),'--seed',str(seed)],'training',seed=seed)
        while active:
            if time.monotonic()-began>cfg['max_seconds']:raise TimeoutError('Six-hour total baseline limit exhausted')
            verify_files(plan['frozen_files'])
            if digest(campaign/'config.json')!=plan['campaign_config_sha256']:raise ValueError('Campaign config changed')
            for key,item in list(active.items()):
                if time.monotonic()-item['started']>cfg['training_seconds_per_worker']+60:
                    raise TimeoutError('Training worker exceeded fixed budget')
                code=item['p'].poll()
                if code is None:continue
                item['log'].close();del active[key]
                if code!=0:raise RuntimeError(f'{key} failed with exit{code}; no automatic retry')
                state,path=load_checkpoint(campaign/f"seed-{item['seed']}")
                if state['iteration']!=480 or state['average_state'] is None:
                    raise ValueError('Training did not produce the declared final current and average')
                workers[str(item['seed'])]=dict(iteration=480,checkpoint=str(path),
                    files={p.name:digest(p) for p in path.iterdir() if p.is_file()},
                    average_fit=state['average_fit']);del state
            publish('training');time.sleep(3)
        # All3workers have stopped before evaluators use all3cores.
        atomic_json(study/'stage-ready.json',dict(complete=True,stage=480,seeds=workers,
            averaging_window=[361,480],training_complete=True,evaluation_complete=False))
        from snapshot_poker_average_extension import create_archive
        publish('verifying_recovery',force=True)
        create_archive(study)
        policies=plan['policies']
        for label,name in [('current480','current.npz'),('average361_480','average.npz')]:
            policies[label]={str(s):dict(path=str(Path(workers[str(s)]['checkpoint'])/name),
                sha256=workers[str(s)]['files'][name]) for s in SEEDS}
        atomic_json(study/'evaluation-plan.json',dict(policies=policies,jobs=jobs(),
            evaluation_deal_seed=cfg['evaluation_deal_seed'],deals=20000,source=str(campaign/'source')))
        pending=deque(jobs())
        while pending or active:
            if time.monotonic()-began>cfg['max_seconds']:raise TimeoutError('Six-hour total baseline limit exhausted')
            verify_files(plan['frozen_files'])
            for key,item in list(active.items()):
                if time.monotonic()-item['started']>cfg['evaluation_seconds_per_cell']+30:
                    raise TimeoutError('Evaluation exceeded fixed cell budget')
                code=item['p'].poll()
                if code is None:continue
                item['log'].close();del active[key]
                if code!=0:raise RuntimeError(f'{key} failed with exit{code}; no automatic retry')
                j=item['job'];rows[key]=verify_results(study/'evaluation'/key,replay=False,
                    expected=dict(candidate_sha256=policies[j['label']][str(j['seed'])]['sha256'],
                        deals=20000,deal_seed=cfg['evaluation_deal_seed'],field=j['field']))
                if not rows[key]['terminal_payoffs_replayed']:raise ValueError('Trusted action replay required')
            while pending and len(active)<3:
                j=pending.popleft();policy=policies[j['label']][str(j['seed'])]
                if digest(policy['path'])!=policy['sha256']:raise ValueError('Evaluation policy changed')
                launch(j['id'],[str(PYTHON),str(ROOT/'scripts/evaluate_average_baseline.py'),
                    '--model',policy['path'],'--source',str(campaign/'source'),'--field',j['field'],
                    '--deals','20000','--deal-seed',str(cfg['evaluation_deal_seed']),
                    '--timeout',str(cfg['evaluation_seconds_per_cell']),
                    '--output',str(study/'evaluation'/j['id'])],'evaluation',job=j)
            publish('evaluating');time.sleep(3)
        final=summarize(rows);final['final_average_quality']={s:w['average_fit'] for s,w in workers.items()};final.update(completed_utc=utc(),result_files={k:digest(study/'evaluation'/k/'results.json') for k in rows})
        atomic_json(study/'evaluation-summary.json',final)
        atomic_json(study/'summary.json',dict(complete=True,completed_utc=utc(),run=study.name,
            training_complete=True,evaluation_complete=True,completed_evaluations=36,total_hands=4320000,
            new_selfplay_iterations_per_seed={'342':60,'343':60,'344':60},averaging_window=[361,480],
            original_four_conditions_pass=final['adoption_against_original120']['original_four_conditions_pass'],
            evaluation_summary_sha256=digest(study/'evaluation-summary.json'),model_promoted=False,
            stop_reason='Fixed480budget and all36evaluations complete; no automatic extension or model replacement'))
        for name in ('summary.json','evaluation-summary.json','backup-ready.json','stage-ready.json'):
            atomic_json(public/name,read(study/name))
        publish('complete','complete',force=True)
    except BaseException as error:
        for item in active.values():
            if alive(item['p'].pid,item['token']):
                os.killpg(item['p'].pid,signal.SIGTERM)
        for item in active.values():
            try:item['p'].wait(timeout=8)
            except subprocess.TimeoutExpired:
                if alive(item['p'].pid,item['token']):os.killpg(item['p'].pid,signal.SIGKILL)
                item['p'].wait()
            item['log'].close()
        failure=dict(observed_utc=utc(),error_type=type(error).__name__,error=str(error),automatic_retry=False,
            model_promoted=False,last_safe_backup='Verified420 window-average recovery; inspect private index for newer completed recovery')
        if plan is not None:
            atomic_json(study/'failure.json',failure)
            publish('stopped','failed',error=str(error),force=True)
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',action='store_true')
    a=p.parse_args()
    if a.prepare_only:prepare(a.output.resolve(),a.config,a.gate)
    else:run(a.output,a.config,a.gate,a.prepared)
