"""Supervise one budget-preserved360->420 continuation and window-average test."""
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
from compact_lightweight_backup import equal_state,verify_archive,load_state
from train_compact_bounded import load_current as load_parent
from train_poker_average_baseline import CHECKPOINT_FORMAT,commit_checkpoint,load_checkpoint
from poker_strategy_average import StrategyReplay
from run_poker_candidate_confirmation import prior_card_ranges,live_poker_jobs
from run_compact_refit_probe import check_card_range
from evaluate_compact_learning_curve import adoption_conditions
from evaluate_average_baseline import interval,verify_results

FOLDER=ROOT/'artifacts/average-baseline'
PARENT=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign'
SEEDS=[342,343,344]
LABELS=['original120','parent360','current420','average361_420']
FIELDS=['public','mixed','noregrets']


def validate_config(cfg):
    fixed=dict(seeds=SEEDS,initial_iteration=360,iterations=420,stages=[372,420],
        traversals=256,buffer_capacity=100000,main_steps=1024,batch_size=512,
        opening_steps=2048,opening_max_steps=8192,max_opening_tv=.1,
        root_repetitions=4,learning_rate=.001,strategy_capacity=100000,reference_hands=512,
        average_fit_steps=8192,average_batch=512,average_max_replay_tv=.15,
        evaluation_deals=20000,evaluation_deal_seed=2037200000,evaluation_labels=LABELS,
        evaluation_fields=FIELDS,max_training_workers=3,max_evaluation_workers=3,
        training_seconds_per_worker=5400,evaluation_seconds_per_cell=2400,max_seconds=21600,
        automatic_retry=False,model_promotion=False,extend_after_420=False)
    if any(cfg.get(k)!=v or type(cfg.get(k))!=type(v) for k,v in fixed.items()):
        raise ValueError('The predeclared baseline/average budget changed')


def jobs():
    return [dict(id=f'{field}-seed-{seed}-{label}',label=label,seed=seed,field=field)
        for label in LABELS for seed in SEEDS for field in FIELDS]


def prepare(study,config_path,gate_path):
    cfg=read(config_path);validate_config(cfg);gate=read(gate_path)
    if gate.get('status')!='complete' or gate.get('passed') is not True:
        raise ValueError('All predeclared strategy gates must pass before6player learning')
    verify_files(gate['source_sha256'])
    prior=prior_card_ranges(study)
    check_card_range(cfg['evaluation_deal_seed'],cfg['evaluation_deals'],{k:v['range'] for k,v in prior.items()})
    index=read(ROOT/'artifacts/offsite-backups.json')
    parents=[r for r in index['backups'] if not r.get('deleted_remote') and r.get('sha256')==cfg['source_recovery_sha256']]
    if len(parents)!=1:raise ValueError('Unique verified compact360 recovery required')
    parent_backup=parents[0];archive=Path(parent_backup['local_archive'])
    if (archive.stat().st_size!=parent_backup['bytes'] or digest(archive)!=parent_backup['sha256']
            or not parent_backup.get('verified_owner_only') or not parent_backup.get('verified_remote_size')):
        raise ValueError('Actual safe parent archive does not match its verified record')
    recovery=verify_archive(archive)
    if not recovery.get('passed'):raise ValueError('Parent recovery failed actual extraction')
    with tarfile.open(archive,'r:gz') as tar:
        parent_manifest=json.load(tar.extractfile('recovery-manifest.json'))
    study.mkdir(parents=True,exist_ok=False);campaign=study/'campaign';campaign.mkdir();(study/'logs').mkdir()
    for name in ('source','tools'):
        shutil.copytree(PARENT/name,campaign/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi','.git'))
    base_manifest=read(PARENT/'source-manifest.json')
    for name,sha in base_manifest['files'].items():
        if digest(campaign/name)!=sha:raise ValueError('Inherited learning source differs')
    # Only these new support modules are added. Original kernel copies remain
    # byte-identical, including the legacy projection and opening specialist.
    helpers=['poker_strategy_average.py','poker_average_inference.py','train_poker_average_baseline.py',
        'compact_lightweight_backup.py','fullhouse_v3_common.py','controlled_poker_refit.py']
    for name in helpers:shutil.copyfile(ROOT/'scripts'/name,campaign/'tools'/name)
    frozen={str(p):digest(p) for name in ('source','tools') for p in (campaign/name).rglob('*') if p.is_file()}
    new_helpers=helpers+['run_poker_average_baseline.py','validate_poker_strategy_average.py',
        'poker_average_runtime.py','evaluate_average_baseline.py','snapshot_poker_average_baseline.py',
        'plan_average_backup_retention.py','preflight_poker_average_baseline.py']
    for name in new_helpers:
        path=ROOT/'scripts'/name;frozen[str(path)]=digest(path)
    frozen.update({str(Path(config_path).resolve()):digest(config_path),str(Path(gate_path).resolve()):digest(gate_path)})
    frozen.update(gate['source_sha256'])
    protected=['compact-v4','bounded-continuation','rich-selfplay','controlled-refit','candidate-confirmation']
    for name in protected:
        p=ROOT/'artifacts'/name/'current.json';frozen[str(p)]=digest(p)
    from evaluate_average_baseline import policy_dependency_hashes
    from poker_average_runtime import policy_for
    policies={'original120':{},'parent360':{}};proof=[]
    parent_paths={};config=dict(cfg,frozen_files=frozen)
    manifest=dict(files={str(p.relative_to(campaign)):sha for p,sha in
        ((Path(k),v) for k,v in frozen.items()) if p.is_relative_to(campaign)},
        driver_sha256=digest(ROOT/'scripts/train_poker_average_baseline.py'),
        parent_source_recovery_sha256=parent_backup['sha256'])
    atomic_json(campaign/'config.json',config);atomic_json(campaign/'source-manifest.json',manifest)
    for seed in SEEDS:
        before,parent_path=load_parent(PARENT,seed)
        if before['iteration']!=360:raise ValueError('Original parent is not360')
        if digest(parent_path/'current.npz')!=parent_manifest['files'][f'seed-{seed}/current.npz']['sha256']:
            raise ValueError('Actual parent model differs from the verified safe archive')
        for k in ('traversals','buffer_capacity','main_steps','batch_size','opening_steps','opening_max_steps',
                  'max_opening_tv','root_repetitions','learning_rate'):
            if before['config'][k]!=cfg[k]:raise ValueError('Inherited regret learning budget changed: '+k)
        folder=campaign/f'seed-{seed}';folder.mkdir();(folder/'diagnostics.jsonl').touch()
        parent_paths[str(seed)]=dict(checkpoint=str(parent_path),files={p.name:digest(p) for p in parent_path.iterdir() if p.is_file()})
        state=dict(before,checkpoint_format=CHECKPOINT_FORMAT,config=config,source_manifest=manifest,
            strategy_replay=StrategyReplay(cfg['strategy_capacity'],seed=seed+110000).state_dict(),
            reference_rng=np.random.default_rng(seed+120000).bit_generator.state,
            average_state=None,average_fit=None,averaging_window=[361,360],historical_mixture_available=False)
        commit_checkpoint(folder,state,lambda p:shutil.copyfile(parent_path/'current.npz',p))
        restored,_=load_checkpoint(folder)
        for key in ('net','buffers','root_replay','rng'):
            if not equal_state(before[key],restored[key]):raise ValueError('Warm-start state changed: '+key)
        proof.append(dict(seed=seed,parent_iteration=360,replay_rows=[b['count'] for b in before['buffers']],
            weights_replay_root_rng_byte_identical=True,parent_policy_sha256=digest(parent_path/'current.npz')))
        for label,original in [('original120',PARENT/f'seed-{seed}/current_00120.npz'),('parent360',parent_path/'current.npz')]:
            target=folder/f'{label}.npz';shutil.copyfile(original,target)
            policy=policy_for(target,campaign/'source')
            policies[label][str(seed)]=dict(path=str(target),sha256=digest(target))
            frozen.update(policy_dependency_hashes(policy,campaign/'source'))
        for path in parent_path.iterdir():
            if path.is_file():frozen[str(path)]=digest(path)
        del state,before,restored
    # Updated frozen map includes all original checkpoint and opponent inputs.
    config['frozen_files']=frozen
    atomic_json(campaign/'config.json',config)
    # Rewrite only new parent copies' config metadata; numerical state is fixed.
    for seed in SEEDS:
        folder=campaign/f'seed-{seed}';pointer=read(folder/'current.json');p=folder/'checkpoints'/pointer['directory']
        state=load_state(p);state['config']=config
        shutil.rmtree(p);commit_checkpoint(folder,state,lambda target,s=seed:shutil.copyfile(PARENT/f'seed-{s}/current_00360.npz',target))
        del state
    plan=dict(created_utc=utc(),study_kind=cfg['study_kind'],config=cfg,
        campaign=str(campaign),campaign_config_sha256=digest(campaign/'config.json'),
        frozen_files=frozen,parent_paths=parent_paths,policies=policies,connection_verification=proof,
        gate_report_sha256=digest(gate_path),gate_report=str(Path(gate_path).resolve()),
        original_regret_learner_unchanged=True,new_selfplay_iterations_per_seed=60,
        averaging_window=[361,420],average_before361_available=False,
        candidate_preselected='average361_420',planned_evaluations=36,planned_hands=4320000,
        independent_card_blocks=20000,prior_card_ranges=prior,model_promoted=False,
        full_paper_reproduction=False,legacy_information_loss_unresolved=True,
        limits='A projection onto legacy51 features; fixed-window average, not all420historical strategies',
        sampling='Uniform legal-action reference for all players; strategy labels weighted t*own-reach; separate RNG',
        budget='Regret replay100k per seat and1024x512 unchanged; strategy replay100k per model; latest recovery only')
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
    primary={f:change('original120','average361_420',f) for f in FIELDS}
    z=statistics.NormalDist().inv_cdf(1-.05/(2*9));diagnostics={}
    for left,right in [('parent360','current420'),('parent360','average361_420'),('current420','average361_420')]:
        for field in FIELDS:
            r=change(left,right,field);h=r['ci95_half_width']/1.96*z
            r.update(simultaneous_lower=r['mean']-h,simultaneous_upper=r['mean']+h,family_comparisons=9)
            diagnostics[f'{field}/{right}-minus-{left}']=r
    return dict(complete=True,completed_evaluations=36,total_hands=36*deals*6,independent_card_blocks=deals,
        absolute_scores={f'{f}/{l}':interval([statistics.mean(b) for b in zip(*arrays(l,f))]) for l in LABELS for f in FIELDS},
        average_minus_original120=primary,adoption_against_original120=adoption_conditions(primary),
        diagnostic_comparisons=diagnostics,new_selfplay_iterations_per_seed=60,averaging_window=[361,420],
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
        public=ROOT/'reports/average-baseline'/study.name;public.mkdir(parents=True,exist_ok=True)
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
                progress[str(seed)]=read(path) if path.exists() else dict(iteration=360,status='prepared')
            record=dict(status=state,phase=phase,observed_utc=utc(),run=study.name,
                study_kind=cfg['study_kind'],supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),
                training_progress=progress,completed_training_workers=len(workers),planned_training_workers=3,
                new_selfplay_iterations_per_seed={s:max(0,r.get('iteration',r.get('last_committed_iteration',360))-360) for s,r in progress.items()},
                planned_new_selfplay_iterations_per_seed=60,training_complete=len(workers)==3,
                averaging_window=[361,420],historical_average_before361_available=False,
                completed_evaluations=len(rows),planned_evaluations=36,verified_hands=len(rows)*20000*6,
                planned_hands=4320000,independent_card_blocks=20000,running=live,error=error,
                small_game_average_gate_passed=True,full_paper_reproduction=False,
                model_promoted=False,automatic_retry=False,study_complete=state=='complete',
                evaluation_complete=len(rows)==36,elapsed_seconds=time.monotonic()-began,max_seconds=cfg['max_seconds'],
                backup_ready=(study/'backup-ready.json').exists(),
                next_action='Inspect failure; no automatic retry' if state=='failed' else
                    'Analyze declared comparisons; no automatic extension or model replacement' if state=='complete'
                    else 'Continue fixed360to420learning and36cell evaluation')
            atomic_json(study/'status.json',record)
            if force or time.monotonic()-last_write>=15 or state!='running':
                atomic_json(ROOT/'reports/average-baseline-status.json',record)
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
            launch(f'train-{seed}',[str(PYTHON),str(ROOT/'scripts/train_poker_average_baseline.py'),
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
                if state['iteration']!=420 or state['average_state'] is None:
                    raise ValueError('Training did not produce the declared final current and average')
                workers[str(item['seed'])]=dict(iteration=420,checkpoint=str(path),
                    files={p.name:digest(p) for p in path.iterdir() if p.is_file()},
                    average_fit=state['average_fit']);del state
            publish('training');time.sleep(3)
        # All3workers have stopped before evaluators use all3cores.
        atomic_json(study/'stage-ready.json',dict(complete=True,stage=420,seeds=workers,
            averaging_window=[361,420],training_complete=True,evaluation_complete=False))
        from snapshot_poker_average_baseline import create_archive
        publish('verifying_recovery',force=True)
        create_archive(study)
        policies=plan['policies']
        for label,name in [('current420','current.npz'),('average361_420','average.npz')]:
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
        final=summarize(rows);final.update(completed_utc=utc(),result_files={k:digest(study/'evaluation'/k/'results.json') for k in rows})
        atomic_json(study/'evaluation-summary.json',final)
        atomic_json(study/'summary.json',dict(complete=True,completed_utc=utc(),run=study.name,
            training_complete=True,evaluation_complete=True,completed_evaluations=36,total_hands=4320000,
            new_selfplay_iterations_per_seed=60,averaging_window=[361,420],
            original_four_conditions_pass=final['adoption_against_original120']['original_four_conditions_pass'],
            evaluation_summary_sha256=digest(study/'evaluation-summary.json'),model_promoted=False,
            stop_reason='Fixed420budget and all36evaluations complete; no automatic extension or model replacement'))
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
            model_promoted=False,last_safe_backup='Original verified compact360; inspect private index for newer completed recovery')
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
