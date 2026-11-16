"""Independent confirmation of one already selected, immutable refit candidate.

No fitting, selfplay, promotion, checkpoint export or Drive operations. Reuse
the audited evaluator, fixed opponents and original four adoption conditions.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import time

import numpy as np

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start, alive
from controlled_poker_refit import read, utc, verify_files
from evaluate_compact_learning_curve import adoption_conditions, process_matches
from evaluate_rich_selfplay import verify_results, interval, policy_dependency_hashes
from rich_poker_runtime import policy_for
from run_compact_refit_probe import check_card_range


FOLDER = ROOT/'artifacts/candidate-confirmation'
CANDIDATE = 'legacy_fewer_steps'
LABELS = ['original120', 'parent360', 'legacy_control', CANDIDATE]
SEEDS = [342, 343, 344]
FIELDS = ['public', 'mixed', 'noregrets']
CELLS = len(LABELS)*len(SEEDS)*len(FIELDS)
DEALS = 20000
PARENT = ROOT/'artifacts/controlled-refit/20261116T004818765266Z'
BACKUP_SHA = '42ec75b328c31d568e971d2231a2f70c710c8eefa0ea8fe0d3b5453b952fcdca'
BUSY_SCRIPTS = {'run_controlled_poker_study.py', 'controlled_poker_refit.py',
    'evaluate_rich_selfplay.py', 'run_rich_poker_study.py', 'train_rich_selfplay.py',
    'run_bounded_poker_study.py', 'train_compact_bounded.py',
    'run_poker_candidate_confirmation.py'}


def validate_config(cfg):
    fixed = dict(selection_study=str(PARENT), candidate=CANDIDATE, labels=LABELS,
        seeds=SEEDS, fields=FIELDS, evaluation_deals=DEALS,
        evaluation_deal_seed=2037100000, max_evaluation_workers=3,
        evaluation_seconds_per_cell=2400, max_seconds=14400,
        new_selfplay_iterations=0, new_fitting_jobs=0,
        automatic_retry=False, model_promotion=False)
    if any(cfg.get(k) != value or type(cfg.get(k)) != type(value) for k,value in fixed.items()):
        raise ValueError('The frozen candidate, cards, criteria and evaluation budget cannot change')


def prior_card_ranges(output, bases=None):
    """Inspect configs AND actual saved cells, including orphaned evaluations."""
    ranges = {}
    for base in bases or [ROOT/'artifacts', ROOT/'reports']:
        for pattern in ['config.json', 'results.json']:
            for path in Path(base).rglob(pattern):
                if path.is_relative_to(output):
                    continue
                row = read(path)
                start, count = (row.get('evaluation_deal_seed'), row.get('evaluation_deals')) \
                    if pattern == 'config.json' else (row.get('deal_seed'), row.get('deals'))
                if type(start) is int and type(count) is int and count > 0:
                    key = f'{start}:{start+count}'
                    item = ranges.setdefault(key, dict(range=[start,start+count], records=0, examples=[]))
                    item['records'] += 1
                    if len(item['examples']) < 3:
                        item['examples'].append(str(path))
    return ranges


def jobs():
    return [dict(id=f'{field}-seed-{seed}-{label}', label=label, seed=seed, field=field)
        for label in LABELS for seed in SEEDS for field in FIELDS]


def live_poker_jobs():
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name) == os.getpid() or process_start(int(proc.name)) is None:
                continue
            args = (proc/'cmdline').read_bytes().decode().split('\0')
            if len(args)>1 and Path(args[1]).name in BUSY_SCRIPTS:
                found.append(dict(pid=int(proc.name), process_start=process_start(int(proc.name)), command=args))
        except (OSError, UnicodeError):
            continue
    return found


def prepare(study, config_path):
    study = Path(study).resolve(); cfg = read(config_path); validate_config(cfg)
    if study.parent != FOLDER or not re.fullmatch(r'\d{8}T\d{6,12}Z',study.name):
        raise ValueError('Use a fresh timestamped candidate-confirmation folder')
    if study.exists():
        raise FileExistsError('Never overwrite or automatically repeat a prepared/started confirmation')
    if live_poker_jobs():
        raise ValueError('A poker worker/evaluator is already alive; no duplicate run')
    summary = read(PARENT/'summary.json'); selection = read(PARENT/'evaluation-summary.json')
    if (not summary['complete'] or summary['completed_evaluations'] != 90
            or summary['evaluation_summary_sha256'] != digest(PARENT/'evaluation-summary.json')
            or not selection['adoption_against_original120'][CANDIDATE]['original_four_conditions_pass']):
        raise ValueError('The selected candidate does not have the completed declared evidence')
    verify_files(read(PARENT/'plan.json')['frozen_files'])
    prior = prior_card_ranges(study)
    check_card_range(cfg['evaluation_deal_seed'],DEALS,{k:v['range'] for k,v in prior.items()})
    index = read(ROOT/'artifacts/offsite-backups.json')
    backed = [r for r in index['backups'] if r.get('sha256')==BACKUP_SHA and not r.get('deleted_remote')]
    if (len(backed)!=1 or backed[0].get('status')!='complete'
            or not all(backed[0].get(k) is True for k in
                ['restoration_verified','verified_remote_size','verified_owner_only'])
            or digest(backed[0]['local_archive'])!=BACKUP_SHA):
        raise ValueError('The existing verified candidate bundle must remain available')
    source = read(PARENT/'stage-ready.json')['policies']
    study.mkdir(parents=True); (study/'policies').mkdir(); (study/'logs').mkdir()
    shutil.copytree(PARENT/'source',study/'source',
        ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc','.git'))
    frozen = {}; policies = {label:{} for label in LABELS}
    for label in LABELS:
        for seed in SEEDS:
            original = Path(source[label][str(seed)]['path'])
            if original.suffix!='.npz' or digest(original)!=source[label][str(seed)]['sha256']:
                raise ValueError('Preselected policy bytes changed')
            with np.load(original, allow_pickle=False) as arrays:
                if any(np.issubdtype(arrays[n].dtype,np.number) and not np.isfinite(arrays[n]).all()
                        for n in arrays.files):
                    raise ValueError('Nonfinite selected policy')
            target = study/'policies'/f'{label}-seed-{seed}.npz';shutil.copyfile(original,target)
            if digest(target)!=digest(original):
                raise ValueError('Copied policy differs from the selected policy')
            policies[label][str(seed)] = dict(path=str(target),sha256=digest(target),
                selected_original_path=str(original), byte_identical=True)
            frozen[str(original)]=digest(original)
            policy=policy_for(target,study/'source')
            frozen.update(policy_dependency_hashes(policy,study/'source'))
    from fullhouse_runtime import Fullhouse
    public=Fullhouse();opponent_registry=ROOT/'artifacts/selfplay/current.json'
    opponent=read(opponent_registry);opponent_path=ROOT/opponent['model']
    if digest(opponent_path)!=opponent['sha256']:
        raise ValueError('NoRegrets opponent bytes changed')
    for path in [Path(public.model),opponent_path,opponent_registry,Path(config_path),Path(__file__),
            ROOT/'scripts/evaluate_compact_learning_curve.py',ROOT/'scripts/controlled_poker_refit.py',
            ROOT/'scripts/run_compact_refit_probe.py',ROOT/'scripts/fullhouse_v3_common.py',
            PARENT/'plan.json',PARENT/'summary.json',PARENT/'evaluation-summary.json',PARENT/'stage-ready.json']:
        frozen[str(path.resolve())]=digest(path)
    for parent_registry in ['controlled-refit','bounded-continuation','rich-selfplay','compact-v4',
            'continuation-study','refit-probe','foundation-study']:
        path=ROOT/'artifacts'/parent_registry/'current.json'
        if path.exists():frozen[str(path)]=digest(path)
    frozen.update({str(p):digest(p) for p in (study/'source').rglob('*') if p.is_file()})
    atomic_json(study/'config.json',cfg);frozen[str(study/'config.json')]=digest(study/'config.json')
    plan=dict(created_utc=utc(),study_kind='independent-frozen-candidate-confirmation',
        config=cfg,policies=policies,source=str(study/'source'),frozen_files=frozen,
        planned_evaluations=CELLS,planned_hands=CELLS*DEALS*6,independent_card_blocks=DEALS,
        primary_comparison=dict(candidate=CANDIDATE,baseline='original120',
            criteria='Unchanged original four conditions; nominal paired95% intervals',
            candidate_preselected_before_new_cards=True,automatic_promotion=False),
        diagnostic_baselines=['parent360','legacy_control'],diagnostic_simultaneous_comparisons=6,
        prior_card_ranges=prior,new_selfplay_iterations=0,new_fitting_jobs=0,
        existing_candidate_backup_sha256=BACKUP_SHA,new_model_archive_required=False,
        backup_scope='Existing verified21-policy bundle contains unchanged candidate weights. It does not contain this later confirmation.',
        no_automatic_retry=True,no_model_promotion=True)
    atomic_json(study/'plan.json',plan)
    return plan


def summarize(rows, deals=DEALS):
    if set(rows)!={j['id'] for j in jobs()}:
        raise ValueError('All36 verified cells required; no result-dependent early selection')
    for field in FIELDS:
        if len({json.dumps(r.get('opponents'),sort_keys=True) for k,r in rows.items()
                if k.startswith(field+'-')})!=1:
            raise ValueError('Opponent composition changed across confirmation cells')
    def arrays(label,field):
        data=[rows[f'{field}-seed-{seed}-{label}']['deal_blocks'] for seed in SEEDS]
        if any(len(v)!=deals or not all(math.isfinite(x) for x in v) for v in data):
            raise ValueError('Card-block coverage or finite values differ')
        return data
    def change(baseline,field):
        per=[[b-a for a,b in zip(x,y)] for x,y in zip(arrays(baseline,field),arrays(CANDIDATE,field))]
        return dict(**interval([statistics.mean(block) for block in zip(*per)]),
            per_seed=[dict(seed=seed,**interval(v)) for seed,v in zip(SEEDS,per)])
    absolute={f'{field}/{label}':interval([statistics.mean(block) for block in zip(*arrays(label,field))])
        for label in LABELS for field in FIELDS}
    primary={field:change('original120',field) for field in FIELDS}
    diagnostics={};z=statistics.NormalDist().inv_cdf(1-.05/(2*6))
    for baseline in ['parent360','legacy_control']:
        for field in FIELDS:
            row=change(baseline,field);half=row['ci95_half_width']/1.96*z
            row.update(simultaneous_lower=row['mean']-half,simultaneous_upper=row['mean']+half,
                family_comparisons=6)
            diagnostics[f'{field}/{CANDIDATE}-minus-{baseline}']=row
    decision=adoption_conditions(primary)
    return dict(complete=True,completed_evaluations=CELLS,total_hands=CELLS*deals*6,
        independent_card_blocks=deals,absolute_scores=absolute,
        candidate_minus_original120=primary,diagnostic_comparisons=diagnostics,
        adoption_against_original120=decision,
        confirmation_pass=decision['original_four_conditions_pass'],
        statistical_unit='One fresh shared card block averaged over seats and three fixed training seeds',
        candidate_preselected=True,new_selfplay_iterations=0,new_fitting_jobs=0,model_promoted=False,
        limitation='Fresh cards confirm only this fixed candidate and fixed seeds/opponents. No general-strength or new-selfplay claim; no automatic promotion or retry.')


def run(study, config_path, prepared=False):
    study=Path(study).resolve();FOLDER.mkdir(parents=True,exist_ok=True)
    lock=(FOLDER/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};results={};handlers={};started=None
    try:
        registry=FOLDER/'current.json'
        if registry.exists():
            prior=read(registry)
            if process_matches(prior['pid'],prior['process_start'],'run_poker_candidate_confirmation.py'):
                raise ValueError('A confirmation supervisor is alive; no duplicate')
        if live_poker_jobs():raise ValueError('Existing poker jobs are alive; do not duplicate')
        if prepared and any((study/n).exists() for n in ['status.json','summary.json','failure.json']):
            raise ValueError('Never restart a previously started confirmation')
        plan=read(study/'plan.json') if prepared else prepare(study,config_path)
        cfg=plan['config'];validate_config(cfg);verify_files(plan['frozen_files'])
        if study.parent!=FOLDER or plan['source']!=str(study/'source'):
            raise ValueError('Prepared study location mismatch')
        started=time.monotonic();pending=deque(jobs());last_public=-math.inf
        atomic_json(registry,dict(folder=str(study),run=study.name,pid=os.getpid(),
            process_start=process_start(os.getpid()),started_utc=utc(),candidate=CANDIDATE,
            new_selfplay_iterations=0,new_fitting_jobs=0))
        def status(state='running',error=None,force=False):
            nonlocal last_public
            live={k:dict(pid=x['process'].pid,process_start=x['start'],command=x['command'])
                for k,x in active.items() if x['process'].poll() is None and alive(x['process'].pid,x['start'])}
            record=dict(status=state,phase='evaluating' if state=='running' else state,observed_utc=utc(),
                run=study.name,study_kind=plan['study_kind'],candidate=CANDIDATE,
                supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),
                completed_evaluations=len(results),planned_evaluations=CELLS,
                verified_hands=len(results)*DEALS*6,planned_hands=CELLS*DEALS*6,
                independent_card_blocks=DEALS,evaluation_deal_seed=cfg['evaluation_deal_seed'],
                running=live,error=error,new_selfplay_iterations=0,new_fitting_jobs=0,
                candidate_weights_changed=False,model_promoted=False,automatic_retry=False,
                study_complete=state=='complete',evaluation_complete=len(results)==CELLS,
                elapsed_seconds=time.monotonic()-started,max_seconds=cfg['max_seconds'],
                existing_candidate_backup_sha256=BACKUP_SHA,new_model_archive_required=False,
                next_action=('Analyze independent confirmation; no automatic promotion or further training'
                    if state=='complete' else 'Inspect failure; no automatic retry' if state!='running'
                    else 'Continue the predeclared36-cell independent evaluation'))
            atomic_json(study/'status.json',record)
            if force or state!='running' or time.monotonic()-last_public>=15:
                atomic_json(ROOT/'reports/candidate-confirmation-status.json',record)
                atomic_json(ROOT/'reports/training-neural-status.json',record);last_public=time.monotonic()
        def interrupted(signum,frame):
            raise InterruptedError('Confirmation interrupted; no automatic retry')
        for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,interrupted)
        status(force=True)
        while pending or active:
            if time.monotonic()-started>cfg['max_seconds']:
                raise TimeoutError('Independent confirmation fixed four-hour budget exhausted')
            for key,item in list(active.items()):
                if time.monotonic()-item['started']>cfg['evaluation_seconds_per_cell']+30:
                    raise TimeoutError('Owned evaluation cell exceeded its fixed budget: '+key)
                code=item['process'].poll()
                if code is None:continue
                item['log'].close();del active[key]
                if code!=0:raise RuntimeError(f'{key} failed with exit{code}; no automatic retry')
                job=item['job'];policy=plan['policies'][job['label']][str(job['seed'])]
                verify_files(plan['frozen_files'])
                rows=verify_results(study/'evaluation'/key,replay=False,expected=dict(
                    candidate_sha256=policy['sha256'],field=job['field'],deals=DEALS,
                    deal_seed=cfg['evaluation_deal_seed']))
                if not rows.get('terminal_payoffs_replayed'):
                    raise ValueError('Evaluation did not verify actual actions through the trusted engine')
                results[key]=rows
            while pending and len(active)<cfg['max_evaluation_workers']:
                job=pending.popleft();policy=plan['policies'][job['label']][str(job['seed'])]
                command=[str(PYTHON),str(ROOT/'scripts/evaluate_rich_selfplay.py'),
                    '--model',policy['path'],'--source',plan['source'],'--field',job['field'],
                    '--deals',str(DEALS),'--deal-seed',str(cfg['evaluation_deal_seed']),
                    '--timeout',str(cfg['evaluation_seconds_per_cell']),
                    '--output',str(study/'evaluation'/job['id'])]
                log=(study/'logs'/(job['id']+'.log')).open('w')
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                active[job['id']]=dict(process=proc,start=process_start(proc.pid),command=command,
                    log=log,started=time.monotonic(),job=job)
            status();time.sleep(3)
        verify_files(plan['frozen_files'])
        final=summarize(results);final.update(completed_utc=utc(),
            result_files={key:digest(study/'evaluation'/key/'results.json') for key in results},
            raw_hands_verified=True,source_provenance_verified=True)
        atomic_json(study/'evaluation-summary.json',final)
        atomic_json(study/'summary.json',dict(complete=True,completed_utc=utc(),
            run=study.name,completed_evaluations=CELLS,total_hands=CELLS*DEALS*6,
            confirmation_pass=final['confirmation_pass'],evaluation_summary_sha256=digest(study/'evaluation-summary.json'),
            new_selfplay_iterations=0,new_fitting_jobs=0,model_promoted=False,
            stop_reason='Predeclared independent confirmation completed; no automatic extension or model replacement',
            no_automatic_retry=True))
        atomic_json(study/'backup-reference.json',dict(existing_archive_sha256=BACKUP_SHA,
            kind='derived-policy-probe',candidate_weights_unchanged=True,new_upload_required=False,
            archived_prior_study_complete=True,archive_contains_this_later_confirmation=False,
            final_confirmation_reports='summary.json and evaluation-summary.json; publish separately',
            remote_sha256_verified=False))
        public=ROOT/'reports/candidate-confirmation'/study.name;public.mkdir(parents=True,exist_ok=True)
        for name in ['summary.json','evaluation-summary.json','backup-reference.json']:
            atomic_json(public/name,read(study/name))
        status('complete',force=True)
        return final
    except BaseException as error:
        if started is not None:
            state='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed'
            atomic_json(study/'failure.json',dict(status=state,observed_utc=utc(),
                error=type(error).__name__+': '+str(error),completed_evaluations=len(results),automatic_retry=False))
            status(state,error=type(error).__name__+': '+str(error),force=True)
        raise
    finally:
        for item in active.values():
            p=item['process']
            if p.poll() is None and alive(p.pid,item['start']):
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for item in active.values():
            try:item['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                if alive(item['process'].pid,item['start']):os.killpg(item['process'].pid,signal.SIGKILL)
            item['log'].close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--run-prepared',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        result=prepare(args.study,args.config)
        print(json.dumps(dict(prepared=True,planned_evaluations=result['planned_evaluations'],new_fitting_jobs=0)))
    else:run(args.study,args.config,args.run_prepared)
