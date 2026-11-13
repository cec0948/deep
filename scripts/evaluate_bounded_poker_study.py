"""Fixed current-only120/240/300/360 evaluation on20,000 new card blocks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from bounded_poker_study_support import check_frozen, read, utc, validate_config
from evaluate_compact_learning_curve import adoption_conditions, comparison, process_matches
from fullhouse_v3_common import ROOT,PYTHON,atomic_json,digest,process_start
from run_compact_refit_probe import check_card_range,interval,previous_ranges,verify_result


def jobs_for(study,cfg):
    campaign=Path(study)/'campaign'
    return [dict(id=f'{field}-seed-{seed}-current-{iteration}',field=field,variant='current',
        iteration=iteration,seed=seed,model=str(campaign/f'seed-{seed}'/f'current_{iteration:05d}.npz'),
        source=str(campaign/'source')) for iteration in cfg['evaluation_iterations']
        for seed in cfg['seeds'] for field in cfg['evaluation_fields']]


def evaluation_parallelism(launch):
    if launch.get('status') not in ('running','complete'):raise ValueError('Training launch failed')
    pids,starts=launch.get('pids',{}),launch.get('process_starts',{})
    live=any(process_matches(pid,starts.get(str(seed)),'train_compact_bounded.py') for seed,pid in pids.items())
    return 3 if launch['status']=='complete' and not live else 1


def summarize(rows,cfg,through):
    iterations=[i for i in cfg['evaluation_iterations'] if i<=through];absolute={};changes={};decisions={}
    for iteration in iterations:
        for field in cfg['evaluation_fields']:
            per=[rows[f'{field}-seed-{seed}-current-{iteration}']['deal_blocks'] for seed in cfg['seeds']]
            if any(len(v)!=cfg['evaluation_deals'] for v in per):raise ValueError('Incomplete card blocks')
            absolute[f'{field}/current-{iteration}']=dict(**interval([statistics.mean(x) for x in zip(*per)]),
                per_seed=[dict(seed=seed,**interval(v)) for seed,v in zip(cfg['seeds'],per)])
    for before,after in [(120,240),(120,300),(120,360),(240,300),(240,360),(300,360)]:
        if after not in iterations:continue
        delta=comparison(rows,cfg,'current',before,'current',after)
        for field,value in delta.items():changes[f'{field}/current-{after}-minus-current-{before}']=value
        if before==120 and after in cfg['stages']:
            decisions[f'current-{after}']=adoption_conditions(delta)
    return dict(complete=True,through_iteration=through,completed_evaluations=len(iterations)*9,
        total_hands=len(iterations)*9*cfg['evaluation_deals']*6,independent_card_blocks=cfg['evaluation_deals'],
        absolute_scores=absolute,paired_changes=changes,decisions=decisions,raw_hands_verified=True,
        statistical_unit='One common card block averaged across six seats and the three fixed training seeds',
        adoption_baseline='Original current120; parentcurrent240 differences are separate progress diagnostics',
        limitations=['Nominal95% intervals conditional on three fixed seeds; no multiplicity adjustment',
            'Shared cards across the predeclared checkpoints are paired research data, not independent final holdouts',
            'Fixed opponents do not establish general or professional strength; no automatic promotion or extension'])


def stop_children(active):
    for item in active.values():
        child=item['process']
        if child.poll() is None and process_start(child.pid)==item['start']:
            try:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=8)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                if child.poll() is None and process_start(child.pid)==item['start']:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        item['log'].close()


def run(study):
    study=Path(study).resolve();cfg=read(study/'config.json');validate_config(cfg);campaign=study/'campaign'
    lock=(study/'evaluation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (study/'evaluation-plan.json').exists():raise FileExistsError('Evaluation cannot be automatically retried')
    prior=previous_ranges(study);prefix=f'reports/bounded-continuation/{study.name}/'
    prior={k:v for k,v in prior.items() if not k.startswith(prefix)}
    check_card_range(cfg['evaluation_deal_seed'],cfg['evaluation_deals'],prior)
    frozen=dict(read(study/'plan.json')['frozen_files'])
    from fullhouse_runtime import BRIDGE,SOURCE,Fullhouse
    public=Fullhouse();registry=read(ROOT/'artifacts/selfplay/current.json')
    for p in [ROOT/'artifacts/selfplay/current.json',ROOT/registry['model'],Path(public.model),Path(BRIDGE),
              ROOT/'configs/fullhouse-model.json',ROOT/'scripts/evaluate_compact.py',ROOT/'scripts/fullhouse_v4_runtime.py',
              ROOT/'scripts/fullhouse_v3_runtime.py',ROOT/'scripts/fullhouse_runtime.py']:
        frozen[str(p.resolve())]=digest(p)
    for p in list(SOURCE.rglob('*.py'))+list((SOURCE/'data').glob('*.npz')):
        frozen[str(p.resolve())]=digest(p)
    jobs=jobs_for(study,cfg);evaluation=study/'evaluation';evaluation.mkdir()
    plan=dict(created_utc=utc(),config=cfg,study_plan_sha256=digest(study/'plan.json'),
        jobs=jobs,frozen_files=frozen,prior_card_ranges=prior)
    atomic_json(study/'evaluation-plan.json',plan);plan_sha=digest(study/'evaluation-plan.json')
    frozen[str(study/'evaluation-plan.json')]=plan_sha
    rows={};active={};pending=jobs[:];proofs={};policies={};expected={};stage_hashes={};completed=[];began=time.monotonic()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def status(state,**extra):
        atomic_json(study/'evaluation-status.json',dict(status=state,updated_utc=utc(),seconds=time.monotonic()-began,
            evaluator_pid=os.getpid(),evaluator_start=process_start(os.getpid()),evaluation_plan_sha256=plan_sha,
            completed_evaluations=len(rows),planned_evaluations=36,verified_hands=len(rows)*cfg['evaluation_deals']*6,
            finished=list(rows),verified_result_files=proofs,pending=[j['id'] for j in pending],
            running={n:dict(pid=v['process'].pid,process_start=v['start']) for n,v in active.items()},
            completed_checkpoint_evaluations=completed[:],**extra))
    def stop(*unused):raise InterruptedError('Bounded evaluation interrupted')
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        while pending or active:
            if time.monotonic()-began>cfg['max_seconds']:raise TimeoutError('Evaluation study time exhausted')
            parent=read(study/'status.json')
            if parent['status'] in ('failed','interrupted'):raise RuntimeError('Supervisor failed')
            if not process_matches(parent['supervisor_pid'],parent['supervisor_start'],'run_bounded_poker_study.py',study):
                raise RuntimeError('Supervisor PID/start/command mismatch')
            launch=read(campaign/'launch.json')
            limit=evaluation_parallelism(launch)
            ready=read(study/'stage-ready.json')['checkpoints']
            for job in pending[:]:
                if len(active)>=limit:break
                entry=ready.get(str(job['iteration']))
                if entry is None:continue
                model=job['model'];sha=entry['models'][str(job['seed'])]
                if digest(model)!=sha:raise ValueError('Verified checkpoint policy changed')
                if entry.get('verification_sha256'):
                    p=study/f"stage-{job['iteration']}-verification.json"
                    if digest(p)!=entry['verification_sha256']:raise ValueError('Stage verification changed')
                    stage_hashes[str(p)]=digest(p)
                check_frozen(frozen);check_frozen(stage_hashes);check_frozen(policies)
                policies[model]=sha
                expected[job['id']]=dict(candidate=sha,evaluator_sha256=frozen[str(ROOT/'scripts/evaluate_compact.py')],
                    runtime_sha256=frozen[str(ROOT/'scripts/fullhouse_v4_runtime.py')],engine_sha256=frozen[str(BRIDGE)],
                    source_features_sha256=digest(campaign/'source/bot/features.py'),
                    opponents=dict(noregrets=registry['sha256'],public=public.digest if job['field'] in ('public','mixed') else None))
                proofpath=evaluation/(job['id']+'.provenance.json')
                atomic_json(proofpath,dict(job=job,expected=expected[job['id']],policy_files={model:sha},stage_evidence=entry))
                out=evaluation/job['id']
                if out.exists():raise FileExistsError('Incomplete evaluation cannot be reused')
                log=(evaluation/(job['id']+'.log')).open('w')
                command=[str(PYTHON),str(ROOT/'scripts/evaluate_compact.py'),'--model',model,'--source',job['source'],
                    '--field',job['field'],'--deals',str(cfg['evaluation_deals']),'--deal-seed',str(cfg['evaluation_deal_seed']),
                    '--timeout',str(cfg['evaluation_seconds_per_cell']),'--output',str(out)]
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job['id']]=dict(process=child,start=process_start(child.pid),log=log,started=time.monotonic(),job=job,proof_sha=digest(proofpath))
                pending.remove(job)
            for name,item in list(active.items()):
                code=item['process'].poll()
                if code is None:
                    if time.monotonic()-item['started']>cfg['evaluation_seconds_per_cell']+30:raise TimeoutError('Evaluation cell timed out')
                    continue
                item['log'].close()
                if code:raise RuntimeError('Evaluation cell failed; no retry: '+name)
                proofpath=evaluation/(name+'.provenance.json')
                if digest(proofpath)!=item['proof_sha']:raise ValueError('Evaluation provenance changed')
                check_frozen(frozen);check_frozen(policies);check_frozen(stage_hashes)
                rows[name]=verify_result(evaluation/name,item['job'],cfg,expected[name])
                proofs[name]=dict(results_sha256=digest(evaluation/name/'results.json'),
                    hands_sha256=digest(evaluation/name/'hands.jsonl.gz'),provenance_sha256=digest(proofpath))
                del active[name]
            for stage in cfg['stages']:
                required=[j['id'] for j in jobs if j['iteration']<=stage]
                if stage not in completed and all(n in rows for n in required):
                    result=summarize(rows,cfg,stage);result.update(config=cfg,completed_utc=utc(),
                        source_provenance_verified=True,evaluation_plan_sha256=plan_sha,
                        verified_result_files={n:proofs[n] for n in required})
                    atomic_json(study/f'stage-{stage}-evaluation.json',result);completed.append(stage)
            status('running')
            if pending or active:time.sleep(2)
        check_frozen(frozen);check_frozen(policies);check_frozen(stage_hashes)
        for name,p in proofs.items():
            for file,key in [('results.json','results_sha256'),('hands.jsonl.gz','hands_sha256')]:
                if digest(evaluation/name/file)!=p[key]:raise ValueError('Completed raw/result bytes changed')
        result=summarize(rows,cfg,360);result.update(config=cfg,completed_utc=utc(),seconds=time.monotonic()-began,
            source_provenance_verified=True,evaluation_plan_sha256=plan_sha,verified_result_files=proofs)
        atomic_json(study/'evaluation-summary.json',result);status('complete');return result
    except BaseException as error:
        status('interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(error));raise
    finally:
        stop_children(active);lock.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args();run(a.study)
