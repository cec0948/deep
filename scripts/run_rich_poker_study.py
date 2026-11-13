"""Supervise a separate rich-input residual family inherited from teacher360.

New-family iterations are1..96, not legacy361..456. This runner never retries,
promotes a model, changes an older registry, or performs remote backup actions.
"""
from __future__ import annotations

import argparse
import datetime
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

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from evaluate_compact_learning_curve import adoption_conditions
from run_compact_refit_probe import check_card_range, interval, previous_ranges


SEEDS = [342, 343, 344]
STAGES = [12, 48, 96]
FIELDS = ['public', 'mixed', 'noregrets']
LABELS = ['original120', 'parent360', 'rich12', 'rich48', 'rich96']
HELPERS = ('run_rich_poker_study.py', 'train_rich_selfplay.py',
           'rich_poker_observation.py', 'rich_poker_traversal.py', 'rich_regret_model.py',
           'fullhouse_v3_common.py', 'compact_lightweight_backup.py',
           'regret_partition.py', 'poker_public_state.py',
           'fullhouse_runtime.py', 'fullhouse_v4_runtime.py',
           'fullhouse_v3_runtime.py', 'rich_poker_runtime.py', 'evaluate_rich_selfplay.py',
           'snapshot_rich_selfplay.py', 'evaluate_compact_learning_curve.py',
           'run_compact_refit_probe.py')
PROTECTED = ('artifacts/compact-v4/current.json', 'artifacts/retraining-loop.json',
             'artifacts/continuation-study/current.json', 'artifacts/refit-probe/current.json',
             'artifacts/foundation-study/current.json', 'artifacts/bounded-continuation/current.json',
             'artifacts/river-strength/current.json')
STAMP = re.compile(r'\d{8}T\d{6,12}Z')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(files):
    if not isinstance(files, dict) or not files:
        raise ValueError('A nonempty frozen file manifest is required')
    for name, sha in files.items():
        if not Path(name).is_absolute() or digest(name) != sha:
            raise ValueError('Frozen rich-study input changed: '+str(name))


def finite(value):
    if isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite rich-study status or diagnostics')


def validate_config(cfg):
    fixed = dict(seeds=SEEDS, stages=STAGES, iterations=96, parent_iteration=360,
                 traversals=64, buffer_capacity=5000, main_steps=256, batch_size=128,
                 checkpoint_every=6, learning_rate=.001)
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError('The approved new-family training design changed')
    if cfg.get('prerequisites_passed') is not True:
        raise ValueError('Rich encoder/model/integration prerequisites have not passed')
    expected = dict(evaluation_fields=FIELDS, evaluation_deals=20000,
                    evaluation_deal_seed=2035000000)
    if any(cfg.get(k, value) != value for k, value in expected.items()):
        raise ValueError('The fixed45-cell evaluation design changed')
    for key, maximum, default in [('max_seconds',21600,21600),
                                  ('training_seconds_per_stage',5400,5400),
                                  ('evaluation_seconds_per_cell',2400,2400)]:
        value = cfg.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid fixed execution budget: '+key)
    if cfg.get('promotion', 'none') != 'none' or cfg.get('automatic_retry', False):
        raise ValueError('Promotion and automatic retry are outside this study')
    for field in ('teachers', 'baselines120'):
        rows = cfg.get(field)
        if not isinstance(rows, dict) or set(rows) != {str(s) for s in SEEDS}:
            raise ValueError('All three verified '+field+' are required')
        for row in rows.values():
            if (not isinstance(row, dict) or not Path(row.get('path','')).is_absolute()
                    or not re.fullmatch('[0-9a-f]{64}', row.get('sha256',''))):
                raise ValueError('Absolute model path and SHA-256 required: '+field)
    if not Path(cfg.get('source','')).is_absolute():
        raise ValueError('The actual frozen parent Python source is required')


def validate_gate(path):
    report = read(path)
    if (report.get('all_three_neural_gates_passed') is not True
            or report.get('status') != 'complete'
            or [row.get('seed') for row in report.get('seed_results',[])] != [71,72,73]
            or any(row.get('shared_neural_approximator') is not True
                   or row.get('tabular_approximator_used') is not False
                   for row in report.get('seed_results',[]))):
        raise ValueError('The actual three-seed shared-neural learning gate must pass')
    return report


def output_path(output):
    output = Path(output).resolve()
    if output.parent != ROOT/'artifacts/rich-selfplay' or not STAMP.fullmatch(output.name):
        raise ValueError('Use a new timestamped artifacts/rich-selfplay directory')
    return output


def command_matches(pid, start, script, flag=None, folder=None):
    if type(pid) is not int or not start or process_start(pid) != str(start):
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        return False
    if not any(Path(arg).name == Path(script).name for arg in args if arg):
        return False
    if flag:
        try:
            at = args.index(flag)
            if Path(args[at+1]).resolve() != Path(folder).resolve(): return False
        except (ValueError, IndexError):
            return False
    return True


def evaluation_parallelism(training, campaign, training_complete=False):
    alive = any(command_matches(row['process'].pid, row['start'], 'train_rich_selfplay.py',
                                '--campaign', campaign)
                for row in training.values() if row['process'].poll() is None)
    return 3 if training_complete and not alive else 1


def prepare(output, config_path):
    output = output_path(output); cfg = read(config_path); validate_config(cfg)
    gate = Path(cfg.get('neural_gate_report', ROOT/'artifacts/neural-information-gate/20261113T073421270063Z/report.json')).resolve()
    validate_gate(gate)
    if cfg.get('frozen_files'): verify_files(cfg['frozen_files'])
    prior = previous_ranges(output)
    # The root configuration's own freshly declared interval is not prior data.
    own = str(Path(config_path).resolve().relative_to(ROOT)) if Path(config_path).resolve().is_relative_to(ROOT) else None
    if own: prior.pop(own, None)
    check_card_range(2035000000, 20000, prior)
    source = Path(cfg['source']).resolve()
    if not (source/'training/game.py').is_file() or not (source/'bot/features.py').is_file():
        raise ValueError('The real six-player frozen source is missing')
    protected = {str(ROOT/name):digest(ROOT/name) for name in PROTECTED if (ROOT/name).is_file()}
    parent=Path(read(ROOT/'artifacts/bounded-continuation/current.json')['folder'])
    ready_parent=read(parent/'stage-ready.json');proof_path=parent/'stage-360-verification.json'
    if (read(parent/'status.json').get('status')!='complete'
            or ready_parent['checkpoints']['360']['verification_sha256']!=digest(proof_path)):
        raise ValueError('The verified completed360 teacher family is required')
    for seed in SEEDS:
        if (cfg['teachers'][str(seed)]['sha256']!=ready_parent['checkpoints']['360']['models'][str(seed)]
                or cfg['baselines120'][str(seed)]['sha256']!=ready_parent['checkpoints']['120']['models'][str(seed)]):
            raise ValueError('Teacher360/original120 provenance does not match the completed parent study')
    output.mkdir(parents=True, exist_ok=False)
    campaign = output/'campaign'; campaign.mkdir(); (output/'logs').mkdir()
    shutil.copytree(source, campaign/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc','.git'))
    (campaign/'teachers').mkdir(); (campaign/'baselines').mkdir()
    teachers, baselines, ready_models = {}, {}, {'original120':{}, 'parent360':{}}
    for seed in SEEDS:
        for key, target, destination in [('teachers',teachers,campaign/'teachers'/f'seed-{seed}.npz'),
                                         ('baselines120',baselines,campaign/'baselines'/f'seed-{seed}-current120.npz')]:
            row = cfg[key][str(seed)]; original = Path(row['path'])
            if digest(original) != row['sha256']: raise ValueError('Verified baseline/teacher bytes changed')
            shutil.copyfile(original,destination)
            if digest(destination) != row['sha256']: raise ValueError('Copied baseline/teacher changed')
            target[str(seed)] = dict(path=str(destination),sha256=row['sha256'])
        ready_models['original120'][str(seed)] = baselines[str(seed)]
        ready_models['parent360'][str(seed)] = teachers[str(seed)]
    root_files = {str(ROOT/'scripts'/name):digest(ROOT/'scripts'/name)
                  for name in HELPERS if (ROOT/'scripts'/name).is_file()}
    if cfg.get('evaluation_enabled',True) and str(ROOT/'scripts/evaluate_rich_selfplay.py') not in root_files:
        raise ValueError('Freeze a ready evaluator before launching enabled evaluations')
    if str(ROOT/'scripts/snapshot_rich_selfplay.py') not in root_files:
        raise ValueError('Freeze the verified snapshot helper before preparing this study')
    files = dict(cfg.get('frozen_files',{})); files.update(root_files); files.update(protected)
    files[str(gate)] = digest(gate)
    for name in ('stage-ready.json','stage-360-verification.json','summary.json','plan.json'):
        files[str(parent/name)]=digest(parent/name)
    files.update({str(path):digest(path) for path in sorted((campaign/'source').rglob('*')) if path.is_file()})
    for group in (teachers,baselines):
        files.update({row['path']:row['sha256'] for row in group.values()})
    worker = dict(cfg, source=str(campaign/'source'), teachers=teachers,
                  seconds_per_training_worker=cfg.get('training_seconds_per_stage',5400),
                  frozen_files=files, parent_iteration=360)
    atomic_json(output/'config.json',cfg); atomic_json(campaign/'config.json',worker)
    frozen = dict(files)
    frozen[str(output/'config.json')] = digest(output/'config.json')
    frozen[str(campaign/'config.json')] = digest(campaign/'config.json')
    plan = dict(created_utc=utc(),config=cfg,campaign=str(campaign),frozen_files=frozen,
                protected_registry_files=protected,neural_gate_report=str(gate),
                neural_gate_sha256=digest(gate),parent_iteration=360,
                family='rich-regret-residual',family_stages=STAGES,
                planned_evaluations=45,planned_hands=5400000,
                independent_card_blocks=20000,no_model_promotion=True,no_automatic_retry=True,
                parent_teacher_knowledge_inherited=True,parent_rng_inherited=False,
                old51_replay_transferred=False,prior_card_ranges=prior)
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'stage-ready.json',dict(updated_utc=utc(),completed_stages=[],policies=ready_models))
    atomic_json(output/'preparation.json',dict(complete=True,prepared_utc=utc(),training_started=False,
                neural_gate_passed=True,prerequisites_passed=True,parent_iteration=360,
                old51_replay_transferred=False,teachers=teachers,baselines120=baselines))
    return cfg,worker,plan


def evaluation_jobs(study):
    study=Path(study); campaign=study/'campaign'
    return [dict(id=f'{field}-seed-{seed}-{label}',field=field,label=label,seed=seed,
                 source=str(campaign/'source'))
            for label in LABELS for seed in SEEDS for field in FIELDS]


def policy_for_job(study, job):
    ready=read(Path(study)/'stage-ready.json')
    entry=ready.get('policies',{}).get(job['label'],{}).get(str(job['seed']))
    if entry is None: return None
    if not Path(entry['path']).is_file() or digest(entry['path']) != entry['sha256']:
        raise ValueError('Immutable ready policy changed')
    return entry


def summarize_evaluations(rows, deals=20000):
    if set(rows) != {job['id'] for job in evaluation_jobs(Path('.'))}:
        raise ValueError('All45 independent result cells are required')
    absolute, changes, decisions = {}, {}, {}
    def arrays(label, field):
        result=[rows[f'{field}-seed-{seed}-{label}']['deal_blocks'] for seed in SEEDS]
        if any(len(a)!=deals for a in result): raise ValueError('Incomplete card-block coverage')
        return result
    for label in LABELS:
        for field in FIELDS:
            values=arrays(label,field)
            absolute[f'{field}/{label}']=dict(**interval([statistics.mean(x) for x in zip(*values)]),
                per_seed=[dict(seed=seed,**interval(v)) for seed,v in zip(SEEDS,values)])
    pairs=[('original120',label) for label in LABELS[1:]]
    pairs += [('parent360',label) for label in LABELS[2:]]+[('rich12','rich48'),('rich48','rich96')]
    for before,after in pairs:
        delta={}
        for field in FIELDS:
            per=[[b-a for a,b in zip(x,y)] for x,y in zip(arrays(before,field),arrays(after,field))]
            item=dict(**interval([statistics.mean(x) for x in zip(*per)]),
                      per_seed=[dict(seed=seed,**interval(v)) for seed,v in zip(SEEDS,per)])
            changes[f'{field}/{after}-minus-{before}']=item;delta[field]=item
        if before=='original120' and after.startswith('rich'):
            decisions[after]=adoption_conditions(delta)
    return dict(complete=True,completed_evaluations=45,total_hands=deals*45*6,
                independent_card_blocks=deals,absolute_scores=absolute,paired_changes=changes,
                decisions=decisions,model_promoted=False,
                statistical_unit='One shared card block averaged over six seats and three fixed seeds',
                adoption_baseline='Originalcurrent120; parent360 and family-stage changes are diagnostics')


def stop_owned(active):
    """Stop only still-owned process groups with matching process-start tokens."""
    for item in active.values():
        child=item['process']
        if child.poll() is None and process_start(child.pid)==item['start']:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for item in active.values():
        child=item['process']
        try: child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if process_start(child.pid)==item['start']:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
            child.wait(timeout=3)


def seed_progress(campaign, active, strict=True):
    rows={}
    for seed in SEEDS:
        folder=Path(campaign)/f'seed-{seed}'
        status=read(folder/'status.json') if (folder/'status.json').exists() else {}
        pointer=read(folder/'current.json') if (folder/'current.json').exists() else {}
        finite(status)
        iteration=status.get('iteration',0); checkpoint=pointer.get('iteration',0)
        if (type(iteration) is not int or type(checkpoint) is not int
                or not 0<=checkpoint<=96 or not 0<=iteration<=96):
            raise ValueError('Invalid new-family iteration or checkpoint state')
        if strict and status.get('status') in ('failed','interrupted'):
            raise RuntimeError('Rich training worker recorded failure: '+str(seed))
        iteration=max(iteration,checkpoint)
        diagnostics=folder/'diagnostics.jsonl'
        if diagnostics.exists():
            text=diagnostics.read_text();lines=text.splitlines()
            if lines and not text.endswith('\n'):
                # The writer commits a full JSON row with a trailing newline.
                # A concurrent reader must wait for a partial tail to finish.
                lines.pop()
            if lines:
                last=read_json_line(lines[-1]);finite(last)
                if last.get('iteration',0)>iteration:
                    # A diagnostic is committed immediately before status.
                    iteration=last['iteration']
        worker=active.get(f'train/{seed}')
        alive=bool(worker and command_matches(worker['process'].pid,worker['start'],
                    'train_rich_selfplay.py','--campaign',campaign))
        rows[str(seed)]=dict(new_family_iteration=iteration,checkpoint_iteration=checkpoint,
            worker_alive=alive,recorded_status=status.get('status','not_started'),
            replay_rows=status.get('replay_rows',[0]*6),fitting=status.get('fitting'),
            last_iteration_seconds=status.get('last_iteration_seconds'))
    return rows


def read_json_line(line):
    try:return json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError('Partial or malformed committed diagnostic line') from error


def public_status(study, state, cfg, plan_sha, active, completed, backups, rows,
                  phase, stage, began, error=None):
    campaign=Path(study)/'campaign';now=utc()
    seeds=seed_progress(campaign,active,strict=state=='running')
    evaluation_complete=len(rows)==45
    complete=state=='complete' and evaluation_complete and completed==STAGES
    running={name:dict(pid=item['process'].pid,process_start=item['start'],command=item['command'])
             for name,item in active.items() if item['process'].poll() is None
             and process_start(item['process'].pid)==item['start']}
    record=dict(status=state,phase=phase,observed_utc=now,run=Path(study).name,
        supervisor_pid=os.getpid(),supervisor_start=process_start(os.getpid()),
        elapsed_seconds=time.monotonic()-began,max_seconds=cfg.get('max_seconds',21600),
        parent_iteration=360,new_family='rich-regret-residual',planned_family_iterations=96,
        current_stage=stage,completed_training_stages=list(completed),
        training_complete=completed==STAGES,evaluation_complete=evaluation_complete,
        study_complete=complete,seed_progress=seeds,running=running,
        completed_evaluations=len(rows),planned_evaluations=45,
        verified_hands=len(rows)*120000,planned_hands=5400000,
        independent_card_blocks=20000,plan_sha256=plan_sha,
        average_policy_available=False,parent_rng_inherited=False,
        teacher_knowledge_inherited=True,old51_replay_transferred=False,
        legacy_exact_resume=False,model_promoted=False,automatic_retry=False,
        local_stage_backups={key:{k:value[k] for k in ('bytes','sha256','checkpoint_mode') if k in value}
                             for key,value in backups.items()},
        next_action=('Analyze the completed fixed study; no automatic extension or promotion' if complete else
                     'Inspect stopped experiment; no automatic retry' if state in ('failed','interrupted') else
                     'Continue the existing fixed96-round family study and its verified evaluation'),
        error=error)
    finite(record);atomic_json(Path(study)/'status.json',record)
    # This is the integrated writer. Do not start a second publisher process.
    return record


def run(output, config_path, prepared=False):
    output=output_path(output);root=output.parent;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry=root/'current.json'
    if registry.exists():
        previous=read(registry)
        if process_start(previous.get('pid'))==previous.get('process_start') and previous.get('process_start'):
            raise RuntimeError('A rich study supervisor is already alive; do not duplicate')
    if prepared:
        cfg=read(output/'config.json');validate_config(cfg);worker=read(output/'campaign/config.json')
        plan=read(output/'plan.json')
        if cfg!=read(config_path) or plan['config']!=cfg:
            raise ValueError('Prepared rich study configuration changed')
        verify_files(plan['frozen_files'])
        if (output/'status.json').exists():
            raise ValueError('An existing run may not be silently resumed or retried')
    else:cfg,worker,plan=prepare(output,Path(config_path).resolve())
    campaign=output/'campaign';plan_sha=digest(output/'plan.json')
    token=process_start(os.getpid())
    atomic_json(registry,dict(folder=str(output),run=output.name,pid=os.getpid(),process_start=token,
        started_utc=utc(),purpose='Authorized rich-input residual learning from teacher360; separate new-family rounds'))
    began=time.monotonic();active={};training={};rows={};completed=[];backups={}
    phase='preparing';stage=0;last_public=-math.inf;eval_jobs=evaluation_jobs(output)
    pending=list(eval_jobs);evaluation_enabled=cfg.get('evaluation_enabled',True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')

    def launch(name, command, job=None):
        path=output/'logs'/(name.replace('/','-')+'.log')
        stream=path.open('w')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        item=dict(process=child,start=process_start(child.pid),log=stream,
                  command=[str(x) for x in command],started=time.monotonic(),job=job)
        active[name]=item;return item

    def consume(name):
        item=active.pop(name);item['log'].close()
        if item['process'].poll()!=0:
            raise RuntimeError('Owned subprocess failed; no retry: '+name)
        return item

    def evaluate_tick(training_done):
        if not evaluation_enabled:return
        from evaluate_rich_selfplay import verify_results
        for name,item in list(active.items()):
            if not name.startswith('eval/'):continue
            code=item['process'].poll()
            if code is None:
                if time.monotonic()-item['started']>cfg.get('evaluation_seconds_per_cell',2400)+30:
                    raise TimeoutError('Rich evaluation cell exhausted its budget: '+name)
                continue
            item=consume(name);job=item['job'];cell=output/'evaluation'/job['id']
            result=verify_results(cell)
            if (result.get('complete') is not True or result.get('field')!=job['field']
                    or result.get('deals')!=20000 or result.get('hands')!=120000
                    or result.get('deal_seed')!=2035000000
                    or Path(result.get('candidate','')).resolve()!=Path(job['model']).resolve()
                    or result.get('candidate_sha256')!=job['ready_sha256']):
                raise ValueError('Evaluation cell does not match its frozen job')
            finite(result);rows[job['id']]=result
        limit=evaluation_parallelism(training,campaign,training_done)
        running=sum(name.startswith('eval/') for name in active)
        for job in pending[:]:
            if running>=limit:break
            policy=policy_for_job(output,job)
            if policy is None:continue
            cell=output/'evaluation'/job['id']
            if cell.exists():raise FileExistsError('A previous/incomplete evaluation may not be reused')
            command=[str(PYTHON),str(ROOT/'scripts/evaluate_rich_selfplay.py'),'--model',policy['path'],
                     '--source',str(campaign/'source'),'--field',job['field'],'--deals','20000',
                     '--deal-seed','2035000000','--timeout',str(cfg.get('evaluation_seconds_per_cell',2400)),
                     '--output',str(cell)]
            actual=dict(job,model=policy['path'],ready_sha256=policy['sha256'])
            launch('eval/'+job['id'],command,actual);pending.remove(job);running+=1
        atomic_json(output/'evaluation-status.json',dict(status='complete' if len(rows)==45 else 'running',
            observed_utc=utc(),completed_evaluations=len(rows),planned_evaluations=45,
            verified_hands=len(rows)*120000,independent_card_blocks=20000,
            finished=sorted(rows),pending=[job['id'] for job in pending],
            running={name[5:]:dict(pid=item['process'].pid,process_start=item['start'])
                     for name,item in active.items() if name.startswith('eval/') and item['process'].poll() is None}))

    def status(state='running', error=None, force=False):
        nonlocal last_public
        record=public_status(output,state,cfg,plan_sha,active,completed,backups,rows,
                             phase,stage,began,error)
        if force or time.monotonic()-last_public>=15 or state!='running':
            atomic_json(ROOT/'reports/training-neural-status.json',record)
            last_public=time.monotonic()

    def monitor(training_done=False):
        if time.monotonic()-began>cfg.get('max_seconds',21600):
            raise TimeoutError('The fixed6-hour rich-study budget is exhausted')
        if digest(output/'plan.json')!=plan_sha:raise ValueError('Immutable study plan changed')
        verify_files(plan['frozen_files'])
        for name,item in active.items():
            if item['process'].poll() not in (None,0):
                raise RuntimeError('Subprocess failed; no automatic retry: '+name)
        evaluate_tick(training_done);status()

    def stop(*unused):raise InterruptedError('Rich study interrupted; no automatic retry')
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        atomic_json(campaign/'launch.json',dict(status='preparing',through=0));status(force=True)
        for stage in STAGES:
            phase='training';training={seed:launch(f'train/{seed}',
                [str(PYTHON),str(ROOT/'scripts/train_rich_selfplay.py'),'--campaign',str(campaign),
                 '--seed',str(seed),'--through',str(stage)]) for seed in SEEDS}
            atomic_json(campaign/'launch.json',dict(status='running',through=stage,
                pids={str(seed):item['process'].pid for seed,item in training.items()},
                process_starts={str(seed):item['start'] for seed,item in training.items()}))
            stage_start=time.monotonic()
            while any(item['process'].poll() is None for item in training.values()):
                if time.monotonic()-stage_start>cfg.get('training_seconds_per_stage',5400)+30:
                    raise TimeoutError('A rich training stage exceeded90minutes')
                monitor();time.sleep(3)
            for seed in SEEDS:consume(f'train/{seed}')
            phase='verifying_stage';status(force=True);verify_files(plan['frozen_files'])
            from snapshot_rich_selfplay import verify_stage
            proof=verify_stage(campaign,stage,worker)
            if not proof.get('passed') or proof.get('family_iteration')!=stage:
                raise ValueError('Rich checkpoint stage verification did not pass')
            atomic_json(output/f'stage-{stage}-verification.json',proof)
            ready=read(output/'stage-ready.json');models={}
            for seed in SEEDS:
                path=campaign/f'seed-{seed}'/f'policy-{stage:05d}.json'
                if not path.is_file():raise ValueError('Verified immutable rich policy descriptor missing')
                models[str(seed)]=dict(path=str(path),sha256=digest(path))
            completed.append(stage);ready['completed_stages']=completed[:]
            ready['policies']['rich'+str(stage)]=models;ready['updated_utc']=utc()
            atomic_json(output/'stage-ready.json',ready)
            phase='snapshotting';status(force=True)
            exporter=launch('snapshot',[str(PYTHON),str(ROOT/'scripts/snapshot_rich_selfplay.py'),
                                      '--study',str(output),'--stage',str(stage)])
            while exporter['process'].poll() is None:monitor();time.sleep(3)
            consume('snapshot');backup=read(output/f'stage-{stage}-backup.json')
            if digest(backup['archive'])!=backup['sha256'] or Path(backup['archive']).stat().st_size!=backup['bytes']:
                raise ValueError('Verified rich stage archive changed')
            backups[str(stage)]=backup;monitor()
        atomic_json(campaign/'launch.json',dict(status='complete',through=96))
        phase='evaluating';status(force=True)
        if not evaluation_enabled:
            phase='evaluation_pending';status('evaluation_pending',force=True)
            return dict(training_complete=True,study_complete=False,evaluation_pending=True)
        while pending or any(name.startswith('eval/') for name in active):
            monitor(training_done=True);time.sleep(3)
        phase='verifying_evaluation';status(force=True);verify_files(plan['frozen_files'])
        result=summarize_evaluations(rows);result.update(completed_utc=utc(),
            source_provenance_verified=True,raw_hands_verified=True,
            result_files={job['id']:digest(output/'evaluation'/job['id']/'results.json') for job in eval_jobs})
        atomic_json(output/'evaluation-summary.json',result)
        atomic_json(output/'summary.json',dict(complete=True,completed_utc=utc(),
            parent_iteration=360,new_family_selfplay_iterations=96,completed_stages=completed,
            completed_evaluations=45,total_hands=5400000,independent_card_blocks=20000,
            evaluation_summary_sha256=digest(output/'evaluation-summary.json'),
            plan_sha256=plan_sha,raw_hands_verified=True,source_provenance_verified=True,
            no_model_promotion=True,no_automatic_retry=True,backups=backups))
        phase='complete';status('complete',force=True);return result
    except BaseException as error:
        previous=phase;phase='interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed'
        try:status(phase,error=type(error).__name__+': '+str(error),force=True)
        except BaseException:
            failure=dict(status=phase,phase=phase,observed_utc=utc(),run=output.name,
                error=type(error).__name__+': '+str(error),failure_stage=previous,
                no_automatic_retry=True,study_complete=False,parent_iteration=360,
                new_family='rich-regret-residual',completed_training_stages=completed)
            atomic_json(output/'failure.json',failure);atomic_json(output/'status.json',failure)
            atomic_json(ROOT/'reports/training-neural-status.json',failure)
        atomic_json(campaign/'launch.json',dict(status=phase,error=str(error)))
        raise
    finally:
        stop_owned(active)
        for item in active.values():item['log'].close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run-prepared',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        prepare(args.output,args.config)
        print(json.dumps(dict(prepared=True,training_started=False,output=str(args.output))))
    else:run(args.output,args.config,args.run_prepared)
