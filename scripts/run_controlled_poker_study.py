"""Predeclared one-factor refit diagnostics, using unchanged parent fit kernels."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import hashlib
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

from controlled_poker_refit import (SEEDS, ARMS, BASELINES, FIELDS, PAIRS, LEGACY_ARMS,
                                    read, utc, verify_files)
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from evaluate_compact_learning_curve import adoption_conditions
from run_compact_refit_probe import interval, previous_ranges, check_card_range

ROOT_FOLDER = ROOT / 'artifacts/controlled-refit'
LABELS = BASELINES + ARMS
FIT_JOBS = len(ARMS) * len(SEEDS)
CELLS = len(LABELS) * len(SEEDS) * len(FIELDS)
HELPERS = ['controlled_poker_refit.py', 'run_controlled_poker_study.py',
           'export_controlled_poker_probe.py', 'plan_controlled_backup_retention.py',
           'compact_lightweight_backup.py', 'fullhouse_v3_common.py',
           'rich_regret_model.py', 'rich_poker_observation.py', 'rich_poker_runtime.py',
           'evaluate_rich_selfplay.py', 'train_rich_selfplay.py',
           'fullhouse_runtime.py', 'fullhouse_v4_runtime.py',
           'regret_partition.py', 'regret_fitting.py', 'scaled_regret_fitting.py',
           'run_compact_refit_probe.py', 'evaluate_compact_learning_curve.py',
           'raw_objective_fitting.py', 'snapshot_rich_selfplay.py']


def validate_config(cfg):
    fixed = dict(seeds=SEEDS, arms=ARMS, evaluation_deals=20000,
                 evaluation_deal_seed=2037000000, parent_iteration=360,
                 new_selfplay_iterations=0, max_fitting_workers=2,
                 max_evaluation_workers=3, max_seconds=21600,
                 fitting_seconds_per_job=2700, evaluation_seconds_per_cell=2400)
    if any(cfg.get(k) != value for k, value in fixed.items()):
        raise ValueError('The predeclared controlled-refit protocol changed')
    if cfg.get('automatic_retry') is not False or cfg.get('model_promotion') is not False:
        raise ValueError('No automatic retries, extensions, or model replacement authorized')


def alive(pid, start):
    if process_start(pid) != start:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def prepare(study, config_path):
    study = Path(study).resolve(); cfg = read(config_path); validate_config(cfg)
    if study.parent != ROOT_FOLDER or not re.fullmatch(r'\d{8}T\d{6,12}Z', study.name):
        raise ValueError('Use a fresh timestamped controlled-refit folder')
    if study.exists():
        raise FileExistsError('A prepared/failed study must not be overwritten or automatically retried')
    legacy = Path(cfg['legacy_parent']).resolve(); rich = Path(cfg['rich_parent']).resolve()
    if not read(legacy / 'summary.json').get('complete') or not read(rich / 'summary.json').get('complete'):
        raise ValueError('Both actual completed parents are required')
    prior = previous_ranges(study); check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], prior)
    study.mkdir(parents=True); (study / 'logs').mkdir(); (study / 'baselines').mkdir()
    shutil.copytree(legacy / 'campaign/source', study / 'source',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbi', '*.nbc', '.git'))
    shutil.copytree(legacy / 'campaign/tools', study / 'legacy-tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    parents = dict(legacy={}, rich={}); frozen = {}; policies = {label: {} for label in BASELINES}
    for family, parent, iteration in [('legacy', legacy, 360), ('rich', rich, 96)]:
        proof = read(parent / f'stage-{iteration}-verification.json')
        if proof.get('passed') is not True:
            raise ValueError('Parent checkpoint verification is missing')
        frozen[str(parent / f'stage-{iteration}-verification.json')] = digest(parent / f'stage-{iteration}-verification.json')
        frozen[str(parent / 'summary.json')] = digest(parent / 'summary.json')
        for seed in SEEDS:
            folder = parent / 'campaign' / f'seed-{seed}'
            pointer = read(folder / 'current.json')
            if pointer['iteration'] != iteration:
                raise ValueError('Parent pointer is not the verified final checkpoint')
            checkpoint = folder / 'checkpoints' / pointer['directory']
            for name, sha in pointer['files'].items():
                if digest(checkpoint / name) != sha:
                    raise ValueError('Parent checkpoint changed')
                frozen[str(checkpoint / name)] = sha
            parents[family][str(seed)] = dict(checkpoint=str(checkpoint))
    for seed in SEEDS:
        for label, path in [('original120', legacy / 'campaign' / f'seed-{seed}/current_00120.npz'),
                            ('parent360', legacy / 'campaign' / f'seed-{seed}/current_00360.npz')]:
            target = study / 'baselines' / f'{label}-seed-{seed}.npz'
            shutil.copyfile(path, target)
            if digest(target) != digest(path):
                raise ValueError('Baseline copy mismatch')
            frozen[str(path)] = digest(path); frozen[str(target)] = digest(target)
            policies[label][str(seed)] = dict(path=str(target), sha256=digest(target))
        parents['legacy'][str(seed)]['model'] = policies['parent360'][str(seed)]['path']
        original = rich / 'campaign' / f'seed-{seed}/policy-00096.json'
        spec = read(original)
        residual = Path(spec['model']); teacher = Path(spec['teacher'])
        if digest(residual) != spec['model_sha256'] or digest(teacher) != spec['teacher_sha256']:
            raise ValueError('Previous rich96 dependencies changed')
        target = study / 'baselines' / f'previous-rich96-seed-{seed}.npz'
        shutil.copyfile(residual, target)
        spec.update(model=str(target), teacher=policies['parent360'][str(seed)]['path'],
                    source=str(study / 'source'))
        descriptor = target.with_suffix('.json'); atomic_json(descriptor, spec)
        policies['previous_rich96'][str(seed)] = dict(path=str(descriptor), sha256=digest(descriptor))
        for p in [original, residual, teacher, target, descriptor]:
            frozen[str(p)] = digest(p)
    for parent_registry in ['compact-v4', 'bounded-continuation', 'rich-selfplay',
                            'continuation-study', 'refit-probe', 'foundation-study']:
        p = ROOT / 'artifacts' / parent_registry / 'current.json'
        if p.exists():
            frozen[str(p)] = digest(p)
    for name in HELPERS:
        path = ROOT / 'scripts' / name
        if not path.is_file():
            raise ValueError('All audited helpers must exist before the plan is frozen: '+name)
        frozen[str(path)] = digest(path)
    from fullhouse_runtime import Fullhouse, SOURCE
    public = Fullhouse()
    opponent_registry = ROOT / 'artifacts/selfplay/current.json'
    opponent = read(opponent_registry); opponent_model = ROOT / opponent['model']
    if digest(opponent_model) != opponent['sha256']:
        raise ValueError('Frozen NoRegrets opponent does not verify')
    for path in [opponent_registry, opponent_model, Path(public.model),
                 ROOT/'configs/fullhouse-model.json',
                 ROOT/'.cache/noregrets/target/release/poker-bridge',
                 SOURCE/'bot/features.py', SOURCE/'bot/deep_cfr_lookup.py',
                 SOURCE/'data/preflop_equity.npz']:
        frozen[str(path)] = digest(path)
    for folder in ['source', 'legacy-tools']:
        frozen.update({str(p): digest(p) for p in (study / folder).rglob('*') if p.is_file()})
    # Frozen legacy kernels must be the exact former source, not a rewritten approximation.
    for name in ['compact_regret.py', 'root_regret_replay.py', 'regret_fitting.py',
                 'scaled_regret_fitting.py', 'regret_partition.py']:
        path = study / 'legacy-tools' / name
        if path.exists() and digest(path) != digest(ROOT / 'scripts' / name):
            raise ValueError('Reference kernel differs from the audited original: '+name)
    atomic_json(study / 'config.json', cfg)
    frozen[str(study / 'config.json')] = digest(study / 'config.json')
    plan = dict(created_utc=utc(), config=cfg, parents=parents, source=str(study / 'source'),
        legacy_tools=str(study / 'legacy-tools'), frozen_files=frozen,
        fixed_replay=True, new_selfplay_iterations=0, planned_fitting_jobs=FIT_JOBS,
        planned_evaluations=CELLS, planned_hands=CELLS*120000, independent_card_blocks=20000,
        primary_pairs=PAIRS, simultaneous_primary_intervals='Bonferroni across5pairs*3fields=15',
        limitation='Input pair uses existing rich96 replay; budget arms use existing360 replay. Cross-group causal attribution and rollout/sampling effects are not identified.',
        baseline_policies=policies, prior_card_ranges=prior,
        no_automatic_retry=True, no_model_promotion=True)
    atomic_json(study / 'plan.json', plan)
    atomic_json(study / 'stage-ready.json', dict(policies=policies, completed_fitting_jobs=[]))
    return plan


def evaluation_jobs():
    return [dict(id=f'{field}-seed-{seed}-{label}', label=label, seed=seed, field=field)
            for label in LABELS for seed in SEEDS for field in FIELDS]


def validate_pairs(fitted):
    for seed in SEEDS:
        rows = [fitted.get(f'{label}-seed-{seed}') for label in LEGACY_ARMS]
        present = [r for r in rows if r]
        if len({r['initial_parameter_sha256'] for r in present}) > 1:
            raise ValueError('Legacy arms have different initial fitting parameters')
        if len({r['opening_sha256'] for r in present}) > 1:
            raise ValueError('Legacy opening expert changed across the controlled comparison')
        if all(rows):
            full = [r for r in rows if r['arm'] != 'legacy_small_replay']
            if any(r['selected_row_sha256'] != full[0]['selected_row_sha256'] for r in full[1:]):
                raise ValueError('Compute-only arms changed the fixed replay rows')
            control, corrected = (fitted[f'{label}-seed-{seed}'] for label in
                                  ['legacy_control', 'legacy_raw_objective'])
            if any(control[k] != corrected[k] for k in
                   ['sampling_cdf_sha256', 'fitting_random_stream_sha256']):
                raise ValueError('Loss-only arm changed the sampling distribution/order')
        a, b = (fitted.get(f'{label}-seed-{seed}') for label in ['rich_masked', 'rich_full'])
        if a and b and any(a[k] != b[k] for k in ['initial_parameter_sha256',
                'selected_row_sha256', 'final_rng_sha256', 'replay_rows']):
            raise ValueError('The input-only pair changed data, initialization, or fitting draws')


def summarize(rows, deals=20000):
    if set(rows) != {j['id'] for j in evaluation_jobs()}:
        raise ValueError('All90 verified cells required for the final comparison')
    for field in FIELDS:
        signatures={json.dumps(r.get('opponents'),sort_keys=True) for key,r in rows.items()
                    if key.startswith(field+'-')}
        if len(signatures)!=1:
            raise ValueError('Opponent composition changed across the controlled comparison')
    def arrays(label, field):
        data = [rows[f'{field}-seed-{seed}-{label}']['deal_blocks'] for seed in SEEDS]
        if any(len(v) != deals for v in data):
            raise ValueError('Card-block coverage differs')
        return data
    def change(before, after, field):
        per = [[b-a for a,b in zip(x,y)] for x,y in zip(arrays(before,field), arrays(after,field))]
        values = [statistics.mean(x) for x in zip(*per)]
        return dict(**interval(values), per_seed=[dict(seed=seed, **interval(v)) for seed,v in zip(SEEDS,per)])
    absolute = {f'{field}/{label}': interval([statistics.mean(x) for x in zip(*arrays(label, field))])
                for label in LABELS for field in FIELDS}
    comparisons = len(PAIRS) * len(FIELDS)
    primary = {}; z = statistics.NormalDist().inv_cdf(1-.05/(2*comparisons))
    for before, after in PAIRS:
        for field in FIELDS:
            item = change(before, after, field)
            half = item['ci95_half_width'] / 1.96 * z
            item.update(simultaneous_lower=item['mean']-half, simultaneous_upper=item['mean']+half,
                        family_comparisons=comparisons)
            primary[f'{field}/{after}-minus-{before}'] = item
    diagnostics = {}; decisions = {}
    for label in ARMS:
        delta = {}
        for field in FIELDS:
            delta[field] = change('original120', label, field)
            diagnostics[f'{field}/{label}-minus-original120'] = delta[field]
            diagnostics[f'{field}/{label}-minus-parent360'] = change('parent360', label, field)
        decisions[label] = adoption_conditions(delta)
    return dict(complete=True, planned_evaluations=CELLS, completed_evaluations=CELLS,
        total_hands=deals*CELLS*6, independent_card_blocks=deals,
        absolute_scores=absolute, primary_one_factor_changes=primary,
        diagnostic_changes=diagnostics, adoption_against_original120=decisions,
        statistical_unit='One shared card block averaged across seats and three fixed seeds',
        new_selfplay_iterations=0, fixed_replay=True, model_promoted=False,
        causal_scope='Only the five declared within-dataset comparisons; no claim about unseen online selfplay or one unique cause of the earlier regression')


def run(study, config_path, prepared=False):
    study = Path(study).resolve(); ROOT_FOLDER.mkdir(parents=True, exist_ok=True)
    lock = (ROOT_FOLDER / 'runner.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    registry = ROOT_FOLDER / 'current.json'
    if registry.exists():
        old = read(registry)
        if alive(old['pid'], old['process_start']):
            raise ValueError('A controlled study is already alive; no duplicate run')
    plan = read(study / 'plan.json') if prepared else prepare(study, config_path)
    cfg = plan['config']; validate_config(cfg); verify_files(plan['frozen_files'])
    if prepared and any((study / name).exists() for name in ['status.json', 'summary.json', 'failure.json']):
        raise ValueError('A previously started study cannot be automatically restarted')
    atomic_json(registry, dict(folder=str(study), run=study.name, pid=os.getpid(),
        process_start=process_start(os.getpid()), started_utc=utc(), fixed_replay=True,
        purpose='User-authorized controlled diagnosis; no new selfplay rounds'))
    began = time.monotonic(); active = {}; fitted = {}; results = {}; ready = read(study / 'stage-ready.json')
    fit_queue = deque((arm, seed) for arm in ARMS for seed in SEEDS)
    evaluation_queue = deque(evaluation_jobs()); last_public = -math.inf
    def launch(name, cmd):
        log = (study / 'logs' / (name.replace('/', '-')+'.log')).open('w')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        active[name] = dict(process=p, start=process_start(p.pid), command=cmd,
                            log=log, started=time.monotonic())
    def status(state='running', error=None, force=False):
        nonlocal last_public
        live = {k:dict(pid=v['process'].pid, process_start=v['start'], command=v['command'])
                for k,v in active.items() if v['process'].poll() is None and alive(v['process'].pid,v['start'])}
        record = dict(status=state, phase=('complete' if state=='complete' else 'fitting_and_evaluating'
            if len(fitted)<FIT_JOBS else 'evaluating'), observed_utc=utc(), run=study.name,
            supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
            study_kind='controlled-fixed-replay-refit', fixed_replay=True,
            new_selfplay_iterations=0, parent_iteration=360,
            completed_fitting_jobs=len(fitted), planned_fitting_jobs=FIT_JOBS,
            fitting_complete=len(fitted)==FIT_JOBS, completed_evaluations=len(results),
            planned_evaluations=CELLS, verified_hands=len(results)*120000,
            planned_hands=CELLS*120000, independent_card_blocks=20000,
            evaluation_complete=len(results)==CELLS, study_complete=state=='complete',
            running=live, error=error, model_promoted=False, automatic_retry=False,
            elapsed_seconds=time.monotonic()-began, max_seconds=cfg['max_seconds'],
            next_action=('Analyze the controlled comparisons; no automatic extension or model replacement'
                if state=='complete' else 'Inspect failure; no automatic retry' if state!='running'
                else 'Continue the existing fixed refit/evaluation plan'))
        atomic_json(study / 'status.json', record)
        if force or state!='running' or time.monotonic()-last_public>=15:
            atomic_json(ROOT/'reports/controlled-refit-status.json', record)
            atomic_json(ROOT/'reports/training-neural-status.json', record); last_public=time.monotonic()
    def interrupt(signum, frame):
        raise InterruptedError('Study interrupted; no automatic retry')
    handlers = {sig:signal.signal(sig, interrupt) for sig in [signal.SIGTERM,signal.SIGINT]}
    try:
        from evaluate_rich_selfplay import verify_results
        status(force=True)
        while fit_queue or evaluation_queue or active:
            if time.monotonic()-began > cfg['max_seconds']:
                raise TimeoutError('The predeclared six-hour study budget expired')
            for name, item in list(active.items()):
                limit = cfg['fitting_seconds_per_job'] if name.startswith('fit/') else cfg['evaluation_seconds_per_cell']+30
                if time.monotonic()-item['started'] > limit:
                    raise TimeoutError('Owned job exceeded its fixed time budget: '+name)
                code = item['process'].poll()
                if code is None:
                    continue
                item['log'].close(); del active[name]
                if code != 0:
                    raise RuntimeError(f'{name} failed with exit{code}; see its log; no retry')
                if name.startswith('fit/'):
                    key = name[4:]; row = read(study/'fitting'/key/'result.json')
                    if not row.get('complete') or digest(row['model']) != row['model_sha256']:
                        raise ValueError('Fitted candidate did not verify')
                    fitted[key] = row; validate_pairs(fitted)
                    ready['policies'].setdefault(row['arm'], {})[str(row['seed'])] = dict(
                        path=row['model'], sha256=row['model_sha256'])
                    ready['completed_fitting_jobs'] = list(fitted); atomic_json(study/'stage-ready.json',ready)
                else:
                    key = name[5:]; job = next(j for j in evaluation_jobs() if j['id']==key)
                    policy = ready['policies'][job['label']][str(job['seed'])]
                    row = verify_results(study/'evaluation'/key, replay=False,
                        expected=dict(candidate_sha256=policy['sha256'], field=job['field'],
                                      deals=20000, deal_seed=cfg['evaluation_deal_seed']))
                    results[key] = row
            while fit_queue and sum(name.startswith('fit/') for name in active)<2:
                arm, seed = fit_queue.popleft()
                launch(f'fit/{arm}-seed-{seed}', [str(PYTHON),str(ROOT/'scripts/controlled_poker_refit.py'),
                    '--study',str(study),'--arm',arm,'--seed',str(seed)])
            eval_limit = 1 if fit_queue or any(n.startswith('fit/') for n in active) else 3
            for job in list(evaluation_queue):
                if sum(name.startswith('eval/') for name in active) >= eval_limit:
                    break
                policy = ready['policies'].get(job['label'],{}).get(str(job['seed']))
                if policy is None:
                    continue
                evaluation_queue.remove(job)
                launch('eval/'+job['id'], [str(PYTHON),str(ROOT/'scripts/evaluate_rich_selfplay.py'),
                    '--model',policy['path'],'--source',plan['source'],'--field',job['field'],
                    '--deals','20000','--deal-seed',str(cfg['evaluation_deal_seed']),
                    '--timeout',str(cfg['evaluation_seconds_per_cell']),
                    '--output',str(study/'evaluation'/job['id'])])
            status(); time.sleep(3)
        verify_files(plan['frozen_files']); validate_pairs(fitted)
        summary = summarize(results); summary.update(completed_utc=utc(),
            result_files={key:digest(study/'evaluation'/key/'results.json') for key in results},
            raw_hands_verified=True, source_provenance_verified=True)
        atomic_json(study/'evaluation-summary.json',summary)
        atomic_json(study/'summary.json',dict(complete=True, completed_utc=utc(),
            fixed_replay=True,new_selfplay_iterations=0,completed_fitting_jobs=FIT_JOBS,
            completed_evaluations=CELLS,total_hands=CELLS*120000,independent_card_blocks=20000,
            evaluation_summary_sha256=digest(study/'evaluation-summary.json'),
            no_automatic_retry=True,no_model_promotion=True))
        from export_controlled_poker_probe import export_probe
        export_probe(study)
        status('complete',force=True)
        return summary
    except BaseException as error:
        state = 'interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed'
        atomic_json(study/'failure.json',dict(status=state,observed_utc=utc(),
            error=type(error).__name__+': '+str(error),completed_fitting_jobs=len(fitted),
            completed_evaluations=len(results),automatic_retry=False))
        status(state,error=type(error).__name__+': '+str(error),force=True)
        raise
    finally:
        for item in active.values():
            if item['process'].poll() is None and alive(item['process'].pid,item['start']):
                try:os.killpg(item['process'].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for item in active.values():
            try:item['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                if alive(item['process'].pid,item['start']):
                    os.killpg(item['process'].pid,signal.SIGKILL)
            item['log'].close()
        for sig, handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-prepared',action='store_true')
    a=p.parse_args()
    if a.prepare_only:
        result=prepare(a.study,a.config);print(json.dumps(dict(prepared=True,study=str(a.study),new_selfplay_iterations=0)))
    else:run(a.study,a.config,a.run_prepared)
