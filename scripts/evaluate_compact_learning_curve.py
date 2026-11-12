"""Evaluate a fixed continued-self-play budget on 20,000 common card blocks.

This helper never trains, changes a controller, promotes a model, or retries a
failed cell. It waits for immutable checkpoint policies, verifies every raw
hand, and treats shared cards as one statistical block across seats and seeds.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import check_card_range, interval, previous_ranges, utc, verify_result


FIELDS = ['public', 'mixed', 'noregrets']
ITERATIONS = [120, 180, 240]
KINDS = ['current', 'mixture']
SEEDS = [342, 343, 344]


def validate_config(cfg):
    if (cfg['seeds'] != SEEDS or cfg['evaluation_iterations'] != ITERATIONS
            or cfg.get('variants') != KINDS or cfg.get('stages') != [180, 240]
            or cfg['evaluation_fields'] != FIELDS or cfg['evaluation_deals'] != 20000):
        raise ValueError('The seeds, fields, current/mixture policies, checkpoints and20,000 cards are fixed')
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], {})
    for key, default, ceiling in (('evaluation_seconds_per_cell', 2400, 3600),
                                  ('max_seconds', 14400, 14400)):
        value = cfg.get(key, default)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError('Invalid evaluation time budget: ' + key)
    if cfg.get('promotion', 'none') != 'none':
        raise ValueError('The learning-curve study cannot automatically promote a model')


def evaluation_jobs(campaign, cfg):
    campaign = Path(campaign).resolve()
    return [dict(id=f'{field}-seed-{seed}-{kind}-{iteration}', field=field,
                 variant=kind, iteration=iteration, seed=seed,
                 model=str(campaign/f'seed-{seed}'/f'{kind}_{iteration:05d}.{ "json" if kind == "mixture" else "npz"}'),
                 source=str(campaign/'source'))
            for iteration in cfg['evaluation_iterations'] for kind in KINDS
            for seed in cfg['seeds'] for field in cfg['evaluation_fields']]


def process_matches(pid, start, script=None, output=None):
    if type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start.isdigit():
        return False
    if process_start(pid) != start:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    except (OSError, UnicodeError):
        return False
    if not any(args) or (script is not None and not any(Path(arg).name == script for arg in args)):
        return False
    if output is not None:
        try:
            position = args.index('--output')
            if Path(args[position+1]).resolve() != Path(output).resolve():
                return False
        except (ValueError, IndexError):
            return False
    return True


def evaluation_parallelism(launch):
    """A stale completion label cannot permit three evaluators beside workers."""
    state = launch.get('status')
    if state not in ('running', 'complete'):
        raise ValueError('Training launch state is not evaluable')
    pids, starts = launch.get('pids', {}), launch.get('process_starts', {})
    if not isinstance(pids, dict) or not isinstance(starts, dict):
        raise ValueError('Invalid training process registry')
    live = any(process_matches(pid, starts.get(str(seed)), 'train_compact_selfplay.py')
               for seed, pid in pids.items())
    return 3 if state == 'complete' and not live else 1


def policy_ready(job):
    path = Path(job['model']); status_path = path.parent/'status.json'
    if not status_path.exists():
        return False
    saved = json.loads(status_path.read_text())
    if saved.get('status') in ('failed', 'interrupted'):
        raise ValueError('Training seed failed before evaluation')
    if type(saved.get('iteration')) is not int or saved['iteration'] < job['iteration']:
        return False
    if not path.is_file():
        raise ValueError('Committed checkpoint is missing its policy artifact')
    return True


def verified_stage_evidence(study, parent, job):
    """Wait for supervisor verification, not just a newly visible NPZ file."""
    study, parent = Path(study), Path(parent)
    ready = json.loads((study/'stage-ready.json').read_text())
    stage = job['iteration']; entry = ready.get('checkpoints', {}).get(str(stage))
    if entry is None:
        return None
    if entry.get('iteration') != stage:
        raise ValueError('Stage readiness does not match the evaluation checkpoint')
    if stage == 120:
        if entry.get('parent_verified') is not True:
            raise ValueError('Original120 parent is not verified ready')
        original = parent/f"seed-{job['seed']}"/Path(job['model']).name
        return dict(stage=stage, candidate_sha256=digest(original),
                    verified_files={str(original.resolve()):digest(original)},
                    scope='Exact original120 policy copy')
    path = study/f'stage-{stage}-verification.json'
    sha = digest(path)
    if sha != entry.get('verification_sha256'):
        raise ValueError('Stage checkpoint verification changed')
    report = json.loads(path.read_text())
    if (report.get('complete') is not True or report.get('passed') is not True
            or report.get('stage') != stage):
        raise ValueError('Checkpoint stage verification did not pass')
    seeds = report.get('seeds', [])
    if [row.get('seed') for row in seeds] != SEEDS:
        raise ValueError('Stage verification did not include every fixed seed')
    row = next(row for row in seeds if row['seed'] == job['seed'])
    if row.get('iteration') != stage:
        raise ValueError('Stage verification seed has wrong iteration')
    return dict(stage=stage, candidate_sha256=row[job['variant']+'_sha256'],
                checkpoint_sha256=row['checkpoint_sha256'],
                verified_files={str(path.resolve()):sha},
                scope='Supervisor-verified complete checkpoint stage')


def freeze_policy(job):
    """Check complete mixture history before any sampled hand can use it."""
    path = Path(job['model']).resolve(); frozen = {str(path): digest(path)}
    if job['variant'] == 'mixture':
        spec = json.loads(path.read_text()); models = spec.get('models', [])
        if (spec.get('model_type') != 'compact_snapshot_mixture'
                or spec.get('training_revision') != 'r1-cpu-v4'
                or spec.get('iteration') != job['iteration']
                or [m.get('iteration') for m in models] != list(range(1, job['iteration']+1))
                or any(m.get('weight') != m.get('iteration') for m in models)):
            raise ValueError('Incomplete or incorrectly weighted checkpoint mixture')
        for item in models:
            snapshot = (path.parent/item['file']).resolve()
            if snapshot.parent != path.parent or not snapshot.is_file():
                raise ValueError('Mixture snapshot is missing or escapes its seed folder')
            sha = digest(snapshot)
            if sha != item.get('sha256'):
                raise ValueError('Mixture snapshot changed')
            frozen[str(snapshot)] = sha
    return frozen


def verify_frozen(frozen):
    for name, sha in frozen.items():
        if digest(name) != sha:
            raise ValueError('Frozen evaluation evidence changed: ' + name)


def comparison(rows, cfg, before_kind, before_iteration, after_kind, after_iteration):
    result = {}
    for field in cfg['evaluation_fields']:
        per_seed = []
        for seed in cfg['seeds']:
            a = rows[f'{field}-seed-{seed}-{before_kind}-{before_iteration}']['deal_blocks']
            b = rows[f'{field}-seed-{seed}-{after_kind}-{after_iteration}']['deal_blocks']
            if len(a) != cfg['evaluation_deals'] or len(b) != len(a):
                raise ValueError('Incomplete paired card coverage')
            per_seed.append([right-left for left, right in zip(a, b)])
        result[field] = dict(**interval([statistics.mean(block) for block in zip(*per_seed)]),
            per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], per_seed)])
    return result


def adoption_conditions(changes):
    public, mixed, weak = [changes[field] for field in FIELDS]
    conditions = dict(public_lower_above_zero=public['lower'] > 0,
        at_least_two_public_seeds_improve=sum(row['mean'] > 0 for row in public['per_seed']) >= 2,
        mixed_lower_above_minus_25=mixed['lower'] >= -25,
        noregrets_lower_above_minus_50=weak['lower'] >= -50)
    return dict(conditions=conditions, original_four_conditions_pass=all(conditions.values()),
                deployment_authorized=False,
                scope='Candidate minus originalcurrent120; adoption evidence only, not the fixed research budget')


def summarize(rows, cfg, through):
    iterations = [i for i in cfg['evaluation_iterations'] if i <= through]
    absolute, paired, decisions = {}, {}, {}
    for iteration in iterations:
        for kind in KINDS:
            for field in cfg['evaluation_fields']:
                per = [rows[f'{field}-seed-{seed}-{kind}-{iteration}']['deal_blocks'] for seed in cfg['seeds']]
                if any(len(values) != cfg['evaluation_deals'] for values in per):
                    raise ValueError('Wrong card count in completed checkpoint')
                absolute[f'{field}/{kind}-{iteration}'] = dict(
                    **interval([statistics.mean(block) for block in zip(*per)]),
                    per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], per)])
            if iteration > 120:
                change = comparison(rows, cfg, 'current', 120, kind, iteration)
                for field, value in change.items():
                    paired[f'{field}/{kind}-{iteration}-minus-current-120'] = value
                decisions[f'{kind}-{iteration}'] = adoption_conditions(change)
    for iteration in iterations:
        for field, value in comparison(rows, cfg, 'current', iteration, 'mixture', iteration).items():
            paired[f'{field}/mixture-{iteration}-minus-current-{iteration}'] = value
    for before, after in ((120, 180), (120, 240), (180, 240)):
        if after not in iterations:
            continue
        for kind in KINDS:
            for field, value in comparison(rows, cfg, kind, before, kind, after).items():
                paired[f'{field}/{kind}-{after}-minus-{kind}-{before}'] = value
    expected = len(iterations)*len(KINDS)*len(cfg['seeds'])*len(cfg['evaluation_fields'])
    return dict(complete=True, through_iteration=through, completed_evaluations=expected,
        total_hands=expected*cfg['evaluation_deals']*6,
        independent_card_blocks=cfg['evaluation_deals'], absolute_scores=absolute,
        paired_changes=paired, decisions=decisions, raw_hands_verified=True,
        statistical_unit='One card block shared across six seats, three fixed training seeds and all checkpoint policies',
        continuation_decision='Fixed predeclared240-round research budget; no performance-adaptive extension',
        limitations=['Common predeclared cards are reused across fixed checkpoints; final cards are not a new independent holdout',
            'Nominal95% intervals condition on three trained seeds and are not adjusted for repeated checkpoint comparisons',
            'Current and mixture candidates use the same originalcurrent120 adoption baseline',
            'The fixed opponent pool does not establish general or professional poker strength',
            'No automatic model promotion, retry, or budget extension'])


def run(study, campaign, config):
    study, campaign, config = Path(study).resolve(), Path(campaign).resolve(), Path(config).resolve()
    cfg = json.loads(config.read_text()); validate_config(cfg)
    if not campaign.is_relative_to(study) or not (study/'plan.json').is_file():
        raise ValueError('A prepared isolated study and child campaign are required')
    study_cfg = json.loads((study/'config.json').read_text())
    if study_cfg != cfg:
        raise ValueError('Study configuration differs from the evaluator configuration')
    prior = previous_ranges(study)
    own_report = f'reports/continuation-study/{study.name}/'
    prior = {name: value for name, value in prior.items() if not name.startswith(own_report)}
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], prior)
    evaluation = study/'evaluation'; evaluation.mkdir(exist_ok=True)
    lock = (study/'evaluation.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (study/'evaluation-plan.json').exists():
        raise FileExistsError('Existing evaluation evidence cannot be automatically retried')
    jobs = evaluation_jobs(campaign, cfg)
    frozen = {str(config): digest(config), str(study/'config.json'): digest(study/'config.json'),
              str(study/'plan.json'): digest(study/'plan.json')}
    names = ['evaluate_compact_learning_curve.py', 'evaluate_compact.py', 'run_compact_refit_probe.py',
             'fullhouse_v3_common.py', 'fullhouse_v4_runtime.py', 'fullhouse_v3_runtime.py',
             'fullhouse_runtime.py', 'regret_partition.py']
    for name in names:
        path = ROOT/'scripts'/name; frozen[str(path)] = digest(path)
    source_manifest = json.loads((campaign/'source-manifest.json').read_text())
    frozen[str(campaign/'source-manifest.json')] = digest(campaign/'source-manifest.json')
    for name, sha in source_manifest['files'].items():
        frozen[str(campaign/name)] = sha
    registry_path = ROOT/'artifacts/selfplay/current.json'; registry = json.loads(registry_path.read_text())
    frozen[str(registry_path)] = digest(registry_path)
    frozen[str(ROOT/registry['model'])] = registry['sha256']
    from fullhouse_runtime import BRIDGE, SOURCE, Fullhouse
    public = Fullhouse(); frozen[str(public.model)] = public.digest; frozen[str(BRIDGE)] = digest(BRIDGE)
    frozen[str(ROOT/'configs/fullhouse-model.json')] = digest(ROOT/'configs/fullhouse-model.json')
    for path in list(SOURCE.rglob('*.py')) + list((SOURCE/'data').glob('*.npz')):
        frozen[str(path)] = digest(path)
    verify_frozen(frozen)
    plan = dict(created_utc=utc(), config=cfg, study_plan_sha256=digest(study/'plan.json'),
                jobs=jobs, prior_card_ranges=prior, frozen_files=frozen,
                card_range=[cfg['evaluation_deal_seed'], cfg['evaluation_deal_seed']+cfg['evaluation_deals']])
    atomic_json(study/'evaluation-plan.json', plan); plan_sha = digest(study/'evaluation-plan.json')
    frozen[str(study/'evaluation-plan.json')] = plan_sha
    parent = Path(json.loads((study/'plan.json').read_text())['parent']).resolve()
    pending = jobs[:]; active = {}; rows = {}; policy_files = {}; expected = {}; completed_stages = []
    verified_result_files = {}; verified_bytes = {}; stage_files = {}
    began = time.monotonic(); env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def status(state, **extra):
        atomic_json(study/'evaluation-status.json', dict(status=state, updated_utc=utc(),
            evaluator_pid=os.getpid(), evaluator_start=process_start(os.getpid()),
            seconds=time.monotonic()-began, evaluation_plan_sha256=plan_sha,
            completed_evaluations=len(rows), planned_evaluations=len(jobs),
            verified_hands=sum(row['hands'] for row in rows.values()), finished=list(rows),
            verified_result_files=verified_result_files.copy(),
            pending=[job['id'] for job in pending],
            running={key:dict(pid=value['process'].pid, process_start=value['process_start']) for key, value in active.items()},
            completed_checkpoint_evaluations=completed_stages[:], **extra))
    def stop(*unused):
        raise InterruptedError('Learning-curve evaluation interrupted')
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            if time.monotonic()-began > cfg.get('max_seconds', 14400):
                raise TimeoutError('Total learning-curve evaluation budget exhausted')
            study_status = json.loads((study/'status.json').read_text())
            if study_status.get('status') in ('failed', 'interrupted'):
                raise RuntimeError('Study supervisor recorded a failure')
            if not process_matches(study_status.get('supervisor_pid'), study_status.get('supervisor_start'), 'run_compact_learning_curve.py', study):
                raise RuntimeError('Study supervisor is not alive with its recorded start token')
            launch = json.loads((campaign/'launch.json').read_text()); limit = evaluation_parallelism(launch)
            for job in pending[:]:
                if len(active) >= limit:
                    break
                if not policy_ready(job):
                    if launch['status'] == 'complete':
                        raise ValueError('Checkpoint policy unavailable after completed training')
                    continue
                stage_proof = verified_stage_evidence(study, parent, job)
                if stage_proof is None:
                    if launch['status'] == 'complete':
                        raise ValueError('Stage verification is missing after completed training')
                    continue
                model = job['model']
                if model not in policy_files:
                    policy_files[model] = freeze_policy(job)
                if policy_files[model][model] != stage_proof['candidate_sha256']:
                    raise ValueError('Evaluation policy differs from its verified checkpoint')
                stage_files.update(stage_proof['verified_files'])
                verify_frozen(frozen); verify_frozen(policy_files[model])
                verify_frozen(stage_files)
                proof = dict(candidate=policy_files[model][model],
                    evaluator_sha256=frozen[str(ROOT/'scripts/evaluate_compact.py')],
                    runtime_sha256=frozen[str(ROOT/'scripts/fullhouse_v4_runtime.py')],
                    engine_sha256=frozen[str(BRIDGE)], source_features_sha256=frozen[str(campaign/'source/bot/features.py')],
                    opponents=dict(noregrets=registry['sha256'], public=public.digest if job['field'] in ('public', 'mixed') else None))
                expected[job['id']] = proof
                output = evaluation/job['id']
                if output.exists():
                    raise FileExistsError('Incomplete evaluation cannot be silently reused: ' + str(output))
                proof_path = evaluation/(job['id']+'.provenance.json')
                atomic_json(proof_path, dict(job=job, expected=proof, policy_files=policy_files[model], stage_evidence=stage_proof))
                log = (evaluation/(job['id']+'.log')).open('w')
                command = [str(PYTHON), str(ROOT/'scripts/evaluate_compact.py'), '--model', model,
                    '--source', job['source'], '--field', job['field'], '--deals', str(cfg['evaluation_deals']),
                    '--deal-seed', str(cfg['evaluation_deal_seed']), '--timeout', str(cfg.get('evaluation_seconds_per_cell', 2400)),
                    '--output', str(output)]
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job['id']] = dict(process=child, process_start=process_start(child.pid),
                    log=log, started=time.monotonic(), job=job, provenance_sha256=digest(proof_path))
                pending.remove(job)
            for name, item in list(active.items()):
                code = item['process'].poll()
                if code is None:
                    if time.monotonic()-item['started'] > cfg.get('evaluation_seconds_per_cell', 2400)+30:
                        raise TimeoutError('Evaluation process exceeded its bound: ' + name)
                    continue
                item['log'].close()
                if code:
                    raise RuntimeError(f'Evaluation cell failed: {name}, exit {code}; no automatic retry')
                proof_path = evaluation/(name+'.provenance.json')
                if digest(proof_path) != item['provenance_sha256']:
                    raise ValueError('Cell provenance evidence changed')
                verify_frozen(frozen); verify_frozen(policy_files[item['job']['model']])
                rows[name] = verify_result(evaluation/name, item['job'], cfg, expected[name])
                proof = dict(results_sha256=digest(evaluation/name/'results.json'),
                    hands_sha256=digest(evaluation/name/'hands.jsonl.gz'),
                    provenance_sha256=digest(proof_path))
                verified_result_files[name] = proof
                for path, sha in [(evaluation/name/'results.json', proof['results_sha256']),
                                  (evaluation/name/'hands.jsonl.gz', proof['hands_sha256']),
                                  (proof_path, proof['provenance_sha256'])]:
                    verified_bytes[str(path)] = sha
                del active[name]
            for through in (180, 240):
                required = [job['id'] for job in jobs if job['iteration'] <= through]
                if through not in completed_stages and all(name in rows for name in required):
                    verify_frozen(stage_files); verify_frozen(verified_bytes)
                    result = summarize(rows, cfg, through)
                    result.update(config=cfg, completed_utc=utc(), evaluation_plan_sha256=plan_sha,
                                  source_provenance_verified=True,
                                  verified_result_files={name:verified_result_files[name] for name in required})
                    atomic_json(study/f'stage-{through}-evaluation.json', result)
                    completed_stages.append(through)
            status('running')
            if pending or active:
                time.sleep(2)
        verify_frozen(frozen); verify_frozen(stage_files); verify_frozen(verified_bytes)
        for files in policy_files.values():
            verify_frozen(files)
        result = summarize(rows, cfg, 240)
        result.update(config=cfg, completed_utc=utc(), seconds=time.monotonic()-began,
            evaluation_plan_sha256=plan_sha, source_provenance_verified=True,
            verified_result_files=verified_result_files,
            verified_policy_files=len(set(name for files in policy_files.values() for name in files)),
            source_frozen_files=len(frozen))
        atomic_json(study/'evaluation-summary.json', result); status('complete')
        print(json.dumps(dict(complete=True, evaluations=len(rows), hands=result['total_hands'])), flush=True)
        return result
    except BaseException as error:
        status('interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed', error=str(error))
        raise
    finally:
        for item in active.values():
            child = item['process']
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=8)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
            item['log'].close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); run(args.study, args.campaign, args.config)
