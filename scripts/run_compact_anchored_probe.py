"""Bounded main-policy anchored refit, evaluated on fresh card blocks.

Only new experiment outputs are written. The original self-play campaign and
its continuation criteria stay fixed; this runner never promotes a model.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import (utc, previous_ranges, check_card_range,
                                     verify_result, summarize)


def run(parent, output, config):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Probe output must be outside the immutable parent run')
    cfg = json.loads(Path(config).read_text()); parent_cfg = json.loads((parent/'config.json').read_text())
    if cfg['seeds'] != parent_cfg['seeds'] or len(cfg['seeds']) < 2:
        raise ValueError('Evaluate every declared parent training seed in its original order')
    if set(cfg['variants']) != {'current', 'anchored'} or len(cfg['variants']) != 2:
        raise ValueError('Exactly current and anchored are required')
    if not math.isfinite(cfg['anchor_weight']) or not 0 < cfg['anchor_weight'] <= 100:
        raise ValueError('Positive bounded anchor weight required')
    if set(cfg['evaluation_fields']) != {'public', 'mixed', 'noregrets'} or len(cfg['evaluation_fields']) != 3:
        raise ValueError('All three fixed opponent fields are required')
    if not 1 <= cfg['parallelism'] <= 3 or not 1 <= cfg['extra_steps'] <= 8192:
        raise ValueError('Invalid bounded compute budget')
    if cfg['batch_size'] != parent_cfg['batch_size'] or cfg['learning_rate'] != parent_cfg['learning_rate']:
        raise ValueError('Parent batch size and learning rate must remain fixed')
    if json.loads((parent/'campaign-status.json').read_text())['status'] != 'complete':
        raise ValueError('Parent campaign must be complete')
    if not json.loads((parent/'verification.json').read_text())['passed']:
        raise ValueError('Parent checkpoint verification must pass')
    prior = previous_ranges(output)
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], prior)
    parent_manifest = json.loads((parent/'source-manifest.json').read_text())
    frozen = {str(parent/name): sha for name, sha in parent_manifest['files'].items()}
    for name in ('config.json', 'source-manifest.json', 'verification.json'):
        frozen[str(parent/name)] = digest(parent/name)
    iteration = parent_cfg['iterations']
    for seed in cfg['seeds']:
        folder = parent/f'seed-{seed}'
        for name in ('latest.pt', f'current_{iteration:05d}.npz'):
            frozen[str(folder/name)] = digest(folder/name)
    for name in ('probe_compact_anchored_refit.py', 'run_compact_refit_probe.py',
                 'evaluate_compact.py', 'fullhouse_v4_runtime.py', 'fullhouse_runtime.py',
                 'fullhouse_v3_runtime.py', 'regret_partition.py', 'fullhouse_v3_common.py'):
        frozen[str(ROOT/'scripts'/name)] = digest(ROOT/'scripts'/name)
    frozen[str(Path(__file__).resolve())] = digest(Path(__file__))
    frozen[str(Path(config).resolve())] = digest(config)
    frozen[str(ROOT/'configs/fullhouse-model.json')] = digest(ROOT/'configs/fullhouse-model.json')
    registry_path = ROOT/'artifacts/selfplay/current.json'
    registry = json.loads(registry_path.read_text()); opponent = ROOT/registry['model']
    frozen[str(registry_path)] = digest(registry_path); frozen[str(opponent)] = registry['sha256']
    engine = ROOT/'.cache/noregrets/target/release/poker-bridge'
    frozen[str(engine)] = digest(engine)
    for name, sha in frozen.items():
        if digest(name) != sha:
            raise ValueError('Frozen artifact changed: '+name)
    from fullhouse_runtime import Fullhouse
    public = Fullhouse()
    frozen[str(public.model)] = public.digest
    output.mkdir(parents=True, exist_ok=False)
    lock = (output/'probe.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output/'logs').mkdir(); (output/'evaluation').mkdir()
    plan = dict(created_utc=utc(), parent=str(parent), parent_iteration=iteration, config=cfg,
                config_source_sha256=digest(config), frozen_files=frozen, prior_card_ranges=prior,
                card_range=[cfg['evaluation_deal_seed'], cfg['evaluation_deal_seed']+cfg['evaluation_deals']],
                note='No original run, self-play registry, controller, or deployment modification')
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    atomic_json(output/'config.json', cfg)
    began = time.monotonic(); active = {}; finished = []; phase = 'fitting'
    event_log = (output/'probe.log').open('a', buffering=1)
    def status(state, pending=(), **extra):
        value = dict(status=state, phase=phase, updated_utc=utc(), seconds=time.monotonic()-began,
                     supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
                     plan_sha256=plan_sha, finished=finished[:], pending=list(pending),
                     running={name:p['process'].pid for name,p in active.items()},
                     completed_evaluations=sum(name.startswith('eval/') for name in finished),
                     planned_evaluations=len(cfg['seeds'])*len(cfg['variants'])*len(cfg['evaluation_fields']), **extra)
        atomic_json(output/'status.json', value)
    def terminate(*unused):
        raise InterruptedError('Probe interrupted')
    previous_signals = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def execute(jobs):
        pending = list(jobs)
        while pending or active:
            while pending and len(active) < cfg['parallelism']:
                job = pending.pop(0); name = job['id']
                log = (output/'logs'/(name.replace('/', '-')+'.log')).open('w')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = dict(process=child, log=log, started=time.monotonic(), timeout=job['timeout'])
                event_log.write(json.dumps(dict(event='started', job=name, pid=child.pid, utc=utc()))+'\n')
            for name, row in list(active.items()):
                result = row['process'].poll()
                if result is None:
                    if time.monotonic()-row['started'] > row['timeout']:
                        raise TimeoutError('Probe job exceeded budget: '+name)
                    continue
                row['log'].close(); del active[name]
                if result:
                    raise RuntimeError(f'Probe job failed: {name}, exit {result}')
                finished.append(name); event_log.write(json.dumps(dict(event='finished', job=name, utc=utc()))+'\n')
            status('running', [j['id'] for j in pending])
            if pending or active:
                time.sleep(2)
    try:
        fit_jobs = [dict(id=f'fit/{seed}', timeout=1200,
                        command=[str(PYTHON), str(ROOT/'scripts/probe_compact_anchored_refit.py'), '--parent', str(parent),
                                 '--output', str(output/f'seed-{seed}'), '--seed', str(seed),
                                 '--extra-steps', str(cfg['extra_steps']),
                                 '--anchor-weight', str(cfg['anchor_weight'])]) for seed in cfg['seeds']]
        execute(fit_jobs)
        for seed in cfg['seeds']:
            result = json.loads((output/f'seed-{seed}/results.json').read_text())
            if not all(result.get(key) is True for key in ('complete', 'opening_unchanged', 'scales_unchanged', 'parent_checkpoint_unchanged', 'teacher_unchanged')):
                raise ValueError('Anchored refit provenance/preservation checks failed')
            if result['seed'] != seed or result['extra_steps'] != cfg['extra_steps']:
                raise ValueError('Wrong refit seed or fit budget')
        phase = 'evaluating'; jobs = []; expected = {}
        common = dict(evaluator_sha256=frozen[str(ROOT/'scripts/evaluate_compact.py')],
                      runtime_sha256=frozen[str(ROOT/'scripts/fullhouse_v4_runtime.py')],
                      engine_sha256=frozen[str(engine)], source_features_sha256=digest(parent/'source/bot/features.py'))
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                model = output/f'seed-{seed}/candidate.npz' if variant == 'anchored' else parent/f'seed-{seed}'/f'current_{iteration:05d}.npz'
                for field in cfg['evaluation_fields']:
                    name = f'{field}-seed-{seed}-{variant}'
                    job = dict(id='eval/'+name, name=name, field=field, variant=variant, seed=seed, model=str(model), timeout=1900,
                               command=[str(PYTHON), str(ROOT/'scripts/evaluate_compact.py'), '--model', str(model),
                                        '--source', str(parent/'source'), '--field', field,
                                        '--deals', str(cfg['evaluation_deals']), '--deal-seed', str(cfg['evaluation_deal_seed']),
                                        '--output', str(output/'evaluation'/name)])
                    jobs.append(job)
                    expected[name] = dict(common, candidate=digest(model),
                        opponents=dict(noregrets=registry['sha256'], public=public.digest if field in ('public','mixed') else None))
        atomic_json(output/'evaluation-plan.json', dict(jobs=jobs, expected=expected))
        evaluation_plan_sha = digest(output/'evaluation-plan.json')
        execute(jobs)
        phase = 'verifying'; status('running')
        rows = {job['name']: verify_result(output/'evaluation'/job['name'], job, cfg, expected[job['name']]) for job in jobs}
        for name, sha in frozen.items():
            if digest(name) != sha:
                raise ValueError('Frozen input/code changed during probe: '+name)
        if digest(output/'plan.json') != plan_sha or digest(output/'evaluation-plan.json') != evaluation_plan_sha:
            raise ValueError('Predeclared experiment plan changed')
        summary = summarize(rows, cfg)
        summary['limitations'][0] = 'Single predeclared candidate; nominal intervals are not adjusted for repeated experimentation'
        summary.update(plan_sha256=plan_sha, evaluation_plan_sha256=evaluation_plan_sha, config=cfg,
                       parent=str(parent), completed_utc=utc(), seconds=time.monotonic()-began,
                       raw_hands_verified=True, source_provenance_verified=True)
        atomic_json(output/'summary.json', summary)
        phase = 'complete'; status('complete')
        print(json.dumps(dict(output=str(output), complete=True, evaluations=len(rows), decisions=summary['decisions'])), flush=True)
        return summary
    except BaseException as error:
        phase = 'interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed'
        status(phase, error=str(error)); raise
    finally:
        for row in active.values():
            if row['process'].poll() is None:
                try: os.killpg(row['process'].pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for row in active.values():
            try: row['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(row['process'].pid, signal.SIGKILL)
                except ProcessLookupError: pass
                row['process'].wait()
            row['log'].close()
        for sig, handler in previous_signals.items(): signal.signal(sig, handler)
        event_log.close(); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output, args.config)
