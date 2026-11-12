"""Run one bounded reference benchmark and independent-opponent diagnostic study.

This study never changes the six-player trainer, controller, model registry or
promotion conditions. Negative scientific results are completed results, not
an instruction to retry with different settings.
"""
import argparse
from bisect import bisect_right
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import check_card_range, interval, previous_ranges, utc


VARIANTS = ('current60', 'current120', 'mixture120')
OPPONENTS = ('calling-station', 'tight-value', 'pressure')


def terminate_children(active):
    """Kill each remaining group, including descendants of an exited leader."""
    for row in active.values():
        try:
            os.killpg(row['process'].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for row in active.values():
        try:
            row['process'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(row['process'].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            row['process'].wait()
        # The leader may already have exited while a descendant ignored TERM.
        try:
            os.killpg(row['process'].pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        row['log'].close()


def validate_config(cfg):
    ref = cfg['reference']
    if ref['seeds'] != [342, 343, 344] or ref['variants'] != list(VARIANTS):
        raise ValueError('All three original seeds and predeclared variants are required')
    if ref['opponents'] != list(OPPONENTS):
        raise ValueError('All three fixed reference opponent families are required')
    if cfg['leduc']['seeds'] != [71, 72, 73] or cfg['leduc']['game'] != 'leduc_poker':
        raise ValueError('The three-seed two-player Leduc benchmark is required')
    for key, maximum in (('parallelism', 3), ('seconds_per_benchmark', 1800),
                         ('seconds_per_ref_eval', 1800), ('max_study_seconds', 3600)):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError('Invalid bounded execution setting: ' + key)
    if not 2 <= ref['evaluation_deals'] <= 10000 or not 0 <= ref.get('trace_hands', 12) <= 100:
        raise ValueError('Invalid evaluation or bounded trace budget')
    check_card_range(ref['evaluation_deal_seed'], ref['evaluation_deals'], {})
    if cfg.get('promotion') != 'none':
        raise ValueError('This diagnostic study cannot promote a model')
    for key, value in (('tabular_max_uniform_ratio', .1), ('deep_average_max_uniform_ratio', .5),
                       ('deep_strategy_max_uniform_ratio', .5)):
        if cfg['leduc'].get(key) != value:
            raise ValueError('Reference sanity conditions must remain predeclared')
    for key in ('evaluation_deal_seed', 'evaluation_deals'):
        if key in cfg and cfg[key] != ref[key]:
            raise ValueError('Top-level card range must match reference configuration')


def verify_reference(path, job, cfg, expected):
    """Independently recompute every payoff block and mixture selection."""
    path = Path(path)
    row = json.loads((path / 'results.json').read_text())
    deals, card_seed = cfg['evaluation_deals'], cfg['evaluation_deal_seed']
    if not row.get('complete') or row.get('opponent') != job['opponent']:
        raise ValueError('Incomplete or wrong opponent result')
    if (row['deals'], row['hands'], row['deal_seed']) != (deals, deals * 6, card_seed):
        raise ValueError('Wrong reference evaluation budget or card range')
    if Path(row['candidate']).resolve() != Path(job['model']).resolve():
        raise ValueError('Wrong candidate path')
    if row['candidate_sha256'] != expected['candidate'] or digest(job['model']) != expected['candidate']:
        raise ValueError('Candidate changed')
    for key in ('evaluator_sha256', 'runtime_sha256', 'engine_sha256',
                'source_features_sha256', 'opponent_source_sha256'):
        if row.get(key) != expected[key]:
            raise ValueError('Reference provenance mismatch: ' + key)
    if row.get('plan_sha256') != digest(path / 'plan.json'):
        raise ValueError('Reference evaluation plan changed')
    for name, key in (('hands.jsonl.gz', 'raw_hands_sha256'), ('traces.json', 'traces_sha256')):
        if row.get(key) != digest(path / name):
            raise ValueError('Reference raw output changed: ' + name)
    own_plan = json.loads((path / 'plan.json').read_text())
    for name, sha in own_plan['frozen_files'].items():
        if digest(name) != sha:
            raise ValueError('Reference input changed: ' + name)
    mixture = job['variant'] == 'mixture120'
    if row['candidate_type'] != ('snapshot_mixture' if mixture else 'compact_advantage'):
        raise ValueError('Wrong candidate policy type')
    cumulative, spec = [], None
    if mixture:
        spec = json.loads(Path(job['model']).read_text()); total = 0
        for policy in spec['models']:
            total += policy['weight']; cumulative.append(total)
    blocks = [[None] * 6 for _ in range(deals)]
    count = 0
    with gzip.open(path / 'hands.jsonl.gz', 'rt') as stream:
        for line in stream:
            hand = json.loads(line); deal, hero = divmod(count, 6)
            if deal >= deals or (hand['deal'], hand['hero']) != (deal, hero):
                raise ValueError('Missing, reordered or duplicated reference hand')
            utilities = hand['utilities']
            if (len(utilities) != 6 or not all(type(v) in (int, float) and math.isfinite(v) for v in utilities)
                    or hand['seed'] != card_seed + deal or sum(utilities) != 0
                    or hand['chips'] != utilities[hero]):
                raise ValueError('Invalid reference hand seed or payoff')
            hand_seed = int.from_bytes(hashlib.sha256(
                f'compact-mixture:{card_seed}:{deal}:{hero}'.encode()).digest()[:8], 'big')
            if hand['hand_seed'] != hand_seed:
                raise ValueError('Wrong reference per-hand random seed')
            if mixture:
                selected = bisect_right(cumulative, random.Random(hand_seed).randrange(cumulative[-1]))
                if hand['policy_iteration'] != spec['models'][selected]['iteration']:
                    raise ValueError('Reference mixture selection mismatch')
            elif hand.get('policy_iteration') is not None:
                raise ValueError('Current policy has a mixture iteration')
            blocks[deal][hero] = hand['chips']; count += 1
    if count != deals * 6:
        raise ValueError('Incomplete reference hand coverage')
    means = [statistics.mean(values) for values in blocks]
    if len(row['deal_blocks']) != deals or any(abs(a-b) > 1e-9 for a, b in zip(means, row['deal_blocks'])):
        raise ValueError('Reference raw payoff blocks disagree')
    metrics = interval(means)
    for key in ('mean', 'ci95_half_width', 'lower', 'upper'):
        if abs(metrics[key] - row['interval'][key]) > 1e-9:
            raise ValueError('Reference reported interval disagrees')
    if abs(metrics['mean'] - row['bb_per_100']) > 1e-9:
        raise ValueError('Reference reported mean disagrees')
    return row


def summarize_reference(rows, cfg):
    absolute, comparisons = {}, {}
    for opponent in cfg['opponents']:
        for variant in cfg['variants']:
            per_seed = [rows[f'{opponent}-seed-{seed}-{variant}']['deal_blocks'] for seed in cfg['seeds']]
            absolute[f'{opponent}/{variant}'] = dict(
                **interval([statistics.mean(v) for v in zip(*per_seed)]),
                per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], per_seed)])
        for before, after in (('current60', 'current120'), ('current120', 'mixture120')):
            changes = [[b-a for a, b in zip(rows[f'{opponent}-seed-{seed}-{before}']['deal_blocks'],
                                           rows[f'{opponent}-seed-{seed}-{after}']['deal_blocks'])]
                       for seed in cfg['seeds']]
            comparisons[f'{opponent}/{after}-minus-{before}'] = dict(
                **interval([statistics.mean(v) for v in zip(*changes)]),
                per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], changes)])
    return dict(completed_evaluations=len(rows), total_hands=sum(row['hands'] for row in rows.values()),
        absolute_scores=absolute, paired_changes=comparisons, raw_hands_verified=True,
        diagnostic_only=True, deployment_authorized=False,
        limitations=['Independent behavior families are transparent heuristic baselines, not proven strong poker opponents',
            'Shared cards form one block across seats, seeds and policies; nominal intervals are not multiplicity-adjusted',
            'Conditional context payoffs are associations, not causal action values',
            'Original four continuation conditions are unchanged and are not replaced by these diagnostics'])


def verify_benchmark(path, job, cfg):
    """Recompute exact exploitability from saved policy tables."""
    import numpy as np
    import pyspiel
    from benchmark_leduc_reference import ExactPolicySpace
    path = Path(path); report = json.loads((path / 'results.json').read_text())
    if (report.get('complete') is not True or report['method'] != job['method']
            or report['seed'] != job['seed'] or report['config'] != cfg):
        raise ValueError('Wrong or incomplete Leduc benchmark')
    if [row['iteration'] for row in report['curve']] != cfg[job['method'] + '_checkpoints']:
        raise ValueError('Leduc checkpoint curve is incomplete')
    for name, sha in report['source_hashes'].items():
        if digest(name) != sha:
            raise ValueError('Leduc source changed: ' + name)
    space = ExactPolicySpace(pyspiel.load_game(cfg['game'], {'players': 2}))
    if json.loads((path / 'information-states.json').read_text()) != space.keys:
        raise ValueError('Leduc policy row order differs')
    initial = space.metrics(space.template.action_probability_array)
    if abs(initial['exploitability'] - report['initial_uniform']['exploitability']) > 1e-9:
        raise ValueError('Leduc uniform baseline differs')
    for row in report['curve']:
        entry = row['policy_file']; artifact = path / entry['name']
        if artifact.parent != path or digest(artifact) != entry['sha256'] or artifact.stat().st_size != entry['bytes']:
            raise ValueError('Leduc saved policy hash or size differs')
        with np.load(artifact, allow_pickle=False) as data:
            if set(data.files) != set(row['metrics']):
                raise ValueError('Leduc policy names differ')
            for name in data.files:
                actual = space.metrics(data[name])
                for key in ('exploitability', 'nash_conv'):
                    if abs(actual[key] - row['metrics'][name][key]) > 1e-8:
                        raise ValueError('Leduc saved-policy metric differs')
    report['saved_policy_metrics_recomputed'] = True
    return report


def summarize_benchmarks(rows, cfg):
    tabular = rows['tabular']
    uniform = tabular['initial_uniform']['exploitability']
    tabular_final = tabular['curve'][-1]['metrics']['average']['exploitability']
    neural = {str(seed): dict(
        current=row['curve'][-1]['metrics']['current']['exploitability'],
        realized_average=row['curve'][-1]['metrics']['realized_average']['exploitability'],
        fitted_average=row['curve'][-1]['metrics']['fitted_average']['exploitability'])
        for seed in cfg['seeds'] for row in [rows[f'deep-{seed}']]}
    checks = dict(tabular_final_at_most_tenth_uniform=tabular_final <= .1 * uniform,
        every_realized_average_at_most_half_uniform=all(x['realized_average'] <= .5 * uniform for x in neural.values()),
        every_fitted_average_at_most_half_uniform=all(x['fitted_average'] <= .5 * uniform for x in neural.values()))
    return dict(initial_uniform_exploitability=uniform, tabular_final_exploitability=tabular_final,
        neural_final_exploitability=neural, checks=checks, all_reference_checks_pass=all(checks.values()),
        saved_policy_metrics_recomputed=True, six_player_adoption_authorized=False,
        interpretation='Fixed-budget two-player Leduc reference checks; no requirement for every checkpoint to improve',
        limitations=['This reduced-budget reference benchmark does not validate the compact six-player architecture',
            'Realized average quality and strategy-network fitting quality are separate measurements',
            'A failed check is retained without automatic rerun or changed criteria'])


def run(config, output):
    config, output = Path(config).resolve(), Path(output).resolve()
    cfg = json.loads(config.read_text()); validate_config(cfg)
    ref = cfg['reference']
    parents = {str(i): (ROOT / cfg['parent' + str(i)]).resolve() for i in (60, 120)}
    if output.parent != ROOT / 'artifacts/foundation-study' or any(output == p or p in output.parents for p in parents.values()):
        raise ValueError('Study output must be a new isolated foundation-study run')
    prior = previous_ranges(output)
    check_card_range(ref['evaluation_deal_seed'], ref['evaluation_deals'], prior)
    frozen = {}
    def freeze(path, expected=None):
        path = Path(path).resolve(); actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError('Frozen artifact already differs: ' + str(path))
        frozen[str(path)] = actual
    for iteration, parent in parents.items():
        parent_cfg = json.loads((parent / 'config.json').read_text())
        if parent_cfg['iterations'] != int(iteration) or parent_cfg['seeds'] != ref['seeds']:
            raise ValueError('Unexpected source campaign')
        if (json.loads((parent / 'campaign-status.json').read_text())['status'] != 'complete'
                or not json.loads((parent / 'verification.json').read_text())['passed']):
            raise ValueError('Source campaign is not verified complete')
        manifest = json.loads((parent / 'source-manifest.json').read_text())
        for name, sha in manifest['files'].items():
            freeze(parent / name, sha)
        for name in ('config.json', 'source-manifest.json', 'verification.json'):
            freeze(parent / name)
        for seed in ref['seeds']:
            folder = parent / f'seed-{seed}'
            for name in ('latest.pt', f'current_{int(iteration):05d}.npz'):
                freeze(folder / name)
            if iteration == '120':
                mixture = folder / 'mixture_00120.json'; freeze(mixture)
                models = json.loads(mixture.read_text())['models']
                if [m['iteration'] for m in models] != list(range(1, 121)) or any(m['weight'] != m['iteration'] for m in models):
                    raise ValueError('Mixture must contain all120 fixed time-weighted policies')
                for model in models:
                    target = (folder / model['file']).resolve()
                    if target.parent != folder:
                        raise ValueError('Snapshot escaped source folder')
                    freeze(target, model['sha256'])
    for name in ('run_poker_foundation_study.py', 'benchmark_leduc_reference.py', 'evaluate_poker_reference.py',
                 'poker_reference_opponents.py', 'fullhouse_v3_common.py', 'run_compact_refit_probe.py',
                 'fullhouse_v4_runtime.py', 'fullhouse_v3_runtime.py', 'fullhouse_runtime.py', 'regret_partition.py'):
        freeze(ROOT / 'scripts' / name)
    freeze(config); freeze(ROOT / 'configs/fullhouse-model.json')
    from fullhouse_runtime import BRIDGE, Fullhouse
    freeze(BRIDGE)
    # Runtime loads the fixed public bundle's equity table even for custom models.
    runtime = Fullhouse()
    for path in runtime.model.parent.glob('*.npz'):
        freeze(path)
    output.parent.mkdir(parents=True, exist_ok=True); output.mkdir(exist_ok=False)
    lock = (output / 'study.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ('logs', 'benchmark', 'evaluation'):
        (output / name).mkdir()
    saved_cfg = dict(cfg, evaluation_deals=ref['evaluation_deals'], evaluation_deal_seed=ref['evaluation_deal_seed'])
    atomic_json(output / 'config.json', saved_cfg)
    atomic_json(output / 'leduc-config.json', cfg['leduc'])
    freeze(output / 'config.json'); freeze(output / 'leduc-config.json')
    benchmark_jobs = [dict(id='benchmark/tabular', name='tabular', method='tabular', seed=cfg['leduc']['seeds'][0])]
    benchmark_jobs += [dict(id=f'benchmark/deep-{seed}', name=f'deep-{seed}', method='deep', seed=seed) for seed in cfg['leduc']['seeds']]
    for job in benchmark_jobs:
        job.update(timeout=cfg['seconds_per_benchmark'], command=[str(PYTHON), 'scripts/benchmark_leduc_reference.py',
            '--config', str(output / 'leduc-config.json'), '--method', job['method'], '--seed', str(job['seed']),
            '--output', str(output / 'benchmark' / job['name'])])
    evaluation_jobs = []; expected = {}
    for variant in ref['variants']:
        parent = parents['60' if variant == 'current60' else '120']
        for seed in ref['seeds']:
            filename = {'current60':'current_00060.npz', 'current120':'current_00120.npz', 'mixture120':'mixture_00120.json'}[variant]
            model = parent / f'seed-{seed}' / filename
            for opponent in ref['opponents']:
                name = f'{opponent}-seed-{seed}-{variant}'
                job = dict(id='eval/' + name, name=name, opponent=opponent, variant=variant, seed=seed,
                    model=str(model), timeout=cfg['seconds_per_ref_eval'], command=[str(PYTHON), 'scripts/evaluate_poker_reference.py',
                        '--model', str(model), '--source', str(parent / 'source'), '--opponent', opponent,
                        '--deals', str(ref['evaluation_deals']), '--deal-seed', str(ref['evaluation_deal_seed']),
                        '--trace-hands', str(ref.get('trace_hands', 12)), '--timeout', str(cfg['seconds_per_ref_eval']),
                        '--output', str(output / 'evaluation' / name)])
                evaluation_jobs.append(job)
                expected[name] = dict(candidate=digest(model), evaluator_sha256=digest(ROOT/'scripts/evaluate_poker_reference.py'),
                    runtime_sha256=digest(ROOT/'scripts/fullhouse_v4_runtime.py'), engine_sha256=digest(BRIDGE),
                    source_features_sha256=digest(parent/'source/bot/features.py'),
                    opponent_source_sha256=digest(ROOT/'scripts/poker_reference_opponents.py'))
    plan = dict(created_utc=utc(), config=saved_cfg, config_source_sha256=digest(config), frozen_files=frozen,
        prior_card_ranges=prior, card_range=[ref['evaluation_deal_seed'], ref['evaluation_deal_seed']+ref['evaluation_deals']],
        benchmark_jobs=benchmark_jobs, evaluation_jobs=evaluation_jobs, expected_reference=expected,
        note='Diagnostic study only; no source campaign, controller or deployment modification')
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    began = time.monotonic(); phase = 'benchmarking'; active = {}; finished = []
    events = (output/'study.log').open('a', buffering=1)
    def status(state, pending=(), **extra):
        atomic_json(output/'status.json', dict(status=state, phase=phase, updated_utc=utc(), seconds=time.monotonic()-began,
            supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()), plan_sha256=plan_sha,
            finished=finished[:], pending=list(pending), running={name:row['process'].pid for name,row in active.items()},
            completed_benchmarks=sum(name.startswith('benchmark/') for name in finished), planned_benchmarks=len(benchmark_jobs),
            completed_evaluations=sum(name.startswith('eval/') for name in finished), planned_evaluations=len(evaluation_jobs), **extra))
    def interrupted(*unused):
        raise InterruptedError('Foundation study interrupted')
    handlers = {sig:signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def execute(jobs):
        pending = list(jobs)
        while pending or active:
            if time.monotonic()-began > cfg['max_study_seconds']:
                raise TimeoutError('Total study time budget exhausted')
            while pending and len(active) < cfg['parallelism']:
                job = pending.pop(0); name = job['id']
                log = (output/'logs'/(name.replace('/', '-')+'.log')).open('w')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = dict(process=child, log=log, started=time.monotonic(), timeout=job['timeout'])
                events.write(json.dumps(dict(event='started', job=name, pid=child.pid, utc=utc()))+'\n')
            for name, row in list(active.items()):
                result = row['process'].poll()
                if result is None:
                    if time.monotonic()-row['started'] > row['timeout']:
                        raise TimeoutError('Study job exceeded budget: ' + name)
                    continue
                row['log'].close()
                if result:
                    raise RuntimeError(f'Study job failed: {name}, exit {result}')
                del active[name]
                finished.append(name); events.write(json.dumps(dict(event='finished', job=name, utc=utc()))+'\n')
            status('running', [job['id'] for job in pending])
            if pending or active:
                time.sleep(2)
    try:
        execute(benchmark_jobs)
        phase = 'verifying_benchmarks'; status('running')
        benchmark_rows = {job['name']: verify_benchmark(output/'benchmark'/job['name'], job, cfg['leduc']) for job in benchmark_jobs}
        if time.monotonic()-began > cfg['max_study_seconds']:
            raise TimeoutError('Total study time budget exhausted')
        benchmarks = summarize_benchmarks(benchmark_rows, cfg['leduc'])
        atomic_json(output/'benchmark-summary.json', benchmarks)
        phase = 'evaluating'; execute(evaluation_jobs)
        phase = 'verifying_evaluations'; status('running')
        rows = {job['name']: verify_reference(output/'evaluation'/job['name'], job, ref, expected[job['name']]) for job in evaluation_jobs}
        if time.monotonic()-began > cfg['max_study_seconds']:
            raise TimeoutError('Total study time budget exhausted')
        for name, sha in frozen.items():
            if digest(name) != sha:
                raise ValueError('Frozen study input changed: ' + name)
        if digest(output/'plan.json') != plan_sha:
            raise ValueError('Predeclared study plan changed')
        summary = dict(complete=True, completed_utc=utc(), seconds=time.monotonic()-began,
            config=saved_cfg, plan_sha256=plan_sha, benchmarks=benchmarks,
            reference_evaluation=summarize_reference(rows, ref), source_provenance_verified=True,
            no_automatic_retry=True, no_model_promotion=True, selfplay_iteration=120,
            next_action='Analyze reference learning and six-player information/evaluation evidence before proposing a new training experiment')
        atomic_json(output/'summary.json', summary); phase='complete'; status('complete')
        print(json.dumps(dict(output=str(output), complete=True, completed_evaluations=len(rows),
            reference_checks=benchmarks['checks'])), flush=True)
        return summary
    except BaseException as error:
        phase = 'interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed'
        status(phase, error=str(error)); raise
    finally:
        terminate_children(active)
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        events.close(); lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args(); run(args.config, args.output)
