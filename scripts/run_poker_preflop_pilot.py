"""One fixed, seed-matched preflop pilot. Preflight/status never launch work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import interval, verify_results
from evaluate_compact_learning_curve import adoption_conditions
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy, fit, write_policy
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges


SCRIPT = Path(__file__).resolve()
CONFIG = ROOT/'configs/poker-preflop-pilot-20261118.json'
CFG = read(CONFIG)
PARENT = ROOT/'artifacts/average-extension'/CFG['source_run']
CAMPAIGN = PARENT/'campaign'
SOURCE = CAMPAIGN/'source'
OUTPUT = ROOT/'artifacts/preflop-pilot/480-v1'
SEEDS = CFG['training_seeds']
FIELDS = CFG['fields']
LABELS = CFG['labels']


def implementation():
    paths = [CONFIG, SCRIPT, SCRIPT.with_name('poker_preflop_specialist.py')]
    return {str(p): digest(p) for p in paths}


def verify_parent():
    verify_files({PARENT/'stage-ready.json': CFG['stage_sha256'],
                  CAMPAIGN/'config.json': CFG['campaign_config_sha256'],
                  PARENT/'evaluation-plan.json': CFG['evaluation_plan_sha256'],
                  PARENT/'evaluation-summary.json': CFG['evaluation_summary_sha256']})
    stage = read(PARENT/'stage-ready.json')
    inputs = read(PARENT/'evaluation-plan.json')['policies']
    parent_cfg = read(CAMPAIGN/'config.json')
    frozen = {p: sha for p, sha in parent_cfg['frozen_files'].items() if Path(p).is_relative_to(CAMPAIGN/'tools')
              or Path(p).is_relative_to(SOURCE)}
    if not frozen or not stage['training_complete'] or set(stage['seeds']) != set(map(str, SEEDS)):
        raise ValueError('Incomplete audited parent inputs')
    verify_files(frozen)
    summary = read(PARENT/'evaluation-summary.json')
    for field in FIELDS:
        key = f'{field}-seed-{SEEDS[0]}-current480'
        path = PARENT/'evaluation'/key/'results.json'
        verify_files({path: summary['result_files'][key]})
        verify_files(read(path)['dependency_hashes'])
    for seed in map(str, SEEDS):
        record = stage['seeds'][seed]
        checkpoint = Path(record['checkpoint'])
        expected = CAMPAIGN/f'seed-{seed}'/'checkpoints/iter-00480'
        pointer = read(CAMPAIGN/f'seed-{seed}'/'current.json')
        if checkpoint != expected or pointer['directory'] != expected.name or pointer['iteration'] != 480 or pointer['files'] != record['files']:
            raise ValueError('Parent checkpoint pointer differs from audited stage')
        verify_files({checkpoint/k: record['files'][k] for k in ('arrays.npz', 'state.json', 'current.npz')})
        for name in ('original120', 'parent420', 'current480'):
            ref = inputs[name][seed]
            verify_files({ref['path']: ref['sha256']})
    if any(parent_cfg[k] != CFG[k] for k in ('batch_size', 'learning_rate', 'main_steps')):
        raise ValueError('Parent fitting settings differ')


def preflight():
    from run_poker_average_extension import live_poker_jobs
    verify_parent()
    ranges = {k: v['range'] for k, v in prior_card_ranges(OUTPUT).items()}
    check_card_range(CFG['evaluation_deal_seed'], CFG['evaluation_deals'], ranges)
    ranges['full-pilot'] = (CFG['evaluation_deal_seed'], CFG['evaluation_deal_seed']+CFG['evaluation_deals'])
    check_card_range(CFG['smoke_deal_seed'], CFG['smoke_deals'], ranges)
    others = [r['pid'] for r in live_poker_jobs() if process_start(r['pid']) is not None]
    free = shutil.disk_usage(PARENT).free
    if others or free < 2**30:
        raise RuntimeError(f'Active poker processes {others} or insufficient local space ({free} bytes)')
    return dict(checked_utc=utc(), parent_run=PARENT.name, parent_files_verified=True,
                source_and_opponents_verified=True, fresh_card_ranges_verified=True, free_bytes=free,
                existing_full_attempt=OUTPUT.exists(), implementation=implementation())


def live(pid, token, arguments):
    if not token or process_start(pid) != token:
        return False
    try:
        proc = Path('/proc')/str(pid)
        cmd = (proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return len(cmd) >= 2 and (proc/'cwd'/cmd[1]).resolve() == SCRIPT and cmd[2:] == arguments
    except (OSError, UnicodeError):
        return False


def seed_models(folder, seed):
    original = read(PARENT/'evaluation-plan.json')['policies']
    return dict(original120=Path(original['original120'][str(seed)]['path']),
                current480=Path(original['current480'][str(seed)]['path']),
                preflop420=folder/'preflop420.json', specialist=folder/'specialist.json')


def seed_worker(study, seed):
    import evaluate_average_baseline as evaluator
    plan = read(study/'plan.json')
    verify_files(plan['implementation'])
    if seed not in SEEDS:
        raise ValueError('Unknown seed')
    folder = study/f'seed-{seed}'
    folder.mkdir()
    def mark(state, **extra):
        atomic_json(folder/'status.json', dict(state=state, observed_utc=utc(), pid=os.getpid(),
                    process_start=process_start(os.getpid()), **extra))
    mark('fitting')
    try:
        trained = fit(CAMPAIGN, seed, folder, plan['steps'])
        atomic_json(folder/'fit.json', trained)
        models = seed_models(folder, seed)
        ancestor = read(PARENT/'evaluation-plan.json')['policies']['parent420'][str(seed)]
        write_policy(models['preflop420'], ancestor['path'], models['current480'])
        write_policy(models['specialist'], trained['model']['path'], models['current480'])
        # This isolated worker supplies street dispatch to the immutable, hash-pinned
        # evaluator. Raw logging, RNG, actions and trusted replay use its original run().
        evaluator.policy_for = evaluator_policy
        completed = {}
        for label, model in models.items():
            for field in FIELDS:
                cell = f'{field}-{label}'
                mark('evaluating', cell=cell, completed_cells=len(completed))
                evaluator.run(model, SOURCE, field, plan['deals'], plan['card_start'],
                              CFG['cell_timeout_seconds'], folder/cell)
                completed[cell] = digest(folder/cell/'results.json')
        verify_files(plan['implementation'])
        atomic_json(folder/'completed.json', completed)
        mark('complete', completed_cells=len(completed))
    except BaseException as exc:
        mark('failed', error=f'{type(exc).__name__}: {exc}')
        raise


def paired(rows, before, after, deals):
    changes = {}
    for field in FIELDS:
        values = np.array([np.subtract(rows[seed, field, after], rows[seed, field, before]) for seed in SEEDS])
        if values.shape != (len(SEEDS), deals) or not np.isfinite(values).all():
            raise ValueError('Incomplete or nonfinite paired card blocks')
        changes[field] = dict(**interval(values.mean(axis=0).tolist()),
                              per_seed=[dict(seed=s, mean=float(v.mean())) for s, v in zip(SEEDS, values)])
    return changes


def review(study):
    plan = read(study/'plan.json')
    verify_files(plan['implementation'])
    verify_parent()
    rows, opponents = {}, {}
    for seed in SEEDS:
        folder = study/f'seed-{seed}'
        files = read(folder/'completed.json')
        for label, model in seed_models(folder, seed).items():
            for field in FIELDS:
                cell = f'{field}-{label}'
                verify_files({folder/cell/'results.json': files[cell]})
                r = verify_results(folder/cell, expected=dict(field=field, deals=plan['deals'], deal_seed=plan['card_start'],
                    hands=6*plan['deals'], candidate=str(model), candidate_sha256=digest(model),
                    big_blind_chips=100, terminal_payoffs_replayed=True))
                if field in opponents and r['opponents'] != opponents[field]:
                    raise ValueError('Opponent pool changed across cells')
                opponents[field] = r['opponents']
                rows[seed, field, label] = r['deal_blocks']
    comparisons = {f'{a}_minus_{b}': paired(rows, b, a, plan['deals']) for a, b in (
        ('specialist', 'current480'), ('specialist', 'original120'), ('preflop420', 'current480'))}
    primary = comparisons['specialist_minus_current480']
    adoption = adoption_conditions(comparisons['specialist_minus_original120'])
    gates = dict(public_lower_above_zero=primary['public']['lower'] > 0,
                 positive_public_seeds=sum(p['mean'] > 0 for p in primary['public']['per_seed']) >= 2,
                 mixed_lower_above_minus_25=primary['mixed']['lower'] >= -25,
                 noregrets_lower_above_minus_50=primary['noregrets']['lower'] >= -50)
    return dict(completed_utc=utc(), smoke=plan['smoke'], completed_fits=len(SEEDS), completed_cells=len(rows),
                verified_hands=len(rows)*6*plan['deals'], independent_card_blocks=plan['deals'], comparisons=comparisons,
                parent_screen=None if plan['smoke'] else gates, existing_original120_conditions=None if plan['smoke'] else adoption,
                eligible_for_independent_confirmation=None if plan['smoke'] else all(gates.values()) and adoption['original_four_conditions_pass'],
                new_selfplay_iterations=0, model_promoted=False)


def supervise(study, smoke):
    # Both launch modes use this lock. A full attempt also permanently claims
    # its directory, so a finished, failed or interrupted run cannot restart.
    with SCRIPT.open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = preflight()
        study.mkdir(parents=True, exist_ok=False)
        plan = dict(smoke=smoke, steps=CFG['smoke_steps' if smoke else 'main_steps'],
                    deals=CFG['smoke_deals' if smoke else 'evaluation_deals'],
                    card_start=CFG['smoke_deal_seed' if smoke else 'evaluation_deal_seed'],
                    implementation=ready['implementation'], parent_run=PARENT.name)
        atomic_json(study/'plan.json', plan)
        # Existing poker supervisors discover reserved card ranges via config.json.
        atomic_json(study/'config.json', dict(evaluation_deal_seed=plan['card_start'], evaluation_deals=plan['deals']))
        workers = []
        def mark(state, **fields):
            atomic_json(study/'status.json', dict(state=state, pid=os.getpid(), process_start=process_start(os.getpid()),
                observed_utc=utc(), workers=[dict(seed=s, pid=p.pid, process_start=t) for s, p, t in workers], **fields))
        def terminate(signum, frame):
            raise InterruptedError(f'Supervisor received signal {signum}')
        def budget_exhausted(signum, frame):
            raise TimeoutError('Study exceeded its time budget')
        previous_term = signal.signal(signal.SIGTERM, terminate)
        previous_alarm = signal.signal(signal.SIGALRM, budget_exhausted)
        mark('starting')
        signal.setitimer(signal.ITIMER_REAL, CFG['max_seconds'])
        try:
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            for seed in SEEDS:
                with (study/f'seed-{seed}.log').open('x') as log:
                    child = subprocess.Popen([str(PYTHON), str(SCRIPT), '_worker', str(study), str(seed)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
                workers.append((seed, child, process_start(child.pid)))
                mark('running')
            while any(p.poll() is None for _, p, _ in workers):
                for seed, child, _ in workers:
                    if child.poll() not in (None, 0):
                        raise RuntimeError(f'Seed {seed} exited {child.returncode}. See seed-{seed}.log')
                time.sleep(1)
            if any(p.returncode for _, p, _ in workers):
                raise RuntimeError('A seed failed. Inspect worker status and logs')
            mark('verifying')
            result = review(study)
            atomic_json(study/'summary.json', result)
            mark('complete', summary_sha256=digest(study/'summary.json'))
            return result
        except BaseException as exc:
            for _, child, token in workers:
                if child.poll() is None and process_start(child.pid) == token:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for _, child, _ in workers:
                if child.poll() is None:
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            mark('failed', error=f'{type(exc).__name__}: {exc}')
            if smoke:
                for path in sorted(study.glob('seed-*.log')):
                    print(path.name, '\n'.join(path.read_text().splitlines()[-12:]), file=sys.stderr)
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.signal(signal.SIGTERM, previous_term)


def status(study):
    if not study.exists():
        return dict(state='not_started', parent_run=PARENT.name)
    if not (study/'status.json').exists():
        return dict(state='claimed_without_status', path=str(study))
    info = read(study/'status.json')
    info['supervisor_live'] = live(info['pid'], info['process_start'], ['run'])
    if info['state'] in ('starting', 'running', 'verifying') and not info['supervisor_live']:
        info['state'] = 'interrupted'
    info.update(completed_fits=0, completed_cells=0, planned_cells=len(SEEDS)*len(FIELDS)*len(LABELS))
    for item in info['workers']:
        seed = item['seed']
        item['live'] = live(item['pid'], item['process_start'], ['_worker', str(study), str(seed)])
        folder = study/f'seed-{seed}'
        path = folder/'status.json'
        if path.exists():
            item['progress'] = read(path)
        info['completed_fits'] += int((folder/'fit.json').is_file())
        info['completed_cells'] += sum(1 for _ in folder.glob('*-*/results.json'))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight', 'smoke', 'run', 'status', 'result', '_worker'])
    parser.add_argument('internal', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == '_worker':
        study = Path(args.internal[0])
        plan = read(study/'plan.json')
        # A worker may only run while its supervisor is alive; never a direct retry.
        owner = read(study/'status.json')
        if os.getppid() != owner['pid'] or not live(owner['pid'], owner['process_start'], ['smoke' if plan['smoke'] else 'run']):
            raise ValueError('Unowned worker launch')
        seed_worker(study, int(args.internal[1]))
        return
    if args.internal:
        parser.error('This action takes no paths or tuning parameters')
    if args.action == 'preflight':
        result = preflight()
    elif args.action == 'status':
        result = status(OUTPUT)
    elif args.action == 'smoke':
        with tempfile.TemporaryDirectory(prefix='poker-preflop-smoke-') as temp:
            study = Path(temp)/'pilot'
            outcome = supervise(study, True)
            result = {k: v for k, v in outcome.items() if k not in ('comparisons', 'existing_original120_conditions', 'parent_screen')}
            result.update(implementation=implementation(), steps_per_seed=CFG['smoke_steps'],
                          parent_rows_used_per_seat={str(s):read(study/f'seed-{s}'/'fit.json')['selected_rows_per_seat'] for s in SEEDS})
    elif args.action == 'run':
        result = supervise(OUTPUT, False)
    else:
        state = status(OUTPUT)
        if state['state'] != 'complete' or digest(OUTPUT/'summary.json') != state['summary_sha256']:
            raise SystemExit(f"Result unavailable ({state['state']})")
        result = read(OUTPUT/'summary.json')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
