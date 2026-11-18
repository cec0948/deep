"""Confirm the frozen preflop pilot on disjoint common cards."""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

import numpy as np

import evaluate_average_baseline as evaluator
import run_poker_preflop_pilot as pilot
from controlled_poker_refit import read, utc, verify_files
from evaluate_compact_learning_curve import adoption_conditions
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy, file_ref
from run_compact_refit_probe import check_card_range
from run_poker_average_extension import live_poker_jobs
from run_poker_candidate_confirmation import prior_card_ranges


SCRIPT = Path(__file__).resolve()
CONFIG = ROOT/'configs/poker-preflop-followup-20261118.json'
CFG = read(CONFIG)
FOLLOWUP = ROOT/'artifacts/preflop-followup/480-v1'
OUTPUT = FOLLOWUP/'confirmation'
SEEDS = pilot.SEEDS
FIELDS = pilot.FIELDS
LABELS = ('original120', 'current480', 'specialist')


def check_no_jobs():
    active = {row['pid'] for row in live_poker_jobs() if row['process_start']}
    names = {'run_poker_preflop_pilot.py', SCRIPT.name, 'run_poker_preflop_online.py'}
    for proc in Path('/proc').glob('[0-9]*'):
        pid = int(proc.name)
        if pid == os.getpid() or not process_start(pid):
            continue
        try:
            argv = (proc/'cmdline').read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if len(argv) > 1 and Path(argv[1]).name in names:
            active.add(pid)
    if active:
        raise RuntimeError(f'Existing poker processes {sorted(active)}')


def check_fresh_cards(study, start, deals):
    previous = {k: v['range'] for k, v in prior_card_ranges(study).items()}
    check_card_range(start, deals, previous)
    return sorted(previous.values())


def frozen_pilot():
    summary = pilot.OUTPUT/'summary.json'
    audit = ROOT/'reports/preflop-pilot/480-v1/final-audit.json'
    verify_files({summary: CFG['pilot_summary_sha256']})
    previous = read(summary)
    if (previous['smoke'] or previous['completed_cells'] != 36
            or previous['new_selfplay_iterations'] != 0
            or not previous['eligible_for_independent_confirmation']
            or read(audit)['summary_sha256'] != digest(summary)):
        raise ValueError('The audited 480-round preflop pilot must be complete and eligible')
    pilot.verify_parent()
    verify_files(read(pilot.OUTPUT/'plan.json')['implementation'])
    ready = {str(p): digest(p) for p in (summary, audit, CONFIG, SCRIPT, pilot.PARENT/'stage-ready.json')}
    models = {label: {} for label in LABELS}
    opponents = {}
    for seed in SEEDS:
        folder = pilot.OUTPUT/f'seed-{seed}'
        fit = read(folder/'fit.json')
        paths = pilot.seed_models(folder, seed)
        if read(paths['specialist']) != dict(format='preflop-street-policy-v1',
                preflop=fit['model'], postflop=file_ref(paths['current480'])):
            raise ValueError('Specialist routing differs from the selected policy')
        for label in LABELS:
            models[label][str(seed)] = file_ref(paths[label])
            for field in FIELDS:
                cell = f'{field}-{label}'
                result = folder/cell/'results.json'
                verify_files({result: read(folder/'completed.json')[cell]})
                row = read(result)
                if (row['candidate_sha256'] != models[label][str(seed)]['sha256']
                        or not row['terminal_payoffs_replayed']):
                    raise ValueError('Frozen policy differs from the verified pilot')
                if opponents.setdefault(field, row['opponents']) != row['opponents']:
                    raise ValueError('Pilot opponent identities differ')
                for path, sha in row['dependency_hashes'].items():
                    if ready.setdefault(path, sha) != sha:
                        raise ValueError('Pilot dependency identities differ')
    verify_files(ready)
    return models, opponents, ready


def evaluate_cell(study, index):
    plan = read(study/'evaluation-plan.json')
    job = plan['jobs'][index]
    ref = plan['policies'][job['label']][str(job['seed'])]
    verify_files({ref['path']: ref['sha256']})
    evaluator.policy_for = evaluator_policy
    evaluator.run(ref['path'], pilot.SOURCE, job['field'], plan['deals'], plan['card_start'],
                  CFG['cell_timeout_seconds'], study/'evaluation'/job['id'])


def run_jobs(folder, commands, workers, phase):
    pending = deque(commands.items())
    active = {}
    done = []
    log_dir = folder/'logs'
    log_dir.mkdir(exist_ok=True)
    status_path = folder/f'{phase}-status.json'
    token = process_start(os.getpid())

    def record(state, **fields):
        atomic_json(status_path, dict(state=state, observed_utc=utc(), pid=os.getpid(),
            process_start=token, planned=len(commands), completed=len(done),
            active={name: dict(pid=p.pid, process_start=start, argv=p.args)
                    for name, (p, start) in active.items()}, **fields))

    def interrupt(signum, frame):
        raise InterruptedError(f'Supervisor signal {signum}')

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1')
    try:
        record('running')
        while pending or active:
            while pending and len(active) < workers:
                name, command = pending.popleft()
                with (log_dir/f'{phase}-{name}.log').open('x') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=env, start_new_session=True)
                active[name] = (child, process_start(child.pid))
            record('running')
            time.sleep(2)
            for name, (child, _) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[name]
                if code:
                    log_tail = '\n'.join((log_dir/f'{phase}-{name}.log').read_text().splitlines()[-8:])
                    raise RuntimeError(f'{phase} {name} exited {code}\n{log_tail}')
                done.append(name)
        record('complete')
    except BaseException as error:
        for child, _ in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child, _ in active.values():
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        active.clear()
        record('failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def evaluate(study, policies, opponents, frozen, start, deals):
    jobs = [dict(id=f'{field}-seed-{seed}-{label}', label=label, seed=seed, field=field)
            for label in policies for field in FIELDS for seed in SEEDS]
    plan = dict(created_utc=utc(), policies=policies, opponents=opponents, jobs=jobs,
                deals=deals, card_start=start, frozen_files=frozen,
                source=str(pilot.SOURCE), planned_hands=len(jobs)*deals*6)
    atomic_json(study/'evaluation-plan.json', plan)
    plan_hash = digest(study/'evaluation-plan.json')
    (study/'evaluation').mkdir()
    commands = {job['id']: [str(PYTHON), str(SCRIPT), '_evaluate', str(study), str(i)]
                for i, job in enumerate(jobs)}
    run_jobs(study, commands, CFG['evaluation_workers'], 'evaluation')
    verify_files({study/'evaluation-plan.json': plan_hash, **frozen})
    rows, digests, fallback = {}, {}, {}
    for job in jobs:
        name, label, field, seed = (job[k] for k in ('id', 'label', 'field', 'seed'))
        ref = policies[label][str(seed)]
        folder = study/'evaluation'/name
        result = evaluator.verify_results(folder, expected=dict(field=field, deals=deals,
            deal_seed=start, hands=deals*6, candidate=ref['path'], candidate_sha256=ref['sha256'],
            opponents=opponents[field], big_blind_chips=100, terminal_payoffs_replayed=True))
        rows[seed, field, label] = result['deal_blocks']
        digests[name] = digest(folder/'results.json')
        behavior = result['behavior']
        fallback[name] = [sum(n for k, n in behavior.items() if k.startswith('opponent/') and k.endswith('/'+suffix))
                          for suffix in ('fallback', 'decisions')]
    absolute = {label: {field: evaluator.interval(np.mean([rows[s, field, label] for s in SEEDS], axis=0).tolist())
                        for field in FIELDS} for label in policies}
    return rows, dict(completed_cells=len(rows), verified_hands=plan['planned_hands'],
        independent_card_blocks=deals, evaluation_deal_seed=start, absolute_scores=absolute,
        result_sha256=digests, evaluation_plan_sha256=plan_hash,
        noregrets_fallback_and_decisions=fallback)


def confirm(study, smoke=False):
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    with (FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_no_jobs()
        models, opponents, frozen = frozen_pilot()
        deals = 4 if smoke else CFG['evaluation_deals']
        start = 2037560000 if smoke else CFG['confirmation_deal_seed']
        previous = check_fresh_cards(study, start, deals)
        if not smoke and shutil.disk_usage(FOLLOWUP).free < 2**30:
            raise RuntimeError('Insufficient space for verified raw evaluation records')
        study.mkdir(exist_ok=False)
        atomic_json(study/'config.json', dict(evaluation_deal_seed=start, evaluation_deals=deals,
            prior_card_ranges=previous, pilot_summary_sha256=CFG['pilot_summary_sha256'],
            criteria='Repeat the four pilot conditions against original120 and current480',
            smoke=smoke, new_selfplay_iterations=0, new_fitting_jobs=0))
        try:
            rows, outcome = evaluate(study, models, opponents, frozen, start, deals)
            changes = {baseline: pilot.paired(rows, baseline, 'specialist', deals)
                       for baseline in ('original120', 'current480')}
            gates = {baseline: adoption_conditions(change)['conditions'] for baseline, change in changes.items()}
            outcome.update(completed_utc=utc(), smoke=smoke, new_fitting_jobs=0,
                new_selfplay_iterations=0, model_promoted=False, specialist_minus=changes,
                conditions=None if smoke else gates,
                independent_confirmation_pass=None if smoke else all(all(g.values()) for g in gates.values()))
            atomic_json(study/'summary.json', outcome)
            return outcome
        except BaseException as error:
            atomic_json(study/'failure.json', dict(observed_utc=utc(), error=f'{type(error).__name__}: {error}'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('smoke', 'run', '_evaluate'))
    parser.add_argument('internal', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == '_evaluate':
        study, index = args.internal
        status = read(Path(study)/'evaluation-status.json')
        if status['pid'] != os.getppid() or process_start(status['pid']) != status['process_start']:
            raise ValueError('Unowned evaluation worker')
        evaluate_cell(Path(study), int(index))
        return
    if args.internal:
        parser.error('No overrides for this fixed study')
    if args.action == 'smoke':
        with tempfile.TemporaryDirectory(prefix='poker-preflop-confirmation-') as temp:
            result = confirm(Path(temp)/'smoke', smoke=True)
    else:
        result = confirm(OUTPUT)
    print(json.dumps({k: v for k, v in result.items() if k not in ('result_sha256', 'noregrets_fallback_and_decisions')},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
