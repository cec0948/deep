"""Finish unscored population hands after the recorded opponent initialization timeout."""
import argparse
import os

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import PYTHON, ROOT, atomic_json, digest, process_start
from run_poker_population import CONFIG, OUTPUT, PUBLIC
from run_poker_preflop_confirmation import run_jobs


BATCH = OUTPUT / 'completion'
WORKER = ROOT / 'scripts/evaluate_poker_diverse_panel.py'
RECOVER = ROOT / 'scripts/finish_poker_panel_cell.py'
LIGHT = {'rich96', 'calling-station', 'tight-value', 'pressure'}


def queue(group):
    first = read(OUTPUT / 'evaluation-status.json')
    if (first['state'] != 'failed' or first['completed'] != 4 or 'Engine response timeout' not in first['error']
            or first['active'] or process_start(first['pid']) == first['process_start']):
        raise ValueError('Unexpected first batch or supervisor still active')
    plan = read(OUTPUT / 'evaluation-plan.json')
    cfg = plan['design']
    reg = read(OUTPUT / 'registration.json')
    verify_files({CONFIG: reg['config_sha256'], OUTPUT/'evaluation-plan.json': reg['panel_plan_sha256'],
                  WORKER: read(OUTPUT/'launch.json')['worker_sha256'], **plan['frozen_files']})
    if group == 'with-engine' and read(PUBLIC/'evaluation-status.json')['state'] != 'complete':
        raise ValueError('Reserve two normal-priority CPU slots for loading the opponent')
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError('Launch at normal process priority')
    batch = BATCH/group
    batch.mkdir(exist_ok=False, parents=True)
    commands, recovered, skipped = {}, [], []
    for job in plan['jobs']:
        field, name = job['field'], job['id']
        if (field in LIGHT) != (group == 'light'):
            continue
        folder = OUTPUT/'evaluation'/name
        if (folder/'results.json').is_file():
            row = read(folder/'results.json')
            if not (row['complete'] and row['terminal_payoffs_replayed'] and
                    row['candidate_sha256'] == plan['policies'][job['label']][str(job['seed'])]['sha256']):
                raise ValueError(f'Unexpected completed record for {name}')
            skipped.append(name)
            continue
        spec = plan['fields'][field]
        args = ['--field', field, '--seed', str(job['seed']), '--label', job['label'],
                '--deal-seed', str(spec['deal_seed']), '--deals', str(spec['deals']),
                '--timeout', str(cfg['timeout_seconds'])]
        if folder.exists():
            receipt = read(folder/'status.json')
            if process_start(receipt['pid']) == receipt['process_start']:
                raise ValueError(f'Interrupted worker for {name} is still alive')
            original = batch/'originals'/name
            original.parent.mkdir(exist_ok=True)
            # Preserve the entire original cell. Only verified results under evaluation count.
            folder.rename(original)
            commands[name] = [str(PYTHON), str(RECOVER), *args, '--original', str(original), '--output', str(folder)]
            recovered.append(name)
        else:
            commands[name] = [str(PYTHON), str(WORKER), *args, '--output', str(folder)]
    atomic_json(batch/'queue.json', dict(utc=utc(), group=group, scheduled=list(commands),
                preserved_complete=skipped, continued_incomplete=recovered,
                frozen_panel_worker_sha256=digest(WORKER), original_failure_sha256=digest(OUTPUT/'evaluation-status.json')))
    run_jobs(batch, commands, 2, 'evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=('light', 'with-engine'))
    queue(parser.parse_args().group)
