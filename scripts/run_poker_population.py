"""Register and run the fresh-card, frozen 480 policy comparison."""
import argparse
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from time import time

import numpy as np

import evaluate_average_baseline as evaluator
import run_poker_preflop_confirmation as previous
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_preflop_specialist import evaluator_policy
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges


CONFIG = ROOT / 'configs/poker-population-20261119.json'
PARENT = previous.FOLLOWUP / 'preflop-dose'
OUTPUT = previous.FOLLOWUP / 'population'
PUBLIC = OUTPUT / 'public-confirmation'
SCRIPT = Path(__file__).resolve()


def register():
    cfg = read(CONFIG)
    previous.check_no_jobs()
    frozen = {str(PARENT / name): cfg[key] for name, key in [
        ('evaluation-plan.json', 'parent_plan_sha256'), ('summary.json', 'parent_summary_sha256'),
        ('final-audit.json', 'parent_audit_sha256')]}
    verify_files(frozen)
    parent = read(PARENT / 'evaluation-plan.json')
    models = {name: parent['policies'][name] for name in cfg['labels']}
    frozen.update(parent['frozen_files'])
    for refs in models.values():
        for model in refs.values():
            policy = evaluator_policy(model['path'], parent['source'])
            frozen.update(policy.dependency_hashes)
            frozen[model['path']] = model['sha256']
    frozen[str(CONFIG)] = digest(CONFIG)
    verify_files(frozen)
    ranges = {k: item['range'] for k, item in prior_card_ranges(OUTPUT).items()}
    for key in ('public_confirmation', 'panel_cards', 'holdout_cards'):
        spec = cfg[key]
        start, count = spec['deal_seed'], spec['deals']
        check_card_range(start, count, ranges)
        ranges[key] = [start, start + count]
    OUTPUT.mkdir(exist_ok=False)
    fields = {field: {**spec, **cfg['holdout_cards' if spec['role'] == 'holdout' else 'panel_cards']}
              for field, spec in cfg['fields'].items()}
    for folder, batch in [(PUBLIC, {'public': {**cfg['public_confirmation'], 'role': 'replication',
                                                'family': 'public'}}), (OUTPUT, fields)]:
        folder.mkdir(exist_ok=True)
        (folder / 'evaluation').mkdir()
        jobs = [dict(id=f'{field}-seed-{seed}-{label}', field=field, seed=seed, label=label)
                for field in batch for seed in cfg['seeds'] for label in cfg['labels']]
        atomic_json(folder / 'evaluation-plan.json', dict(registered_utc=utc(), design=cfg,
                    policies=models, source=parent['source'], seeds=cfg['seeds'], fields=batch,
                    jobs=jobs, frozen_files=frozen, planned_hands=sum(batch[j['field']]['deals']*6 for j in jobs)))
    atomic_json(OUTPUT / 'registration.json', dict(registered_utc=utc(), config_sha256=digest(CONFIG),
                public_plan_sha256=digest(PUBLIC / 'evaluation-plan.json'),
                panel_plan_sha256=digest(OUTPUT / 'evaluation-plan.json'), range_count=len(ranges)-3))


def run_batch(folder):
    registration = read(OUTPUT / 'registration.json')
    plan_file = folder / 'evaluation-plan.json'
    plan_sha = registration['public_plan_sha256' if folder == PUBLIC else 'panel_plan_sha256']
    verify_files({plan_file: plan_sha, CONFIG: registration['config_sha256']})
    plan = read(plan_file)
    verify_files(plan['frozen_files'])
    cfg = plan['design']
    commands = {}
    worker = ROOT / 'scripts/evaluate_poker_diverse_panel.py'
    for job in plan['jobs']:
        spec = plan['fields'][job['field']]
        if folder == PUBLIC:
            commands[job['id']] = [str(PYTHON), str(SCRIPT), 'public-cell', job['id']]
        else:
            commands[job['id']] = [str(PYTHON), str(worker), '--label', job['label'],
                     '--seed', str(job['seed']), '--field', job['field'],
                     '--deal-seed', str(spec['deal_seed']), '--deals', str(spec['deals']),
                     '--timeout', str(cfg['timeout_seconds']), '--output', str(folder / 'evaluation' / job['id'])]
    # Exclusive creation is also the guard against restarting a failed or live batch.
    with (folder / 'launch.json').open('x') as handle:
        import json
        json.dump(dict(utc=utc(), plan_sha256=plan_sha, worker_sha256=digest(SCRIPT if folder == PUBLIC else worker)), handle)
    previous.run_jobs(folder, commands, cfg['workers'], 'evaluation')
    verify_files({plan_file: plan_sha, **plan['frozen_files']})


def public_cell(job_id):
    plan = read(PUBLIC / 'evaluation-plan.json')
    job, = [job for job in plan['jobs'] if job['id'] == job_id]
    model = plan['policies'][job['label']][str(job['seed'])]
    verify_files({model['path']: model['sha256']})
    spec = plan['fields']['public']
    evaluator.policy_for = evaluator_policy
    evaluator.run(model['path'], plan['source'], 'public', spec['deals'], spec['deal_seed'],
                  plan['design']['timeout_seconds'], PUBLIC / 'evaluation' / job_id)


def public_summary():
    plan = read(PUBLIC / 'evaluation-plan.json')
    verify_files({PUBLIC / 'evaluation-plan.json': read(OUTPUT / 'registration.json')['public_plan_sha256']})
    spec = plan['fields']['public']
    payoffs = {label: [] for label in plan['policies']}
    raw_checks = {}
    for job in plan['jobs']:
        folder = PUBLIC / 'evaluation' / job['id']
        ref = plan['policies'][job['label']][str(job['seed'])]
        row = evaluator.verify_results(folder, expected=dict(field='public', deal_seed=spec['deal_seed'],
                    deals=spec['deals'], candidate_sha256=ref['sha256'], terminal_payoffs_replayed=True))
        payoffs[job['label']].append(row['deal_blocks'])
        raw_checks[job['id']] = dict(result=digest(folder/'results.json'), raw=row['raw_hands_sha256'])
    by_seed = np.asarray(payoffs['steps256']) - np.asarray(payoffs['steps0'])
    blocks = by_seed.mean(axis=0)
    change = float(blocks.mean())
    width = NormalDist().inv_cdf(.975)*float(blocks.std(ddof=1))/np.sqrt(len(blocks))
    result = dict(verified_utc=utc(), cards=spec, independent_deals=len(blocks),
                  planned_hands=plan['planned_hands'], completed_hands=len(blocks)*len(plan['jobs'])*6,
                  unit='bb/100', scores={label:float(np.mean(rows)) for label,rows in payoffs.items()},
                  difference=change, paired_ci95=[change-width,change+width], differences_by_seed=by_seed.mean(axis=1).tolist(),
                  raw_checks=raw_checks, new_selfplay=0, new_fitting=0, model_promoted=False)
    atomic_json(PUBLIC / 'summary.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register','public','panel','public-cell','public-summary'])
    parser.add_argument('job', nargs='?')
    args = parser.parse_args()
    if args.action == 'register':
        register()
    elif args.action == 'public-cell':
        public_cell(args.job)
    elif args.action == 'public-summary':
        public_summary()
    else:
        run_batch(PUBLIC if args.action == 'public' else OUTPUT)
