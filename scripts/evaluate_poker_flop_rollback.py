"""Isolate flop policy replacement on frozen, previously observed evaluation deals."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

import evaluate_average_baseline as verification
import evaluate_poker_opponent_diversity as previous
from controlled_poker_refit import read, utc, verify_files
from evaluate_poker_postflop_snapshots import ready_engine
from fullhouse_v3_common import PYTHON, ROOT, atomic_json, digest, process_start
from poker_flop_rollback import FlopRollbackPolicy, write_policy
from run_poker_preflop_confirmation import run_jobs
from score_poker_flop_rollback import summarize_study


STUDY = ROOT / 'artifacts/preflop-followup/480-v1/flop-rollback-20261123'
PRIOR = previous.OUTPUT
LABEL = 'flop480late504'
SCRIPT = Path(__file__).resolve()
CONFIG = ROOT / 'configs/poker-flop-rollback-20261123.json'
CONFIG_HASH = '75027475d933dd7020e610c78c17a92abfbbf001feae5521cea25051e825d8af'
PLAN_HASH = 'a851844aee89e4032992840e40b2bf41ac765cc132dd5be9f94fee09e9540b43'
REFERENCE_HASH = '691897dda94b12155ed6cf4141e866a02d3604b633e6a00186d768b41d5ab2d8'
SUMMARY_HASH = '070c481106dd77ba6bbc9dcd20b6d9969cc0e6d7b1291c3e5d91ae074348e515'


def reference_inputs(prior_plan, summary, dependencies):
    reference_cells, opponents = {}, {}
    early = {j['id'] for j in read(PRIOR/'reference-plan.json')['jobs']}
    for job in prior_plan['jobs']:
        field, seed, label, name = (job[key] for key in ('field', 'seed', 'label', 'id'))
        if label not in (previous.BASE, previous.POOL):
            continue
        folder = PRIOR / 'evaluation' / name
        reference_cells[name] = summary['cell_result_sha256'][name]
        verify_files({folder/'results.json': reference_cells[name]})
        result = read(folder/'results.json')
        expected = dict(field=field, deals=prior_plan['fields'][field]['deals'],
                        deal_seed=prior_plan['fields'][field]['deal_seed'], training_seed=seed, label=label,
                        candidate_sha256=prior_plan['policies'][label][str(seed)]['sha256'],
                        evaluation_plan_sha256=REFERENCE_HASH if name in early else PLAN_HASH,
                        complete=True, terminal_payoffs_replayed=True, raw_hands_verified=True)
        if any(result.get(k) != v for k,v in expected.items()):
            raise ValueError(f'Incorrect archived cell identity: {name}')
        key = f'{field}-seed-{seed}'
        if opponents.setdefault(key, result['opponents']) != result['opponents']:
            raise ValueError(f'Inconsistent archived opponent bindings for {key}')
        for path, sha in result['dependency_hashes'].items():
            if dependencies.setdefault(path, sha) != sha:
                raise ValueError(f'Inconsistent archived dependency: {path}')
        for filename, sha in [('hands.jsonl.gz',result['raw_hands_sha256']),
                              ('provenance.json',result['provenance_sha256'])]:
            dependencies[str(folder/filename)] = sha
        dependencies[str(folder/'results.json')] = reference_cells[name]
    return reference_cells, opponents


def register():
    verify_files({CONFIG:CONFIG_HASH})
    cfg = read(CONFIG)
    old_paths = {CONFIG:CONFIG_HASH, PRIOR.parent/'final-audit.json':cfg['source_final_audit_sha256'],
                 PRIOR/'evaluation-plan.json':PLAN_HASH, PRIOR/'reference-plan.json':REFERENCE_HASH,
                 PRIOR/'summary.json':SUMMARY_HASH}
    verify_files(old_paths)
    prior_plan, summary = read(PRIOR/'evaluation-plan.json'), read(PRIOR/'summary.json')
    dependencies = dict(prior_plan['frozen_files'])
    dependencies.update({str(k):v for k,v in old_paths.items()})
    refs, opponents = reference_inputs(prior_plan, summary, dependencies)
    own = (SCRIPT, SCRIPT.with_name('poker_flop_rollback.py'), SCRIPT.with_name('score_poker_flop_rollback.py'),
           SCRIPT.with_name('run_poker_preflop_confirmation.py'))
    dependencies.update({str(p):digest(p) for p in own})
    verify_files(dependencies)
    fields, seeds = prior_plan['fields'], prior_plan['seeds']
    jobs = [dict(id=f'{field}-seed-{seed}-{LABEL}', field=field, seed=seed, label=LABEL)
            for field in fields for seed in seeds]
    hands = sum(6*fields[j['field']]['deals'] for j in jobs)
    blocks = sum(n for _,n in {(v['deal_seed'],v['deals']) for v in fields.values()})
    if (len(jobs),len(refs),hands,blocks) != (27,54,648000,26000):
        raise ValueError('Evaluation geometry does not match the registered design')
    STUDY.mkdir(parents=True, exist_ok=True)
    model_dir = STUDY/'policies'
    model_dir.mkdir(exist_ok=False)
    policies = {}
    for seed in seeds:
        key = str(seed)
        path = model_dir/f'seed-{seed}.json'
        write_policy(path, seed, prior_plan['policies'][previous.BASE][key], prior_plan['policies'][previous.POOL][key])
        candidate = FlopRollbackPolicy(path, prior_plan['source'])
        policies[key] = dict(path=str(path),sha256=digest(path))
        dependencies.update(verification.policy_dependency_hashes(candidate, prior_plan['source']))
    verify_files(dependencies)
    plan = dict(registered_utc=utc(),config_sha256=CONFIG_HASH,fields=fields,seeds=seeds,policies={LABEL:policies},jobs=jobs,
                previous_plan_sha256=PLAN_HASH,previous_reference_plan_sha256=REFERENCE_HASH,
                previous_summary_sha256=SUMMARY_HASH,source=prior_plan['source'],frozen_files=dependencies,
                previous_cells=refs,opponents=opponents,expected_new_hands=hands,expected_reference_hands=2*hands,
                unique_original_card_blocks=blocks,workers=4,cell_timeout_seconds=7200,
                purpose='Previously observed deals used to select the flop diagnostic; paired attribution only.',
                policy='480 preflop and flop, registered 504 snapshot mixture on turn and river; identical private slot per hand.',
                primary='Public paired difference, flop480late504 minus diverse504, with nominal 95% interval.',
                secondary='Public difference versus specialist480; family-balanced +100 cap and +50/+200/uncapped sensitivity and all named tables against both comparators.',
                training_updates=0, model_promoted=False, automatic_retry=False)
    (STUDY/'evaluation').mkdir()
    atomic_json(STUDY/'evaluation-plan.json',plan)
    atomic_json(STUDY/'bound.json',dict(evaluation_plan_sha256=digest(STUDY/'evaluation-plan.json'),bound_utc=utc()))


def bound_plan():
    sha = read(STUDY/'bound.json')['evaluation_plan_sha256']
    verify_files({STUDY/'evaluation-plan.json':sha})
    return read(STUDY/'evaluation-plan.json'), sha


def worker(field, seed):
    plan, plan_hash = bound_plan()
    specs = [j for j in plan['jobs'] if j['field'] == field and j['seed'] == seed]
    if len(specs) != 1:
        raise ValueError('Unscheduled evaluation cell')
    out = STUDY/'evaluation'/specs[0]['id']
    out.mkdir(exist_ok=False)
    def progress(phase, **values):
        atomic_json(out/'status.json', dict(phase=phase, observed_utc=utc(), pid=os.getpid(),
                    process_start=process_start(os.getpid()), **values))
    try:
        bots, ids, blueprint, dependencies = previous.inputs(seed, field)
        frozen = plan['frozen_files']
        model = plan['policies'][LABEL][str(seed)]
        candidate = FlopRollbackPolicy(model['path'], plan['source'])
        dependencies.update(verification.policy_dependency_hashes(candidate, plan['source']))
        dependencies.update({str(p):frozen[str(p)] for p in (SCRIPT,Path(previous.__file__))})
        if (any(frozen.get(p) != sha for p,sha in dependencies.items())
                or ids != plan['opponents'][f'{field}-seed-{seed}']):
            raise ValueError('Worker opponent or runtime differs from the registered bytes')
        verify_files(dependencies)
        field_spec = plan['fields'][field]
        start, deals = field_spec['deal_seed'], field_spec['deals']
        prov = dict(label=LABEL,field=field,training_seed=seed,deal_seed=start,deals=deals,hands=6*deals,
                    candidate=str(candidate.model),candidate_sha256=model['sha256'],candidate_type=candidate.model_type,
                    opponents=ids,dependency_hashes=dependencies,big_blind_chips=100,
                    first_deal_clockwise_lineup=previous.table(field,start),varies_by_card=field=='varied',
                    evaluation_plan_sha256=plan_hash,statistical_unit='One original card block across all hero seats and fixed training seeds')
        atomic_json(out/'provenance.json',prov)
        counts, choices, blocks = Counter(), Counter(), []
        raw = out/'hands.jsonl.gz'
        with ready_engine(blueprint, STUDY) as engine:
            deadline = time.monotonic()+plan['cell_timeout_seconds']
            progress('evaluating',completed_card_blocks=0)
            with gzip.open(raw,'xt',compresslevel=1) as stream:
                for i,card in enumerate(range(start,start+deals)):
                    total = 0
                    table = previous.table(field,card)
                    for hero in range(6):
                        row = previous.panel.play(engine,candidate,bots,table,card,hero,counts,deadline)
                        slot = candidate.snapshot_iteration
                        if slot not in previous.SNAPSHOTS:
                            raise ValueError('Missing or unscheduled private hand snapshot')
                        choices[slot] += 1
                        row.update(deal=i,snapshot_iteration=slot)
                        stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                        total += row['chips']
                    blocks.append(total/6)
                    if (i+1)%100 == 0:
                        progress('evaluating',completed_card_blocks=i+1)
        stats = verification.interval(blocks)
        result = dict(prov,deal_blocks=blocks,bb_per_100=stats['mean'],ci95_half_width=stats['ci95_half_width'],
                      behavior=dict(counts),snapshot_hand_counts=dict(choices))
        progress('verifying_raw_hands',verified_hands=0)
        check = verification._verify_rows(result,raw,replay=True,deadline=deadline,progress=progress)
        verify_files(dependencies)
        result.update(complete=True,raw_hands_verified=True,terminal_payoffs_replayed=True,verification=check,
                      completed_utc=utc(),raw_hands_sha256=digest(raw),provenance_sha256=digest(out/'provenance.json'))
        atomic_json(out/'results.json',result)
        progress('complete',verified_hands=6*deals,results_sha256=digest(out/'results.json'))
    except BaseException as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}',automatic_retry=False)
        raise


def run():
    plan, ident = bound_plan()
    verify_files(plan['frozen_files'])
    target = STUDY/'launch.json'
    with target.open('x') as launch:
        json.dump(dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid()),plan_sha256=ident), launch)
    commands = {j['id']:[str(PYTHON),str(SCRIPT),'_worker',j['field'],str(j['seed'])] for j in plan['jobs']}
    run_jobs(STUDY,commands,plan['workers'],'evaluation')
    summarize_study(STUDY, PRIOR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['register','run','summarize','_worker'])
    parser.add_argument('args',nargs='*')
    args = parser.parse_args()
    if args.mode == '_worker' and len(args.args) == 2:
        worker(args.args[0],int(args.args[1]))
    elif args.args or args.mode == '_worker':
        parser.error('Unexpected worker arguments')
    else:
        {'register':register,'run':run,'summarize':lambda: summarize_study(STUDY, PRIOR)}[args.mode]()
