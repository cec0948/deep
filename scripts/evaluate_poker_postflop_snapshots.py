"""Paired evaluation of a frozen preflop policy and prospective postflop snapshots."""
import argparse
from collections import Counter
from contextlib import closing, contextmanager
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import select
from statistics import NormalDist
import time

import numpy as np

import evaluate_average_baseline as baseline_eval
import evaluate_poker_diverse_panel as panel
import poker_capture_postflop_snapshots as snapshots
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges
from run_poker_preflop_confirmation import run_jobs
from score_poker_population import score_panel


CONFIG = ROOT/'configs/poker-postflop-averaging-evaluation-20261120.json'
CONFIG_SHA = '32364b5cb013d692b62fbc975c7cb261d70b878712bc7bbe0a58681ac546874b'
STUDY = ROOT/'artifacts/preflop-followup/480-v1/postflop-averaging-20261120'
OUTPUT = STUDY/'population'
SCRIPT = Path(__file__).resolve()
SEEDS = (342, 343, 344)
BASE, MIX, END, EXTRA = 'specialist480', 'mixture504', 'slow504', 'steps256'


def fixed_design():
    verify_files({CONFIG: CONFIG_SHA})
    cfg = read(CONFIG)
    fields = {name: dict(family=spec['family'], role=spec['role'], **cfg[spec['cards']])
              for name, spec in cfg['fields'].items()}
    jobs = [dict(id=f'{field}-seed-{seed}-{label}', field=field, seed=seed, label=label)
            for field in fields for seed in SEEDS
            for label in ([BASE, MIX, END, EXTRA] if field == 'public' else [BASE, MIX, END])]
    hands = sum(fields[j['field']]['deals'] * 6 for j in jobs)
    unique = sum(n for _, n in {(s['deal_seed'], s['deals']) for s in fields.values()})
    if (len(jobs), hands, unique) != (cfg['expected_cells'], cfg['expected_hands'], cfg['expected_unique_card_blocks']):
        raise ValueError('Evaluation coverage differs from the preregistration')
    return cfg, fields, jobs


def register():
    cfg, fields, jobs = fixed_design()
    old = ROOT/'configs/poker-population-20261119.json'
    verify_files({old: cfg['fixed_opponent_design_sha256'], panel.DOSE_PLAN: panel.DOSE_SHA, panel.RICH_STAGE: panel.RICH_SHA})
    if any(read(old)['fields'][name] != {k: spec[k] for k in ('role','family')} for name,spec in fields.items()):
        raise ValueError('Opponent family assignment differs from registered population study')
    prev = {k:v['range'] for k,v in prior_card_ranges(STUDY).items()}
    for key in ('public_cards', 'panel_cards', 'holdout_cards'):
        start, count = (cfg[key][k] for k in ('deal_seed','deals'))
        check_card_range(start, count, prev)
        prev[key] = [start, start+count]
    OUTPUT.mkdir(exist_ok=False)
    # prior_card_ranges discovers each reservation through the standard config schema.
    for key in ('public_cards', 'panel_cards', 'holdout_cards'):
        reservation = OUTPUT/'reservations'/key
        reservation.mkdir(parents=True)
        atomic_json(reservation/'config.json', dict(evaluation_deal_seed=cfg[key]['deal_seed'], evaluation_deals=cfg[key]['deals']))
    atomic_json(OUTPUT/'design.json', dict(config=cfg, fields=fields, jobs=jobs))
    atomic_json(OUTPUT/'registration.json', dict(created_utc=utc(), config_sha256=CONFIG_SHA,
                design_sha256=digest(OUTPUT/'design.json'), checked_ranges=sorted(prev.values())))


def load_candidate(label, ref, source):
    verify_files({ref['path']: ref['sha256']})
    if label == MIX:
        c = snapshots.ScheduledPostflopMixture(ref['path'], source)
        d = Path(ref['path']).parent
        c.dependency_hashes.update({str(p):digest(p) for p in [d/'plan.json', *(d.glob('iter-*/receipt.json'))]})
        return c
    return evaluator_policy(ref['path'], source)


def bind():
    cfg, fields, jobs = fixed_design()
    reg = read(OUTPUT/'registration.json')
    design_path = OUTPUT/'design.json'
    verify_files({design_path: reg['design_sha256'], CONFIG: reg['config_sha256'], panel.DOSE_PLAN: panel.DOSE_SHA})
    if read(design_path) != dict(config=cfg, fields=fields, jobs=jobs):
        raise ValueError('Registered assignment changed')
    stage_file = STUDY/'stage-ready.json'
    stage = read(stage_file)
    reference = read(STUDY/'evaluation-design-reference.json')
    if reference['evaluation_config_sha256'] != CONFIG_SHA or reference['before_evaluation'] is not True:
        raise ValueError('Training registration points to another evaluation')
    verify_files({STUDY/'plan.json': reference['training_plan_sha256'],
                  STUDY/'registration.json': reference['training_registration_sha256']})
    if (stage.get('training_complete') is not True or stage.get('iteration') != 504
            or stage.get('warm_start_weight_chain_verified') is not True
            or set(stage['seed_verification']) != {str(s) for s in SEEDS}):
        raise ValueError('Expected a completed 504 stage')
    prior = read(panel.DOSE_PLAN)
    models = {label:stage['policies'][label] for label in (BASE,MIX,END)}
    models[EXTRA] = prior['policies'][EXTRA]
    if models[BASE] != prior['policies']['steps0']:
        raise ValueError('Reference differs from the audited 480 specialist')
    source = prior['source']
    dependencies = {str(p):digest(p) for p in (CONFIG, SCRIPT, stage_file, panel.DOSE_PLAN,
        panel.RICH_STAGE, Path(panel.__file__), Path(snapshots.__file__), ROOT/'scripts/score_poker_population.py',
        ROOT/'scripts/poker_reference_opponents.py', STUDY/'evaluation-design-reference.json',
        STUDY/'plan.json', STUDY/'registration.json', OUTPUT/'registration.json', design_path)}
    for seed in SEEDS:
        sid = str(seed)
        spec = read(models[MIX][sid]['path'])
        final = read(models[END][sid]['path'])
        parent = read(models[BASE][sid]['path'])
        if (spec['seed'] != seed or [r['iteration'] for r in spec['postflop']] != [486,492,498,504]
                or spec['preflop']['sha256'] != parent['preflop']['sha256']
                or final['preflop']['sha256'] != spec['preflop']['sha256']
                or final['postflop']['sha256'] != spec['postflop'][-1]['sha256']):
            raise ValueError('Stage policies violate the frozen preflop or capture schedule')
        for label, refs in models.items():
            c = load_candidate(label, refs[sid], source)
            dependencies.update(baseline_eval.policy_dependency_hashes(c, source))
    verify_files(dependencies)
    plan_path = OUTPUT/'evaluation-plan.json'
    if plan_path.exists():
        raise FileExistsError(plan_path)
    atomic_json(plan_path, dict(created_utc=utc(), design_sha256=reg['design_sha256'],
                fields=fields, jobs=jobs, policies=models, seeds=list(SEEDS), source=source,
                frozen_files=dependencies, planned_hands=cfg['expected_hands'], stage_sha256=digest(stage_file)))
    (OUTPUT/'evaluation').mkdir()
    atomic_json(OUTPUT/'bound.json', dict(evaluation_plan_sha256=digest(plan_path), bound_utc=utc()))


def verify_bound():
    plan_path = OUTPUT/'evaluation-plan.json'
    verify_files({plan_path:read(OUTPUT/'bound.json')['evaluation_plan_sha256']})
    return read(plan_path)


@contextmanager
def ready_engine(opponent, lock_dir):
    if opponent is None:
        with closing(Bridge()) as engine:
            yield engine
        return
    # Loading the 352 MiB opponent occurs before the binary reads its first request.
    # Serialize this phase and time it separately from the normal 30s action deadline.
    with (lock_dir/'engine-startup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        engine = Bridge(opponent)
        try:
            engine.p.stdin.write('{"op":"reset","seed":7,"button":0}\n')
            engine.p.stdin.flush()
            if not select.select([engine.p.stdout], [], [], 180)[0]:
                engine.p.kill()
                raise TimeoutError('NoRegrets initialization did not complete')
            line = engine.p.stdout.readline()
            reply = json.loads(line)
            if not reply['ok'] or reply['state']['terminal']:
                raise RuntimeError('NoRegrets initialization reset failed')
        except BaseException:
            engine.close()
            raise
    with closing(engine):
        yield engine


def one_cell(job, plan, engine, strategies, opponents, opponent_deps):
    label, seed, field = (job[k] for k in ('label','seed','field'))
    spec = plan['fields'][field]
    start, deals = spec['deal_seed'], spec['deals']
    model = plan['policies'][label][str(seed)]
    candidate = load_candidate(label, model, plan['source'])
    deps = dict(opponent_deps)
    deps.update(baseline_eval.policy_dependency_hashes(candidate,plan['source']))
    deps[str(SCRIPT)] = plan['frozen_files'][str(SCRIPT)]
    verify_files(deps)
    out = OUTPUT/'evaluation'/job['id']
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    deadline = began+read(CONFIG)['cell_timeout_seconds']
    def progress(phase, **data):
        atomic_json(out/'status.json', dict(phase=phase, observed_utc=utc(), pid=os.getpid(),
                process_start=process_start(os.getpid()), elapsed_seconds=time.monotonic()-began, **data))
    prov = dict(label=label, training_seed=seed, field=field, deal_seed=start, deals=deals, hands=deals*6,
                candidate=str(candidate.model), candidate_sha256=model['sha256'], candidate_type=candidate.model_type,
                opponents=opponents, dependency_hashes=deps, big_blind_chips=100,
                statistical_unit='One original card block with six seats and three fixed training seeds',
                first_deal_clockwise_lineup=panel.lineup(field,start), varies_by_card=field=='varied',
                evaluation_plan_sha256=digest(OUTPUT/'evaluation-plan.json'))
    atomic_json(out/'provenance.json',prov)
    counts, choices, block_means = Counter(), Counter(), []
    raw = out/'hands.jsonl.gz'
    try:
        progress('evaluating', completed_card_blocks=0)
        with gzip.open(raw, 'xt', compresslevel=1) as stream:
            for deal, card in enumerate(range(start,start+deals)):
                table = panel.lineup(field,card)
                total = 0
                for hero in range(6):
                    row = panel.play(engine,candidate,strategies,table,card,hero,counts,deadline)
                    row['deal'] = deal
                    if label == MIX:
                        # play calls begin_hand once; all later decisions use that private choice.
                        idx = candidate.postflop.index(candidate.active)
                        row['snapshot_iteration'] = (486,492,498,504)[idx]
                        choices[row['snapshot_iteration']] += 1
                    total += row['chips']
                    stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                block_means.append(total/6)
                if (deal+1)%100 == 0:
                    progress('evaluating', completed_card_blocks=deal+1)
        stats = baseline_eval.interval(block_means)
        result = dict(prov, deal_blocks=block_means, bb_per_100=stats['mean'], ci95_half_width=stats['ci95_half_width'],
                      behavior=dict(counts), snapshot_hand_counts=dict(choices))
        progress('verifying_raw_hands', verified_hands=0)
        check = baseline_eval._verify_rows(result, raw, replay=True, deadline=deadline, progress=progress)
        verify_files(deps)
        result.update(verification=check, raw_hands_verified=True, terminal_payoffs_replayed=True, complete=True,
                      raw_hands_sha256=digest(raw), provenance_sha256=digest(out/'provenance.json'), completed_utc=utc())
        atomic_json(out/'results.json',result)
        progress('complete', verified_hands=deals*6, results_sha256=digest(out/'results.json'))
    except BaseException as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}',automatic_retry=False)
        raise


def worker(field, seed, label):
    plan = verify_bound()
    jobs = [j for j in plan['jobs'] if j['field']==field and j['seed']==seed and label in (j['label'],'all')]
    if not jobs or (label=='all') != (field!='public'):
        raise ValueError('Unscheduled field, policy, or seed')
    _, strategies, ids, op_model, deps = panel.frozen_inputs('steps0',seed,field)
    with ready_engine(op_model, OUTPUT) as engine:
        for job in jobs:
            one_cell(job,plan,engine,strategies,ids,deps)


def run():
    plan = verify_bound()
    verify_files(plan['frozen_files'])
    (OUTPUT/'launch.json').open('x').close()
    atomic_json(OUTPUT/'launch.json',dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid()),
                plan_sha256=digest(OUTPUT/'evaluation-plan.json')))
    workers = read(CONFIG)['workers']
    for public in (True,False):
        commands={}
        for job in plan['jobs']:
            field,seed = job['field'],job['seed']
            if (field=='public') != public:
                continue
            label = job['label'] if public else 'all'
            group = f'{field}-seed-{seed}-{label}'
            commands.setdefault(group,[str(PYTHON),str(SCRIPT),'_worker',field,str(seed),label])
        run_jobs(OUTPUT,commands,workers,'public' if public else 'panel')
    verify_files(plan['frozen_files'])
    summarize()


def nominal(blocks):
    delta = float(blocks.mean())
    width = NormalDist().inv_cdf(.975)*float(blocks.std(ddof=1))/math.sqrt(len(blocks))
    return dict(mean=delta,paired_ci95=[delta-width,delta+width])


def summarize():
    plan = verify_bound()
    verify_files(plan['frozen_files'])
    groups, rows, hashes, mix_counts = plan['fields'],{}, {}, {}
    expected_jobs = fixed_design()[2]
    if plan['jobs'] != expected_jobs:
        raise ValueError('Missing or unexpected scheduled cells')
    for job in expected_jobs:
        field,label,seed,name=(job[k] for k in ('field','label','seed','id'))
        spec=groups[field]
        out=OUTPUT/'evaluation'/name
        r=baseline_eval.verify_results(out, expected=dict(label=label,field=field,training_seed=seed,deal_seed=spec['deal_seed'],
                deals=spec['deals'],candidate_sha256=plan['policies'][label][str(seed)]['sha256'], terminal_payoffs_replayed=True))
        cells=rows.setdefault(field,{}).setdefault(label,{})
        cells[seed]=r['deal_blocks']
        hashes[name]=digest(out/'results.json')
        if label==MIX:
            choices = r['snapshot_hand_counts']
            if set(choices) != {'486','492','498','504'} or sum(choices.values())!=r['hands']:
                raise ValueError('Incomplete private snapshot selections')
            mix_counts[name]=choices
    def contrast(old, new):
        panel_rows={f:{k:rows[f][k] for k in (old,new)} for f in groups}
        return score_panel(panel_rows,groups,list(SEEDS),baseline=old,candidate=new)
    public={}
    for old,new in [(BASE,MIX),(BASE,END),(END,MIX),(EXTRA,MIX),(BASE,EXTRA)]:
        old_matrix=np.asarray([rows['public'][old][s] for s in SEEDS])
        new_matrix=np.asarray([rows['public'][new][s] for s in SEEDS])
        differences = new_matrix-old_matrix
        public[f'{new}_minus_{old}'] = dict(baseline=old,candidate=new,baseline_bb100=float(old_matrix.mean()),
                candidate_bb100=float(new_matrix.mean()),by_seed=differences.mean(axis=1).tolist(),
                **nominal(differences.mean(axis=0)))
    result=dict(complete=True,verified_utc=utc(),completed_cells=len(hashes),completed_hands=plan['planned_hands'],
                unique_original_card_blocks=read(CONFIG)['expected_unique_card_blocks'],
                primary=contrast(BASE,MIX),diagnostics=dict(current_vs_480=contrast(BASE,END),mixture_vs_current=contrast(END,MIX)),
                public_comparisons=public,cell_result_sha256=hashes,snapshot_hand_counts=mix_counts,
                plan_sha256=digest(OUTPUT/'evaluation-plan.json'),model_promoted=False)
    atomic_json(OUTPUT/'summary.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','bind','run','summarize','_worker'))
    parser.add_argument('arguments',nargs='*')
    args=parser.parse_args()
    if args.action=='_worker':
        field,seed,label=args.arguments
        worker(field,int(seed),label)
    else:
        if args.arguments:
            parser.error('Unexpected arguments for this phase')
        {'register':register,'bind':bind,'run':run,'summarize':summarize}[args.action]()
