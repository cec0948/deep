"""Masked external evaluation on the registered diverse-opponent and standard training arms."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np

import evaluate_average_baseline as evaluation
import evaluate_poker_diverse_panel as panel
import evaluate_poker_postflop_snapshots as previous
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import SOURCE as PUBLIC_SOURCE
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges
from run_poker_preflop_confirmation import run_jobs
from score_poker_population import score_panel


CONFIG = ROOT/'configs/poker-opponent-diversity-evaluation-20261123.json'
CONFIG_SHA = '61cacc2449818c98444e1c9ed5dc140fb4cc88610576df9d2a96441dee6e4dc6'
STUDY = ROOT/'artifacts/preflop-followup/480-v1/opponent-diversity-20261123'
OUTPUT = STUDY/'population'
REPORT = ROOT/'reports/preflop-followup/480-v1/opponent-diversity-20261123'
SCRIPT = Path(__file__).resolve()
PARENT = STUDY.parent/'postflop-averaging-20261120'
BASE, CONTROL, POOL = 'specialist480', 'control504', 'diverse504'
NOVEL = 'solverlike'
SNAPSHOTS = (486, 492, 498, 504)


def design():
    verify_files({CONFIG: CONFIG_SHA})
    cfg = read(CONFIG)
    fields = {name: dict(role=spec['role'], family=spec['family'], **cfg['ranges'][spec['cards']])
              for name, spec in cfg['fields'].items()}
    jobs = [dict(id=f'{field}-seed-{seed}-{label}', field=field, seed=seed, label=label)
            for field in fields for seed in cfg['seeds'] for label in cfg['labels']]
    unique = sum(n for _, n in {(spec['deal_seed'], spec['deals']) for spec in fields.values()})
    if (len(jobs), sum(fields[j['field']]['deals']*6 for j in jobs), unique) != (
            cfg['cells'], cfg['hands'], cfg['unique_original_card_blocks']):
        raise ValueError('Registered evaluation counts disagree')
    return dict(config=cfg, fields=fields, jobs=jobs)


def load(label, ref, source):
    return previous.load_candidate(previous.BASE if label == BASE else previous.MIX, ref, source)


def table(field, card):
    return (NOVEL,)*5 if field == NOVEL else panel.lineup(field, card)


def inputs(seed, field):
    if field == NOVEL:
        from poker_solverlike_opponent import SolverLikeOpponent
        bot = SolverLikeOpponent()
        if bot.digest != read(CONFIG)['upstream_solverlike_sha256']:
            raise ValueError('Unverified external SolverLike benchmark')
        return {NOVEL: bot}, {NOVEL:bot.digest}, None, dict(bot.dependency_hashes)
    _, bots, ids, blueprint, deps = panel.frozen_inputs('steps0', seed, field)
    return bots, ids, blueprint, deps


def register():
    info = design()
    cfg = info['config']
    # Stage the design and parent bindings before opening a single evaluation card.
    prior_plan = PARENT/'population/evaluation-plan.json'
    prior_stage = PARENT/'stage-ready.json'
    solver = PUBLIC_SOURCE/'bots/adversarial/solver_like.py'
    verify_files({prior_plan:cfg['previous_evaluation_plan_sha256'], prior_stage:cfg['previous_stage_sha256'],
                  solver:cfg['upstream_solverlike_sha256']})
    prior = read(prior_plan)
    verify_files(prior['frozen_files'])
    root_reg = read(STUDY/'registration.json')
    verify_files({ROOT/'configs/poker-opponent-diversity-20261123.json':root_reg['config_sha256']})
    claimed = {tuple(pair) for pair in root_reg['fresh_evaluation_cards'].values()}
    if claimed != {(item['deal_seed'], item['deals']) for item in cfg['ranges'].values()}:
        raise ValueError('Fresh cards differ from the registered training and evaluation design')
    previous_ranges = {k:v['range'] for k,v in prior_card_ranges(STUDY).items()}
    for name, item in cfg['ranges'].items():
        start, n = item['deal_seed'], item['deals']
        check_card_range(start, n, previous_ranges)
        previous_ranges[name] = [start, start+n]
    if [s for s in info['fields'] if s != NOVEL] != list(prior['fields']):
        raise ValueError('Existing table set changed')
    for field, spec in prior['fields'].items():
        if any(info['fields'][field][k] != spec[k] for k in ('family','role')):
            raise ValueError('Registered family weights or roles changed')
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT/'evaluation').mkdir()
    atomic_json(OUTPUT/'design.json', info)
    reg = dict(registered_utc=utc(), config_sha256=CONFIG_SHA, design_sha256=digest(OUTPUT/'design.json'),
               previous_and_new_card_ranges=sorted(previous_ranges.values()),
               solverlike_sha256=cfg['upstream_solverlike_sha256'])
    atomic_json(OUTPUT/'registration.json',reg)
    REPORT.mkdir(parents=True,exist_ok=True)
    atomic_json(REPORT/'evaluation-registration.json',reg)
    deps = {str(p):digest(p) for p in (CONFIG, SCRIPT, prior_plan, prior_stage, STUDY/'registration.json', OUTPUT/'design.json',
                                      OUTPUT/'registration.json', Path(panel.__file__), Path(previous.__file__),
                                      Path(evaluation.__file__), ROOT/'scripts/score_poker_population.py', solver)}
    models = {BASE:prior['policies'][previous.BASE], CONTROL:prior['policies'][previous.MIX]}
    for label, seed_refs in models.items():
        for ref in seed_refs.values():
            c = load(label,ref,prior['source'])
            deps.update(evaluation.policy_dependency_hashes(c,prior['source']))
    verify_files(deps)
    be = [j for j in info['jobs'] if j['id'] in cfg['early_reference_jobs']]
    if len(be) != len(cfg['early_reference_jobs']) or any(j['label']==POOL for j in be):
        raise ValueError('Early reference assignment changed')
    pre = dict(created_utc=utc(),design_sha256=reg['design_sha256'],fields=info['fields'],jobs=be,
               policies=models,seeds=cfg['seeds'],source=prior['source'],frozen_files=deps)
    atomic_json(OUTPUT/'reference-plan.json',pre)
    atomic_json(OUTPUT/'reference-bound.json',dict(plan_sha256=digest(OUTPUT/'reference-plan.json')))


def get_plan(early=False):
    stem = 'reference' if early else 'evaluation'
    plan_file = OUTPUT/(stem+'-plan.json')
    verify_files({plan_file:read(OUTPUT/(stem+'-bound.json'))['plan_sha256']})
    return read(plan_file),digest(plan_file)


def bind():
    info = design()
    pre, prehash = get_plan(True)
    verify_files(pre['frozen_files'])
    stage_path = STUDY/'stage-ready.json'
    stage = read(stage_path)
    if not stage['training_complete'] or stage['iteration'] != 504:
        raise ValueError('Treatment training is incomplete')
    policies = dict(pre['policies'], **{POOL:stage['policies']['poolMixture504']})
    if stage['policies'][BASE] != policies[BASE]:
        raise ValueError('Treatment has a different 480 reference')
    dependencies = dict(pre['frozen_files'])
    dependencies.update({str(p):digest(p) for p in [stage_path,OUTPUT/'reference-plan.json',
                        ROOT/'scripts/poker_solverlike_opponent.py']})
    for s in info['config']['seeds']:
        sid = str(s)
        ref = policies[POOL][sid]
        c = load(POOL,ref,pre['source'])
        ctrl = read(policies[CONTROL][sid]['path'])
        cand = read(ref['path'])
        if (ctrl['preflop']['sha256'] != cand['preflop']['sha256']
                or [r['iteration'] for r in cand['postflop']] != list(SNAPSHOTS)
                or [r['iteration'] for r in ctrl['postflop']] != list(SNAPSHOTS)
                or cand['seed'] != s or ctrl['seed'] != s):
            raise ValueError('Treatment differs in the nominal policy identity or mixture schedule')
        dependencies.update(evaluation.policy_dependency_hashes(c,pre['source']))
    verify_files(dependencies)
    plan = dict(created_utc=utc(), reference_plan_sha256=prehash, design_sha256=pre['design_sha256'],
                fields=info['fields'],jobs=info['jobs'],policies=policies,source=pre['source'],
                seeds=pre['seeds'],frozen_files=dependencies,stage_sha256=digest(stage_path))
    target = OUTPUT/'evaluation-plan.json'
    if target.exists():
        raise FileExistsError(target)
    atomic_json(target,plan)
    atomic_json(OUTPUT/'evaluation-bound.json',dict(plan_sha256=digest(target)))


def one_cell(job, plan, plan_hash, engine, bots, opponent_ids, opponent_deps):
    label,seed,field=(job[k] for k in ('label','seed','field'))
    spec=plan['fields'][field]
    start, n = spec['deal_seed'],spec['deals']
    ref=plan['policies'][label][str(seed)]
    candidate=load(label,ref,plan['source'])
    deps=dict(opponent_deps)
    deps.update(evaluation.policy_dependency_hashes(candidate,plan['source']))
    deps[str(SCRIPT)] = plan['frozen_files'][str(SCRIPT)]
    verify_files(deps)
    out=OUTPUT/'evaluation'/job['id']
    out.mkdir()
    timeout=time.monotonic()+read(CONFIG)['cell_timeout_seconds']
    def progress(phase, **values):
        atomic_json(out/'status.json',dict(phase=phase,observed_utc=utc(),pid=os.getpid(),
                    process_start=process_start(os.getpid()), **values))
    p=dict(label=label,training_seed=seed,field=field,deal_seed=start,deals=n,hands=6*n,
           candidate=str(candidate.model),candidate_sha256=ref['sha256'],candidate_type=candidate.model_type,
           opponents=opponent_ids,dependency_hashes=deps,big_blind_chips=100,
           first_deal_clockwise_lineup=table(field,start),varies_by_card=field=='varied',
           design_sha256=plan['design_sha256'],evaluation_plan_sha256=plan_hash,
           statistical_unit='Original card block; six hero seats and three fixed model seeds are repeats')
    atomic_json(out/'provenance.json',p)
    sums=[]
    counts, snapshots=Counter(),Counter()
    raw=out/'hands.jsonl.gz'
    try:
        progress('evaluating',completed_card_blocks=0)
        with gzip.open(raw,'xt',compresslevel=1) as stream:
            for i,card in enumerate(range(start,start+n)):
                seats=table(field,card)
                total=0
                for seat in range(6):
                    row=panel.play(engine,candidate,bots,seats,card,seat,counts,timeout)
                    row['deal']=i
                    if label != BASE:
                        slot=SNAPSHOTS[candidate.postflop.index(candidate.active)]
                        row['snapshot_iteration']=slot
                        snapshots[slot]+=1
                    stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    total+=row['chips']
                sums.append(total/6)
                if (i+1)%100 == 0:
                    progress('evaluating',completed_card_blocks=i+1)
        est=evaluation.interval(sums)
        result=dict(p,deal_blocks=sums,bb_per_100=est['mean'],ci95_half_width=est['ci95_half_width'],
                    behavior=dict(counts),snapshot_hand_counts=dict(snapshots))
        progress('verifying_raw_hands',verified_hands=0)
        verification=evaluation._verify_rows(result,raw,replay=True,deadline=timeout,progress=progress)
        verify_files(deps)
        result.update(complete=True,raw_hands_verified=True,terminal_payoffs_replayed=True,verification=verification,
                      raw_hands_sha256=digest(raw),provenance_sha256=digest(out/'provenance.json'),completed_utc=utc())
        atomic_json(out/'results.json',result)
        progress('complete',verified_hands=n*6,results_sha256=digest(out/'results.json'))
    except BaseException as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}',automatic_retry=False)
        raise


def worker(early,field,seed,label):
    plan,ident=get_plan(early)
    already=set(read(CONFIG)['early_reference_jobs'])
    jobs=[j for j in plan['jobs'] if j['field']==field and j['seed']==seed
          and label in ('all',j['label']) and (j['id'] in already)==early]
    if not jobs or (label=='all') != (field!='public'):
        raise ValueError('Unexpected assignment')
    bots,ids,blueprint,deps=inputs(seed,field)
    with previous.ready_engine(blueprint,OUTPUT) as engine:
        for job in jobs:
            one_cell(job,plan,ident,engine,bots,ids,deps)


def run(early):
    cfg=read(CONFIG)
    plan,ident=get_plan(early)
    verify_files(plan['frozen_files'])
    phase='early-reference' if early else 'full'
    marker=OUTPUT/(phase+'-launch.json')
    with marker.open('x'): pass
    atomic_json(marker,dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid()),plan_sha256=ident))
    if early:
        os.nice(10)
    reserved=set(cfg['early_reference_jobs'])
    for public in ([True] if early else [True,False]):
        commands={}
        for job in plan['jobs']:
            f,s,label=(job[k] for k in ('field','seed','label'))
            if (f=='public')!=public or (job['id'] in reserved)!=early:
                continue
            group_label=label if public else 'all'
            key=f'{f}-seed-{s}-{group_label}'
            commands.setdefault(key,[str(PYTHON),str(SCRIPT),'_early' if early else '_worker',f,str(s),group_label])
        workers=1 if early else cfg['workers']
        if not early and read(OUTPUT/'early-public-status.json')['state']=='running':
            workers-=1
        run_jobs(OUTPUT,commands,workers,'early-public' if early else 'public' if public else 'panel')
    verify_files(plan['frozen_files'])
    if not early:
        summarize()


def summarize():
    cfg=read(CONFIG)
    early,h1=get_plan(True)
    plan,h2=get_plan()
    registered=design()
    if plan['jobs']!=registered['jobs'] or plan['fields']!=registered['fields']:
        raise ValueError('Plan differs from registered design')
    verify_files(plan['frozen_files'])
    early_ids={j['id'] for j in early['jobs']}
    rows,hashes,snapshot_counts={}, {}, {}
    for j in plan['jobs']:
        f,s,label,k=(j[key] for key in ('field','seed','label','id'))
        spec=plan['fields'][f]
        c=OUTPUT/'evaluation'/k
        r=evaluation.verify_results(c,expected=dict(label=label,training_seed=s,field=f,
            deal_seed=spec['deal_seed'],deals=spec['deals'],candidate_sha256=plan['policies'][label][str(s)]['sha256'],
            terminal_payoffs_replayed=True,evaluation_plan_sha256=h1 if k in early_ids else h2))
        rows.setdefault(f,{}).setdefault(label,{})[s]=r['deal_blocks']
        hashes[k]=digest(c/'results.json')
        if label!=BASE:
            choices=r['snapshot_hand_counts']
            if set(choices)!={str(slot) for slot in SNAPSHOTS} or sum(choices.values())!=r['hands']:
                raise ValueError('Incomplete private snapshot selections')
            other=snapshot_counts.setdefault(f'{f}/{s}',choices)
            if other!=choices:
                raise ValueError('Treatment and control did not draw the same hand-private snapshots')
    def compare(old,new):
        return score_panel({f:{l:rows[f][l] for l in (old,new)} for f in rows},plan['fields'],plan['seeds'],
                           baseline=old,candidate=new)
    public={}
    for old,new in ((CONTROL,POOL),(BASE,POOL),(BASE,CONTROL)):
        a,b=(np.asarray([rows['public'][label][s] for s in plan['seeds']]) for label in (old,new))
        public[f'{new}_minus_{old}']=dict(baseline_bb100=float(a.mean()),candidate_bb100=float(b.mean()),
                by_seed=(b-a).mean(axis=1).tolist(),**previous.nominal((b-a).mean(axis=0)))
    result=dict(complete=True,verified_utc=utc(),completed_cells=len(hashes),
        completed_hands=sum(plan['fields'][j['field']]['deals']*6 for j in plan['jobs']),
        unique_original_card_blocks=cfg['unique_original_card_blocks'],primary=compare(CONTROL,POOL),
        treatment_vs_parent=compare(BASE,POOL),control_vs_parent=compare(BASE,CONTROL),public_comparisons=public,
        cell_result_sha256=hashes,snapshot_hand_counts=snapshot_counts,plan_sha256=h2,reference_plan_sha256=h1,
        model_promoted=False)
    atomic_json(OUTPUT/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','early','bind','run','summarize','_worker','_early'))
    parser.add_argument('args',nargs='*')
    cli=parser.parse_args()
    if cli.mode.startswith('_'):
        field,seed,label=cli.args
        worker(cli.mode=='_early',field,int(seed),label)
    else:
        if cli.args:
            parser.error('Unexpected arguments')
        {'register':register,'early':lambda:run(True),'bind':bind,'run':lambda:run(False),'summarize':summarize}[cli.mode]()
