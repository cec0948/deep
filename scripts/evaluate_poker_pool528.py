"""Matched six-seat external evaluation of current, hand-mixture and reference-average policies."""
import argparse
from collections import Counter
import gzip
import itertools
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

import evaluate_average_baseline as verification
import evaluate_poker_opponent_diversity as prior
from controlled_poker_refit import read, utc, verify_files
from evaluate_poker_postflop_snapshots import ready_engine
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_average_runtime import WindowAveragePolicy
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges
from run_poker_preflop_confirmation import run_jobs


STUDY = prior.STUDY.parent / 'pool-extension-528-20261126'
OUTPUT = STUDY / 'main'
CONFIG = ROOT / 'configs/poker-pool-extension-528-20261126.json'
SCRIPT = Path(__file__).resolve()
SEEDS = (342, 343, 344)
TABLES = {'public': (2060100000, 4000), 'mixed': (2060200000, 1000), 'noregrets': (2060200000, 1000)}
DEPLOYED = ('current480', 'mixture504', 'mixture528', 'current504', 'current528')
AVERAGES = ('average480', 'average504', 'average528')
EARLY = ('current480','mixture504','current504','average480','average504')
SLOTS = {'mixture504': (486,492,498,504), 'mixture528': (510,516,522,528)}
SOURCE_STAGE = '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9'
SOURCE_PLAN = 'a851844aee89e4032992840e40b2bf41ac765cc132dd5be9f94fee09e9540b43'


def jobs():
    return [dict(id=f'{field}-seed-{seed}-{label}', field=field, seed=seed, label=label)
            for field in TABLES for seed in SEEDS for label in DEPLOYED + (AVERAGES if field == 'public' else ())]


def is_early(job):
    return job['field'] == 'public' and job['seed'] == 342 and job['label'] in EARLY


def candidate(label, ref, source, seed):
    verify_files({ref['path']: ref['sha256']})
    if label in AVERAGES:
        model = WindowAveragePolicy(ref['path'],source)
        if (model.metadata['averaging_window'] != [361, int(label[-3:])] or model.metadata['seed'] != seed or
                model.metadata['historical_mixture_available'] is not False):
            raise ValueError('Incorrect auxiliary policy or collection window')
        return model
    model = prior.previous.load_candidate(prior.previous.MIX if label in SLOTS else prior.previous.BASE, ref, source)
    if label in SLOTS:
        spec = read(ref['path'])
        if spec['seed'] != seed or tuple(entry['iteration'] for entry in spec['postflop']) != SLOTS[label]:
            raise ValueError('Incorrect private snapshot schedule')
    return model


def average_ref(folder, stage):
    pointer = read(folder/'current.json')
    if pointer['iteration'] != stage:
        raise ValueError('Incorrect native average checkpoint')
    sha = pointer['files']['average.npz']
    path = folder/'checkpoints'/pointer['directory']/'average.npz'
    verify_files({path: sha})
    return dict(path=str(path),sha256=sha)


def register():
    prior_plan_path, stage_path = prior.OUTPUT/'evaluation-plan.json', prior.STUDY/'stage-ready.json'
    verify_files({prior_plan_path:SOURCE_PLAN,stage_path:SOURCE_STAGE})
    registration = STUDY/'registration.json'
    reg = read(registration)
    verify_files({CONFIG:reg['config_sha256']})
    cfg = read(CONFIG)
    if (cfg['cards']['public'] != list(TABLES['public']) or cfg['cards']['matched_panel'] != list(TABLES['mixed'])
            or cfg['external']['cells'] != 54 or cfg['external']['hands'] != 756000 or cfg['seeds'] != list(SEEDS)):
        raise ValueError('Main policy/card design differs from root registration')
    prior_plan, stage = read(prior_plan_path), read(stage_path)
    train = read(prior.STUDY/'training-plan.json')
    verify_files(prior_plan['frozen_files'])
    expected = ((2060100000,4000), (2060200000,1000))
    used = {key:row['range'] for key,row in prior_card_ranges(STUDY).items()}
    for start,n in expected:
        check_card_range(start,n,used)
        used[f'new-{start}'] = [start,start+n]
    policies = {name:stage['policies'][key] for name,key in
                (('current480','specialist480'),('current504','pool504'),('mixture504','poolMixture504'))}
    policies['average480'] = {str(s):average_ref(Path(train['campaign'])/f'seed-{s}',480) for s in SEEDS}
    policies['average504'] = {str(s):average_ref(prior.STUDY/'pool504'/f'seed-{s}',504) for s in SEEDS}
    source = prior_plan['source']
    frozen = {str(p):digest(p) for p in (SCRIPT, CONFIG, registration, prior_plan_path, stage_path,
                  prior.STUDY/'training-plan.json', Path(prior.__file__), Path(prior.panel.__file__),
                  Path(prior.previous.__file__),Path(verification.__file__), SCRIPT.with_name('run_poker_preflop_confirmation.py'))}
    opponents = {}
    for seed in SEEDS:
        for label, refs in policies.items():
            model = candidate(label,refs[str(seed)],source,seed)
            frozen.update(verification.policy_dependency_hashes(model,source))
        for field in TABLES:
            _, ids, _, deps = prior.inputs(seed,field)
            frozen.update(deps)
            opponents[f'{field}/{seed}'] = ids
    verify_files(frozen)
    schedule = jobs()
    hands = sum(TABLES[j['field']][1]*6 for j in schedule)
    if (len(schedule),hands,len([j for j in schedule if is_early(j)])) != (54,756000,5):
        raise ValueError('Wrong registered geometry')
    plan = dict(registered_utc=utc(),seeds=SEEDS,fields=TABLES,jobs=schedule,source=source,policies=policies,
                opponents=opponents,frozen_files=frozen,source_stage_sha256=SOURCE_STAGE,
                prior_ranges_checked=len(used)-2,planned_hands=hands,unique_original_cards=5000,
                primary='Public mixture528 minus mixture504 nominal two-sided paired 95% interval',
                secondary='All available matched policies by field, including current, native all-street reference averages on public, and comparisons to the fixed specialist480.',
                missing_480_mix='The parent campaign preserved 420/480 endpoints; there is no pre-480 four-slot capture under this specialist.',
                early_ids=[j['id'] for j in schedule if is_early(j)])
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/'evaluation').mkdir()
    atomic_json(OUTPUT/'reference-plan.json',plan)
    atomic_json(OUTPUT/'reference-bound.json',dict(sha256=digest(OUTPUT/'reference-plan.json')))


def load_plan(phase):
    stem = 'reference' if phase=='early' else 'evaluation'
    file = OUTPUT/(stem+'-plan.json')
    sha = read(OUTPUT/(stem+'-bound.json'))['sha256']
    verify_files({file:sha})
    return read(file), sha


def bind():
    plan, earlier = load_plan('early')
    verify_files(plan['frozen_files'])
    stage_path = STUDY/'stage-ready.json'
    stage = read(stage_path)
    if not stage['training_complete'] or stage['iteration'] != 528:
        raise ValueError('528 training stage is incomplete')
    for label in ('current528','mixture528','average528'):
        refs = stage['policies'][label]
        if set(refs) != set(map(str,SEEDS)):
            raise ValueError(f'Missing seed for {label}')
        for seed in SEEDS:
            ref = refs[str(seed)]
            model = candidate(label,ref,plan['source'],seed)
            plan['frozen_files'].update(verification.policy_dependency_hashes(model,plan['source']))
        plan['policies'][label] = refs
    for seed in SEEDS:
        frozen = read(plan['policies']['current480'][str(seed)]['path'])['preflop']['sha256']
        for label in ('current504','current528','mixture504','mixture528'):
            if read(plan['policies'][label][str(seed)]['path'])['preflop']['sha256'] != frozen:
                raise ValueError(f'{label} did not preserve the specialist preflop')
    plan['reference_plan_sha256'] = earlier
    plan['training_stage_sha256'] = digest(stage_path)
    plan['frozen_files'][str(stage_path)] = plan['training_stage_sha256']
    verify_files(plan['frozen_files'])
    with (OUTPUT/'binding.lock').open('x'):
        atomic_json(OUTPUT/'evaluation-plan.json',plan)
        atomic_json(OUTPUT/'evaluation-bound.json',dict(sha256=digest(OUTPUT/'evaluation-plan.json')))


def run_cell(job,plan,sha,engine,bots,ids,opponent_deps):
    field,seed,label = (job[k] for k in ('field','seed','label'))
    folder = OUTPUT/'evaluation'/job['id']
    folder.mkdir(exist_ok=False)
    def progress(phase,**details):
        atomic_json(folder/'status.json',dict(phase=phase,pid=os.getpid(),process_start=process_start(os.getpid()),
                                               observed_utc=utc(),**details))
    try:
        ref = plan['policies'][label][str(seed)]
        model = candidate(label,ref,plan['source'],seed)
        deps = dict(opponent_deps)
        deps.update(verification.policy_dependency_hashes(model,plan['source']))
        deps[str(SCRIPT)] = digest(SCRIPT)
        if (ids!=plan['opponents'][f'{field}/{seed}'] or
                any(plan['frozen_files'].get(k)!=v for k,v in deps.items())):
            raise ValueError('Runtime dependency differs from frozen plan')
        verify_files(deps)
        start,count = plan['fields'][field]
        provenance = dict(label=label,field=field,training_seed=seed,deal_seed=start,deals=count,hands=6*count,
                          big_blind_chips=100,candidate=str(model.model),candidate_type=model.model_type,
                          candidate_sha256=ref['sha256'],opponents=ids,dependency_hashes=deps,evaluation_plan_sha256=sha)
        atomic_json(folder/'provenance.json',provenance)
        deadline=time.monotonic()+7200
        progress('evaluating',completed_card_blocks=0)
        raw=folder/'hands.jsonl.gz'
        counts,choices,blocks=Counter(),Counter(),[]
        with gzip.open(raw,'xt',compresslevel=1) as stream:
            for i,card in enumerate(range(start,start+count)):
                payoff=0
                for seat in range(6):
                    row=prior.panel.play(engine,model,bots,prior.table(field,card),card,seat,counts,deadline)
                    row['deal']=i
                    if label in SLOTS:
                        slot = model.postflop.index(model.active)
                        row['snapshot_iteration']=SLOTS[label][slot]
                        choices[slot]+=1
                    stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    payoff+=row['chips']
                blocks.append(payoff/6)
                if (i+1)%100==0:
                    progress('evaluating',completed_card_blocks=i+1)
        estimate=verification.interval(blocks)
        result=dict(provenance,deal_blocks=blocks,bb_per_100=estimate['mean'],ci95_half_width=estimate['ci95_half_width'],
                    snapshot_slot_counts=dict(choices),behavior=dict(counts))
        progress('replaying_terminals')
        validation=verification._verify_rows(result,raw,replay=True,deadline=deadline,progress=progress)
        verify_files(deps)
        result.update(complete=True,raw_hands_verified=True,terminal_payoffs_replayed=True,verification=validation,
                      raw_hands_sha256=digest(raw),provenance_sha256=digest(folder/'provenance.json'),completed_utc=utc())
        atomic_json(folder/'results.json',result)
        progress('complete',verified_hands=6*count)
    except BaseException as error:
        progress('failed',error=f'{type(error).__name__}: {error}',automatic_retry=False)
        raise


def worker(phase, field, seed, group):
    plan,sha=load_plan(phase)
    jobs_here=[j for j in plan['jobs'] if j['field']==field and j['seed']==seed and
               (group=='all' or group==j['label']) and is_early(j)==(phase=='early')]
    if not jobs_here:
        raise ValueError('Unscheduled evaluation batch')
    bots,ids,blueprint,deps=prior.inputs(seed,field)
    with ready_engine(blueprint,OUTPUT) as engine:
        for job in jobs_here:
            run_cell(job,plan,sha,engine,bots,ids,deps)


def run(phase, workers):
    if not 1<=workers<=4 or (phase=='early' and workers!=1):
        raise ValueError('Invalid CPU budget')
    plan,sha=load_plan(phase)
    verify_files(plan['frozen_files'])
    marker=OUTPUT/(phase+'-launch.json')
    with marker.open('x') as f:
        json.dump(dict(utc=utc(),pid=os.getpid(),start_token=process_start(os.getpid()),plan_sha256=sha),f)
    commands={}
    for job in plan['jobs']:
        if is_early(job)!=(phase=='early'):
            continue
        field,seed,label=(job[k] for k in ('field','seed','label'))
        group=label if field=='public' else 'all'
        name=f'{field}-seed-{seed}-{group}'
        commands[name]=[str(PYTHON),str(SCRIPT),'worker',phase,field,str(seed),group]
    run_jobs(OUTPUT,commands,workers,phase)


def interval(blocks, z):
    n=len(blocks)
    if n<2 or not np.all(np.isfinite(blocks)):
        raise ValueError('Incomplete or nonfinite card-level returns')
    m=float(np.mean(blocks))
    h=float(z*np.std(blocks,ddof=1)/math.sqrt(n))
    return dict(mean=m,ci95=[m-h,m+h])


def summarize():
    plan,sha=load_plan('full')
    _,pre=load_plan('early')
    verify_files(plan['frozen_files'])
    data,hashes,slots={},{},{}
    for job in plan['jobs']:
        field, seed, label, name=(job[k] for k in ('field','seed','label','id'))
        start,n=plan['fields'][field]
        folder=OUTPUT/'evaluation'/name
        expected=dict(field=field,training_seed=seed,label=label,deal_seed=start,deals=n,hands=6*n,
                      candidate_sha256=plan['policies'][label][str(seed)]['sha256'],terminal_payoffs_replayed=True,
                      evaluation_plan_sha256=pre if is_early(job) else sha)
        result=verification.verify_results(folder,expected=expected)
        if result['opponents']!=plan['opponents'][f'{field}/{seed}']:
            raise ValueError('Wrong opponent identities')
        if label in SLOTS:
            counts=result['snapshot_slot_counts']
            if (set(counts)!=set('0123') or sum(counts.values())!=6*n or
                    slots.setdefault((field,seed),counts)!=counts):
                raise ValueError('Unmatched hand-private strategy draws')
        hashes[name]=digest(folder/'results.json')
        data.setdefault(field,{}).setdefault(label,{})[seed]=result['deal_blocks']
    scores,comparisons={},{}
    z=NormalDist().inv_cdf(.975)
    total_pairs=sum(math.comb(len(rows),2) for rows in data.values())
    family_z=NormalDist().inv_cdf(1-.05/(2*total_pairs))
    for field,policies in data.items():
        matrices={label:np.array([seeds[s] for s in SEEDS],dtype=float) for label,seeds in policies.items()}
        if any(x.shape!=(len(SEEDS),TABLES[field][1]) for x in matrices.values()):
            raise ValueError('Incomplete card matrix')
        scores[field]={label:interval(x.mean(axis=0),z) for label,x in matrices.items()}
        comparisons[field]={}
        for base,new in itertools.combinations(policies,2):
            change=matrices[new]-matrices[base]
            cards=change.mean(axis=0)
            comparisons[field][f'{new}_minus_{base}']=dict(
                baseline=base,candidate=new,by_seed={str(s):float(change[i].mean()) for i,s in enumerate(SEEDS)},
                **interval(cards,z),simultaneous_ci95=interval(cards,family_z)['ci95'])
    # Store all contrasts in chronological/functional order (e.g. mixture528 minus mixture504).
    total_hands=sum(TABLES[j['field']][1]*6 for j in plan['jobs'])
    result=dict(complete=True,verified_utc=utc(),seeds=SEEDS,fields=TABLES,completed_cells=len(hashes),
                completed_hands=total_hands,unique_original_card_blocks=5000,absolute_bb100=scores,
                paired_comparisons=comparisons,primary=comparisons['public']['mixture528_minus_mixture504'],
                primary_comparison='public mixture528 minus mixture504',secondary_pair_family=total_pairs,
                statistical_unit='Original card block after averaging six hero seats and three fixed policy seeds.',
                estimates='Nominal paired 95% intervals; secondary simultaneous intervals apply across all field/policy-pair contrasts.',
                cell_result_sha256=hashes,reference_plan_sha256=pre,evaluation_plan_sha256=sha,
                source_stage_sha256=plan['source_stage_sha256'],training_stage_sha256=plan['training_stage_sha256'],
                unavailable_480_mix=plan['missing_480_mix'],model_promoted=False)
    if (len(hashes),total_hands)!=(54,756000):
        raise ValueError('Incomplete study')
    atomic_json(OUTPUT/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','bind','early','full','worker','summarize'))
    parser.add_argument('args',nargs='*')
    cli=parser.parse_args()
    if cli.mode=='worker':
        phase,field,seed,group=cli.args
        worker(phase,field,int(seed),group)
    elif cli.mode in ('early','full'):
        run(cli.mode,int(cli.args[0]))
    else:
        if cli.args:
            parser.error('Unexpected arguments')
        {'register':register,'bind':bind,'summarize':summarize}[cli.mode]()
