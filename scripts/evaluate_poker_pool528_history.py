"""Test fixed postflop history opponents with the preflop they had during pool training."""
import argparse
from contextlib import closing
from copy import copy
import gzip
import json
from pathlib import Path

import numpy as np

import evaluate_poker_pool528_crossplay as cross
from audit_postflop_averaging import verify_terminals
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import atomic_json, digest, PYTHON
from poker_preflop_specialist import evaluator_policy, write_policy
from run_poker_preflop_confirmation import run_jobs


OUTPUT = cross.STUDY / 'training-opponents'
PUBLIC = cross.PUBLIC.parent / 'training-opponents'
SCRIPT = Path(__file__).resolve()
DESIGN = 'd88c206f2837af68a4a344010dce08930e3c31a8adee17a78c0facb196f07571'


def bind():
    verify_files({OUTPUT/'design.json': DESIGN, PUBLIC/'design.json': DESIGN})
    plan, key = cross.plan()
    design = read(OUTPUT/'design.json')
    history = read(cross.ORIGINAL/'training-plan.json')['historical_policies']
    if design['cards'] != [cross.START, cross.DEALS] or design['hero'] != list(cross.LABELS):
        raise ValueError('Different registered card or policy design')
    dependencies = dict(plan['frozen_files'])
    dependencies[str(SCRIPT)] = digest(SCRIPT)
    refs = {}
    for seed in cross.SEEDS:
        sid = str(seed)
        base = read(plan['policies'][cross.LABELS[0]][sid]['path'])
        refs[sid] = {}
        for stage in design['opponents']:
            ref = history[sid][str(stage)]
            verify_files({ref['path']:design['archived_opponents_sha256'][sid][str(stage)],
                          base['preflop']['path']:base['preflop']['sha256']})
            folder = OUTPUT/'policies'/sid
            folder.mkdir(parents=True, exist_ok=True)
            path = write_policy(folder/f'past-{stage}.json',base['preflop']['path'],ref['path'])
            refs[sid][str(stage)] = str(path)
            bot = evaluator_policy(path,plan['source'])
            dependencies.update(bot.dependency_hashes)
            dependencies[str(path)] = digest(path)
    verify_files(dependencies)
    payload = dict(bound_utc=utc(),cross_plan_sha256=key,study_sha256=DESIGN,opponents=refs,
                   frozen_files=dependencies)
    atomic_json(OUTPUT/'plan.json',payload)
    atomic_json(OUTPUT/'bound.json',dict(sha256=digest(OUTPUT/'plan.json')))


def verify_cell(folder, players, labels, engine, expected, key):
    rows, decisions, sampled = 0, 0, 0
    totals = np.zeros(cross.DEALS, dtype=np.int64)
    raw = folder/'hands.jsonl.gz'
    with gzip.open(raw,'rt') as source:
        for line in source:
            row = json.loads(line)
            index, seat = divmod(rows,6)
            card = cross.START + index
            policy_labels = cross.seated(labels,seat)
            slots = [cross.private_slot(card,s) if label in cross.SNAPSHOTS else None
                     for s,label in enumerate(policy_labels)]
            if ((row['seed'],row['deal'],row['hero'],row['slots']) != (card,index,seat,slots)
                    or row['chips'] != row['utilities'][seat] or not index < cross.DEALS):
                raise ValueError('Original cards, roles or private snapshot differ')
            if index % 233 == 0:
                sample = cross.play(engine,cross.seated(players,seat),policy_labels,card,seat)
                if sample != row:
                    raise ValueError('Sampled model transcript differs')
                sampled += 1
            totals[index] += row['chips']
            rows += 1
            decisions += row['steps']
    blocks = totals / 6
    terminal = verify_terminals(raw)
    if (terminal != dict(hands=rows,decisions=decisions) or rows != 6*cross.DEALS or
            not np.array_equal(expected,blocks)):
        raise ValueError('Incomplete terminal or paired card replay')
    atomic_json(folder/'results.json',dict(complete=True,completed_utc=utc(),plan_sha256=key,
                hands=rows,actions=decisions,sampled_hands=sampled,raw_sha256=digest(raw),
                blocks=blocks.tolist()))


def run_seed(seed):
    bound = read(OUTPUT/'bound.json')['sha256']
    verify_files({OUTPUT/'plan.json':bound})
    local = read(OUTPUT/'plan.json')
    verify_files(local['frozen_files'])
    plan, key = cross.plan()
    if key != local['cross_plan_sha256']:
        raise ValueError('Wrong study plan')
    sid = str(seed)
    with closing(Bridge()) as engine:
        for stage, path in local['opponents'][sid].items():
            old = evaluator_policy(path,plan['source'])
            for label in cross.LABELS:
                hero = cross.load(plan['policies'][label][sid],label,plan['source'])
                players = [[copy(p) for _ in range(6)] for p in (old,hero)]
                labels = [['past'+stage]*6,[label]*6]
                folder = OUTPUT/'evaluation'/f'{label}-vs-{stage}-{seed}'
                folder.mkdir()
                blocks = []
                with gzip.open(folder/'hands.jsonl.gz','xt',compresslevel=1) as stream:
                    for card in range(cross.START,cross.START+cross.DEALS):
                        chips = 0
                        for seat in range(6):
                            row = cross.play(engine,cross.seated(players,seat),cross.seated(labels,seat),card,seat)
                            stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                            chips += row['chips']
                        blocks.append(chips/6)
                verify_cell(folder,players,labels,engine,blocks,bound)
    verify_files(local['frozen_files'])


def score():
    bound = read(OUTPUT/'bound.json')['sha256']
    absolute, contrasts, hashes = {}, {}, {}
    hands = actions = sampled = 0
    for stage in read(OUTPUT/'design.json')['opponents']:
        data = {}
        for label in cross.LABELS:
            rows = []
            for seed in cross.SEEDS:
                folder = OUTPUT/'evaluation'/f'{label}-vs-{stage}-{seed}'
                r = read(folder/'results.json')
                if not r['complete'] or r['plan_sha256'] != bound:
                    raise ValueError('Unverified stage result')
                verify_files({folder/'hands.jsonl.gz':r['raw_sha256']})
                rows.append(r['blocks'])
                hashes[folder.name] = digest(folder/'results.json')
                hands += r['hands']
                actions += r['actions']
                sampled += r['sampled_hands']
            data[label] = np.array(rows)
        absolute[stage] = {label:cross.estimate(array,9) for label,array in data.items()}
        contrasts[stage] = {base:cross.estimate(data['mixture528']-data[base],6)
                            for base in ('mixture504','specialist480')}
    if hands != 162000 or len(hashes) != 27:
        raise ValueError('Incomplete history panel')
    summary = dict(complete=True,verified_utc=utc(),plan_sha256=bound,design_sha256=DESIGN,
                   terminal_replayed_hands=hands,terminal_replayed_actions=actions,
                   policy_replayed_sample_hands=sampled,original_cards_shared_with_crossplay=cross.DEALS,
                   means_bb100=absolute, paired_528_minus_reference_bb100=contrasts,
                   simultaneous_two_sided_95_family_sizes=dict(means=9,paired_contrasts=6),cell_sha256=hashes)
    for folder in (OUTPUT,PUBLIC):
        atomic_json(folder/'summary.json',summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('bind','run','seed','summary'))
    parser.add_argument('--seed',type=int,choices=cross.SEEDS)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    args = parser.parse_args()
    if args.mode == 'bind':
        with (OUTPUT/'bind.lock').open('x'):
            bind()
    elif args.mode == 'seed':
        run_seed(args.seed)
    elif args.mode == 'run':
        (OUTPUT/'evaluation').mkdir()
        (OUTPUT/'logs').mkdir(exist_ok=True)
        jobs = {f'seed-{s}':[str(PYTHON),str(SCRIPT),'seed','--seed',str(s)] for s in cross.SEEDS}
        run_jobs(OUTPUT,jobs,args.workers,'evaluation')
        score()
    else:
        score()
