"""Follow-up matched flop action values against four observed training-window opponents."""
import argparse
import ast
from contextlib import closing
import gzip
from itertools import chain
import json
from pathlib import Path

import numpy as np

from audit_postflop_averaging import net_fingerprint
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import PYTHON, atomic_json, digest
from poker_flop_action_values import BUTTON, _play, _support, future_deck, source_cards
from poker_multivalued_turn import Rules
from run_poker_flop_diagnostic import PARENT, REPORT, STUDY, bound
from run_poker_preflop_confirmation import run_jobs
from sample_poker_flop_diagnostic import load_policies
from score_poker_flop_diagnostic import action_values, clustered_intervals, probabilities

SUB = STUDY/'selfplay-window'
PUBLIC = REPORT/'selfplay-window'
SCRIPT = Path(__file__).resolve()
SEEDS = (342,343,344)
SLOTS = (480,486,492,498)
NEW = tuple(f'self{i}' for i in SLOTS[1:])
NAMES = ('mixture_minus_480_on_window','public_minus_window_on_mixture_change')


def provenance(seed, audit, mapping):
    sid = str(seed)
    directory = PARENT/'slow504'/f'seed-{seed}'
    ready = read(directory/'ready.json')
    diag = directory/'diagnostics.jsonl'
    verify_files({diag:ready['diagnostics_sha256']})
    records = {line['iteration']:line for line in (json.loads(s) for s in diag.read_text().splitlines())}
    start = read(directory/'start-verification.json')
    plan = read(PARENT/'population/evaluation-plan.json')
    policy480 = read(plan['policies']['specialist480'][sid]['path'])
    manifest = Path(plan['policies']['mixture504'][sid]['path'])
    snapshots = {r['iteration']:r for r in read(manifest)['postflop']}
    cycles, files = {}, {}
    for n in SLOTS:
        if n == 480:
            ref = policy480['postflop']
            path = Path(ref['path'])
            expected = start['weights']['general']
            if ref['sha256'] != start['street_weight_sha256']['postflop']:
                raise ValueError('Unmatched starting policy')
        else:
            ref = snapshots[n]
            path = manifest.parent/ref['file']
            expected = records[n]['output_weights']['general']
            if audit['seeds'][sid]['checkpoint_sha256'][str(n)] != ref['sha256']:
                raise ValueError('Captured opponent differs from training audit')
        files[str(path)] = ref['sha256']
        verify_files({path:ref['sha256']})
        if net_fingerprint(path,mapping) != expected or expected != records[n+1]['input_weights']['general']:
            raise ValueError(f'Opponent does not match the actual start of round {n+1}')
        cycles[str(n)] = dict(start_round=n+1,last_round=n+6,postflop_export_sha256=ref['sha256'],
                              round_input_weight_fingerprint=expected)
    files[str(diag)] = ready['diagnostics_sha256']
    return cycles, files


def register():
    cfg = bound()
    audit_path = PARENT/'training-audit.json'
    training_audit = read(audit_path)
    cfg_source = Path(read(PARENT/'plan.json')['campaign'])/'tools/compact_regret.py'
    parsed = ast.parse(cfg_source.read_text())
    mapping = next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='ARRAY_MAP' for target in node.targets))
    sources = [SCRIPT, SCRIPT.parent/'poker_flop_action_values.py', SCRIPT.parent/'sample_poker_flop_diagnostic.py',
               SCRIPT.parent/'score_poker_flop_diagnostic.py', audit_path, cfg_source, STUDY/'registration.json',
               STUDY/'summary.json',STUDY/'final-audit.json',PARENT/'population/evaluation-plan.json']
    files = {str(path):digest(path) for path in sources}
    cycles = {}
    for seed in SEEDS:
        folder = STUDY/f'seed-{seed}'
        old = read(folder/'rollout-complete.json')
        files.update({str(folder/'states.jsonl'):old['source_sha256'],str(folder/'values.jsonl.gz'):old['values_sha256']})
        cycles[str(seed)], refs = provenance(seed,training_audit,mapping)
        files.update(refs)
    verify_files(files)
    if cfg['paired_repetitions']!=8 or cfg['eligible_original_deals_per_seed']!=1024:
        raise ValueError('Unknown primary study budget')
    reg=dict(registered_utc=utc(),primary_diagnostic_already_scored=True, design='follow-up control, designed after primary results',
             seeds=list(SEEDS),states_per_seed=1024,repeats=8,unique_deals=3072,additional_deals=0,
             prefix_and_hero_continuation='frozen specialist480; all five opponents replaced after the same forced first flop action',
             fields_new=list(NEW),fields_reused=['self480','public'],four_opponent_policy_iterations=list(SLOTS),
             per_state_window_weights=[.25]*4,root_probability_difference='mean(486,492,498,504) minus 480',
             contrasts=list(NAMES),inference='equal fixed seed means; original deal clusters; two Bonferroni simultaneous normal 95% intervals and nominal paired intervals',
             frozen_cycles_by_seed=cycles, trace='original source deal and runouts, action RNG, forced concrete actions and replicate-0 audit traces from the primary protocol',
             units='big blinds per sampled first hero flop',limitation='four cycle-entry weights approximate training opponents; beta value targets, other 20 policies and reach distributions not reproduced',
             fit_or_policy_update=False, frozen_files=files)
    SUB.mkdir(exist_ok=False)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic_json(SUB/'registration.json',reg)
    atomic_json(PUBLIC/'registration.json',reg)


def write_line(stream, value):
    stream.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')


def worker(seed,reg):
    directory=SUB/f'seed-{seed}'
    directory.mkdir(exist_ok=False)
    root_policy, policies, _ = load_policies(seed)
    active_policies = {f'self{n}':policies[str(n)] for n in SLOTS[1:]}
    source = STUDY/f'seed-{seed}'/'states.jsonl'
    vpath, tpath = directory/'values.jsonl.gz',directory/'traces.jsonl.gz'
    count = hands = trace_count = 0
    with closing(Bridge()) as bridge, closing(Rules()) as rules, source.open() as states, \
            gzip.open(vpath,'xt',compresslevel=2) as out, gzip.open(tpath,'xt',compresslevel=2) as traces:
        for line in states:
            state=json.loads(line)
            if state['seed'] != seed:
                raise ValueError('Unexpected seed')
            header = {k:state[k] for k in ('state_id','seed','deal_seed','hero')}
            fixed=source_cards(bridge,state['deal_seed'])
            support = _support(state)
            legal = state['legal_actions']
            for name, opponent in active_policies.items():
                payoffs = [[] if i in support else None for i in range(len(legal))]
                for rep in range(reg['repeats']):
                    deck=future_deck(fixed,state['deal_seed'],rep)
                    for i in support:
                        u,actions=_play(rules,state,deck,legal[i],root_policy,opponent,rep,rep==0)
                        payoffs[i].append(u[state['hero']])
                        hands+=1
                        if rep==0:
                            write_line(traces,dict(header,lineup=name,action_index=i,replicate=rep,deck=deck,
                                                button=BUTTON,prefix=state['prefix'],actions=actions,utilities=u))
                            trace_count+=1
                write_line(out,dict(header,lineup=name,legal_actions=legal,repeats=reg['repeats'],action_payoffs_chips=payoffs))
            count+=1
            if count%128==0:
                atomic_json(directory/'progress.json',dict(utc=utc(),states=count,terminal_branches=hands))
    if count != reg['states_per_seed'] or hands!=trace_count*reg['repeats']:
        raise ValueError('Incomplete reroll coverage')
    atomic_json(directory/'receipt.json',dict(utc=utc(),states=count,terminal_branches=hands,trace_count=trace_count,
                                           states_sha256=digest(source),values_sha256=digest(vpath),traces_sha256=digest(tpath)))


def summarize(reg):
    effects, seed_ids, deals = [], [], []
    hands=0
    proofs={}
    for seed in SEEDS:
        old, new=STUDY/f'seed-{seed}',SUB/f'seed-{seed}'
        receipt=read(new/'receipt.json')
        verify_files({new/'values.jsonl.gz':receipt['values_sha256'],new/'traces.jsonl.gz':receipt['traces_sha256'],
                      old/'states.jsonl':receipt['states_sha256']})
        hands+=receipt['terminal_branches']
        proofs[str(seed)]=receipt
        local = {}
        with (old/'states.jsonl').open() as file:
            for line in file:
                state=json.loads(line)
                if state['state_id'] in local or state['seed']!=seed:
                    raise ValueError('Repeated or unmatched sample')
                p=probabilities(state)
                local[state['state_id']] = (state,p[1:].mean(axis=0)-p[0],np.any(p>0,axis=0),{})
        with gzip.open(old/'values.jsonl.gz','rt') as existing, gzip.open(new/'values.jsonl.gz','rt') as now:
            for line in chain(existing,now):
                row=json.loads(line)
                state, delta, support, gains=local[row['state_id']]
                field=row['lineup']
                if field=='tight-value':
                    continue
                if (field in gains or field not in ('self480','public',*NEW) or row['seed']!=seed
                        or row['deal_seed']!=state['deal_seed'] or row['legal_actions']!=state['legal_actions']
                        or row['hero']!=state['hero']):
                    raise ValueError('Primary and supplementary raw rows do not match')
                q=action_values(row,support,reg['repeats'],100)
                gains[field]=float(delta @ np.where(support,q-q[support][0],0))
        for state, _, _, gains in local.values():
            window=np.mean([gains[f'self{n}'] for n in SLOTS])
            effects.append([window,gains['public']-window])
            seed_ids.append(seed)
            deals.append(state['deal_seed'])
    if len(set(deals))!=reg['unique_deals'] or len(deals)!=reg['unique_deals']:
        raise ValueError('Wrong card coverage')
    result=dict(utc=utc(),registration_sha256=digest(SUB/'registration.json'),primary_results_known=True,
                unique_original_deals=len(deals),additional_terminal_branches=hands,
                additional_action_fits=0,additional_selfplay_rounds=0,units=reg['units'],receipts=proofs,
                contrasts=clustered_intervals(effects,seed_ids,deals,SEEDS,NAMES))
    atomic_json(SUB/'summary.json',result)
    atomic_json(PUBLIC/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('register','run','_worker'))
    parser.add_argument('seed',nargs='?',type=int)
    args=parser.parse_args()
    if args.phase=='register':
        register()
    else:
        reg=read(SUB/'registration.json')
        verify_files(reg['frozen_files'])
        if args.phase=='_worker' and args.seed in SEEDS:
            worker(args.seed,reg)
        elif args.phase=='run' and args.seed is None:
            with (SUB/'launch').open('x'):
                pass
            run_jobs(SUB,{f'seed-{s}':[str(PYTHON),str(SCRIPT),'_worker',str(s)] for s in SEEDS},3,'rollout')
            verify_files(reg['frozen_files'])
            summarize(reg)
        else:
            parser.error('Unexpected seed')
