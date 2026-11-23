"""Run the predeclared public-only flop rollback comparison on 5,000 fresh card blocks."""
import argparse
from collections import Counter
from contextlib import closing
import gzip
import json
import math
import os
from pathlib import Path
from statistics import NormalDist, mean, stdev
import subprocess
import time

import evaluate_average_baseline as baseline
import evaluate_poker_opponent_diversity as original
from audit_poker_public_divergence import binary as paired_terminal_reader
from audit_postflop_averaging import hero_policy, private_slot, replay, terminal_auditor
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_flop_rollback import FlopRollbackPolicy
from run_poker_preflop_confirmation import run_jobs

MAIN = original.STUDY.parent/'flop-rollback-20261123'
LOCAL = MAIN/'fresh-public'
PUBLIC = ROOT/'reports'/LOCAL.relative_to(ROOT/'artifacts')
SCRIPT = Path(__file__).resolve()
CONFIG = LOCAL/'config.json'
CONFIG_SHA = '2a08221765644288d5c17a066c8b083af04808db485a91dc322fa3ec28dcd044'
WRAPPER_SHA = 'f5a19bea33f000ea15918664ce24d8286a9c012f216785fc5ac7254330f615be'
SEEDS = (342, 343, 344)
POLICIES = ('specialist480','diverse504','flop480late504')
PAIRS = ((2,1),(2,0),(1,0))
START, COUNT = 2050100000, 5000


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(label, ref, source):
    if label == POLICIES[2]:
        return FlopRollbackPolicy(ref['path'], source)
    return original.load(label, ref, source)


class IndependentHybrid:
    """Use the independently reconstructed hand mixture only after the flop."""
    def __init__(self, parent, late, source):
        self.parent=hero_policy(parent['path'],source,False)
        self.late=hero_policy(late['path'],source,True)

    def begin_hand(self, seed):
        self.late.begin_hand(seed)

    def decide(self, state, seed):
        policy=self.late if state['street'] in ('turn','river') else self.parent
        return policy.decide(state,seed)


def bind():
    verify_files({CONFIG:CONFIG_SHA, PUBLIC/'config.json':CONFIG_SHA,
                  SCRIPT.with_name('poker_flop_rollback.py'):WRAPPER_SHA})
    cfg = read(CONFIG)
    require((cfg['evaluation_deal_seed'],cfg['evaluation_deals'],cfg['labels'],cfg['seeds'],
             cfg['total_hands'],cfg['total_cells']) == (START,COUNT,list(POLICIES),list(SEEDS),270000,9),
            'Fresh registration differs from the locked design')
    verify_files(cfg['source_stage_and_policy_sha256'])
    main_plan = MAIN/'evaluation-plan.json'
    main_sha = read(MAIN/'bound.json')['evaluation_plan_sha256']
    verify_files({main_plan:main_sha})
    plan, prior = read(main_plan), read(original.OUTPUT/'evaluation-plan.json')
    for field in plan['fields'].values():
        require(max(START,field['deal_seed']) >= min(START+COUNT,field['deal_seed']+field['deals']),
                'Fresh card overlaps a main-panel card')
    require(plan['source']==prior['source'] and plan['seeds']==list(SEEDS), 'Unrelated source engine or seeds')
    policies={p:prior['policies'][p] for p in POLICIES[:2]}
    policies[POLICIES[2]]=plan['policies'][POLICIES[2]]
    stage=read(original.STUDY/'stage-ready.json')['policies']
    require(policies[POLICIES[0]]==stage['specialist480'] and policies[POLICIES[1]]==stage['poolMixture504'],
            'Wrong parents in registered evaluation')
    own = [SCRIPT, CONFIG, main_plan, MAIN/'bound.json', original.OUTPUT/'evaluation-plan.json',
           Path(original.panel.__file__),Path(original.__file__),Path(baseline.__file__),
           SCRIPT.with_name('poker_flop_rollback.py'),
           SCRIPT.with_name('audit_poker_public_divergence.py'),SCRIPT.with_name('audit_postflop_averaging.py'),
           SCRIPT.with_name('run_poker_preflop_confirmation.py'),ROOT/'native/noregrets-json/public_divergence.rs',
           ROOT/'native/noregrets-json/terminal_audit.rs']
    verifier=terminal_auditor()
    paired=paired_terminal_reader()
    frozen={str(p):digest(p) for p in (*own,verifier,paired)}
    opponents={}
    for seed in SEEDS:
        key=str(seed)
        _, ids, blueprint, deps=original.inputs(seed,'public')
        require(blueprint is None and ids==plan['opponents'][f'public-seed-{seed}'], 'Wrong public opponent')
        opponents[key]=ids
        frozen.update(deps)
        for label in POLICIES:
            ref=policies[label][key]
            verify_files({ref['path']:ref['sha256']})
            if label==POLICIES[2]:
                manifest=read(ref['path'])
                require(manifest['specialist480']==policies[POLICIES[0]][key] and manifest['diverse504']==policies[POLICIES[1]][key],
                        'Hybrid references a different baseline or snapshot mixture')
            candidate=load(label,ref,plan['source'])
            frozen.update(baseline.policy_dependency_hashes(candidate,plan['source']))
    verify_files(frozen)
    payload=dict(utc=utc(),design_sha256=CONFIG_SHA,main_plan_sha256=main_sha,policies=policies,
                 source=plan['source'],opponents=opponents,terminal_auditor=str(verifier),paired_auditor=str(paired),
                 frozen_files=frozen,evaluation_deal_seed=START,evaluation_deals=COUNT)
    target=LOCAL/'evaluation-plan.json'
    with (LOCAL/'binding.lock').open('x'):
        if target.exists():
            raise FileExistsError(target)
        atomic_json(target,payload)
        bound=dict(utc=utc(),plan_sha256=digest(target),design_sha256=CONFIG_SHA,main_plan_sha256=main_sha,
                   terminal_verifier_sha256=frozen[str(verifier)],paired_verifier_sha256=frozen[str(paired)])
        atomic_json(LOCAL/'bound.json',bound)
        atomic_json(PUBLIC/'bound.json',bound)


def plan_and_hash():
    h=read(LOCAL/'bound.json')['plan_sha256']
    verify_files({CONFIG:CONFIG_SHA,LOCAL/'evaluation-plan.json':h})
    return read(LOCAL/'evaluation-plan.json'),h


def verify_terminals(binary, raw):
    with subprocess.Popen(['gzip','-dc',raw],stdout=subprocess.PIPE) as unzip:
        try:
            result=json.loads(subprocess.check_output([binary],stdin=unzip.stdout))
        finally:
            unzip.stdout.close()
        require(unzip.wait()==0, 'Corrupt raw ledger')
    require(result['hands']==6*COUNT, 'Incorrect number of terminal settlements')
    return result


def worker(label, seed):
    require(label in POLICIES and seed in SEEDS,'Out-of-design cell')
    plan,sha = plan_and_hash()
    parent=read(LOCAL/'launch.json')
    require(parent['pid']==os.getppid() and process_start(parent['pid'])==parent['process_start'],
            'Worker belongs to another supervisor')
    key=str(seed)
    out=LOCAL/'evaluation'/f'public-seed-{seed}-{label}'
    out.mkdir(exist_ok=False)
    def status(phase, **fields):
        atomic_json(out/'status.json',dict(phase=phase,utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid()),**fields))
    try:
        status('loading')
        bots,ids,blueprint,deps=original.inputs(seed,'public')
        ref=plan['policies'][label][key]
        candidate=load(label,ref,plan['source'])
        deps.update(baseline.policy_dependency_hashes(candidate,plan['source']))
        require(ids==plan['opponents'][key] and blueprint is None and
                all(plan['frozen_files'].get(f)==v for f,v in deps.items()),'Changed evaluation dependencies')
        verify_files(deps)
        deps.update({str(SCRIPT):plan['frozen_files'][str(SCRIPT)]})
        provenance=dict(label=label,training_seed=seed,field='public',deal_seed=START,deals=COUNT,hands=6*COUNT,
                        opponents=ids,candidate_sha256=ref['sha256'],candidate=ref['path'],candidate_type=candidate.model_type,
                        first_deal_clockwise_lineup=list(original.table('public',START)),big_blind_chips=100,
                        design_sha256=CONFIG_SHA,evaluation_plan_sha256=sha,dependency_hashes=deps)
        atomic_json(out/'provenance.json',provenance)
        chips=[]; counts=Counter(); slots=Counter()
        raw=out/'hands.jsonl.gz'
        deadline=time.monotonic()+7200
        status('evaluating',completed_card_blocks=0)
        with original.previous.ready_engine(blueprint,LOCAL) as engine,gzip.open(raw,'xt',compresslevel=1) as stream:
            for i, card in enumerate(range(START,START+COUNT)):
                table=original.table('public',card)
                total=0
                for hero in range(6):
                    hand=original.panel.play(engine,candidate,bots,table,card,hero,counts,deadline)
                    hand['deal']=i
                    if label!=POLICIES[0]:
                        slot=original.SNAPSHOTS[candidate.postflop.index(candidate.active)]
                        slots[slot]+=1
                        hand['snapshot_iteration']=slot
                    total+=hand['chips']
                    stream.write(json.dumps(hand,separators=(',',':'),allow_nan=False)+'\n')
                chips.append(total/6)
                if (i+1)%100==0:
                    status('evaluating',completed_card_blocks=i+1)
        status('verifying_raw_hands')
        terminal=verify_terminals(plan['terminal_auditor'],raw)
        verify_files(deps)
        result=dict(provenance,deal_blocks=chips,bb_per_100=mean(chips),behavior=dict(counts),snapshot_counts=dict(slots),
                    completed_utc=utc(),complete=True,terminal_payoffs_replayed=True,verification=terminal,
                    raw_hands_sha256=digest(raw),provenance_sha256=digest(out/'provenance.json'))
        atomic_json(out/'results.json',result)
        status('complete',verified_hands=6*COUNT,results_sha256=digest(out/'results.json'))
    except BaseException as err:
        status('failed',error=f'{type(err).__name__}: {err}',automatic_retry=False)
        raise


def interval(blocks):
    require(len(blocks)==COUNT and all(map(math.isfinite,blocks)), 'Invalid independent card vector')
    score=mean(blocks)
    err=stdev(blocks)/math.sqrt(len(blocks))
    half=NormalDist().inv_cdf(.975)*err
    return dict(mean_bb_per_100=score,paired_card_se=err,nominal_95_bb_per_100=[score-half,score+half])


def summarize():
    plan,plan_sha=plan_and_hash()
    verify_files(plan['frozen_files'])
    data={label:{} for label in POLICIES}
    receipts={}
    total_decisions=0
    sample_hands=sample_actions=0
    category_deltas={}
    for seed in SEEDS:
        paths=[]
        refs={label:plan['policies'][label][str(seed)] for label in POLICIES}
        auditors={POLICIES[0]:hero_policy(refs[POLICIES[0]]['path'],plan['source'],False),
                  POLICIES[1]:hero_policy(refs[POLICIES[1]]['path'],plan['source'],True),
                  POLICIES[2]:IndependentHybrid(refs[POLICIES[0]],refs[POLICIES[1]],plan['source'])}
        with closing(Bridge()) as rules:
            for label in POLICIES:
                folder=LOCAL/'evaluation'/f'public-seed-{seed}-{label}'
                result=read(folder/'results.json')
                provenance=read(folder/'provenance.json')
                expected=dict(label=label,training_seed=seed,deal_seed=START,deals=COUNT,hands=COUNT*6,
                              opponents=plan['opponents'][str(seed)],evaluation_plan_sha256=plan_sha,design_sha256=CONFIG_SHA,
                              candidate_sha256=refs[label]['sha256'],field='public',big_blind_chips=100)
                require(all(result[k]==provenance[k]==v for k,v in expected.items()) and result['complete'] and
                        result['terminal_payoffs_replayed'], 'Cell does not match bound model, cards or opponent')
                raw=folder/'hands.jsonl.gz'
                verify_files({raw:result['raw_hands_sha256'],folder/'provenance.json':result['provenance_sha256']})
                paths.append(raw)
                receipts[folder.name]=digest(folder/'results.json')
                sums=[0]*COUNT
                bots={'public':Fullhouse(),'hero':auditors[label]}
                row_count=0
                snapshots=Counter()
                with gzip.open(raw,'rt') as stream:
                    for line in stream:
                        row=json.loads(line)
                        deal,hero=divmod(row_count,6)
                        card=START+deal
                        hs=baseline.hand_seed(card,hero)
                        require(row_count<6*COUNT and (row['seed'],row['deal'],row['hero'],row['hand_seed'])==(card,deal,hero,hs),
                                'Missing, duplicated or mismatched card/seat/hand RNG')
                        if label==POLICIES[0]:
                            require('snapshot_iteration' not in row,'Unexpected parent hand snapshot')
                        else:
                            require(row['snapshot_iteration']==private_slot(hs),'Wrong frozen hand-private slot')
                            snapshots[str(row['snapshot_iteration'])]+=1
                        if deal%499==0:
                            trace=replay(row,'public',rules,bots)
                            sample_hands+=1
                            sample_actions+=len(trace)
                        sums[deal]+=row['utilities'][hero]
                        total_decisions+=row['steps']
                        row_count+=1
                require(row_count==6*COUNT and snapshots==result['snapshot_counts'] and
                        [x/6 for x in sums]==result['deal_blocks'] and
                        math.isclose(sum(sums)/(6*COUNT),result['bb_per_100'],abs_tol=1e-10),
                        'Card-level ledger disagrees with the cell')
                data[label][seed]=sums
        # Order: hybrid-parent, diverse-parent, hybrid-diverse. Validate every actor, move and chip,
        # then independently reconcile all THREE paired conditional trajectories to the saved card ledger.
        check=json.loads(subprocess.check_output([plan['paired_auditor'],*paths,str(START),str(COUNT)]))
        require(check['policy_chips']==[sum(data[label][seed]) for label in POLICIES], 'Replayed chip totals differ')
        pairs=((2,0),(1,0),(2,1))
        for j,(a,b) in enumerate(pairs):
            count,gain=check['counts'][j],check['chip_differences'][j]
            require(sum(count)==6*COUNT and sum(gain)==sum(data[POLICIES[a]][seed])-sum(data[POLICIES[b]][seed])
                    and count[0]==count[4]==0,'Ledger or comparison routing changed')
            if (a,b)==(2,0):
                require(count[1]==0,'Rollback changed a hero action before turn relative to its 480 parent')
            category_deltas[f'{seed}:{POLICIES[a]} minus {POLICIES[b]}'] = count
    means={p:[sum(data[p][s][i] for s in SEEDS)/18 for i in range(COUNT)] for p in POLICIES}
    scores={p:interval(v) for p,v in means.items()}
    pairs={}
    for a,b in PAIRS:
        l,r=POLICIES[a],POLICIES[b]
        paired=interval([x-y for x,y in zip(means[l],means[r],strict=True)])
        paired['fixed_seed_deltas_bb_per_100']={str(s):(sum(data[l][s])-sum(data[r][s]))/(6*COUNT) for s in SEEDS}
        require(math.isclose(paired['mean_bb_per_100'],scores[l]['mean_bb_per_100']-scores[r]['mean_bb_per_100'],
                             abs_tol=1e-10),'Non-additive paired scores')
        pairs[f'{l} minus {r}']=paired
    report=dict(complete=True,verified_utc=utc(),plan_sha256=plan_sha,design_sha256=CONFIG_SHA,
                field='public',card_start_inclusive=START,card_end_inclusive=START+COUNT-1,
                independent_card_blocks=COUNT,fixed_model_seeds=list(SEEDS),seat_rotations=6,recorded_hands=270000,
                complete_cells=9,policy_scores_bb_per_100=scores,paired_comparisons=pairs,
                inference='Mean paired payoff per card after averaging the 18 fixed seed and seat repeats, two-sided nominal 95% intervals.',
                terminal_chip_replayed_hands=270000,terminal_actions=total_decisions,
                independently_policy_replayed_hands=sample_hands,independently_policy_replayed_actions=sample_actions,
                independent_street_divergence_counts=category_deltas,ordered_street_groups=['preflop','flop','turn','river','opponent_first','identical'],
                result_sha256=receipts,new_fitting=0,new_selfplay=0,model_promoted=False)
    for folder in (LOCAL,PUBLIC): atomic_json(folder/'summary.json',report)


def run(workers):
    require(1<=workers<=3, 'One to three workers expected')
    plan,h = plan_and_hash()
    verify_files(plan['frozen_files'])
    (LOCAL/'evaluation').mkdir(exist_ok=False)
    with (LOCAL/'launch.json').open('x') as stream:
        json.dump(dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid()),
                       plan_sha256=h,concurrent_workers=workers),stream)
    commands={f'public-seed-{s}-{p}':[str(PYTHON),str(SCRIPT),'_worker',p,str(s)] for s in SEEDS for p in POLICIES}
    run_jobs(LOCAL,commands,workers,'evaluation')
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['bind','run','_worker','summarize'])
    parser.add_argument('args',nargs='*')
    args=parser.parse_args()
    if args.mode=='_worker' and len(args.args)==2:
        worker(args.args[0],int(args.args[1]))
    elif args.mode=='run' and len(args.args)==1:
        run(int(args.args[0]))
    elif args.args:
        parser.error('Unsupported arguments')
    elif args.mode=='bind':
        bind()
    elif args.mode=='summarize':
        summarize()
    else:
        parser.error('Worker or concurrency arguments required')
