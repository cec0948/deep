"""Bounded common-card evaluation of explicit rich or legacy current policies.

Every terminal payoff is replayed from the recorded concrete actions through
the trusted rules engine before results.json is committed. Cards are the
statistical blocks; seats, seeds and policy versions repeat those same blocks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from fullhouse_runtime import Bridge, Fullhouse, ROOT, SOURCE
from fullhouse_v3_common import atomic_json, digest, process_start
from rich_poker_runtime import policy_for


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def semantic_seed(deal_seed, hero, state):
    value = dict(protocol='rich-common-card-evaluation-v1', deal_seed=deal_seed,
                 hero=hero, actor=state['seat_to_act'], action_log=state['action_log'])
    return int.from_bytes(hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':')).encode()).digest()[:8], 'big')


def hand_seed(deal_seed, hero):
    return int.from_bytes(hashlib.sha256(f'rich-evaluation-hand:{deal_seed}:{hero}'.encode()).digest()[:8], 'big')


def interval(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('At least two finite independent card blocks required')
    mean = statistics.mean(values)
    half = 1.96*statistics.stdev(values)/math.sqrt(len(values))
    return dict(mean=mean, ci95_half_width=half, lower=mean-half, upper=mean+half)


def check_dependencies(expected):
    for name, sha in expected.items():
        if digest(name) != sha:
            raise ValueError('Frozen evaluation dependency changed: '+name)


def policy_dependency_hashes(policy, source):
    result = dict(getattr(policy, 'dependency_hashes', {}))
    result[str(Path(policy.model).resolve())] = digest(policy.model)
    if source is not None:
        path = Path(source).resolve()/'bot/features.py'
        result[str(path)] = digest(path)
    for name in ('evaluate_rich_selfplay.py','rich_poker_runtime.py',
                 'rich_poker_observation.py','poker_public_state.py',
                 'fullhouse_runtime.py','fullhouse_v4_runtime.py','regret_partition.py'):
        path = ROOT/'scripts'/name
        result[str(path)] = digest(path)
    for path in (ROOT/'.cache/noregrets/target/release/poker-bridge',
                 ROOT/'configs/fullhouse-model.json', SOURCE/'bot/features.py',
                 SOURCE/'bot/deep_cfr_lookup.py', SOURCE/'data/preflop_equity.npz'):
        result[str(path)] = digest(path)
    return result


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Evaluation time budget exhausted; no automatic retry')


def _validate_reply(state, reply):
    if not isinstance(reply, dict) or reply.get('ok') is not True:
        raise ValueError('Candidate returned an invalid action response')
    probabilities = [row.get('probability') for row in reply.get('distribution', [])]
    if (not probabilities or any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
                                 for v in probabilities)
            or not math.isclose(sum(probabilities), 1., abs_tol=1e-6)):
        raise ValueError('Invalid candidate action distribution')
    # The trusted engine separately enforces every concrete action's legality.
    return reply


def _verify_rows(result, raw_path, *, replay, deadline=None, progress=None):
    deals, start = result['deals'], result['deal_seed']
    blocks = [[None]*6 for _ in range(deals)]
    engine = Bridge() if replay else None
    count = decisions = 0
    try:
        with gzip.open(raw_path, 'rt') as stream:
            for line in stream:
                if deadline is not None:
                    _check_deadline(deadline)
                row = json.loads(line)
                deal, hero = divmod(count, 6)
                if deal >= deals or (row.get('deal'),row.get('hero'),row.get('seed')) != (deal,hero,start+deal):
                    raise ValueError('Repeated, missing or out-of-order card/seat record')
                utilities = row.get('utilities')
                if (not isinstance(utilities,list) or len(utilities) != 6
                        or any(type(v) is not int for v in utilities) or sum(utilities) != 0
                        or row.get('chips') != utilities[hero]
                        or row.get('hand_seed') != hand_seed(start+deal,hero)):
                    raise ValueError('Invalid raw utility/hand seed record')
                actions = row.get('actions')
                if (not isinstance(actions,list) or len(actions) != row.get('steps')
                        or len(actions) > 512):
                    raise ValueError('Missing or excessive concrete action trace')
                if engine is not None:
                    state = engine.call(dict(op='reset',seed=start+deal,button=0))
                    for action in actions:
                        if state['terminal'] or action.get('seat') != state['seat_to_act']:
                            raise ValueError('Raw action replay has the wrong acting seat')
                        state = engine.call(dict(op='step',action=action)); decisions += 1
                    if not state['terminal'] or state['utilities'] != utilities:
                        raise ValueError('Trusted replay terminal payoffs differ from raw record')
                blocks[deal][hero] = row['chips']; count += 1
                if progress is not None and count % 600 == 0:
                    progress('verifying_raw_hands',verified_hands=count,planned_hands=deals*6)
    finally:
        if engine is not None:
            engine.close()
    if count != deals*6 or result['hands'] != count:
        raise ValueError('Incomplete raw hand coverage')
    means = [statistics.mean(row) for row in blocks]
    stats = interval(means)
    if (len(result['deal_blocks']) != deals
            or any(not math.isclose(a,b,rel_tol=0,abs_tol=1e-9)
                   for a,b in zip(means,result['deal_blocks']))
            or not math.isclose(stats['mean'],result['bb_per_100'],rel_tol=0,abs_tol=1e-8)
            or not math.isclose(stats['ci95_half_width'],result['ci95_half_width'],rel_tol=0,abs_tol=1e-8)):
        raise ValueError('Raw card-block aggregate differs from reported result')
    return dict(hands=count,card_blocks=deals,concrete_decisions_replayed=decisions,
                terminal_payoffs_replayed=replay,chip_conservation_verified=True,
                aggregate_recomputed=True)


def verify_results(output, *, replay=False, expected=None):
    """Recheck frozen bytes and every raw payoff; optionally replay actions again."""
    output = Path(output)
    result = json.loads((output/'results.json').read_text())
    if not result.get('complete') or not result.get('raw_hands_verified'):
        raise ValueError('Result has not completed raw-record verification')
    if expected:
        for key,value in expected.items():
            if result.get(key) != value:
                raise ValueError('Unexpected evaluation provenance: '+key)
    check_dependencies(result['dependency_hashes'])
    if (digest(output/'hands.jsonl.gz') != result['raw_hands_sha256']
            or digest(output/'provenance.json') != result['provenance_sha256']):
        raise ValueError('Verified raw/provenance bytes changed')
    _verify_rows(result,output/'hands.jsonl.gz',replay=replay)
    return result


def run(model, source, field, deals, deal_seed, timeout, output):
    if field not in ('public','mixed','noregrets'):
        raise ValueError('Unknown fixed opponent field')
    if (type(deals) is not int or not 2 <= deals <= 20000 or type(deal_seed) is not int
            or not 0 <= deal_seed < deal_seed+deals <= 2**32
            or not 0 < timeout <= 7200):
        raise ValueError('Invalid bounded evaluation design')
    output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    began = time.monotonic(); deadline = began+timeout
    import os
    pid = os.getpid(); token = process_start(pid)
    def progress(phase,**values):
        atomic_json(output/'status.json',dict(status='running',phase=phase,
            observed_utc=utc(),pid=pid,process_start=token,
            elapsed_seconds=time.monotonic()-began,planned_card_blocks=deals,**values))
    engine = None
    try:
        progress('preparing')
        policy = policy_for(model, source)
        public = Fullhouse() if field in ('public','mixed') else None
        registry_path = ROOT/'artifacts/selfplay/current.json'
        registry = json.loads(registry_path.read_text())
        opponent_model = ROOT/registry['model']
        if digest(opponent_model) != registry['sha256']:
            raise ValueError('NoRegrets opponent changed')
        dependencies = policy_dependency_hashes(policy, source)
        dependencies[str(registry_path)] = digest(registry_path)
        if field in ('mixed','noregrets'):
            dependencies[str(opponent_model.resolve())] = registry['sha256']
        if public:
            dependencies[str(public.model)] = public.digest
        check_dependencies(dependencies)
        opponents = dict(noregrets=registry['sha256'],public=public.digest if public else None)
        provenance = dict(created_utc=utc(),candidate=str(Path(policy.model).resolve()),
            candidate_sha256=digest(policy.model),candidate_type=policy.model_type,
            field=field,deals=deals,hands=deals*6,deal_seed=deal_seed,opponents=opponents,
            semantic_rng='Shared SHA256 of card seed, hero, actor and complete public action log',
            dependency_hashes=dependencies,model_promoted=False,
            statistical_unit='One original card block; six seats repeat that block')
        atomic_json(output/'provenance.json',provenance)
        engine = Bridge(opponent_model if field in ('mixed','noregrets') else None)
        blocks=[];counts=Counter();peak=0
        progress('evaluating',completed_card_blocks=0,completed_hands=0)
        with gzip.open(output/'hands.jsonl.gz','wt',compresslevel=1) as stream:
            for deal in range(deals):
                chips=[]
                for hero in range(6):
                    _check_deadline(deadline)
                    state=engine.call(dict(op='reset',seed=deal_seed+deal,button=0));actions=[]
                    fixed_hand_seed=hand_seed(deal_seed+deal,hero)
                    if hasattr(policy,'begin_hand'):policy.begin_hand(fixed_hand_seed)
                    while not state['terminal']:
                        _check_deadline(deadline)
                        if len(actions)>=512:raise RuntimeError('Nonterminating evaluation hand')
                        actor=state['seat_to_act'];seed=semantic_seed(deal_seed+deal,hero,state)
                        if actor==hero:
                            reply=_validate_reply(state,policy.decide(state,seed))
                            counts[f'hero/{state["street"]}/{reply["action"]["kind"]}']+=1
                        elif field=='public' or (field=='mixed' and (actor-hero)%6%2==1):
                            reply=_validate_reply(state,public.decide(state,seed))
                        else:
                            reply=engine.call(dict(op='blueprint',seed=seed))
                            counts[f'opponent/{state["street"]}/fallback']+=int(reply.get('fallback',False))
                            counts[f'opponent/{state["street"]}/decisions']+=1
                        action=dict(reply['action'],seat=actor);actions.append(action)
                        state=engine.call(dict(op='step',action=action))
                    utilities=state['utilities']
                    if len(utilities)!=6 or any(type(v) is not int for v in utilities) or sum(utilities)!=0:
                        raise ValueError('Terminal chip conservation failed')
                    chips.append(utilities[hero])
                    stream.write(json.dumps(dict(deal=deal,hero=hero,seed=deal_seed+deal,
                        chips=utilities[hero],utilities=utilities,steps=len(actions),actions=actions,
                        hand_seed=fixed_hand_seed),allow_nan=False)+'\n')
                #100chips/BB means mean chips is also BB per100hands.
                blocks.append(statistics.mean(chips))
                if (deal+1)%100==0 or deal+1==deals:
                    check_dependencies(dependencies)
                    rss=0
                    for process in (pid,engine.p.pid):
                        lines=Path(f'/proc/{process}/status').read_text().splitlines()
                        rss+=next((int(s.split()[1])*1024 for s in lines if s.startswith('VmRSS:')),0)
                    peak=max(peak,rss)
                    if rss>8*1024**3:raise MemoryError('Evaluation memory budget exceeded')
                    progress('evaluating',completed_card_blocks=deal+1,completed_hands=(deal+1)*6)
                    atomic_json(output/'progress.json',dict(observed_utc=utc(),completed_deals=deal+1,
                        planned_deals=deals,elapsed_seconds=time.monotonic()-began))
        engine.close();engine=None
        statistics_row=interval(blocks)
        result=dict(complete=True,field=field,deals=deals,hands=deals*6,deal_seed=deal_seed,
            candidate=str(Path(policy.model).resolve()),candidate_sha256=digest(policy.model),
            candidate_type=policy.model_type,opponents=opponents,
            bb_per_100=statistics_row['mean'],ci95_half_width=statistics_row['ci95_half_width'],
            deal_blocks=blocks,big_blind_chips=100,behavior=dict(counts),
            elapsed_seconds=time.monotonic()-began,peak_sampled_rss_bytes=peak,
            evaluator_sha256=digest(__file__),runtime_sha256=digest(ROOT/'scripts/rich_poker_runtime.py'),
            engine_sha256=digest(ROOT/'.cache/noregrets/target/release/poker-bridge'),
            source_features_sha256=digest(Path(source)/'bot/features.py') if source else None,
            dependency_hashes=dependencies,source_provenance_verified=True,
            raw_hands_verified=False,terminal_payoffs_replayed=False,
            statistical_unit='Original card block; seats and training seeds are repeated observations',
            limitations=['Conditional nominal95%card-block intervals; no multiplicity adjustment',
                'Fixed public and NoRegrets pool does not establish professional or general strength',
                'NoRegrets lookup fallback counts are reported; it is not certified as a strong opponent',
                'Current policies only; no average strategy or automatic model promotion'])
        progress('verifying_raw_hands',verified_hands=0,planned_hands=deals*6)
        verification=_verify_rows(result,output/'hands.jsonl.gz',replay=True,deadline=deadline,progress=progress)
        check_dependencies(dependencies);_check_deadline(deadline)
        result.update(raw_hands_verified=True,terminal_payoffs_replayed=True,verification=verification,
            raw_hands_sha256=digest(output/'hands.jsonl.gz'),
            provenance_sha256=digest(output/'provenance.json'),completed_utc=utc(),
            elapsed_seconds=time.monotonic()-began)
        atomic_json(output/'results.json',result)
        atomic_json(output/'status.json',dict(status='complete',phase='complete',observed_utc=utc(),
            pid=pid,process_start=token,completed_evaluations=1,verified_hands=deals*6,
            elapsed_seconds=time.monotonic()-began,results_sha256=digest(output/'results.json')))
        return result
    except BaseException as error:
        atomic_json(output/'status.json',dict(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',
            phase='stopped',observed_utc=utc(),pid=pid,process_start=token,
            elapsed_seconds=time.monotonic()-began,error_type=type(error).__name__,error=str(error),automatic_retry=False))
        raise
    finally:
        if engine is not None:engine.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--source',type=Path)
    parser.add_argument('--field',choices=['public','mixed','noregrets'],required=True)
    parser.add_argument('--deals',type=int,default=20000);parser.add_argument('--deal-seed',type=int,default=2035000000)
    parser.add_argument('--timeout',type=float,default=2400);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.model,args.source,args.field,args.deals,args.deal_seed,args.timeout,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('deal_blocks','behavior','dependency_hashes')},allow_nan=False))


if __name__=='__main__':main()
