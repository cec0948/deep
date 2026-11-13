"""Bounded paired *conditional river* panel; never a full-game strength claim.

Selection uses only the frozen current360 baseline versus five public policies.
The first four eligible cases per model are retained, at most one per model and
deal block. Every arm then replays the identical trusted-engine seed and public
prefix. No hidden deck or other players' hole cards are supplied to the hero.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

from fullhouse_runtime import Bridge,Fullhouse
from fullhouse_v4_runtime import FullhouseV4
from poker_public_state import PublicState
from poker_river_runtime import RiverSearchPolicy,legal_action

ROOT=Path(__file__).resolve().parents[1]
FORMAT='river-strength-pilot-v1'
ARM_NAMES=('baseline','search-1s','search-5s','search-10s')
T95={1:12.7062047364,2:4.3026527297,3:3.1824463053,4:2.7764451052,
     5:2.5705818356,6:2.4469118511,7:2.3646242510,8:2.3060041352,
     9:2.2621571629,10:2.2281388520,11:2.2009851601}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def validate_config(config):
    expected=dict(format=FORMAT,deal_seed_start=2032111300,cases_per_model=4,
                  model_seeds=[342,343,344],hero_seats=list(range(6)),search_seconds=[1,5,10],button=0)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Predeclared conditional-panel design changed')
    for key,maximum in [('max_deal_blocks',128),('max_seconds',1800)]:
        if type(config.get(key)) is not int or not 1<=config[key]<=maximum:
            raise ValueError('Invalid fixed scan/execution budget')
    models=config.get('models')
    if not isinstance(models,list) or [row.get('seed') for row in models]!=config['model_seeds']:
        raise ValueError('Exactly the three declared current360 models are required')
    for row in models:
        if set(row)!={'seed','model','source'} or not Path(row['model']).is_file() or not Path(row['source']).is_dir():
            raise ValueError('Each model must identify an existing model and frozen source')
    frozen=config.get('expected_hashes')
    if not isinstance(frozen,dict) or not frozen or any(not Path(p).is_absolute() for p in frozen):
        raise ValueError('Frozen absolute source/model hashes required before execution')
    required={str(Path(__file__).resolve()),str(ROOT/'scripts/poker_public_state.py'),
              str(ROOT/'scripts/poker_river_runtime.py'),str(ROOT/'scripts/river_subgame_solver.py'),
              str(ROOT/'scripts/fullhouse_runtime.py'),str(ROOT/'scripts/fullhouse_v4_runtime.py'),
              str(ROOT/'scripts/regret_partition.py'),str(ROOT/'configs/fullhouse-model.json'),
              str(ROOT/'.cache/noregrets/target/release/poker-bridge'),
              str(ROOT/'.cache/fullhouse-bot/bot/features.py'),
              str(ROOT/'.cache/fullhouse-bot/bot/deep_cfr_lookup.py'),
              str(ROOT/'.cache/fullhouse-bot/data/preflop_equity.npz'),
              str(Path(config.get('public_opponent_model') or ROOT/'.cache/fullhouse-bot/data/deep_cfr_model.npz').resolve())}
    required.update(str(Path(m['model']).resolve()) for m in models)
    required.update(str(Path(m['source']).resolve()/'bot/features.py') for m in models)
    if not required.issubset(frozen):raise ValueError('Frozen manifest omits a policy/rules/runtime dependency')


def check_frozen(config):
    for name,sha in config['expected_hashes'].items():
        if not isinstance(sha,str) or digest(name)!=sha:
            raise ValueError('Frozen input changed: '+name)


def semantic_seed(deal_seed,hero,state):
    """Same public decision gets the same RNG across arms, independent of steps."""
    key=dict(protocol='river-pilot-public-decision-rng-v1',deal_seed=deal_seed,hero=hero,
             actor=state['seat_to_act'],public_action_history=state['action_log'])
    return int.from_bytes(hashlib.sha256(json.dumps(key,sort_keys=True,separators=(',',':')).encode()).digest()[:8],'big')


def hand_seed(deal_seed,hero,model_seed):
    return int.from_bytes(hashlib.sha256(f'river-pilot-hand:{deal_seed}:{hero}:{model_seed}'.encode()).digest()[:8],'big')


def eligible(state,hero):
    if state.get('terminal') or state.get('street')!='river' or state.get('seat_to_act')!=hero:
        return False
    active=[p for p in state.get('players',[]) if not p['is_folded']]
    return len(active)==2 and all(p['stack']>0 and not p['is_all_in'] for p in active)


def validate_reply(state,reply):
    if not isinstance(reply,dict) or reply.get('ok') is not True:
        raise ValueError('Policy returned an error')
    json.dumps(reply,allow_nan=False)
    rows=reply.get('distribution')
    if not isinstance(rows,list) or not rows or not legal_action(state,reply.get('action',{})):
        raise ValueError('Policy returned an illegal action or missing distribution')
    probabilities=[row.get('probability') for row in rows]
    if any(type(p) not in (int,float) or not math.isfinite(p) or p<0 for p in probabilities):
        raise ValueError('Nonfinite or negative action probability')
    if not math.isclose(sum(probabilities),1.,abs_tol=1e-6) or any(not legal_action(state,row.get('action',{})) for row in rows):
        raise ValueError('Invalid action distribution')
    diagnostic=reply.get('river_search',{})
    reason=str(diagnostic.get('reason',''))
    if diagnostic.get('worker_error') or reason.startswith(('AssertionError:','TypeError:','KeyError:','RuntimeError:',
        'FloatingPointError:','OverflowError:','MemoryError:','ZeroDivisionError:')) or 'nonfinite' in reason.lower():
        raise RuntimeError('Unexpected search failure: '+reason)
    return reply


def _check_time(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Fixed panel budget exhausted; no retry')


def select_cases(config,engine,baseline,opponent,deadline,progress,on_case=None):
    """No outcomes or search policies participate in the eligibility selection."""
    cases=[];counts=Counter();attempted_hands=0
    for offset in range(config['max_deal_blocks']):
        deal_seed=config['deal_seed_start']+offset
        for model_seed in config['model_seeds']:
            if counts[model_seed]>=config['cases_per_model']:continue
            for hero in config['hero_seats']:
                _check_time(deadline);attempted_hands+=1
                state=engine.call(dict(op='reset',seed=deal_seed,button=config['button']))
                prefix=[]
                for _ in range(513):
                    _check_time(deadline)
                    if state['terminal']:break
                    if eligible(state,hero):
                        public=PublicState.from_bridge(state)
                        case=dict(id=f'model-{model_seed}-block-{offset}-hero-{hero}',model_seed=model_seed,
                            deal_block=offset,deal_seed=deal_seed,hero=hero,button=config['button'],
                            prefix=prefix,root_observation=public.to_observation(),
                            root_public_state=public.public_only().to_json(),hand_seed=hand_seed(deal_seed,hero,model_seed))
                        cases.append(case);counts[model_seed]+=1
                        if on_case is not None:on_case(cases)
                        progress('selecting',cases=len(cases),selected_per_model=dict(counts),
                                 scanned_deal_blocks=offset+1,baseline_hands_scanned=attempted_hands)
                        break
                    actor=state['seat_to_act'];policy=baseline[model_seed] if actor==hero else opponent
                    reply=validate_reply(state,policy.decide(state,semantic_seed(deal_seed,hero,state)))
                    action=dict(reply['action'],seat=actor);prefix.append(action)
                    state=engine.call(dict(op='step',action=action))
                else:raise RuntimeError('Nonterminating baseline hand')
                # At most one selected case per(model,dealblock).
                if cases and cases[-1]['model_seed']==model_seed and cases[-1]['deal_block']==offset:break
            if all(counts[s]==config['cases_per_model'] for s in config['model_seeds']):
                return cases,dict(scanned_deal_blocks=offset+1,baseline_hands_scanned=attempted_hands,
                                  selected_per_model=dict(counts))
        progress('selecting',cases=len(cases),selected_per_model=dict(counts),
                 scanned_deal_blocks=offset+1,baseline_hands_scanned=attempted_hands)
    raise ValueError(f"Insufficient predeclared eligible coverage within{config['max_deal_blocks']} card blocks; no range extension or retry")


def replay_case(case,engine,policy,opponent,arm,deadline,stream,progress):
    if hasattr(policy,'begin_hand'):policy.begin_hand(case['hand_seed'])
    elif arm!='baseline':raise ValueError('Search policy must clear its hand-local cache via begin_hand')
    state=engine.call(dict(op='reset',seed=case['deal_seed'],button=case['button']))
    for action in case['prefix']:
        _check_time(deadline);state=engine.call(dict(op='step',action=action))
    if PublicState.from_bridge(state).to_observation()!=case['root_observation']:
        raise ValueError('Trusted seed/prefix did not reproduce selected public/hero state')
    diagnostics=[];decisions=[];steps=0
    while not state['terminal']:
        _check_time(deadline)
        if steps>=512:raise RuntimeError('Nonterminating paired continuation')
        actor=state['seat_to_act'];seed=semantic_seed(case['deal_seed'],case['hero'],state)
        started=time.monotonic()
        reply=validate_reply(state,(policy if actor==case['hero'] else opponent).decide(state,seed))
        elapsed=time.monotonic()-started;_check_time(deadline)
        action=dict(reply['action'],seat=actor)
        row=dict(case=case['id'],model_seed=case['model_seed'],deal_block=case['deal_block'],arm=arm,
                 continuation_step=steps,actor=actor,seed=seed,action=action,
                 public_action_history=state['action_log'],elapsed_seconds=elapsed)
        if actor==case['hero']:
            diagnostic=dict(reply.get('river_search',{}));diagnostic['measured_total_seconds']=elapsed
            diagnostics.append(diagnostic);row['river_search']=diagnostic
            decisions.append(dict(seed=seed,public_action_history=state['action_log'],action=action))
        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
        state=engine.call(dict(op='step',action=action));steps+=1
        progress('evaluating',active_case=case['id'],arm=arm,continuation_steps=steps,
                 hero_decisions_completed=len(diagnostics))
    utilities=state.get('utilities')
    if not isinstance(utilities,list) or len(utilities)!=6 or any(type(v) is not int for v in utilities) or sum(utilities)!=0:
        raise ValueError('Invalid terminal utilities or chip conservation')
    return dict(case=case['id'],model_seed=case['model_seed'],deal_block=case['deal_block'],
                hero=case['hero'],arm=arm,chips=utilities[case['hero']],big_blind_chips=100,
                utilities=utilities,continuation_steps=steps,hero_decisions=decisions,diagnostics=diagnostics)


def percentile(values,q):
    if not values:return None
    values=sorted(values);position=(len(values)-1)*q;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def clustered_interval(block_values):
    values=[statistics.mean(v) for _,v in sorted(block_values.items())]
    n=len(values)
    if not n:return dict(mean_bb=None,lower95_bb=None,upper95_bb=None,independent_deal_blocks=0)
    mean=statistics.mean(values)
    if n==1:return dict(mean_bb=mean,lower95_bb=None,upper95_bb=None,independent_deal_blocks=1)
    if n-1 not in T95:raise ValueError('Panel unexpectedly exceeded12 independent selected cases')
    half=T95[n-1]*statistics.stdev(values)/math.sqrt(n)
    return dict(mean_bb=mean,lower95_bb=mean-half,upper95_bb=mean+half,independent_deal_blocks=n)


def summarize(cases,records):
    if not cases:raise ValueError('Cannot summarize an empty conditional panel')
    expected={(c['id'],arm) for c in cases for arm in ARM_NAMES}
    keyed={(row['case'],row['arm']):row for row in records}
    if len(keyed)!=len(records) or set(keyed)!=expected:raise ValueError('Missing or repeated paired continuation')
    comparisons={};arm_stats={}
    for arm in ARM_NAMES:
        selected=[keyed[c['id'],arm] for c in cases];diagnostics=[d for row in selected for d in row['diagnostics']]
        elapsed=[d['measured_total_seconds'] for d in diagnostics]
        reasons=Counter(str(d.get('reason','baseline')) for d in diagnostics)
        arm_stats[arm]=dict(cases=len(selected),hero_decisions=len(diagnostics),
            search_attempted=sum(bool(d.get('attempted')) for d in diagnostics),
            search_used=sum(bool(d.get('used')) for d in diagnostics),
            fallback_decisions=sum(not bool(d.get('used')) for d in diagnostics) if arm!='baseline' else 0,
            timeouts=sum('deadline' in str(d.get('reason','')) for d in diagnostics),
            outside_scope=sum(d.get('reason')=='outside_two_active_player_river' for d in diagnostics),
            reasons=dict(reasons),elapsed_p50_seconds=percentile(elapsed,.5),
            elapsed_p95_seconds=percentile(elapsed,.95),elapsed_p99_seconds=percentile(elapsed,.99),
            cfr_iterations=sum(int(d.get('solver_iterations',0)) for d in diagnostics))
        if arm=='baseline':continue
        blocks=defaultdict(list);per_case=[]
        for case in cases:
            delta=(keyed[case['id'],arm]['chips']-keyed[case['id'],'baseline']['chips'])/100
            blocks[case['deal_block']].append(delta)
            per_case.append(dict(case=case['id'],model_seed=case['model_seed'],deal_block=case['deal_block'],
                                 delta_chips=round(delta*100),delta_bb=delta))
        comparisons[arm+'-minus-baseline']=dict(**clustered_interval(blocks),
            case_weighted_mean_bb=statistics.mean(p['delta_bb'] for p in per_case),
            per_case=per_case,block_mean_deltas_bb={str(k):statistics.mean(v) for k,v in blocks.items()})
    return dict(complete=True,scope='Paired conditional river continuation panel only',
        full_game_strength_proven=False,full_game_bb_per_100_reported=False,model_promotion=False,
        selected_cases=len(cases),model_seeds=sorted({c['model_seed'] for c in cases}),
        independent_deal_blocks=len({c['deal_block'] for c in cases}),completed_arm_continuations=len(records),
        comparisons=comparisons,arms=arm_stats,
        statistical_unit='Unique original dealblock; selectedmodels/heroseats within a dealblock are repeated observations',
        interval_scope='Descriptive Student-t95% intervals on selected dealblock mean conditional BB shifts; small non-generalizing panel',
        limitations=['Cases condition on reaching an eligible river under baseline policies, not complete six-player hands',
            'First-four-per-model selection and common cards do not create12 independent observations',
            'Supplied blueprint history marginals and omitted folded-card removal remain approximations',
            'No conclusion of generally stronger play, professional strength, or authorization to deploy'])


def run(config_path,output):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text());validate_config(config);check_frozen(config)
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'artifacts'):raise ValueError('Pilot data must stay in private artifacts')
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'status.json').exists():raise ValueError('Existing execution may not be overwritten or automatically retried')
    started=time.monotonic();deadline=started+config['max_seconds'];engine=None
    pid=os.getpid();stat=Path(f'/proc/{pid}/stat').read_text();token=stat[stat.rfind(')')+2:].split()[19]
    def progress(phase,**fields):
        atomic_json(output/'status.json',dict(status='running',phase=phase,observed_utc=utc(),
            pid=pid,process_start=token,elapsed_seconds=time.monotonic()-started,
            max_seconds=config['max_seconds'],scope='conditional_river_panel_only',**fields))
    def alarm(signum,frame):raise TimeoutError('Fixed panel execution budget exhausted; no retry')
    old=signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,config['max_seconds'])
    try:
        atomic_json(output/'config.json',config)
        atomic_json(output/'provenance.json',dict(format=FORMAT,created_utc=utc(),config_source=str(config_path),
            config_source_sha256=digest(config_path),frozen_hashes=config['expected_hashes'],
            selection='First4eligible permodel,≤1case per(model,dealblock), baselineonly, fixedscan≤128',
            semantic_rng='SHA256(deal_seed,hero,actor,complete public action history); shared across arms',
            hidden_deck_access=False,model_promotion=False,training=False,automatic_retry=False))
        progress('preparing')
        opponent=Fullhouse(config.get('public_opponent_model'))
        baseline={row['seed']:FullhouseV4(row['model'],row['source']) for row in config['models']}
        search={row['seed']:{f'search-{seconds}s':RiverSearchPolicy(row['model'],row['source'],
            seconds=seconds,opponent_policy=opponent) for seconds in config['search_seconds']} for row in config['models']}
        engine=Bridge()
        def save_partial(cases):
            atomic_json(output/'cases.json',dict(complete=False,cases=cases,
                scope='conditional_river_panel_only',automatic_retry=False))
        cases,coverage=select_cases(config,engine,baseline,opponent,deadline,progress,on_case=save_partial)
        assert len(cases)==12
        atomic_json(output/'cases.json',dict(cases=cases,coverage=coverage,scope='conditional_river_panel_only'))
        check_frozen(config);records=[]
        with gzip.open(output/'private-hands.jsonl.gz','wt',compresslevel=1) as stream:
            for case in cases:
                for arm in ARM_NAMES:
                    check_frozen(config);policy=baseline[case['model_seed']] if arm=='baseline' else search[case['model_seed']][arm]
                    row=replay_case(case,engine,policy,opponent,arm,deadline,stream,progress);records.append(row)
                    atomic_json(output/'continuations'/f"{case['id']}-{arm}.json",row)
                    progress('evaluating',completed_arm_continuations=len(records),planned_arm_continuations=48)
        check_frozen(config);_check_time(deadline)
        result=summarize(cases,records);result.update(completed_utc=utc(),elapsed_seconds=time.monotonic()-started,
            config_sha256=digest(output/'config.json'),source_provenance_verified=True,
            private_hands_sha256=digest(output/'private-hands.jsonl.gz'),cases_sha256=digest(output/'cases.json'))
        atomic_json(output/'results.json',result)
        atomic_json(output/'status.json',dict(status='complete',phase='complete',observed_utc=utc(),pid=pid,
            process_start=token,elapsed_seconds=time.monotonic()-started,selected_cases=12,
            completed_arm_continuations=48,independent_deal_blocks=result['independent_deal_blocks'],
            scope='conditional_river_panel_only',full_game_strength_proven=False,results_sha256=digest(output/'results.json')))
        return result
    except BaseException as error:
        atomic_json(output/'status.json',dict(status='interrupted' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',
            phase='stopped',observed_utc=utc(),pid=pid,process_start=token,elapsed_seconds=time.monotonic()-started,
            error_type=type(error).__name__,error=str(error),automatic_retry=False,scope='conditional_river_panel_only'))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if engine is not None:engine.close()
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.config,args.output)
    print(json.dumps({k:result[k] for k in ['complete','selected_cases','independent_deal_blocks','completed_arm_continuations','scope']},allow_nan=False))
