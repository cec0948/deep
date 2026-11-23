"""Verify the historical 480 control, ancestor policies and the online opponent-pool trial."""
from __future__ import annotations

import argparse
import ast
import hashlib
from collections import Counter
from unittest.mock import patch
import json
from pathlib import Path

import numpy as np
import torch

import audit_postflop_averaging as previous
import train_poker_fixed_preflop_window as canonical
from audit_poker_population import check
from compact_lightweight_backup import equal_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import collect_game_reference, load_checkpoint
from train_poker_warm_preflop import connect, weights

STUDY = previous.BASE/'opponent-diversity-20261123'
CONTROL = previous.STUDY
ARTIFACTS = ROOT/'artifacts'
BOUNDED = ARTIFACTS/'bounded-continuation/20261113T014811152621Z'
AVERAGE = ARTIFACTS/'average-baseline/20261116T084130Z'
CONTINUATION = ARTIFACTS/'average-continuation/20261117T032100Z'
EXTENSION = ARTIFACTS/'average-extension/20261117T063900Z'
SEEDS = (342,343,344)
HISTORY = (300,360,420)
CAPTURES = (486,492,498,504)
CONTROL_STAGE = '288ec2d5f9068221b5922cfeac29301f226e030734e67b15a4399748224d071e'
CONTROL_PLAN = '533a88378fe0a4b27cd98ece6d493ca62600c66ee48265fcdbc57cac379b594c'


def checkpoint(study, seed, t):
    return study/'campaign'/f'seed-{seed}'/'checkpoints'/f'iter-{t:05d}'


def historical_sources():
    transfer = {p:read(p/'plan.json') for p in (AVERAGE,CONTINUATION,EXTENSION)}
    bounded_stage = read(BOUNDED/'stage-ready.json')['checkpoints']
    stage420 = read(CONTINUATION/'stage-ready.json')['seeds']
    stage480 = read(EXTENSION/'stage-ready.json')['seeds']
    proof = {}
    for seed in SEEDS:
        key = str(seed)
        source = {}
        for t in (300,360):
            path = checkpoint(BOUNDED,seed,t)/'current.npz'
            receipt = BOUNDED/f'stage-{t}-verification.json'
            ref = next(d for d in read(receipt)['seeds'] if d['seed']==seed)
            sha = digest(path)
            check(bounded_stage[str(t)]['verification_sha256']==digest(receipt)
                  and ref['iteration']==t and Path(ref['checkpoint_directory'])==path.parent
                  and sha==bounded_stage[str(t)]['models'][key]==ref['current_sha256'],
                  'Older self-play model differs from the verified bounded run')
            source[str(t)]=sha
        for parent,child in ((BOUNDED,AVERAGE),(AVERAGE,CONTINUATION),(CONTINUATION,EXTENSION)):
            parent_ref=transfer[child]['parent_paths'][key]
            parent_dir=Path(parent_ref['checkpoint'])
            t=int(parent_dir.name.split('-')[-1])
            check(parent_dir==checkpoint(parent,seed,t), 'Checkpoint belongs to a different seed or study')
            ready=next(item for item in transfer[child]['connection_verification'] if item['seed']==seed)
            loaded_sha = ready['parent_policy_sha256'] if child==AVERAGE else ready['current_policy_sha256']
            check(parent_ref['files']['current.npz']==loaded_sha==digest(parent_dir/'current.npz')
                  ==digest(checkpoint(child,seed,t)/'current.npz'), 'Broken lineage at training study boundary')
            if child==AVERAGE:
                check(t==360 and ready['weights_replay_root_rng_byte_identical'], 'Bad 360 transfer')
            else:
                check(ready['iteration']==t and all(ready['checks'].values()), 'Bad continuation transfer')
        path = checkpoint(CONTINUATION,seed,420)/'current.npz'
        source['420']=stage420[key]['files']['current.npz']
        check(digest(path)==source['420']==digest(checkpoint(EXTENSION,seed,420)/'current.npz') and
              stage480[key]['files']['current.npz']==digest(checkpoint(EXTENSION,seed,480)/'current.npz'),
              '420 or 480 is not the expected continuing lineage')
        check(len(set(source.values()))==len(HISTORY), 'Repeated historical opponent')
        proof[key]=source
    return proof


def revalidate_control():
    verify_files({CONTROL/'stage-ready.json':CONTROL_STAGE,CONTROL/'plan.json':CONTROL_PLAN})
    # Historical JSON object keys (e.g. iteration numbers) are strings on disk.
    result=json.loads(json.dumps(previous.training(),allow_nan=False))
    original=read(CONTROL/'training-audit.json')
    check({k:v for k,v in result.items() if k!='verified_utc'}==
          {k:v for k,v in original.items() if k!='verified_utc'}, 'Original control audit failed to reproduce')
    return dict(verified_utc=utc(),control_stage_sha256=CONTROL_STAGE,control_plan_sha256=CONTROL_PLAN,
                reproduced_control_training_sha256=digest(CONTROL/'training-audit.json'),
                historical_postflop_sha256=historical_sources(),control_total_regret_rows=result['total_new_regret_rows'],
                common_parent=read(CONTROL/'plan.json')['parent_policies'])


def first_control(seed):
    """Reproduce one saved update in memory, using the new traversal with all opponents on-policy."""
    from poker_opponent_population import pool_update
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan=read(CONTROL/'plan.json')
    ref=plan['parent_policies'][str(seed)]
    networks,parent,proof=connect(seed,ref)
    from bot.features import load_preflop_equity
    from training import r1_support as r1

    cfg={**parent['config'],'learning_rate':.00025}
    buffers,root,strategy,rng,reference_rng=canonical.restore_replay(parent)
    origin=read(CONTROL/'slow504'/f'seed-{seed}'/'start-verification.json')
    check(all(proof[k]==origin[k] for k in proof), 'Control start not reproduced')
    model_in=weights(networks)
    equity=load_preflop_equity()
    r1.WEIGHT_HORIZON=cfg['weight_horizon']
    r1.AVERAGE_SAMPLER='importance'
    before=r1.pack_rng(rng)
    observed=collect_game_reference(networks,strategy,reference_rng,481,cfg['reference_hands'],equity)
    check(equal_state(before,r1.pack_rng(rng)), 'Auxiliary reference draws affect learner RNG')
    row=pool_update(networks,buffers,root,rng,481,cfg,equity,seed,{},current_only=True)
    expected=json.loads((CONTROL/'slow504'/f'seed-{seed}'/'diagnostics.jsonl').read_text().splitlines()[0])
    fields=['new_regret_rows','root_rows','traverser','traversals','fit_rows_by_street','fit_draws_by_street',
            'general_steps','fit_draws','main_loss','opening_steps','preflop_steps']
    check(all(row[k]==expected[k] for k in fields) and observed==expected['reference_collection']
          and model_in==expected['input_weights'] and weights(networks)==expected['output_weights'],
          'New traversal with on-policy assignment did not reproduce the historic control')
    return dict(seed=seed,round=481,weights=weights(networks),input_weights=model_in,
                verified_fields=fields,validated_reference=observed,new_regret_rows=row['new_regret_rows'],
                identical_main_loss=row['main_loss'],postflop_gradient_steps=row['general_steps'])


def reference_assignments(seed, iteration, count=256):
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([0x504f4f4c,seed,iteration])))
    values = generator.integers(0,6,size=(count,6))
    out = np.zeros((count,6), dtype='<i2')
    for raw, policy in zip((3,4,5),HISTORY,strict=True):
        out[values==raw] = policy
    out[:,(iteration-1)%6] = -1
    return out


class BatchLedger:
    def add_batch(self, features, regrets, iteration):
        check(not hasattr(self,'features'), 'Collector wrote multiple unrelated batches')
        self.features=features
        self.regrets=regrets


def audit_tree_dispatch(seed, round_id=None):
    """Observe the unsimplified six-seat game state and each policy actually called by the routing code."""
    from poker_opponent_population import historical_inputs, load_historical, pool_traversals
    net,parent,_=connect(seed,read(CONTROL/'plan.json')['parent_policies'][str(seed)])
    from training import traverse_fast as game
    from training.game import S_STREET, S_DEALER, S_TOACT, S_FOLDED, S_STACKS
    from bot.features import load_preflop_equity
    inputs,_=historical_inputs(SEEDS)
    hist=load_historical(net, inputs[str(seed)])
    rng,old_rng,disabled_rng=[np.random.default_rng(39214126+seed) for _ in range(3)]
    count=48
    if round_id is None:
        round_id=481+seed-SEEDS[0]
    traverser=(round_id-1)%6
    equity=load_preflop_equity()
    options=dict(opponent_exploration=0.,strategy_buffer=None,randomize_stacks=False)
    original,disabled=BatchLedger(),BatchLedger()
    score1=game.run_traversals_fast(count,6,traverser,net,original,round_id,old_rng,equity,**options)
    score2=pool_traversals(seed,{},True,{},count,6,traverser,net,disabled,round_id,disabled_rng,equity,**options)
    check(score1==score2 and equal_state(old_rng.bit_generator.state,disabled_rng.bit_generator.state)
          and np.array_equal(original.features,disabled.features) and np.array_equal(original.regrets,disabled.regrets),
          'Dose-zero collection differs in sampled features, regrets, return, or terminal RNG')

    references=reference_assignments(seed,round_id,count)
    calls=Counter()
    seen_dealers=set()
    players_postflop=set()
    unequal_stacks_postflop=0
    cursor=dict(tree=-1,actor=None)
    class RecordedPolicy:
        def __init__(self, identity, owner):
            self.identity=identity
            self.owner=owner
        def strategy(self, x, legal):
            actor,street=cursor['actor']
            declared=int(references[cursor['tree'], actor])
            expected=0 if street==0 or declared<0 else declared
            check(self.identity==expected,'Traverser, street or per-tree absolute seat assignment changed')
            calls[street,declared] += 1
            return self.owner.strategy(x,legal)

    native_new,native_encode,native_factory=game.new_hand,game._encode_fast,game.NumpyPolicy
    def start(n, dealer, generator, stacks):
        cursor['tree']+=1
        seen_dealers.add(dealer)
        return native_new(n,dealer,generator,stacks)
    def observe(state, holes, deck, seat, mask):
        nonlocal unequal_stacks_postflop
        check(seat==int(state[S_TOACT]) and int(state[S_DEALER]) in seen_dealers, 'Wrong acting seat')
        cursor['actor']=(seat,int(state[S_STREET]))
        if state[S_STREET]>0:
            live=np.asarray(state[S_FOLDED:S_FOLDED+6])==0
            players_postflop.add(int(live.sum()))
            if len(set(state[S_STACKS:S_STACKS+6][live]))>1:
                unequal_stacks_postflop+=1
        return native_encode(state,holes,deck,seat,mask)
    tagged={key:RecordedPolicy(key,policy) for key,policy in hist.items()}
    proof={}
    mixed=BatchLedger()
    with patch.object(game,'new_hand',start),patch.object(game,'_encode_fast',observe), \
            patch.object(game,'NumpyPolicy',lambda n:RecordedPolicy(0,native_factory(n))):
        pool_traversals(seed,tagged,False,proof,count,6,traverser,net,mixed,round_id,rng,equity,**options)
    visits={str(code):sum(v for (street,role),v in calls.items() if role==code and street!=0)
            for code in (-1,0,*HISTORY)}
    check(len(seen_dealers)==6 and cursor['tree']+1==count and min(visits.values())>0
          and visits==proof['postflop_node_visits'] and np.isfinite(mixed.regrets).all()
          and 2 in players_postflop and max(players_postflop)>=4 and unequal_stacks_postflop>0,
          'The independent tree fixture did not exercise all historic and current roles')
    return dict(seed=seed,traverser=traverser,fixture_trees=count,observed_actions=sum(calls.values()),
                dealer_positions=sorted(seen_dealers), verified_postflop_dispatches=visits,
                postflop_players_observed=sorted(players_postflop),
                postflop_decisions_with_unequal_live_stacks=unequal_stacks_postflop,
                verified_preflop_dispatches=sum(v for (street,_),v in calls.items() if street==0),
                reference_assignment_sha256=hashlib.sha256(references.tobytes()).hexdigest(),
                inactive_pool_identical_sampled_regrets_features_and_rng=True,
                exact_inactive_fixture_rows=len(original.features))


def verify_seed(seed, plan, stage, mapping):
    key=str(seed)
    worker=STUDY/'pool504'/f'seed-{seed}'
    control=CONTROL/'slow504'/f'seed-{seed}'
    cap=STUDY/'snapshots'/f'seed-{seed}'
    start,ready,pointer=(read(worker/name) for name in ('start-verification.json','ready.json','current.json'))
    rows=[json.loads(line) for line in (worker/'diagnostics.jsonl').read_text().splitlines()]
    control_rows=[json.loads(line) for line in (control/'diagnostics.jsonl').read_text().splitlines()]
    check(len(rows)==24 and [r['iteration'] for r in rows]==list(range(481,505))
          and start==read(control/'start-verification.json') and start['iteration']==480
          and ready==stage['arms']['pool504'][key] and ready['files']==pointer['files'] and pointer['iteration']==504
          and read(worker/'status.json')['status']=='complete', 'Incomplete or mismatched training bounds')
    verify_files({worker/'start-verification.json':ready['start_sha256'],
                  worker/'diagnostics.jsonl':ready['diagnostics_sha256'],
                  **{Path(ready['checkpoint'])/name:sha for name,sha in ready['files'].items()}})
    parent_folder=Path(plan['campaign'])/f'seed-{seed}'
    parent,parent_ck=load_checkpoint(parent_folder)
    final,end=load_checkpoint(worker)
    old,end_control=load_checkpoint(control)
    config=final['config']
    diff={k for k in set(config)|set(old['config']) if config.get(k)!=old['config'].get(k)}
    check(diff=={'training_arm','opponent_assignment','historical_policies'} and config['training_arm']=='pool504'
          and config['opponent_assignment']=='independent-seat-per-traversal-v1'
          and config['historical_policies']==plan['historical_policies'][key]
          and parent['seed']==final['seed']==old['seed']==seed and parent['iteration']==480
          and final['iteration']==old['iteration']==504 and final['source_manifest']==old['source_manifest']==parent['source_manifest']
          and read(parent_folder/'current.json')['files']==start['checkpoint_files'],
          'Treatment and matched control have different starting assets, data, or training configuration')
    del old
    refs=stage['policies']
    parent_policy,endpoint,mix=(read(previous.checked_ref(refs[k][key])) for k in ('specialist480','pool504','poolMixture504'))
    pre,post=(previous.checked_ref(parent_policy[k]) for k in ('preflop','postflop'))
    pre_sha=digest(pre)
    prior={'preflop':previous.net_fingerprint(pre,mapping),'general':previous.net_fingerprint(post,mapping)}
    check(post==parent_ck/'current.npz' and prior==start['weights'] and start['replay_root_reference_and_rng_verified']
          and refs['pool504'][key]==ready['policy'] and refs['poolMixture504'][key]==ready['mixture']
          and Path(endpoint['postflop']['path'])==end/'current.npz'
          and endpoint['preflop']['sha256']==mix['preflop']['sha256']==pre_sha
          and ready['policy']==refs['pool504'][key] and digest(cap/mix['preflop']['file'])==pre_sha,
          'Mismatched preflop, checkpoint or policy metadata')
    verify_export(previous.checked_ref(endpoint['preflop']),final['specialist_net'],mapping)
    verify_export(previous.checked_ref(endpoint['postflop']),final['net'],mapping)
    verify_export(post,parent['net'],mapping)
    captured={r['iteration']:r for r in mix['postflop']}
    snapshot_plan=read(cap/'plan.json')
    ctrlmix=read(previous.checked_ref(read(CONTROL/'stage-ready.json')['arms']['slow504'][key]['mixture']))
    check(mix['format']==ctrlmix['format'] and mix['selection']==ctrlmix['selection']=='Equal probability, one private draw per hand'
          and list(captured)==list(CAPTURES)==snapshot_plan['iterations'] and mix['seed']==seed and snapshot_plan['worker']==str(worker)
          and mix['plan_sha256']==digest(cap/'plan.json') and snapshot_plan['parent_policy_sha256']==refs['specialist480'][key]['sha256'],
          'Wrong four-capture deployment or snapshot stream')
    visits,selections=Counter(),Counter()
    by_seat=[0]*6
    capture_hashes={}
    budget_fields=('seed','iteration','traverser','traversals','root_rows','general_steps','preflop_steps','opening_steps',
                   'learning_rate','fit_draws','frozen_preflop_probes','buffer_rows')
    for row,reference in zip(rows,control_rows,strict=True):
        t=row['iteration']
        seat=(t-1)%6
        weights_out=row['output_weights']
        check(all(row[k]==reference[k] for k in budget_fields)
              and row['input_weights']==prior and weights_out['preflop']==start['weights']['preflop']
              and weights_out['general']!=prior['general'] and row['frozen_preflop_max_probability_error']==0
              and row['new_regret_rows']>0 and np.isfinite(row['main_loss'])
              and min(row['fit_rows_by_street'])>0 and min(row['fit_draws_by_street'])>0
              and sum(row['fit_draws_by_street'])==row['fit_draws']==768*512
              and row['reference_collection']['hands']==512 and row['reference_collection']['rows']==100000
              and row['traverser']==seat and row['root_rows']==676 and row['traversals']==256,
              f'Input, output, fit or sampling budget mismatch on {seed}/{t}')
        a=reference_assignments(seed,t)
        data=row['opponent_population']
        counts={str(x):int((a==x).sum()) for x in (0,*HISTORY)}
        check(data['assignment_sha256']==hashlib.sha256(a.tobytes()).hexdigest()
              and counts==data['opponent_seat_assignments'] and sum(counts.values())==5*256
              and set(data['postflop_node_visits'])=={'-1',*counts} and min(data['postflop_node_visits'].values())>0,
              f'Traverser or immutable seat assignment mismatch on {seed}/{t}')
        visits.update(data['postflop_node_visits'])
        selections.update(counts)
        by_seat[seat]+=row['new_regret_rows']
        if t in captured:
            item=captured[t]
            path=cap/item['file']
            receipt=read(path.parent/'receipt.json')
            check(path==cap/f'iter-{t:05d}/current.npz' and digest(path)==receipt['export_sha256']==item['sha256']
                  and path.stat().st_size==receipt['export_bytes'] and previous.net_fingerprint(path,mapping)==weights_out['general']
                  and receipt['iteration']==t and receipt['start_sha256']==ready['start_sha256']
                  and receipt['plan_sha256']==mix['plan_sha256'], 'Corrupt or unbound capture')
            capture_hashes[t]=item['sha256']
        prior=weights_out
    check(len(set(capture_hashes.values()))==4 and capture_hashes[504]==digest(end/'current.npz')!=digest(end_control/'current.npz')
          and rows[0]['reference_collection']==control_rows[0]['reference_collection']
          and rows[0]['output_weights']['general']!=control_rows[0]['output_weights']['general'],
          'Incorrect match, capture or failure to diverge on the new training data')
    for seat,(begin,after) in enumerate(zip(parent['buffers'],final['buffers'],strict=True)):
        check(after['count']==100000 and after['total_seen']-begin['total_seen']==by_seat[seat]
              and np.array_equal(np.unique(after['iterations'][after['iterations']>480]),
                                 [t for t in range(481,505) if (t-1)%6==seat]),
              'Reservoir is missing or misattributing new game data')
    check(final['strategy_replay']['size']==100000 and final['strategy_replay']['total_seen']==rows[-1]['reference_collection']['total_seen']
          and np.sum(final['root_replay']['count'])-np.sum(parent['root_replay']['count'])==24*676,
          'Replay bookkeeping differs from verified source and the 24 online updates')
    return dict(seed=seed,rounds=24,traversals=24*256,postflop_adam_updates=24*768,
                postflop_node_visits=dict(visits),per_tree_seat_selections=dict(selections),
                new_regret_rows_by_seat=by_seat,new_regret_rows=sum(by_seat),replay_rows_per_seat=100000,
                parent_480=refs['specialist480'][key]['sha256'],unchanged_preflop_sha256=pre_sha,
                preflop_probes_each_round=start['frozen_preflop_probes'],snapshot_export_sha256=capture_hashes,
                final_weights=prior,actual_control_treatment_training_config_differences=sorted(diff))


def verify_training():
    config_path=ROOT/'configs/poker-opponent-diversity-20261123.json'
    reg,plan,stage=(read(STUDY/name) for name in ('registration.json','training-plan.json','stage-ready.json'))
    cfg=read(config_path)
    verify_files({config_path:reg['config_sha256'],STUDY/'training-plan.json':read(STUDY/'training-bound.json')['plan_sha256'],
                  CONTROL/'stage-ready.json':CONTROL_STAGE,CONTROL/'plan.json':CONTROL_PLAN})
    verify_files(plan['frozen_files'])
    verify_files(reg['frozen_files'])
    lineage=read(STUDY/'control-lineage-audit.json')
    proof_files=[STUDY/n for n in ('control-lineage-audit.json','control-round-audit.json','dispatch-audit.json')]
    check(stage['training_complete'] and stage['iteration']==504 and plan['parent_policies']==lineage['common_parent']
          and stage['policies']['specialist480']==read(CONTROL/'stage-ready.json')['policies']['specialist480']==plan['parent_policies']
          and reg['control_training_plan_sha256']==lineage['control_plan_sha256']==CONTROL_PLAN
          and tuple(plan['seeds'])==SEEDS and tuple(plan['capture_iterations'])==CAPTURES
          and (cfg['end'],cfg['start'],cfg['replay_capacity_per_seat'],cfg['batch_size'],cfg['learning_rate'])==(504,480,100000,512,.00025)
          and cfg['treatment_opponents']['training_external_opponents']==[] and cfg['treatment_opponents']['current_probability']==.5,
          'Registered starts, control, opponent history, or completion record differ')
    for seed,refs in plan['historical_policies'].items():
        check({t:r['sha256'] for t,r in refs.items()}==lineage['historical_postflop_sha256'][seed]
              and set(refs)==set(map(str,HISTORY)), 'A foreign/future weight was introduced')
    code=ast.parse((Path(plan['campaign'])/'tools/compact_regret.py').read_text())
    mapping=next(ast.literal_eval(node.value) for node in code.body if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='ARRAY_MAP' for t in node.targets))
    seeds={str(s):verify_seed(s,plan,stage,mapping) for s in SEEDS}
    return dict(verified_utc=utc(),training_complete=True,stage_sha256=digest(STUDY/'stage-ready.json'),
                training_plan_sha256=digest(STUDY/'training-plan.json'),matched_control_stage_sha256=CONTROL_STAGE,
                independent_proofs={p.name:digest(p) for p in proof_files},seeds=seeds,
                registered_new_rounds_per_seed=24,total_online_traversals=sum(v['traversals'] for v in seeds.values()),
                total_new_regret_rows=sum(v['new_regret_rows'] for v in seeds.values()),
                total_postflop_adam_updates=sum(v['postflop_adam_updates'] for v in seeds.values()),
                total_opponent_seat_assignments=sum(sum(v['per_tree_seat_selections'].values()) for v in seeds.values()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=['ancestry','control-round','dispatch','training'])
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    if args.phase=='ancestry':
        result=revalidate_control()
    elif args.phase=='control-round':
        result=dict(verified_utc=utc(),seeds={s:first_control(s) for s in SEEDS},
                    retained_new_models=0,
                    scope='One in-memory frozen-control parity update for each seed. No weights enter either study arm.')
    elif args.phase=='dispatch':
        result=dict(verified_utc=utc(),seeds={s:audit_tree_dispatch(s) for s in SEEDS},
                    other_three_traversers={s:audit_tree_dispatch(s,484+s-SEEDS[0]) for s in SEEDS})
    else:
        result=verify_training()
    atomic_json(args.output,result)


if __name__=='__main__':
    main()
