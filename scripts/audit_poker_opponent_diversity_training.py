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


def audit_tree_dispatch(seed):
    """Observe the unsimplified six-seat game state and each policy actually called by the routing code."""
    from poker_opponent_population import historical_inputs, load_historical, pool_traversals
    net,parent,_=connect(seed,read(CONTROL/'plan.json')['parent_policies'][str(seed)])
    from training import traverse_fast as game
    from training.game import S_STREET, S_DEALER, S_TOACT
    from bot.features import load_preflop_equity
    inputs,_=historical_inputs(SEEDS)
    hist=load_historical(net, inputs[str(seed)])
    rng,old_rng,disabled_rng=[np.random.default_rng(39214126+seed) for _ in range(3)]
    count=48
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
        check(seat==int(state[S_TOACT]) and int(state[S_DEALER]) in seen_dealers, 'Wrong acting seat')
        cursor['actor']=(seat,int(state[S_STREET]))
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
          and visits==proof['postflop_node_visits'] and np.isfinite(mixed.regrets).all(),
          'The independent tree fixture did not exercise all historic and current roles')
    return dict(seed=seed,traverser=traverser,fixture_trees=count,observed_actions=sum(calls.values()),
                dealer_positions=sorted(seen_dealers), verified_postflop_dispatches=visits,
                verified_preflop_dispatches=sum(v for (street,_),v in calls.items() if street==0),
                reference_assignment_sha256=hashlib.sha256(references.tobytes()).hexdigest(),
                inactive_pool_identical_sampled_regrets_features_and_rng=True,
                exact_inactive_fixture_rows=len(original.features))


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
        result=dict(verified_utc=utc(),seeds={s:audit_tree_dispatch(s) for s in SEEDS})
    else:
        result=verify_training()
    atomic_json(args.output,result)
