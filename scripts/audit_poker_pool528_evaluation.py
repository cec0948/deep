"""Rebuild the registered external 528 comparisons from complete, paired hand ledgers."""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
from functools import lru_cache
import gzip
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import time

import numpy as np

from audit_poker_population import check, close_to, expected_identities
from audit_postflop_averaging import replay, terminal_auditor, verify_terminals
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_average_runtime import WindowAveragePolicy
from poker_preflop_specialist import evaluator_policy

BASE = ROOT/'artifacts/preflop-followup/480-v1'
STUDY = BASE/'pool-extension-528-20261126'
MAIN = STUDY/'main'
OLD = BASE/'opponent-diversity-20261123'
SOURCE_SHA = '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9'
DOSE = BASE/'preflop-dose/evaluation-plan.json'
DOSE_SHA = '642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'
CONFIG = ROOT/'configs/poker-pool-extension-528-20261126.json'
CONFIG_SHA = 'c96fa5f97aecb52071f8e61e68ec9fe34a00d16de6f46b71a9902c7e91f7ca01'
SEEDS = (342,343,344)
TABLES = {'public':(2060100000,4000),'mixed':(2060200000,1000),'noregrets':(2060200000,1000)}
DEPLOYED = ('current480','mixture504','mixture528','current504','current528')
AUX = ('average480','average504','average528')
SLOTS = {'mixture504':(486,492,498,504),'mixture528':(510,516,522,528)}
EARLY = {'current480','mixture504','current504','average480','average504'}


def entries():
    for field in TABLES:
        for seed in SEEDS:
            for label in DEPLOYED + (AUX if field=='public' else ()):
                yield f'{field}-seed-{seed}-{label}', field, seed, label


def slot_number(private_seed):
    entropy = b'scheduled-postflop-choice-v1\0' + private_seed.to_bytes(8,'big')
    return random.Random(entropy).randrange(4)


class SampleStreetPolicy:
    def __init__(self, preflop, postflop):
        self.preflop, self.postflop = preflop, postflop
        self.index = None

    def begin_hand(self, seed):
        self.index = slot_number(seed) if len(self.postflop)==4 else 0

    def decide(self, state, seed):
        return (self.preflop if state['street']=='preflop' else self.postflop[self.index]).decide(state,seed)


class SampleAverage(WindowAveragePolicy):
    """Use the frozen encoder/translator, but rebuild the masked network forward pass."""

    def _strategy(self, state):
        features, mask = self.features.encode_state_dict(state,self.lookup._equity_tables)
        legal = np.flatnonzero(mask)
        check(np.array_equal(features[46:51],mask) and len(legal)>0,'Incorrect legal feature mask')
        y = np.asarray(features,dtype=np.float64)
        for weight,bias in self.layers[:-1]:
            y = np.maximum(weight.astype(np.float64) @ y + bias,0)
        weight,bias = self.layers[-1]
        logits = (weight.astype(np.float64) @ y + bias)[legal]
        relative = np.exp(logits - max(logits))
        result = np.zeros(len(mask),dtype=np.float32)
        result[legal] = relative/relative.sum()
        check(np.allclose(result,super()._strategy(state),rtol=0,atol=2e-5),'Auxiliary forward pass disagrees')
        return result


@lru_cache(maxsize=12)
def single_net(path, source):
    return evaluator_policy(path,source)


def expected_ref(label, seed):
    stages = {'current480':'specialist480','current504':'pool504','mixture504':'poolMixture504'}
    if label in stages:
        return read(OLD/'stage-ready.json')['policies'][stages[label]][str(seed)]
    if label in ('average480','average504'):
        campaign = Path(read(OLD/'training-plan.json')['campaign']) if label=='average480' else OLD/'pool504'
        worker = campaign/f'seed-{seed}'
        pointer = read(worker/'current.json')
        stage = int(label[-3:])
        check(pointer['iteration']==stage and pointer['directory']==f'iter-{stage:05d}','Wrong auxiliary stage')
        return dict(path=str(worker/'checkpoints'/pointer['directory']/'average.npz'),sha256=pointer['files']['average.npz'])
    stage = read(STUDY/'stage-ready.json')
    check(stage['iteration']==528 and stage['training_complete'],'528 export not complete')
    return stage['policies'][label][str(seed)]


def sample_policy(label, seed, source, ref):
    check(ref == expected_ref(label,seed),'Different source for sampled policy')
    path = Path(ref['path'])
    verify_files({path:ref['sha256']})
    if label in AUX:
        model = SampleAverage(path,source)
        check(model.metadata['seed']==seed and model.metadata['averaging_window']==[361,int(label[-3:])]
              and model.metadata['historical_mixture_available'] is False,'Wrong all-street average')
        return model
    spec = read(path)
    frozen = read(expected_ref('current480',seed)['path'])['preflop']['sha256']
    check(spec['preflop']['sha256']==frozen,'Changed frozen preflop')
    if label in SLOTS:
        verify_files({path.parent/'plan.json':spec['plan_sha256']})
        check(spec['seed']==seed and tuple(r['iteration'] for r in spec['postflop'])==SLOTS[label]
              and spec['postflop'][-1]['sha256']==read(expected_ref('current'+label[-3:],seed)['path'])['postflop']['sha256'],
              'Different private snapshot schedule or different endpoint')
        parts = [spec['preflop'],*spec['postflop']]
        paths = [path.parent/part['file'] for part in parts]
    else:
        parts = [spec['preflop'],spec['postflop']]
        paths = [Path(part['path']) for part in parts]
    verify_files({path:part['sha256'] for path,part in zip(paths,parts,strict=True)})
    preflop, *postflop = (single_net(str(p),source) for p in paths)
    return SampleStreetPolicy(preflop,postflop)


def load_plan(stem):
    path = MAIN/f'{stem}-plan.json'
    sha = read(MAIN/f'{stem}-bound.json')['sha256']
    verify_files({path:sha})
    plan = read(path)
    verify_files(plan['frozen_files'])
    check(plan['seeds']==list(SEEDS) and {f:tuple(r) for f,r in plan['fields'].items()}==TABLES
          and plan['planned_hands']==756000 and plan['unique_original_cards']==5000 and plan['source_stage_sha256']==SOURCE_SHA,
          'Changed external design')
    check([(j['id'],j['field'],j['seed'],j['label']) for j in plan['jobs']]==list(entries()),'Incorrect scheduled jobs')
    check(set(plan['early_ids'])=={name for name,f,s,p in entries() if f=='public' and s==342 and p in EARLY},
          'Different early reference set')
    if stem=='evaluation':
        check(plan['reference_plan_sha256']==digest(MAIN/'reference-plan.json') and
              plan['training_stage_sha256']==digest(STUDY/'stage-ready.json'),'Final binding differs')
    return plan,sha


def verify_cell(name, field, seed, label, plan, sha, parent, engine, public_bot, blueprint_hash):
    folder = MAIN/'evaluation'/name
    ref = plan['policies'][label][str(seed)]
    start,n = TABLES[field]
    result, provenance, status = (read(folder/f) for f in ('results.json','provenance.json','status.json'))
    verify_files({folder/'provenance.json':result['provenance_sha256'],folder/'hands.jsonl.gz':result['raw_hands_sha256']})
    verify_files(result['dependency_hashes'])
    identity = dict(field=field,training_seed=seed,label=label,deal_seed=start,deals=n,hands=6*n,big_blind_chips=100,
                    candidate=ref['path'],candidate_sha256=ref['sha256'],evaluation_plan_sha256=sha)
    check(all(result[k]==provenance[k]==v for k,v in identity.items())
          and result['dependency_hashes']==provenance['dependency_hashes'] and result['opponents']==provenance['opponents'],
          'Wrong policy, opponents, card range or provenance')
    check(result['complete'] and status['phase']=='complete' and status['verified_hands']==6*n and
          result['raw_hands_verified'] and result['terminal_payoffs_replayed'] and result['verification']['hands']==6*n
          and result['verification']['card_blocks']==n and
          all(result['verification'][k] for k in ('terminal_payoffs_replayed','chip_conservation_verified','aggregate_recomputed')),
          'Incomplete cell')
    ids,roles = expected_identities(dict(field=field,seed=seed),parent,False)
    check(result['opponents']==plan['opponents'][f'{field}/{seed}']==ids and
          all(plan['frozen_files'].get(k)==v for k,v in result['dependency_hashes'].items()),'Changed runtime identities')
    if 'public' in roles:
        check(public_bot.digest==ids['public'],'Wrong public runtime')
    if 'noregrets' in roles:
        check(blueprint_hash==ids['noregrets'],'Wrong blueprint runtime')
    hero = sample_policy(label,seed,plan['source'],ref)
    blocks = np.zeros(n,dtype=np.int64)
    slots,streets = Counter(),Counter()
    hands = actions = sampled = sampled_actions = 0
    raw = folder/'hands.jsonl.gz'
    with gzip.open(raw,'rt') as fp:
        for line in fp:
            row = json.loads(line)
            block,seat = divmod(hands,6)
            card = start+block
            private_seed = hand_seed(card,seat)
            check(block<n and (row['deal'],row['hero'],row['seed'],row['hand_seed'])==(block,seat,card,private_seed),
                  f'Wrong cards or seat at {name}/{hands}')
            if label in SLOTS:
                selection = slot_number(private_seed)
                check(row['snapshot_iteration']==SLOTS[label][selection],'Incorrect actual mixture snapshot')
                slots[str(selection)] += 1
            else:
                check('snapshot_iteration' not in row,'A non-mixture used a slot')
            pay = row['utilities']
            check(len(pay)==6 and all(type(c) is int for c in pay) and sum(pay)==0 and
                  type(row['chips']) is int and pay[seat]==row['chips'] and 0<len(row['actions'])==row['steps']<=512,
                  'Invalid game ledger')
            if block%389==0 or block==n-1:
                trace = replay(row,field,engine,dict(hero=hero,public=public_bot))
                streets.update(street for street,actor,_,_ in trace if actor==seat)
                sampled += 1
                sampled_actions += len(trace)
            blocks[block] += pay[seat]
            hands += 1
            actions += row['steps']
    check(hands==6*n and result['verification']['concrete_decisions_replayed']==actions and
          result['snapshot_slot_counts']==dict(slots),'Unmatched ledger counts')
    check(verify_terminals(raw)==dict(hands=hands,decisions=actions),'Native terminal replay count differs')
    means = blocks/6
    close_to(result['deal_blocks'],means,'Published original-deal values differ')
    close_to(result['bb_per_100'],means.mean(),'Published cell payoff differs')
    close_to(result['ci95_half_width'],1.96*means.std(ddof=1)/math.sqrt(n),'Published cell standard error differs')
    return means,dict(hands=hands,actions=actions,sample_hands=sampled,sample_actions=sampled_actions,
                      sampled_hero_streets=dict(streets),slot_counts=dict(slots),raw_sha256=digest(raw),
                      results_sha256=digest(folder/'results.json'),candidate_sha256=ref['sha256'])


def band(cards,z):
    center = float(np.mean(cards))
    margin = z*math.sqrt(float(np.sum((cards-center)**2))/(len(cards)*(len(cards)-1)))
    return [center-margin,center+margin]


def aggregate(matrices, records, plans):
    summary = read(MAIN/'summary.json')
    check(summary['complete'] and summary['completed_cells']==54 and summary['completed_hands']==756000
          and summary['unique_original_card_blocks']==5000 and summary['seeds']==list(SEEDS)
          and not summary['model_promoted'] and summary['secondary_pair_family']==48
          and summary['cell_result_sha256']=={k:r['results_sha256'] for k,r in records.items()}
          and summary['reference_plan_sha256']==plans['reference'][1] and summary['evaluation_plan_sha256']==plans['evaluation'][1]
          and summary['source_stage_sha256']==SOURCE_SHA and summary['training_stage_sha256']==digest(STUDY/'stage-ready.json'),
          'Summary source, design or counts differ')
    z = NormalDist().inv_cdf(.975)
    z_family = NormalDist().inv_cdf(1-.025/48)
    tested = 0
    for field in TABLES:
        labels = DEPLOYED+(AUX if field=='public' else ())
        data = {label:np.stack([matrices[f'{field}-seed-{s}-{label}'] for s in SEEDS]) for label in labels}
        check(set(data)==set(summary['absolute_bb100'][field]),'Different absolute score set')
        for label,policy in data.items():
            score = summary['absolute_bb100'][field][label]
            cards = policy.mean(axis=0)
            close_to(score['mean'],float(cards.mean()),'Different absolute score')
            close_to(score['ci95'],band(cards,z),'Different absolute interval')
        comparisons = summary['paired_comparisons'][field]
        check(len(comparisons)==math.comb(len(labels),2),'Invalid pair family')
        for before,after in combinations(labels,2):
            change = data[after]-data[before]
            cards = change.mean(axis=0)
            row = comparisons[f'{after}_minus_{before}']
            check(row['baseline']==before and row['candidate']==after,'Different comparison direction')
            close_to(row['mean'],cards.mean(),'Different paired estimate')
            close_to(row['ci95'],band(cards,z),'Different paired interval')
            close_to(row['simultaneous_ci95'],band(cards,z_family),'Different simultaneous interval')
            close_to([row['by_seed'][str(s)] for s in SEEDS],change.mean(axis=1),'Different fixed-seed deltas')
            tested += 1
    primary = summary['paired_comparisons']['public']['mixture528_minus_mixture504']
    check(tested==48 and summary['primary']==primary,'Different primary or contrast count')
    all_streets = {'preflop','flop','turn','river'}
    sample_streets = {}
    for label in DEPLOYED+AUX:
        counts = Counter()
        for name,_,_,p in entries():
            if p==label:
                counts.update(records[name]['sampled_hero_streets'])
        check(set(counts)==all_streets,'Missing betting street in independent samples for '+label)
        sample_streets[label] = dict(counts)
    receipt = dict(complete=True, verified_utc=utc(),cells=len(records),hands=sum(r['hands'] for r in records.values()),
                   unique_original_card_blocks=5000,cards_per_field={f:n for f,(_,n) in TABLES.items()},
                   all_native_terminal_replays=True,concrete_decisions_replayed=sum(r['actions'] for r in records.values()),
                   all_mixture_snapshot_ids_independently_rebuilt=True,
                   sampled_policy_hands=sum(r['sample_hands'] for r in records.values()),
                   sampled_all_actor_actions=sum(r['sample_actions'] for r in records.values()),
                   hero_action_sample_by_street=sample_streets,absolute_estimates_verified=18,paired_contrasts_verified=tested,
                   primary_public_mixture528_minus_mixture504=primary,source_stage_sha256=SOURCE_SHA,
                   training_stage_sha256=summary['training_stage_sha256'],evaluation_summary_sha256=digest(MAIN/'summary.json'),
                   audit_script_sha256=digest(Path(__file__)),rules_reader_sha256=digest(ROOT/'native/noregrets-json/terminal_audit.rs'))
    check(receipt['hands']==756000,'Wrong final hand count')
    atomic_json(STUDY/'external-audit.json',dict(receipt,cells_verified=records))
    report = ROOT/'reports'/STUDY.relative_to(ROOT/'artifacts')
    report.mkdir(parents=True,exist_ok=True)
    atomic_json(report/'external-audit.json',receipt)


def main(follow):
    verify_files({DOSE:DOSE_SHA,CONFIG:CONFIG_SHA,OLD/'stage-ready.json':SOURCE_SHA})
    terminal_auditor()
    parent = read(DOSE)
    plans = {'reference':load_plan('reference')}
    blueprint = ROOT/read(ROOT/'artifacts/selfplay/current.json')['model']
    blueprint_sha = digest(blueprint)
    public_bot = Fullhouse()
    records,matrices = {},{}
    with ExitStack() as stack:
        engines = {}
        while len(records)<54:
            if 'evaluation' not in plans and (MAIN/'evaluation-bound.json').is_file():
                plans['evaluation'] = load_plan('evaluation')
            for name,field,seed,label in entries():
                if name in records:
                    continue
                phase = 'reference' if field=='public' and seed==342 and label in EARLY else 'evaluation'
                progress = MAIN/'evaluation'/name/'status.json'
                if phase not in plans or not progress.is_file():
                    continue
                status = read(progress)
                check(status['phase']!='failed',f'Evaluator failed: {name}')
                if status['phase']!='complete':
                    continue
                need_blueprint = field!='public'
                if need_blueprint not in engines:
                    engines[need_blueprint] = stack.enter_context(closing(Bridge(blueprint if need_blueprint else None)))
                plan,sha = plans[phase]
                matrices[name],records[name] = verify_cell(name,field,seed,label,plan,sha,parent,engines[need_blueprint],
                                                            public_bot,blueprint_sha)
                print(f'{utc()} {len(records)}/54 {name} verified',flush=True)
            if len(records)<54:
                check(follow,'One or more cells are pending')
                time.sleep(8)
    while not (MAIN/'summary.json').is_file():
        check(follow,'Published external summary is pending')
        time.sleep(6)
    aggregate(matrices,records,plans)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow',action='store_true',help='Read this already-running evaluation as its cells complete.')
    main(parser.parse_args().follow)
