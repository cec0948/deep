"""Compare matched 24-round endpoints on a preselected prefix of the public-table evaluation."""
import argparse
from contextlib import closing
import gzip
from itertools import islice
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

import evaluate_average_baseline as evaluator
from audit_poker_population import replay_decisions, sample_policies
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy


STUDY = ROOT/'artifacts/preflop-followup/480-v1/postflop-averaging-20261120'
OLD = STUDY.parent/'fixed-preflop-504'
WORK = STUDY/'learning-rate-contrast'
REPORTS = ROOT/'reports/preflop-followup/480-v1/postflop-averaging-20261120/learning-rate-contrast'
CONFIG = ROOT/'configs/poker-postflop-averaging-evaluation-20261120.json'
SEEDS = (342, 343, 344)
START = 2038400000
COUNT = 4000
STRIDE = 313
OLD_PLAN = 'fc6d7ef96c866e10cebb70f9c42b90a84140e5049db92fcaeb18d897643a86a4'
OLD_AUDIT = '23f7f524a8d9f78460228005ea1da37b5ed96b51cad74b5ff2dfa99a1007a9b3'
PUBLIC_PLAN = '32364b5cb013d692b62fbc975c7cb261d70b878712bc7bbe0a58681ac546874b'


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def register():
    check(not (STUDY/'stage-ready.json').exists() and
          not any((STUDY/'population').rglob('results.json')), 'Register before training or evaluation completes')
    verify_files({OLD/'evaluation-plan.json':OLD_PLAN, OLD/'final-audit.json':OLD_AUDIT, CONFIG:PUBLIC_PLAN})
    original = read(OLD/'evaluation-plan.json')
    check(read(CONFIG)['public_cards'] == dict(deal_seed=START, deals=20000), 'Expected primary card range')
    for seed in SEEDS:
        for label in ('fixed504','specialist480'):
            ref = original['policies'][label][str(seed)]
            verify_files({ref['path']: ref['sha256']})
        old_ref = original['policies']['fixed504'][str(seed)]
        old_cell = read(OLD/'evaluation'/f'public-seed-{seed}-fixed504'/'results.json')
        check(old_cell['candidate_sha256'] == old_ref['sha256'] and old_cell['terminal_payoffs_replayed'],
              'Previous evaluated endpoint does not match')
    settings = dict(registered_utc=utc(), field='public', deal_seed=START, deals=COUNT,
        training_seeds=SEEDS, seats=6, contrast='slow504 minus fixed504',
        learning_rates=dict(slow504=.00025,fixed504=.001), training_rounds_per_seed=24,
        additional_cells=3, additional_hands=COUNT*6*len(SEEDS), additional_unique_card_blocks=0,
        subset='First 4,000 of the same 20,000 public card blocks; diagnostic overlapping the main study',
        interval='One prespecified two-sided nominal 95% normal paired interval over original cards; average seats and fixed seeds first',
        baseline='specialist480 on identical cards, descriptive', recorded_policy_replay_card_stride=STRIDE,
        frozen_public_design_sha256=PUBLIC_PLAN, previous_eval_plan_sha256=OLD_PLAN, previous_audit_sha256=OLD_AUDIT,
        new_training_plan_sha256=digest(STUDY/'plan.json'), implementation_sha256=digest(__file__),
        previous_weights_sha256={seed:original['policies']['fixed504'][str(seed)]['sha256'] for seed in SEEDS})
    WORK.mkdir()
    REPORTS.mkdir(parents=True,exist_ok=True)
    atomic_json(WORK/'design.json', settings)
    atomic_json(REPORTS/'design.json', settings)


def design():
    verify_files({WORK/'design.json':digest(REPORTS/'design.json'), OLD/'evaluation-plan.json':OLD_PLAN,
                  CONFIG:PUBLIC_PLAN})
    frozen = read(WORK/'design.json')
    verify_files({Path(__file__):frozen['implementation_sha256'], STUDY/'plan.json':frozen['new_training_plan_sha256']})
    return frozen, read(OLD/'evaluation-plan.json')


def run_old():
    frozen, old = design()
    (WORK/'evaluation').mkdir()
    evaluator.policy_for = evaluator_policy
    for seed in SEEDS:
        ref = old['policies']['fixed504'][str(seed)]
        verify_files({ref['path']:frozen['previous_weights_sha256'][str(seed)]})
        destination = WORK/'evaluation'/f'public-seed-{seed}-fixed504'
        evaluator.run(ref['path'],old['source'],'public',COUNT,START,3600,destination)
    atomic_json(WORK/'endpoints-complete.json',dict(utc=utc(), cells=len(SEEDS), hands=COUNT*6*len(SEEDS)))


def blocks_for(folder, ref, source, public_sha):
    # Validate the full frozen archive even when only its preselected prefix is scored.
    result = read(folder/'results.json')
    verify_files({folder/'results.json':read(folder/'status.json')['results_sha256'],
                  folder/'hands.jsonl.gz':result['raw_hands_sha256'],
                  folder/'provenance.json':result['provenance_sha256'], ref['path']: ref['sha256']})
    provenance = read(folder/'provenance.json')
    verify_files(provenance['dependency_hashes'])
    check(provenance['candidate_sha256'] == result['candidate_sha256'] == ref['sha256']
          and result['field'] == 'public' and result['opponents']['public'] == public_sha
          and result['complete'] and result['terminal_payoffs_replayed'] and result['raw_hands_verified']
          and result['deal_seed'] == START and result['deals'] in (COUNT,20000) and result['big_blind_chips']==100,
          'Mismatched model, public table or cards')
    check(provenance['opponents'] == result['opponents'] and provenance['dependency_hashes'] == result['dependency_hashes'],
          'Result and provenance disagree')
    bots, _ = sample_policies({'public'}, ref, source, 0, {'public':public_sha})
    blocks, replayed = np.zeros(COUNT,dtype=np.int64), 0
    count = 0
    with gzip.open(folder/'hands.jsonl.gz','rt') as stream, closing(Bridge()) as engine:
        for i,line in enumerate(islice(stream,COUNT*6)):
            row = json.loads(line)
            number,hero = divmod(i,6)
            check((row['seed'],row['deal'],row['hero'],row['hand_seed']) ==
                  (START+number, number, hero, evaluator.hand_seed(START+number,hero)), 'Card/seat order changed')
            check(len(row['utilities'])==6 and all(type(v) is int for v in row['utilities'])
                  and sum(row['utilities']) == 0 and row['chips'] == row['utilities'][hero], 'Invalid hand payoff')
            blocks[number] += row['chips']
            if number % STRIDE == 0 or number == COUNT-1:
                replay_decisions(row, 'public', engine, bots)
                replayed += 1
            count += 1
    check(count==COUNT*6 and np.allclose(blocks/6, result['deal_blocks'][:COUNT],atol=1e-8,rtol=0),
          'Incomplete or mismatched hand ledger')
    return blocks/6, dict(result_sha256=digest(folder/'results.json'), raw_sha256=result['raw_hands_sha256'],
                          verified_hands=count, policy_replayed_hands=replayed)


def summarize():
    _, old = design()
    plan = read(STUDY/'population/evaluation-plan.json')
    public_sha = old['opponents']['public']['public']
    scores, proofs = {}, {}
    for label in ('fixed504', 'slow504', 'specialist480'):
        scores[label] = []
        for seed in SEEDS:
            name = f'public-seed-{seed}-{label}'
            source_plan = old if label=='fixed504' else plan
            source_folder = WORK if label=='fixed504' else STUDY/'population'
            ref = source_plan['policies'][label][str(seed)]
            if label=='specialist480':
                check(ref == old['policies']['specialist480'][str(seed)], 'Starting policy differs')
            blocks, proofs[name] = blocks_for(source_folder/'evaluation'/name, ref, old['source'], public_sha)
            scores[label].append(blocks)
    means = {label:np.array(value).mean(axis=0) for label,value in scores.items()}
    paired = means['slow504']-means['fixed504']
    difference = float(paired.mean())
    half = float(NormalDist().inv_cdf(.975)*paired.std(ddof=1)/np.sqrt(COUNT))
    summary = dict(utc=utc(), design_sha256=digest(WORK/'design.json'), independent_card_blocks=COUNT,
        additional_unique_card_blocks=0, additional_evaluated_hands=COUNT*6*len(SEEDS),
        scores_bb100={k:float(v.mean()) for k,v in means.items()}, slow_minus_prior_fixed504_bb100=difference,
        nominal_paired_ci95_bb100=[difference-half, difference+half], differences_by_training_seed={
            s:float(np.mean(scores['slow504'][i]-scores['fixed504'][i])) for i,s in enumerate(SEEDS)},
        checked_prefix_rows=sum(v['verified_hands'] for v in proofs.values()),
        independent_policy_and_terminal_replay_hands=sum(v['policy_replayed_hands'] for v in proofs.values()),
        result_proofs=proofs, generated_trajectories_can_diverge_after_first_fit=True,
        scope='Matched fixed-budget endpoint public-table diagnostic on a subset of the main evaluation; one table, three fixed training seeds.')
    atomic_json(WORK/'summary.json',summary)
    atomic_json(REPORTS/'summary.json',summary)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','evaluate','summarize'))
    mode = parser.parse_args().mode
    {'register':register,'evaluate':run_old,'summarize':summarize}[mode]()
