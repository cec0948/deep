"""Run the fixed continuation of the selected 480 specialist, then its paired evaluation."""
import fcntl
import json
from pathlib import Path
import shutil
from statistics import NormalDist

import run_poker_preflop_confirmation as confirmation
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from run_poker_preflop_pilot import CAMPAIGN, FIELDS, SEEDS, paired
from train_poker_warm_preflop import ARM, FINAL, GENERAL_STEPS, INITIAL, OUTPUT, PARENT, PREFLOP_STEPS


WORKER = Path(__file__).with_name('train_poker_warm_preflop.py').resolve()
START, DEALS = 2037600000, 20000
PAIR_SHA = 'd52a9e719b80108bc54230a001c991c136b4bf82c6f30131ea869e629d79975f'
BUNDLE_SHA = '424b8e76a31f359f21de47ec02c58fd3a902ab2639f9f7e9a8785ff253d63f2e'


def verify_training(plan):
    summaries, assets = {}, {}
    for seed in SEEDS:
        folder = OUTPUT/ARM/f'seed-{seed}'
        ready = read(folder/'ready.json')
        start = read(folder/'start-verification.json')
        saved = read(folder/'current.json')
        verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256'],
                      folder/'start-verification.json':ready['start_sha256'],
                      ready['policy']['path']:ready['policy']['sha256'],
                      ready['preflop_export']['path']:ready['preflop_export']['sha256']})
        verify_files({Path(ready['checkpoint'])/name:sha for name,sha in ready['files'].items()})
        if (saved['iteration'] != FINAL or ready['files'] != saved['files']
                or start['deployed_policy_sha256'] != plan['parent_policies'][str(seed)]['sha256']
                or not start['replay_root_reference_and_rng_verified']):
            raise ValueError('Parent policy or restored training state differs')
        steps = [json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
        if [r['iteration'] for r in steps] != list(range(INITIAL+1,FINAL+1)):
            raise ValueError('Missing or duplicate selfplay iteration')
        previous = start['weights']
        for row in steps:
            if (row['seed'] != seed or row['input_weights'] != previous
                    or any(row['output_weights'][k] == previous[k] for k in previous)
                    or row['traverser'] != (row['iteration']-1)%6 or row['traversals'] != 256
                    or (row['general_steps'],row['preflop_steps']) != (GENERAL_STEPS,PREFLOP_STEPS)
                    or row['buffer_rows'] != [100000]*6 or row['reference_collection']['rows'] != 100000
                    or row['new_regret_rows'] <= 0):
                raise ValueError('Training weight continuity, sampling or fixed budget differs')
            previous = row['output_weights']
        assets[str(seed)] = ready
        summaries[str(seed)] = dict(iteration=FINAL, start=start, updates=len(steps),
            general_steps=GENERAL_STEPS*len(steps), preflop_steps=PREFLOP_STEPS*len(steps),
            traversals=sum(r['traversals'] for r in steps), new_regret_rows=sum(r['new_regret_rows'] for r in steps),
            opening_samples=sum(r['root_rows'] for r in steps), reference_hands=sum(r['reference_collection']['hands'] for r in steps),
            final_weights=previous, final_opening_tv=steps[-1]['opening_tv'], final_average=steps[-1]['average_fit'])
    return dict(training_complete=True, iteration=FINAL, arms={ARM:assets}, completed_utc=utc(),
                warm_start_weight_chain_verified=True, seed_verification=summaries)


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUTPUT/'summary.json').is_file():
            return read(OUTPUT/'summary.json')
        confirmation.check_no_jobs()
        models, opponents, frozen = confirmation.frozen_pilot()
        comparison = confirmation.FOLLOWUP/'specialist-comparison'
        reference = confirmation.OUTPUT/'backup-reference.json'
        frozen.update({str(comparison/'summary.json'):PAIR_SHA,
            str(comparison/'final-audit.json'):'87b043b60fac3769beef6f7eb9851489209962dc92e1d5fddc9765175680335c'})
        verify_files(frozen)
        if (read(comparison/'final-audit.json')['summary_sha256'] != PAIR_SHA
                or read(comparison/'evaluation-plan.json')['policies'][PARENT] != models['specialist']
                or BUNDLE_SHA not in {r['sha256'] for r in read(reference)['existing_backups']}):
            raise ValueError('Selected 480 specialist differs from the audited direct comparison')
        previous = confirmation.check_fresh_cards(OUTPUT,START,DEALS)
        parent_cfg = read(CAMPAIGN/'config.json')
        frozen.update({p:h for p,h in parent_cfg['frozen_files'].items()
                       if Path(p).is_relative_to(CAMPAIGN/'source') or Path(p).is_relative_to(CAMPAIGN/'tools')})
        paths = [Path(__file__).resolve(), WORKER, reference, CAMPAIGN/'config.json',
                 comparison/'evaluation-plan.json', ROOT/'scripts/run_poker_preflop_online.py',
                 ROOT/'scripts/train_poker_average_continuation.py', ROOT/'scripts/poker_strategy_average.py',
                 ROOT/'scripts/compact_lightweight_backup.py', ROOT/'scripts/snapshot_poker_preflop_online.py']
        frozen.update({str(p):digest(p) for p in paths})
        verify_files(frozen)
        if shutil.disk_usage(OUTPUT.parent).free < 1024**3:
            raise RuntimeError('Insufficient space for the completed study and recovery')
        OUTPUT.mkdir(exist_ok=False)
        try:
            plan = dict(created_utc=utc(), initial_iteration=INITIAL, final_iteration=FINAL, arms=[ARM],
                seeds=SEEDS, fields=FIELDS, campaign=str(CAMPAIGN), parent_policies=models['specialist'],
                parent_archive_sha256=confirmation.pilot.CFG['restore_dependency_480_sha256'],
                parent_policy_bundle_sha256=BUNDLE_SHA, frozen_files=frozen,
                general_steps=GENERAL_STEPS, preflop_steps=PREFLOP_STEPS, main_batch=512,
                training='Continue each previous street network; fresh Adam moments per fit; shared opening expert',
                statistical_unit='One common original card block, averaging six seats and three fixed training seeds',
                target='Warm 492 minus the selected fixed 480 candidate; report all fields and individual seed effects',
                evaluation_deal_seed=START, evaluation_deals=DEALS, planned_evaluation_cells=18,
                interval='Nominal 95% per field with secondary simultaneous 95% intervals for three fields',
                automatic_retry=False, model_promotion=False, automatic_extension=False)
            atomic_json(OUTPUT/'plan.json',plan)
            plan_sha = digest(OUTPUT/'plan.json')
            atomic_json(OUTPUT/'config.json',dict(evaluation_deal_seed=START,evaluation_deals=DEALS,
                prior_card_ranges=previous,initial_iteration=INITIAL,final_iteration=FINAL))
            commands = {f'{ARM}-seed-{s}':[str(PYTHON),str(WORKER),'worker',str(s)] for s in SEEDS}
            confirmation.run_jobs(OUTPUT,commands,len(SEEDS),'training')
            verify_files({**frozen,str(OUTPUT/'plan.json'):plan_sha})
            stage = verify_training(plan)
            atomic_json(OUTPUT/'stage-ready.json',stage)
            policies = {PARENT:models['specialist'], ARM:{s:v['policy'] for s,v in stage['arms'][ARM].items()}}
            frozen = {**frozen,str(OUTPUT/'stage-ready.json'):digest(OUTPUT/'stage-ready.json'),str(OUTPUT/'plan.json'):plan_sha}
            rows, outcome = confirmation.evaluate(OUTPUT,policies,opponents,frozen,START,DEALS)
            change = paired(rows,PARENT,ARM,DEALS)
            z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
            simultaneous = {}
            for field,estimate in change.items():
                half = estimate['ci95_half_width']*z/1.96
                simultaneous[field] = dict(lower=estimate['mean']-half,upper=estimate['mean']+half)
            outcome.update(completed_utc=utc(), completed=True, new_selfplay_iterations_per_seed=FINAL-INITIAL,
                warm_start_weight_chain_verified=True, trained_minus_frozen_parent=change,
                simultaneous_95_bonferroni=simultaneous, model_promoted=False, automatic_extension=False)
            atomic_json(OUTPUT/'summary.json',outcome)
            return outcome
        except BaseException as error:
            atomic_json(OUTPUT/'failure.json',dict(observed_utc=utc(),error=f'{type(error).__name__}: {error}'))
            raise


if __name__ == '__main__':
    print(json.dumps(run()['trained_minus_frozen_parent'],indent=2,allow_nan=False))
