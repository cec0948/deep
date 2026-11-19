"""Pair the frozen 480 specialist with archived 492 specialist hands on the 492 cards."""
import fcntl
import json
from pathlib import Path

import run_poker_preflop_confirmation as confirmation
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest
from run_poker_preflop_pilot import FIELDS, SEEDS, SOURCE, paired


ONLINE = confirmation.FOLLOWUP / 'online'
OUTPUT = confirmation.FOLLOWUP / 'specialist-comparison'
CONFIRMATION_SHA = '52d1a10ff493680189afd12fdef4408b4bd8dc980ce56786d7ad7dcba4dfdff1'
ONLINE_SHA = 'a9c0be44cedbe403f7cd943856a11c86413aa1e74dea3ee53068eb76d7e80a51'


def inputs():
    confirmed = confirmation.OUTPUT
    frozen = {str(confirmed / 'summary.json'): CONFIRMATION_SHA,
              str(ONLINE / 'summary.json'): ONLINE_SHA}
    verify_files(frozen)
    first, online = read(confirmed / 'summary.json'), read(ONLINE / 'summary.json')
    for folder, outcome in ((confirmed, first), (ONLINE, online)):
        frozen[str(folder / 'evaluation-plan.json')] = outcome['evaluation_plan_sha256']
    verify_files(frozen)
    original = read(confirmed / 'evaluation-plan.json')
    plan = read(ONLINE / 'evaluation-plan.json')
    policies, opponents, pilot_files = confirmation.frozen_pilot()
    if (not first['independent_confirmation_pass'] or not online['completed']
            or online['completed_cells'] != 36 or plan['source'] != str(SOURCE)
            or plan['card_start'] != confirmation.CFG['online_deal_seed']
            or plan['deals'] != confirmation.CFG['evaluation_deals']
            or plan['opponents'] != opponents or original['opponents'] != opponents
            or policies['specialist'] != original['policies']['specialist']):
        raise ValueError('The confirmed policy or completed 492 evaluation differs from the design')
    verify_files(pilot_files)
    verify_files(plan['frozen_files'])
    frozen.update(pilot_files)
    frozen.update(plan['frozen_files'])
    frozen[str(Path(__file__).resolve())] = digest(__file__)
    names = {}
    for seed in SEEDS:
        for field in FIELDS:
            name = f'{field}-seed-{seed}-specialist492'
            result_path = ONLINE / 'evaluation' / name / 'results.json'
            names[name] = online['result_sha256'][name]
            verify_files({result_path: names[name]})
            record = read(result_path)
            frozen[str(result_path)] = names[name]
            frozen[str(result_path.parent / 'hands.jsonl.gz')] = record['raw_hands_sha256']
            frozen[str(result_path.parent / 'provenance.json')] = record['provenance_sha256']
            verify_files(record['dependency_hashes'])
    verify_files(frozen)
    return policies['specialist'], plan, names, frozen


def compare():
    with (confirmation.FOLLOWUP / 'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / 'summary.json').is_file():
            return read(OUTPUT / 'summary.json')
        confirmation.check_no_jobs()
        specialist, online, reused, frozen = inputs()
        deals, start = online['deals'], online['card_start']
        OUTPUT.mkdir(exist_ok=False)
        try:
            atomic_json(OUTPUT / 'design.json', dict(created_utc=utc(),
                comparison='specialist492 minus fixed specialist480, paired within original card block',
                reason_for_reusing_cards='Extend the completed 492 evaluation with one frozen policy',
                independent_of_prior_492_evaluation=False, first_card=start, last_card=start+deals-1,
                deals=deals, fields=FIELDS, training_seeds=SEEDS,
                new_cells=len(reused), reused_cells=len(reused), reused_result_sha256=reused,
                fixed_480_confirmation_sha256=CONFIRMATION_SHA, online_492_summary_sha256=ONLINE_SHA,
                new_selfplay_iterations=0, new_fitting_jobs=0, model_promotion=False))
            frozen[str(OUTPUT / 'design.json')] = digest(OUTPUT / 'design.json')
            rows, result = confirmation.evaluate(OUTPUT, {'specialist480': specialist},
                online['opponents'], frozen, start, deals)
            for seed in SEEDS:
                ref = online['policies']['specialist492'][str(seed)]
                for field in FIELDS:
                    name = f'{field}-seed-{seed}-specialist492'
                    existing = confirmation.evaluator.verify_results(ONLINE / 'evaluation' / name,
                        expected=dict(field=field, deals=deals, deal_seed=start, hands=deals*6,
                            candidate=ref['path'], candidate_sha256=ref['sha256'],
                            opponents=online['opponents'][field], big_blind_chips=100,
                            terminal_payoffs_replayed=True))
                    rows[seed, field, 'specialist492'] = existing['deal_blocks']
            changes = paired(rows, 'specialist480', 'specialist492', deals)
            result['absolute_scores']['specialist492'] = read(ONLINE / 'summary.json')['absolute_scores']['specialist492']
            result.update(completed_utc=utc(), new_cells=result['completed_cells'],
                new_hands=result['verified_hands'], reused_cells=len(reused), reused_hands=len(reused)*deals*6,
                reused_result_sha256=reused, specialist492_minus_specialist480=changes,
                source_492_summary_sha256=ONLINE_SHA, source_480_confirmation_sha256=CONFIRMATION_SHA,
                assessment='Paired comparison on the previously evaluated 492 card set',
                new_selfplay_iterations=0, new_fitting_jobs=0, model_promoted=False)
            verify_files(frozen)
            atomic_json(OUTPUT / 'summary.json', result)
            return result
        except BaseException as exc:
            atomic_json(OUTPUT / 'failure.json', dict(observed_utc=utc(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    print(json.dumps(compare()['specialist492_minus_specialist480'], indent=2, allow_nan=False))
