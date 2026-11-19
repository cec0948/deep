"""Evaluate 492 preflop plus 480 postflop against two frozen endpoints on their common cards."""
import fcntl
from pathlib import Path
from statistics import NormalDist

import run_poker_preflop_confirmation as confirmation
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest
from poker_preflop_specialist import file_ref, write_policy
from run_poker_preflop_pilot import FIELDS, SEEDS, SOURCE, paired


SOURCE_STUDY = confirmation.FOLLOWUP/'warm-start'
OUTPUT = confirmation.FOLLOWUP/'preflop-swap'
BASE, HYBRID, WARM = 'specialist480', 'hybrid492pre480post', 'warm492'
START, DEALS = 2037600000, 20000
SOURCE_SUMMARY = 'b1f9a3bf1721d87c535955b1f4e038ee111a3f8589b8e4db7a6d2b0fb76b7c9c'
SOURCE_AUDIT = '78ef335bb0895535c0e67f38d621b6480148032325a04c63411ffe23579d044a'


def prepare_sources():
    frozen = {str(SOURCE_STUDY/'summary.json'): SOURCE_SUMMARY,
              str(SOURCE_STUDY/'final-audit.json'): SOURCE_AUDIT}
    verify_files(frozen)
    summary = read(SOURCE_STUDY/'summary.json')
    audit = read(SOURCE_STUDY/'final-audit.json')
    frozen[str(SOURCE_STUDY/'evaluation-plan.json')] = summary['evaluation_plan_sha256']
    verify_files(frozen)
    plan = read(SOURCE_STUDY/'evaluation-plan.json')
    if (audit['summary_sha256'] != SOURCE_SUMMARY or audit['raw_hand_records_reaggregated'] != 2160000
            or set(plan['policies']) != {BASE, WARM} or plan['source'] != str(SOURCE)
            or (plan['card_start'], plan['deals'], summary['completed_cells']) != (START, DEALS, 18)):
        raise ValueError('The original paired study differs')
    frozen.update(plan['frozen_files'])
    rows = {}
    for label, refs in plan['policies'].items():
        for seed in SEEDS:
            ref = refs[str(seed)]
            for field in FIELDS:
                name = f'{field}-seed-{seed}-{label}'
                cell = SOURCE_STUDY/'evaluation'/name
                verify_files({cell/'results.json': summary['result_sha256'][name]})
                result = read(cell/'results.json')
                if (not result['terminal_payoffs_replayed'] or not result['complete']
                        or result['candidate_sha256'] != ref['sha256'] or result['candidate'] != ref['path']
                        or result['opponents'] != plan['opponents'][field] or result['deal_seed'] != START
                        or result['hands'] != DEALS*6):
                    raise ValueError('Different reference cell: '+name)
                frozen[str(cell/'results.json')] = summary['result_sha256'][name]
                frozen[str(cell/'provenance.json')] = result['provenance_sha256']
                frozen[str(cell/'hands.jsonl.gz')] = result['raw_hands_sha256']
                for path, sha in result['dependency_hashes'].items():
                    if frozen.setdefault(path, sha) != sha:
                        raise ValueError('Inconsistent reference dependency')
                rows[seed, field, label] = result['deal_blocks']
    frozen[str(Path(__file__).resolve())] = digest(__file__)
    verify_files(frozen)
    return plan, summary, rows, frozen


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT/'summary.json').is_file():
            return read(OUTPUT/'summary.json')
        confirmation.check_no_jobs()
        old_plan, old_result, endpoints, frozen = prepare_sources()
        OUTPUT.mkdir(exist_ok=False)
        try:
            hybrids = {}
            for seed in map(str, SEEDS):
                old = read(old_plan['policies'][BASE][seed]['path'])
                new = read(old_plan['policies'][WARM][seed]['path'])
                directory = OUTPUT/f'seed-{seed}'
                directory.mkdir()
                spec = write_policy(directory/'policy.json', new['preflop']['path'], old['postflop']['path'])
                if read(spec) != dict(format='preflop-street-policy-v1',
                                      preflop=new['preflop'], postflop=old['postflop']):
                    raise ValueError('Crossed street weights do not match their original endpoints')
                hybrids[seed] = file_ref(spec)
                frozen[str(spec)] = hybrids[seed]['sha256']
            atomic_json(OUTPUT/'design.json', dict(created_utc=utc(), first_card=START, deals=DEALS,
                source_summary_sha256=SOURCE_SUMMARY, source_audit_sha256=SOURCE_AUDIT,
                preflop_from=WARM, postflop_from=BASE, new_selfplay_iterations=0, new_fitting_jobs=0,
                independent_new_cards=False, new_cells=9, reused_cells=18, model_promotion=False,
                contrast_order=['hybrid_minus_480','492_minus_hybrid'],
                inference='Paired original card blocks, average seats and fixed seeds; simultaneous 95% across three fields per contrast'))
            frozen[str(OUTPUT/'design.json')] = digest(OUTPUT/'design.json')
            rows, result = confirmation.evaluate(OUTPUT, {HYBRID: hybrids}, old_plan['opponents'], frozen, START, DEALS)
            rows.update(endpoints)
            changes = {'hybrid_minus_480': paired(rows, BASE, HYBRID, DEALS),
                       '492_minus_hybrid': paired(rows, HYBRID, WARM, DEALS)}
            z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
            simultaneous = {}
            for contrast, estimates in changes.items():
                simultaneous[contrast] = {}
                for field, estimate in estimates.items():
                    half = z * estimate['ci95_half_width']/1.96
                    simultaneous[contrast][field] = dict(lower=estimate['mean']-half, upper=estimate['mean']+half)
            result['absolute_scores'].update(old_result['absolute_scores'])
            result.update(completed=True, completed_utc=utc(), new_cells=9, reused_cells=18,
                          new_hands=9*DEALS*6, reused_hands=18*DEALS*6, reused_result_sha256=old_result['result_sha256'],
                          source_summary_sha256=SOURCE_SUMMARY, source_audit_sha256=SOURCE_AUDIT,
                          comparisons=changes, simultaneous_95=simultaneous, new_selfplay_iterations=0,
                          new_fitting_jobs=0, model_promoted=False, independent_new_cards=False)
            verify_files(frozen)
            atomic_json(OUTPUT/'summary.json', result)
            return result
        except BaseException as exc:
            atomic_json(OUTPUT/'failure.json', dict(observed_utc=utc(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    run()
