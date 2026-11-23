"""Check the street-specific intervention, its matched raw outcomes, and original-deal statistics."""
import argparse
from collections import Counter
from contextlib import ExitStack, closing
import gzip
import hashlib
import json
from statistics import NormalDist
import subprocess
import time

import numpy as np

from audit_poker_opponent_diversity import AuditedSolverLike, BASE, POP as PREVIOUS_PANEL, RANGES, SEEDS, STRIDE, TREATMENT, SOLVER_SHA, bots_for
from audit_poker_population import check, close_to, matched_role
from audit_poker_public_divergence import CATEGORIES, binary as pair_replay_binary
from audit_postflop_averaging import audit_pair, checked_ref, file_sha, hero_policy, paired_stats, private_slot, replay
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy

STUDY = PREVIOUS_PANEL.parent.parent/'flop-rollback-20261123'
PREVIOUS = PREVIOUS_PANEL.parent
HYBRID = 'flop480late504'
LABELS = (BASE, TREATMENT, HYBRID)
PREVIOUS_AUDIT = 'e792012f135f80a212ad1b1f876992307f878a6bfd98bcbd3e0537a16ad12e88'
PREVIOUS_PLAN = 'a851844aee89e4032992840e40b2bf41ac765cc132dd5be9f94fee09e9540b43'
PREVIOUS_REFERENCE = '691897dda94b12155ed6cf4141e866a02d3604b633e6a00186d768b41d5ab2d8'
PREVIOUS_SUMMARY = '070c481106dd77ba6bbc9dcd20b6d9969cc0e6d7b1291c3e5d91ae074348e515'
STREETS = ('preflop', 'flop', 'turn', 'river')


class IndependentHybrid:
    """Load the two checkpoints' weights directly, independent of the intervention's runtime."""
    def __init__(self, parent, mixture, source):
        old = read(checked_ref(parent))
        path = checked_ref(mixture)
        late = read(path)
        self.preflop = evaluator_policy(checked_ref(old['preflop']), source)
        self.flop = evaluator_policy(checked_ref(old['postflop']), source)
        self.later = {}
        for snapshot in late['postflop']:
            weights = path.parent/snapshot['file']
            verify_files({weights: snapshot['sha256']})
            self.later[snapshot['iteration']] = evaluator_policy(weights, source)
        check(tuple(self.later) == (486, 492, 498, 504), 'Different hand-private mixture')
        check(late['preflop']['sha256'] == old['preflop']['sha256'], 'Different preflop weights in source policies')
        self.seen = Counter()

    def begin_hand(self, seed):
        self.slot = private_slot(seed)

    def decide(self, observation, seed):
        street = observation['street']
        check(street in STREETS, 'Unexpected betting street')
        self.seen[street] += 1
        if street == 'preflop':
            policy = self.preflop
        elif street == 'flop':
            policy = self.flop
        else:
            policy = self.later[self.slot]
        return policy.decide(observation, seed)


def cell_metadata(folder, name, ref, expected_plan, spec, seed, label):
    result = read(folder/'results.json')
    provenance = read(folder/'provenance.json')
    status = read(folder/'status.json')
    hands = spec['deals']*6
    verify_files({folder/'provenance.json': result['provenance_sha256'],
                  folder/'hands.jsonl.gz': result['raw_hands_sha256'],
                  folder/'results.json': status['results_sha256']})
    seen = dict(field=name, training_seed=seed, label=label, deal_seed=spec['deal_seed'],
                deals=spec['deals'], hands=hands, candidate=ref['path'], candidate_sha256=ref['sha256'],
                big_blind_chips=100, evaluation_plan_sha256=expected_plan)
    check(all(provenance[k] == result[k] == v for k, v in seen.items()), 'Different policy, card, or plan')
    check(status['phase'] == 'complete' and status['verified_hands'] == hands and result['complete']
          and result['raw_hands_verified'] and result['terminal_payoffs_replayed']
          and result['verification']['hands'] == hands and result['verification']['card_blocks'] == spec['deals']
          and all(result['verification'][k] for k in ('terminal_payoffs_replayed','chip_conservation_verified','aggregate_recomputed')),
          'Incomplete or unverified ledger')
    check(result['dependency_hashes'] == provenance['dependency_hashes'] and
          result['opponents'] == provenance['opponents'], 'Provenance and summary conflict')
    verify_files(result['dependency_hashes'])
    return result


def stream_table(field, seed, spec, paths, records, engine, opponents, policies):
    """Reaggregate every original deal. Rebuild every actor's actions on a fixed full-hand sample."""
    sums = np.zeros((3, spec['deals']), dtype=np.int64)
    steps = Counter()
    snapshots = Counter()
    slot_choices = bytearray()
    sample_actions = Counter()
    sampled = 0
    with ExitStack() as files:
        streams = [files.enter_context(gzip.open(path, 'rt')) for path in paths]
        total = 0
        for i, lines in enumerate(zip(*streams, strict=True)):
            block, seat = divmod(i, 6)
            card = spec['deal_seed'] + block
            check(block < spec['deals'], 'Extra hand')
            private_seed = hand_seed(card, seat)
            slot = private_slot(private_seed)
            snapshots[str(slot)] += 1
            slot_choices.append((486,492,498,504).index(slot))
            sample = block % STRIDE == 0 or block == spec['deals']-1
            sampled += sample
            for arm, (line, label) in enumerate(zip(lines, LABELS, strict=True)):
                row = json.loads(line)
                check((row['deal'], row['hero'], row['seed'], row['hand_seed']) == (block, seat, card, private_seed),
                      'Common card or clockwise seat schedule changed')
                check(row.get('snapshot_iteration') == (None if label == BASE else slot), 'Wrong private hand slot')
                payouts = row['utilities']
                check(len(payouts) == 6 and all(type(p) is int for p in payouts) and sum(payouts) == 0
                      and row['chips'] == payouts[seat] and 0 < len(row['actions']) == row['steps'] <= 512,
                      'Invalid terminal ledger')
                steps[label] += row['steps']
                sums[arm, block] += payouts[seat]
                if sample:
                    replay(row, field, engine, dict(opponents, hero=policies[label]))
                    sample_actions[label] += row['steps']
            total += 1
    check(total == spec['deals']*6, 'Missing matched hands')
    for i, label in enumerate(LABELS):
        r = records[label]
        check(({} if label == BASE else dict(snapshots)) == r['snapshot_hand_counts'] and
              steps[label] == r['verification']['concrete_decisions_replayed'], 'Wrong counts of slots/decisions')
        close_to(r['deal_blocks'], sums[i]/6, 'Different original-card scores')
        close_to(r['bb_per_100'], sums[i].sum()/total, 'Different bb/100')
    return sums/6, dict(hands_per_policy=total, decisions=steps, sampled_hands_per_policy=sampled,
                        sampled_actions=sample_actions, private_snapshot_counts=snapshots,
                        private_slot_order_sha256=hashlib.sha256(slot_choices).hexdigest())


def matched_replays(executable, paths, spec, blocks):
    """Independently settle each policy hand and locate its first changed action on common history."""
    command = [str(executable), *map(str, paths), str(spec['deal_seed']), str(spec['deals'])]
    result = json.loads(subprocess.check_output(command, text=True))
    chips = result['policy_chips']
    check(np.array_equal(chips, np.rint(blocks.sum(axis=1)*6)), 'Raw poker-engine scores differ')
    pairings = ((2, 0), (1, 0), (2, 1))
    groups = {}
    for idx, (right, left) in enumerate(pairings):
        counts = dict(zip(CATEGORIES, result['counts'][idx], strict=True))
        gains = dict(zip(CATEGORIES, result['chip_differences'][idx], strict=True))
        check(sum(counts.values()) == spec['deals']*6 and sum(gains.values()) == chips[right]-chips[left]
              and counts['preflop'] == counts['opponent_first'] == gains['identical'] == 0,
              'Inconsistent shared histories or terminal differences')
        if (right, left) == (2, 0):
            check(counts['flop'] == 0, 'Rollback diverged from 480 on a common flop')
        elif (right, left) == (2, 1):
            check(counts['turn'] == counts['river'] == 0, 'Rollback diverged from diverse504 outside the flop')
        groups[f'{LABELS[right]}_minus_{LABELS[left]}'] = dict(hands=counts, chip_difference=gains)
    return groups


def check_plan():
    verify_files({PREVIOUS_PANEL/'evaluation-plan.json': PREVIOUS_PLAN,
                  PREVIOUS_PANEL/'reference-plan.json': PREVIOUS_REFERENCE,
                  PREVIOUS/'final-audit.json': PREVIOUS_AUDIT, PREVIOUS_PANEL/'summary.json': PREVIOUS_SUMMARY})
    old = read(PREVIOUS_PANEL/'evaluation-plan.json')
    source_proof = read(PREVIOUS/'final-audit.json')
    summary = read(PREVIOUS_PANEL/'summary.json')
    plan = read(STUDY/'evaluation-plan.json')
    bound = read(STUDY/'bound.json')['evaluation_plan_sha256']
    verify_files({STUDY/'evaluation-plan.json': bound})
    verify_files(plan['frozen_files'])
    config_path = ROOT/'configs/poker-flop-rollback-20261123.json'
    config = read(config_path)
    check(plan['previous_plan_sha256'] == config['source_plan_sha256'] == PREVIOUS_PLAN
          and plan['previous_reference_plan_sha256'] == PREVIOUS_REFERENCE
          and plan['previous_summary_sha256'] == config['source_summary_sha256'] == PREVIOUS_SUMMARY
          and config['source_final_audit_sha256'] == PREVIOUS_AUDIT, 'Different source study')
    check(plan['fields'] == old['fields'] and set(plan['fields']) == set(RANGES) and plan['source'] == old['source']
          and plan['seeds'] == config['fixed_training_seeds'] == list(SEEDS) and
          plan['training_updates'] == config['training_updates'] == 0 and not plan['model_promoted'],
          'Fields, seeds, model source or intervention changed')
    check(all((spec['deal_seed'], spec['deals']) == RANGES[name] for name,spec in plan['fields'].items())
          and plan['unique_original_card_blocks'] == sum(n for _,n in set(RANGES.values())) == 26000,
          'Different card and seat schedule')
    ids = {f'{f}-seed-{s}-{HYBRID}': (f, s, HYBRID) for f in plan['fields'] for s in SEEDS}
    check({j['id']: (j['field'],j['seed'],j['label']) for j in plan['jobs']} == ids
          and len(plan['jobs']) == len(ids) == 27 and
          config['main']['new_hands'] == plan['expected_new_hands'] ==
          sum(plan['fields'][f]['deals']*6 for f,_,_ in ids.values()) == 648000,
          'Different new games geometry')
    reference_ids = {f'{f}-seed-{s}-{p}': summary['cell_result_sha256'][f'{f}-seed-{s}-{p}']
                     for f in plan['fields'] for s in SEEDS for p in (BASE,TREATMENT)}
    check(plan['previous_cells'] == reference_ids and len(reference_ids) == config['main']['reference_cells_reused'] == 54
          and config['main']['reference_hands_reused'] == plan['expected_reference_hands'] == 1296000,
          'Archived reference cells differ')
    refs = {BASE: old['policies'][BASE], TREATMENT: old['policies'][TREATMENT], HYBRID:plan['policies'][HYBRID]}
    for seed in SEEDS:
        key = str(seed)
        manifest = read(checked_ref(refs[HYBRID][key]))
        check(manifest['format'] == 'specialist480-flop-diverse504-turn-river-v1' and
              manifest['training_seed'] == seed and
              all(manifest[p] == {k:refs[p][key][k] for k in ('path','sha256')} for p in (BASE,TREATMENT)),
              'Manifest does not bind the intended source policies')
    return plan, bound, refs, source_proof


def wait_for(path):
    """Wait for the evaluator's published receipt and stop on a cell or supervisor failure."""
    supervisor = STUDY/'evaluation-status.json'
    while not path.exists():
        for status_path in (supervisor, path.parent/'status.json'):
            if status_path.exists():
                status = read(status_path)
                check('failed' not in (status.get('phase'),status.get('state')), f'Evaluation stopped: {status_path}')
        time.sleep(3)
    if path.name == 'results.json':
        while read(path.parent/'status.json')['phase'] != 'complete':
            check(read(path.parent/'status.json')['phase'] != 'failed', f'Evaluation stopped at {path.parent}')
            time.sleep(1)


def verify_panel(follow=False):
    plan, bound, refs, source_proof = check_plan()
    cfg = ROOT/'configs/poker-flop-rollback-20261123.json'
    parent_path = PREVIOUS.parent/'preflop-dose/evaluation-plan.json'
    verify_files({parent_path:'642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'})
    parent = read(parent_path)
    reg = read(ROOT/'artifacts/selfplay/current.json')
    blueprint_path = ROOT/reg['model']
    public = Fullhouse()
    check(reg['sha256'] == file_sha(blueprint_path) == parent['opponents']['public']['noregrets'] and
          public.digest == parent['opponents']['public']['public'], 'Reference opponent differs')
    known = {'solverlike':AuditedSolverLike(), 'public':public}
    executable = pair_replay_binary()
    by_seed = {}
    for seed in SEEDS:
        key = str(seed)
        by_seed[seed] = {p:hero_policy(refs[p][key]['path'], plan['source'], p == TREATMENT) for p in (BASE,TREATMENT)}
        by_seed[seed][HYBRID] = IndependentHybrid(refs[BASE][key], refs[TREATMENT][key], plan['source'])
    matrices, proofs, hashes = {}, {}, {}
    with ExitStack() as stack:
        ordinary = stack.enter_context(closing(Bridge()))
        with_blueprint = stack.enter_context(closing(Bridge(blueprint_path)))
        for field, spec in plan['fields'].items():
            roles = {matched_role(field, spec['deal_seed'], 0, seat) for seat in range(1,6)}
            engine = with_blueprint if 'noregrets' in roles else ordinary
            matrices[field] = {p:[] for p in LABELS}
            for seed in SEEDS:
                key = str(seed)
                bots, expected_ids = bots_for(field, seed, parent, known)
                if field == 'solverlike':
                    expected_ids = {'solverlike':SOLVER_SHA}
                paths, records = [], {}
                for label in LABELS:
                    name = f'{field}-seed-{seed}-{label}'
                    folder = (STUDY if label == HYBRID else PREVIOUS_PANEL)/'evaluation'/name
                    ident = bound if label == HYBRID else PREVIOUS_REFERENCE if name == 'public-seed-342-specialist480' else PREVIOUS_PLAN
                    if follow and label == HYBRID:
                        wait_for(folder/'results.json')
                    record = cell_metadata(folder, field, refs[label][key], ident, spec, seed, label)
                    hashes[name] = digest(folder/'results.json')
                    raw = folder/'hands.jsonl.gz'
                    check(record['opponents'] == expected_ids == plan['opponents'][f'{field}-seed-{seed}']
                          and list(record['first_deal_clockwise_lineup']) ==
                          [matched_role(field,spec['deal_seed'],0,seat) for seat in range(1,6)]
                          and record['varies_by_card'] == (field == 'varied'), 'Different opponent or seating assignment')
                    if label != HYBRID:
                        proof = source_proof['cells'][name]
                        check(hashes[name] == plan['previous_cells'][name] == proof['result_sha256'] and
                              record['raw_hands_sha256'] == proof['raw_sha256'], 'Reference receipt no longer matches')
                    paths.append(raw)
                    records[label] = record
                blocks, tape = stream_table(field, seed, spec, paths, records, engine, bots, by_seed[seed])
                tape['paired_first_difference'] = matched_replays(executable, paths, spec, blocks)
                tape['raw_sha256'] = {p:records[p]['raw_hands_sha256'] for p in LABELS}
                proofs[f'{field}/seed-{seed}'] = tape
                for idx,label in enumerate(LABELS):
                    matrices[field][label].append(blocks[idx])
            matrices[field] = {p:np.asarray(items) for p,items in matrices[field].items()}
    routed = {str(s):dict(by_seed[s][HYBRID].seen) for s in SEEDS}
    check(all(set(d) == set(STREETS) and min(d.values()) > 0 for d in routed.values()),
          'No independent policy replay on one of the betting streets')
    receipt = dict(config_sha256=digest(cfg), plan_sha256=bound, audited_reference_sha256=PREVIOUS_AUDIT,
                   verified_new_cells=27, verified_archived_cells=54, new_terminal_replay_hands=648000,
                   archived_terminal_replay_hands=1296000, original_deals=26000,
                   terminal_replay_decisions_by_policy={p:sum(t['decisions'][p] for t in proofs.values()) for p in LABELS},
                   policy_sample_hands_per_arm=sum(t['sampled_hands_per_policy'] for t in proofs.values()),
                   sampled_decisions_by_policy={p:sum(t['sampled_actions'][p] for t in proofs.values()) for p in LABELS},
                   sample_card_stride=STRIDE, sampled_hybrid_hero_decisions_by_street_and_seed=routed,
                   terminal_binary_sha256=digest(executable), result_sha256=hashes, cell_audits=proofs)
    return plan, matrices, receipt


def check_public_recovery(scores, stated):
    gap = (scores[BASE] - scores[TREATMENT]).mean(axis=0)
    gain = (scores[HYBRID] - scores[TREATMENT]).mean(axis=0)
    d, g = float(gap.mean()), float(gain.mean())
    z = NormalDist().inv_cdf(.975)
    half = z*gap.std(ddof=1) / np.sqrt(len(gap))
    close_to(stated['denominator_bb100'], d, 'Wrong public reference deficit')
    close_to(stated['numerator_bb100'], g, 'Wrong public deficit reduction')
    close_to(stated['denominator_ci95'], [d-half, d+half], 'Wrong public deficit interval')
    ratio = g/d if d > 0 else None
    if ratio is None:
        check(stated['fraction'] is None, 'No positive reference deficit to recover')
    else:
        close_to(stated['fraction'], ratio, 'Wrong recovered share')
    if d-half <= 0:
        check(stated['paired_fieller_ci95'] is None, 'Unbounded recovery ratio')
    else:
        lo, hi = stated['paired_fieller_ci95']
        check(lo <= ratio <= hi and stated['interval_status'] == 'bounded', 'Wrong Fieller confidence set')
        # At both endpoints a paired test of gain - r*gap == 0 is exactly at its critical value.
        for value, sign in ((lo, 1), (hi, -1)):
            residual = gain - value*gap
            close_to(residual.mean(), sign*z*residual.std(ddof=1)/np.sqrt(len(gap)),
                     'Recovery interval ignores denominator variance or pairing')
    return dict(fraction=ratio, paired_fieller_ci95=stated['paired_fieller_ci95'], gap_bb100=d, gain_bb100=g)


def audit(follow=False):
    plan, matrices, receipt = verify_panel(follow=follow)
    path = STUDY/'summary.json'
    if follow:
        wait_for(path)
    summary = read(path)
    check(summary['complete'] and not summary['model_promoted'] and summary['plan_sha256'] == receipt['plan_sha256']
          and summary['previous_plan_sha256'] == PREVIOUS_PLAN and summary['previous_summary_sha256'] == PREVIOUS_SUMMARY
          and summary['previous_reference_plan_sha256'] == PREVIOUS_REFERENCE and
          (summary['new_hands'], summary['reference_hands'], summary['completed_hands']) == (648000,1296000,1944000)
          and (summary['new_cells'], summary['reference_cells'], summary['completed_cells']) == (27,54,81)
          and summary['unique_original_card_blocks'] == 26000 and summary['cell_result_sha256'] == receipt['result_sha256'],
          'Final report references different cells or registered design')
    tapes = {key.replace('/seed-', '/'): dict(counts=tape['private_snapshot_counts'],
                                              hand_choice_sha256=tape['private_slot_order_sha256'])
             for key,tape in receipt['cell_audits'].items()}
    check(tapes == summary['snapshot_hand_assignments'], 'Summary does not bind complete private hand-slot tapes')
    public = {}
    data = matrices['public']
    for left, right in ((TREATMENT,HYBRID), (BASE,HYBRID), (BASE,TREATMENT)):
        key = f'{right}_minus_{left}'
        verified = paired_stats(data[left], data[right], NormalDist().inv_cdf(.975))
        reported = summary['public_comparisons'][key]
        check((reported['baseline'], reported['candidate']) == (left, right), 'Incorrect public contrast')
        for column, stat in [('mean',verified['mean']),('paired_ci95',verified['interval']),
                             ('baseline_bb100',data[left].mean()),('candidate_bb100',data[right].mean())]:
            close_to(reported[column], stat, 'Different public paired contrast or interval')
        close_to([reported['by_seed'][str(s)] for s in SEEDS], verified['fixed_seed_changes'], 'Incorrect per-seed mean')
        public[key] = verified
    check(summary['primary_public_comparison'] == f'{HYBRID}_minus_{TREATMENT}' and
          set(summary['public_comparisons']) == set(public), 'Wrong primary comparison')
    panels = {left:audit_pair(summary[f'treatment_vs_{left}'], plan['fields'], matrices, left, HYBRID)
              for left in (TREATMENT, BASE)}
    receipt.update(verified_utc=utc(), summary_sha256=digest(path), audit_source_sha256=digest(__file__),
                   independently_recomputed_public=public, independently_recomputed_tables=panels,
                   public_gap_recovery=check_public_recovery(data,summary['public_gap_recovery']),
                   scope='Previously observed cards motivated the rollback. A terminal return includes all later actions.')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow', action='store_true', help='Check each cell when it finishes')
    atomic_json(STUDY/'final-audit.json', audit(follow=parser.parse_args().follow))
