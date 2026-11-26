"""Check the three historical training opponents and the payoff contrasts from raw chips."""

import gzip
import hashlib
import json
import math
import random
import statistics

from audit_poker_population import check
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest

BASE = ROOT / 'artifacts/preflop-followup/480-v1'
PARENT = BASE / 'opponent-diversity-20261123'
STUDY = BASE / 'pool-extension-528-20261126'
OUTPUT = STUDY / 'training-opponents'
PUBLIC = ROOT / 'reports' / OUTPUT.relative_to(ROOT / 'artifacts')
REGISTERED = 'd88c206f2837af68a4a344010dce08930e3c31a8adee17a78c0facb196f07571'
LABELS = ('specialist480', 'mixture504', 'mixture528')
OPPONENTS = (300, 360, 420)
SEEDS = (342, 343, 344)
START, CARDS = 2060300000, 1000


def check_policy_sources():
    plan_file = OUTPUT / 'plan.json'
    plan = read(plan_file)
    key = read(OUTPUT / 'bound.json')['sha256']
    original = PARENT / 'training-plan.json'
    before_launch = read(STUDY / 'source-state-audit.json')['files']
    verify_files(
        {
            plan_file: key,
            OUTPUT / 'design.json': REGISTERED,
            PUBLIC / 'design.json': REGISTERED,
            original: before_launch[str(original.relative_to(ROOT))],
            STUDY / 'crossplay/evaluation-plan.json': plan['cross_plan_sha256'],
        }
    )
    verify_files(plan['frozen_files'])
    reg, parent = read(OUTPUT / 'design.json'), read(original)
    fork = read(STUDY / 'training-plan.json')
    cross = read(STUDY / 'crossplay/evaluation-plan.json')
    left, right = (read(folder / 'stage-ready.json')['policies'] for folder in (PARENT, STUDY))
    check(
        tuple(reg['hero']) == LABELS
        and tuple(reg['opponents']) == OPPONENTS
        and tuple(reg['seeds']) == SEEDS
        and reg['cards'] == [START, CARDS]
        and read(STUDY / 'crossplay/design.json')['chips_per_big_blind'] == 100,
        'Design changed',
    )
    check(
        parent['historical_policies'] == fork['historical_policies']
        and cross['policies']
        == {
            'specialist480': left['specialist480'],
            'mixture504': left['poolMixture504'],
            'mixture528': right['mixture528'],
        },
        'Opponent pool or compared hero policies differ from archived stages',
    )
    hashes = {}
    for seed in SEEDS:
        sid = str(seed)
        reference = parent['parent_policies'][sid]
        check(reference == left['specialist480'][sid], 'Different fixed specialist')
        verify_files({reference['path']: reference['sha256']})
        preflop = read(reference['path'])['preflop']
        for round_ in OPPONENTS:
            n = str(round_)
            postflop = parent['historical_policies'][sid][n]
            policy_path = plan['opponents'][sid][n]
            policy = read(policy_path)
            check(
                policy == {'format': 'preflop-street-policy-v1', 'preflop': preflop, 'postflop': postflop}
                and reg['archived_opponents_sha256'][sid][n] == postflop['sha256']
                and all(plan['frozen_files'][ref['path']] == ref['sha256'] for ref in (preflop, postflop)),
                'Evaluation does not reproduce the training-time street routing',
            )
            verify_files({preflop['path']: preflop['sha256'], postflop['path']: postflop['sha256']})
            hashes[f'{seed}-{round_}'] = {street: policy[street]['sha256'] for street in ('preflop', 'postflop')}
    return key, hashes


def tapes():
    values = {}
    for card in range(START, START + CARDS):
        for seat in range(6):
            key = hashlib.sha256(f'rich-evaluation-hand:{card}:{seat}'.encode()).digest()[:8]
            slot = random.Random(b'scheduled-postflop-choice-v1\0' + key).randrange(4)
            values[card, seat] = (int.from_bytes(key, 'big'), slot)
    return values


def read_chips(label, opponent, seed, expected_hash, plan_sha, hand_tapes):
    folder = OUTPUT / 'evaluation' / f'{label}-vs-{opponent}-{seed}'
    result = folder / 'results.json'
    verify_files({result: expected_hash})
    metadata = read(result)
    raw = folder / 'hands.jsonl.gz'
    verify_files({raw: metadata['raw_sha256']})
    check(metadata['complete'] and metadata['plan_sha256'] == plan_sha, 'Unbound cell')
    chips = [0] * CARDS
    hands = decisions = 0
    with gzip.open(raw, 'rb') as source:
        for line in source:
            hand = json.loads(line)
            card_index, seat = divmod(hands, 6)
            card = START + card_index
            check(
                card_index < CARDS and (hand['seed'], hand['deal'], hand['hero']) == (card, card_index, seat),
                'Mismatched paired deals or repeated seat',
            )
            key, slot = hand_tapes[card, seat]
            expected_slots = [None] * 6
            if label != 'specialist480':
                expected_slots[seat] = slot
            pay = hand['utilities']
            check(
                len(pay) == 6
                and all(type(v) is int for v in pay)
                and sum(pay) == 0
                and hand['chips'] == pay[seat]
                and hand['hand_seed'] == key
                and hand['slots'] == expected_slots
                and 0 < hand['steps'] == len(hand['actions']) <= 512,
                'Invalid raw chip ledger or policy role',
            )
            chips[card_index] += pay[seat]
            hands += 1
            decisions += hand['steps']
    check(
        hands == metadata['hands'] == 6 * CARDS
        and metadata['actions'] == decisions
        and [v / 6 for v in chips] == metadata['blocks'],
        'Stored aggregates differ from raw utilities',
    )
    return chips, hands, decisions


def check_estimate(seed_totals, reported, comparisons):
    # Chip sums span six seats. One big blind is 100 chips, so bb/100 equals chips/hand.
    fixed_seed_means = {str(s): statistics.mean(row) / 6 for s, row in zip(SEEDS, seed_totals, strict=True)}
    card_means = [sum(row) / 18 for row in zip(*seed_totals, strict=True)]
    mean = statistics.mean(card_means)
    se = statistics.stdev(card_means) / math.sqrt(CARDS)
    radius = statistics.NormalDist().inv_cdf(1 - 0.05 / (2 * comparisons)) * se
    values = {'mean': mean, 'lower': mean - radius, 'upper': mean + radius}
    check(
        all(math.isclose(reported[k], v, abs_tol=1e-8, rel_tol=0) for k, v in values.items())
        and all(math.isclose(reported['by_seed'][s], v, abs_tol=1e-8, rel_tol=0) for s, v in fixed_seed_means.items()),
        'Published mean, fixed-seed averages or simultaneous limits differ',
    )
    return values


def main():
    plan_sha, policies = check_policy_sources()
    proof = read(OUTPUT / 'summary.json')
    verify_files({PUBLIC / 'summary.json': digest(OUTPUT / 'summary.json')})
    expected_hashes = proof['cell_sha256']
    expected_names = {f'{h}-vs-{v}-{s}' for h in LABELS for v in OPPONENTS for s in SEEDS}
    check(
        set(expected_hashes) == expected_names
        and proof['complete']
        and proof['plan_sha256'] == plan_sha
        and proof['design_sha256'] == REGISTERED
        and proof['original_cards_shared_with_crossplay'] == CARDS
        and proof['simultaneous_two_sided_95_family_sizes'] == {'means': 9, 'paired_contrasts': 6},
        'Incomplete or changed historical panel',
    )
    check(
        {d.name for d in (OUTPUT / 'evaluation').iterdir() if d.is_dir()} == expected_names,
        'Unexpected historical panel cells',
    )
    hand_tapes = tapes()
    hands = decisions = 0
    contrasts, means = {}, {}
    for opponent in OPPONENTS:
        n = str(opponent)
        by_label, contrasts[n], means[n] = {}, {}, {}
        for label in LABELS:
            totals = []
            for seed in SEEDS:
                chips, count, actions = read_chips(
                    label, opponent, seed, expected_hashes[f'{label}-vs-{opponent}-{seed}'], plan_sha, hand_tapes
                )
                hands += count
                decisions += actions
                totals.append(chips)
            by_label[label] = totals
            means[n][label] = check_estimate(totals, proof['means_bb100'][n][label], 9)
        for base in ('mixture504', 'specialist480'):
            differences = [
                [a - b for a, b in zip(x, y, strict=True)]
                for x, y in zip(by_label['mixture528'], by_label[base], strict=True)
            ]
            contrasts[n][base] = check_estimate(
                differences, proof['paired_528_minus_reference_bb100'][n][base], 6
            )
    check(
        hands == 162000 == proof['terminal_replayed_hands'] and decisions == proof['terminal_replayed_actions'],
        'Incorrect total hands or actions',
    )
    result = {
        'complete': True,
        'verified_utc': utc(),
        'source_training_plan_sha256': digest(PARENT / 'training-plan.json'),
        'study_plan_sha256': plan_sha,
        'summary_sha256': digest(OUTPUT / 'summary.json'),
        'training_opponent_weights_by_seed_and_iteration': policies,
        'raw_ledger_cells': len(expected_hashes),
        'hands': hands,
        'actions': decisions,
        'distinct_deals_shared_with_crossplay': CARDS,
        'fixed_seeds': list(SEEDS),
        'simultaneous_two_sided_95_family_sizes': {'means': 9, 'paired_contrasts': 6},
        'means_bb100': means,
        'paired_528_minus_reference_bb100': contrasts,
    }
    for directory in (OUTPUT, PUBLIC):
        atomic_json(directory / 'paired-audit.json', result)


if __name__ == '__main__':
    main()
