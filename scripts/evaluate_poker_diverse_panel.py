"""Evaluate one immutable specialist against a seat-balanced six-player table."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import statistics
import time

from evaluate_average_baseline import (_check_deadline, _validate_reply, _verify_rows,
    check_dependencies, hand_seed, interval, policy_dependency_hashes, semantic_seed, utc)
from fullhouse_runtime import Bridge, Fullhouse, ROOT
from fullhouse_v3_common import atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import PROFILES, ReferenceOpponent, checked_action
from rich_poker_runtime import RichPolicy


DOSE_PLAN = ROOT/'artifacts/preflop-followup/480-v1/preflop-dose/evaluation-plan.json'
DOSE_SHA = '642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'
RICH_STAGE = ROOT/'artifacts/rich-selfplay/20261113T075620400993Z/stage-ready.json'
RICH_SHA = '8970ce73ceabe645ab6c98bd5419d3e23ed19c9b82edf0a8700820471e8ff4e6'
FIELDS = ('public', 'noregrets', 'rich96', *PROFILES, 'mixed', 'varied')
VARIED = ('public', 'noregrets', 'rich96', 'tight-value', 'pressure')
MIXED = ('public', 'noregrets', 'public', 'noregrets', 'public')


def lineup(field, deal_seed):
    """Opponents in clockwise order relative to hero. Balance the varied roles across cards."""
    if field == 'mixed':
        return MIXED
    if field == 'varied':
        offset = deal_seed % len(VARIED)
        return VARIED[offset:] + VARIED[:offset]
    if field not in FIELDS:
        raise ValueError('Unknown table')
    return (field,) * 5


def frozen_inputs(label, seed, field):
    check_dependencies({str(DOSE_PLAN): DOSE_SHA})
    plan = json.loads(DOSE_PLAN.read_text())
    ref = plan['policies'][label][str(seed)]
    check_dependencies({ref['path']: ref['sha256']})
    candidate = evaluator_policy(ref['path'], plan['source'])
    dependencies = policy_dependency_hashes(candidate, plan['source'])
    dependencies[str(DOSE_PLAN)] = DOSE_SHA
    dependencies[str(Path(__file__).resolve())] = digest(__file__)
    rules_file = Path(__file__).with_name('poker_reference_opponents.py').resolve()
    dependencies[str(rules_file)] = digest(rules_file)
    strategies = {}
    opponents = {}
    bridge_model = None
    for role in sorted(set(lineup(field, 0))):
        if role == 'noregrets':
            registry = ROOT/'artifacts/selfplay/current.json'
            registry_data = json.loads(registry.read_text())
            bridge_model = ROOT/registry_data['model']
            expected = plan['opponents']['noregrets']['noregrets']
            if registry_data['sha256'] != expected:
                raise ValueError('NoRegrets reference differs from the original opponent')
            dependencies[str(registry)] = digest(registry)
            dependencies[str(bridge_model)] = expected
            opponents[role] = expected
        elif role == 'public':
            strategies[role] = Fullhouse()
            expected = plan['opponents']['public']['public']
            if strategies[role].digest != expected:
                raise ValueError('Public opponent differs from prior study')
            dependencies[str(strategies[role].model)] = expected
            opponents[role] = expected
        elif role == 'rich96':
            check_dependencies({str(RICH_STAGE): RICH_SHA})
            rich_ref = json.loads(RICH_STAGE.read_text())['policies']['rich96'][str(seed)]
            check_dependencies({rich_ref['path']: rich_ref['sha256']})
            rich = RichPolicy(rich_ref['path'], None)
            if (rich.spec['iteration'], rich.spec['seed']) != (96, seed):
                raise ValueError('Incorrect rich96 opponent provenance')
            strategies[role] = rich
            dependencies[str(RICH_STAGE)] = RICH_SHA
            dependencies.update(rich.dependency_hashes)
            opponents[role] = rich_ref['sha256']
        else:
            strategies[role] = ReferenceOpponent(role)
            opponents[role] = {'profile': role, 'source_sha256': dependencies[str(rules_file)]}
    for name, sha in dependencies.items():
        if name in plan['frozen_files'] and plan['frozen_files'][name] != sha:
            raise ValueError('Original dependency identity changed: '+name)
    check_dependencies(dependencies)
    return candidate, strategies, opponents, bridge_model, dependencies


def play(engine, candidate, strategies, seats, card, hero, counts, deadline):
    state = engine.call({'op': 'reset', 'seed': card, 'button': 0})
    session_seed = hand_seed(card, hero)
    actions = []
    if hasattr(candidate, 'begin_hand'):
        candidate.begin_hand(session_seed)
    while not state['terminal']:
        _check_deadline(deadline)
        if len(actions) >= 512:
            raise ValueError('Nonterminating hand')
        actor = state['seat_to_act']
        rng = semantic_seed(card, hero, state)
        if actor == hero:
            role = 'hero'
            reply = _validate_reply(state, candidate.decide(state, rng))
        else:
            role = seats[(actor - hero) % 6 - 1]
            if role == 'noregrets':
                reply = engine.call({'op': 'blueprint', 'seed': rng})
                counts['noregrets/decisions'] += 1
                counts['noregrets/fallback'] += int(reply.get('fallback', False))
            elif role in PROFILES:
                reply = strategies[role].decide(state, rng)
            else:
                reply = _validate_reply(state, strategies[role].decide(state, rng))
        action = checked_action(state, reply['action'])
        counts[f'{role}/{state["street"]}/{action["kind"]}'] += 1
        concrete = dict(action, seat=actor)
        actions.append(concrete)
        state = engine.call({'op': 'step', 'action': concrete})
    utilities = state['utilities']
    if len(utilities) != 6 or any(type(v) is not int for v in utilities) or sum(utilities) != 0:
        raise ValueError('Invalid terminal utilities')
    return dict(hero=hero, seed=card, hand_seed=session_seed, actions=actions, steps=len(actions),
                chips=utilities[hero], utilities=utilities)


def run(label, seed, field, start, deals, output, timeout=7200):
    if (label not in ('steps0', 'steps256') or seed not in (342, 343, 344) or field not in FIELDS
            or not 2 <= deals <= 20000 or not 0 <= start < start+deals <= 2**32 or not 0 < timeout <= 7200):
        raise ValueError('Invalid bounded evaluation configuration')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+timeout
    pid = os.getpid()
    start_token = process_start(pid)
    def status(phase, **values):
        atomic_json(output/'status.json', dict(phase=phase, pid=pid, process_start=start_token,
                    observed_utc=utc(), elapsed_seconds=time.monotonic()-started, **values))
    try:
        status('preparing')
        candidate, strategies, opponents, bridge_model, deps = frozen_inputs(label, seed, field)
        provenance = dict(label=label, training_seed=seed, candidate=str(candidate.model),
            candidate_sha256=deps[str(candidate.model)], candidate_type=candidate.model_type,
            field=field, deal_seed=start, deals=deals, hands=deals*6, big_blind_chips=100,
            opponents=opponents, dependency_hashes=deps, evaluator_sha256=digest(__file__),
            first_deal_clockwise_lineup=lineup(field, start), varies_by_card=field=='varied',
            semantic_rng='SHA256 of card seed, hero seat, actor and public action log',
            statistical_unit='One original card block; hero seats and training seeds are repeated observations')
        atomic_json(output/'provenance.json', provenance)
        cards = []
        counts = Counter()
        raw_path = output/'hands.jsonl.gz'
        engine = Bridge(bridge_model)
        try:
            status('evaluating', completed_card_blocks=0)
            with gzip.open(raw_path, 'xt', compresslevel=1) as stream:
                for deal, card in enumerate(range(start, start+deals)):
                    seats = lineup(field, card)
                    chips = []
                    for hero in range(6):
                        row = play(engine, candidate, strategies, seats, card, hero, counts, deadline)
                        row['deal'] = deal
                        chips.append(row['chips'])
                        stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
                    cards.append(statistics.mean(chips))
                    if (deal+1) % 100 == 0:
                        status('evaluating', completed_card_blocks=deal+1)
        finally:
            engine.close()
        stats = interval(cards)
        result = dict(provenance, bb_per_100=stats['mean'], ci95_half_width=stats['ci95_half_width'],
                      behavior=dict(counts), deal_blocks=cards)
        status('verifying_raw_hands')
        verification = _verify_rows(result, raw_path, replay=True, deadline=deadline, progress=status)
        check_dependencies(deps)
        _check_deadline(deadline)
        result.update(complete=True, verification=verification, raw_hands_verified=True,
                      terminal_payoffs_replayed=True, raw_hands_sha256=digest(raw_path),
                      provenance_sha256=digest(output/'provenance.json'), completed_utc=utc(),
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'results.json', result)
        status('complete', verified_hands=deals*6, results_sha256=digest(output/'results.json'))
        return result
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}', automatic_retry=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True, choices=('steps0', 'steps256'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--field', required=True, choices=FIELDS)
    parser.add_argument('--deal-seed', type=int, required=True)
    parser.add_argument('--deals', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=7200)
    args = parser.parse_args()
    run(args.label, args.seed, args.field, args.deal_seed, args.deals, args.output, args.timeout)
