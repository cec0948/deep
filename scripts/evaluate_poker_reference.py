"""Common-card evaluation against transparent, independent behavior baselines.

No training, promotion, or interpretation as professional poker strength occurs.
Every hero seat plays the same deck schedule in each comparison. Diagnostic
payoffs conditional on encountered contexts are observational, not causal.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

from fullhouse_runtime import Bridge, BRIDGE
from fullhouse_v3_common import ROOT, atomic_json, digest
from fullhouse_v4_runtime import policy_for
from poker_reference_opponents import PROFILES, SPEC, ReferenceOpponent, checked_action, observation


def interval(values):
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** .5 if len(values) > 1 else None
    return dict(mean=mean, ci95_half_width=half, lower=mean-half if half is not None else None,
                upper=mean+half if half is not None else None, blocks=len(values))


def context(state):
    owed = min(state['amount_owed'], state['your_stack'])
    odds = owed / max(1, state['pot'] + owed)
    pressure = 'free' if owed == 0 else 'low-price' if odds <= .20 else 'medium-price' if odds <= .35 else 'high-price'
    alive = sum(not p['is_folded'] for p in state['players'])
    players = 'heads-up' if alive <= 2 else 'multiway'
    stack_pot = state['your_stack'] / max(1, state['pot'])
    spr = 'low-spr' if stack_pot <= 2 else 'medium-spr' if stack_pot <= 6 else 'high-spr'
    return f"{state['street']}/{pressure}/{players}/{spr}"


def hand_seed(deal_seed, deal, hero):
    return int.from_bytes(hashlib.sha256(f'compact-mixture:{deal_seed}:{deal}:{hero}'.encode()).digest()[:8], 'big')


def verify_raw(path, deals, deal_seed):
    """Recompute payoff blocks from the complete sequential raw record."""
    values = [[] for _ in range(deals)]
    count = 0
    with gzip.open(path, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            deal, hero = divmod(count, 6)
            if deal >= deals or row['deal'] != deal or row['hero'] != hero or row['seed'] != deal_seed + deal:
                raise ValueError('Missing, reordered, or duplicated raw hand')
            utilities = row['utilities']
            if (len(utilities) != 6 or any(type(x) is not int for x in utilities)
                    or sum(utilities) != 0 or row['chips'] != utilities[hero]
                    or not -10000 <= row['chips'] <= 50000
                    or row['hand_seed'] != hand_seed(deal_seed, deal, hero)
                    or not 1 <= row['steps'] <= 512):
                raise ValueError('Invalid raw terminal payoff/provenance')
            values[deal].append(row['chips']); count += 1
    if count != deals * 6:
        raise ValueError('Incomplete raw hand record')
    # The engine uses100chips perbigblind, so chips/hand equals bb/100hands.
    return [statistics.mean(row) for row in values]


def run(model, source, opponent, deals, deal_seed, output, trace_hands=12, timeout=1800):
    model, source, output = Path(model).resolve(), Path(source).resolve(), Path(output).resolve()
    if deals < 2 or type(deal_seed) is not int or not 0 <= deal_seed < 2**64-deals:
        raise ValueError('Invalid bounded evaluation/card range')
    if not 0 <= trace_hands <= 120 or not 0 < timeout <= 3600:
        raise ValueError('Invalid trace/time budget')
    if output == source or source in output.parents or output == model.parent or model.parent in output.parents:
        raise ValueError('Evaluation output must be outside immutable model/source inputs')
    policy = policy_for(model, source)
    reference = ReferenceOpponent(opponent)
    frozen = {str(model): digest(model), str(BRIDGE): digest(BRIDGE)}
    for relative in ('scripts/evaluate_poker_reference.py', 'scripts/poker_reference_opponents.py',
                     'scripts/fullhouse_runtime.py', 'scripts/fullhouse_v4_runtime.py',
                     'scripts/fullhouse_v3_common.py', 'scripts/regret_partition.py',
                     'configs/fullhouse-model.json'):
        frozen[str(ROOT/relative)] = digest(ROOT/relative)
    for item in source.rglob('*.py'):
        frozen[str(item)] = digest(item)
    if hasattr(policy, 'models'):
        for item in policy.models:
            path = model.parent/item['file']
            if digest(path) != item['sha256']:
                raise ValueError('Mixture snapshot changed before evaluation')
            frozen[str(path)] = item['sha256']
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(model=str(model), source=str(source), opponent=opponent, opponent_spec=SPEC,
                deals=deals, deal_seed=deal_seed, trace_hands=trace_hands, timeout=timeout,
                frozen_files=frozen, eval7_version=importlib.metadata.version('eval7'),
                random_schedule='candidate seed=deal*100000+hero*1000+step; reference uses a profile-separated SHA256 stream',
                seat_schedule='button0, hero0..5, same shuffled52carddeck for allsixhero seats perdeal',
                extra_trace_payoff_note='Payoff is whole-hand utility; context associations do not establish action causality')
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    start = time.monotonic(); counts = Counter(); contexts = defaultdict(list)
    seats = defaultdict(list); blocks = []; traces = []; engine = Bridge()
    try:
        with gzip.open(output/'hands.jsonl.gz', 'wt', compresslevel=1) as raw:
            for deal in range(deals):
                payoffs = []
                for hero in range(6):
                    state = engine.call(dict(op='reset', seed=deal_seed+deal, button=0))
                    independent_seed = hand_seed(deal_seed, deal, hero)
                    if hasattr(policy, 'begin_hand'):
                        policy.begin_hand(independent_seed)
                    steps = 0; seen_contexts = set(); hero_actions = Counter(); trace = []
                    while not state['terminal']:
                        if time.monotonic()-start > timeout:
                            raise TimeoutError('Reference evaluation time budget exceeded')
                        seat = state['seat_to_act']; seed = deal*100000+hero*1000+steps
                        is_hero = seat == hero
                        reply = policy.decide(state, seed) if is_hero else reference.decide(state, seed)
                        action = checked_action(observation(state), reply['action'])
                        tag = context(state)
                        role = 'hero' if is_hero else 'opponent'
                        counts[f'{role}/{tag}/{action["kind"]}'] += 1
                        if is_hero:
                            seen_contexts.add(tag); hero_actions[f'{state["street"]}/{action["kind"]}'] += 1
                        if deal*6+hero < trace_hands:
                            trace.append(dict(step=steps, seat=seat, role=role, street=state['street'],
                                acting_player_cards=state['your_cards'], public_board=state['community_cards'],
                                pot=state['pot'], to_call=state['amount_owed'], stack=state['your_stack'],
                                raise_bounds=state['raise_bounds'], context=tag, action=action,
                                distribution=reply.get('distribution')))
                        state = engine.call(dict(op='step', action=dict(action, seat=seat)))
                        steps += 1
                        if steps > 512:
                            raise RuntimeError('Nonterminating hand')
                    if sum(state['utilities']) != 0:
                        raise AssertionError('Chip conservation failed')
                    chips = state['utilities'][hero]
                    row = dict(deal=deal, hero=hero, seed=deal_seed+deal, chips=chips,
                        utilities=state['utilities'], steps=steps, hand_seed=independent_seed,
                        policy_iteration=getattr(policy, 'active_iteration', None),
                        hero_contexts=sorted(seen_contexts), hero_actions=dict(hero_actions))
                    raw.write(json.dumps(row)+'\n'); payoffs.append(chips); seats[hero].append(chips)
                    for tag in seen_contexts:
                        contexts[tag].append(chips)
                    if trace:
                        traces.append(dict(deal=deal, hero=hero, seed=deal_seed+deal, actions=trace, utilities=state['utilities']))
                blocks.append(statistics.mean(payoffs))
                if (deal+1) % 50 == 0 or deal+1 == deals:
                    atomic_json(output/'progress.json', dict(completed_deals=deal+1, planned_deals=deals,
                        elapsed_seconds=time.monotonic()-start))
    finally:
        engine.close()
    verified = verify_raw(output/'hands.jsonl.gz', deals, deal_seed)
    if verified != blocks:
        raise ValueError('Raw payoffs and computed deal blocks disagree')
    for name, expected in frozen.items():
        if digest(name) != expected:
            raise ValueError('Frozen evaluation input changed: '+name)
    if digest(output/'plan.json') != plan_sha:
        raise ValueError('Evaluation plan changed')
    atomic_json(output/'traces.json', traces)
    result = dict(complete=True, field='reference/'+opponent, opponent=opponent, deals=deals,
        hands=deals*6, deal_seed=deal_seed, candidate=str(model), candidate_sha256=policy.digest,
        candidate_type=policy.model_type, bb_per_100=statistics.mean(blocks), interval=interval(blocks),
        deal_blocks=blocks, per_seat={str(k): interval(v) for k, v in seats.items()},
        behavior=dict(counts), context_outcomes={k:dict(hands=len(v),mean_hand_chips=statistics.mean(v)) for k,v in contexts.items()},
        context_outcome_interpretation='Each hand counted once per encountered context; contexts overlap; whole-hand conditional payoffs are associations, not action values or causal effects',
        elapsed_seconds=time.monotonic()-start, raw_hands_verified=True,
        plan_sha256=plan_sha, raw_hands_sha256=digest(output/'hands.jsonl.gz'),
        traces_sha256=digest(output/'traces.json'), opponent_source_sha256=digest(ROOT/'scripts/poker_reference_opponents.py'),
        evaluator_sha256=digest(Path(__file__)), runtime_sha256=digest(ROOT/'scripts/fullhouse_v4_runtime.py'),
        engine_sha256=digest(BRIDGE), source_features_sha256=digest(source/'bot/features.py'),
        source_provenance_verified=True,
        limitations=SPEC['limitations']+['Nominal95%deal-block intervals; shared decks are not independent across seats, seeds or policies',
            'These three fixed behavior families expand regression coverage but do not establish general strength',
            'Per-context utilities require followup controlled comparisons before attribution',
            'No training, automatic promotion, or change to original four continuation conditions'])
    atomic_json(output/'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--opponent', choices=PROFILES, required=True)
    parser.add_argument('--deals', type=int, default=1000)
    parser.add_argument('--deal-seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-hands', type=int, default=12)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    result = run(args.model, args.source, args.opponent, args.deals, args.deal_seed,
                 args.output, args.trace_hands, args.timeout)
    print(json.dumps({k:result[k] for k in ('complete','opponent','hands','bb_per_100','interval','elapsed_seconds')}), flush=True)
