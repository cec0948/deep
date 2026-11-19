"""Independently audit the 480-round specialist's 0/64/256-step fixed-replay study."""
from contextlib import ExitStack, closing
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np

from audit_poker_warm_preflop import DEALS, FIELDS, SEATS, SEEDS, FirstActionDifferences, check, check_estimate
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest


STUDIES = ROOT / 'artifacts/preflop-followup/480-v1'
STUDY = STUDIES / 'preflop-dose'
SOURCE_SUMMARY = '263c20db53b9ef78a0ba1ac6081f58c323b8c3a2d3500a218ca8ab7bfd874dcf'
SOURCE_AUDIT = '7c476c13e5b91f377a896716762a03269ec2045eb083b44efd642fe12edb1a0a'
SOURCE_PLAN = 'ff9bf107066817372f1529962d6743a69e056a98b35b97e6e87199f312be7399'
START = 2037700000
LABELS = ('steps0', 'steps64', 'steps256')
CONTRASTS = {'64_minus_0': (0, 1), '256_minus_0': (0, 2), '256_minus_64': (1, 2)}
STEPS = (0, 64, 256)
BATCH = 512
REPLAY_STRIDE = 250


def verify_fit(seed, plan, previous, fit):
    from compact_regret import ARRAY_MAP
    from poker_preflop_specialist import preflop_rows
    from regret_partition import base_strata
    from scaled_regret_fitting import ScaledRegretSampler
    from train_poker_average_continuation import load_checkpoint

    ref = {label: plan['policies'][label][str(seed)] for label in LABELS}
    check(ref['steps0'] == previous['policies']['specialist480'][str(seed)] and fit['policies'] == ref,
          f'{seed}: unexpected parent or fitted weights')
    parent = read(ref['steps0']['path'])
    campaign = Path(previous['source']).parent
    replay, checkpoint = load_checkpoint(campaign / f'seed-{seed}')
    source_stage = campaign.parent/'stage-ready.json'
    warm_plan = STUDIES/'warm-start'/'evaluation-plan.json'
    check(fit['source_plan'] == dict(path=str(warm_plan), sha256=SOURCE_PLAN)
          and fit['source_stage'] == dict(path=str(source_stage), sha256=previous['frozen_files'][str(source_stage)])
          and fit['seed'] == seed and fit['iteration'] == 480 and fit['new_selfplay_iterations'] == 0,
          'Different fitted source')
    verify_files({source_stage: fit['source_stage']['sha256']})
    stage = read(source_stage)['seeds'][str(seed)]
    check(stage['iteration'] == 480 and Path(stage['checkpoint']) == checkpoint
          and stage['files'] == fit['connection']['checkpoint_files'], 'Different stage or checkpoint inputs')
    verify_files({checkpoint/name: sha for name, sha in stage['files'].items()})
    check((replay['seed'], replay['iteration']) == (seed, 480)
          and checkpoint/'current.npz' == Path(parent['postflop']['path']), 'Wrong replay provenance')
    check(len(replay['buffers']) == SEATS and all(b['count'] == 100000 for b in replay['buffers']),
          'Unexpected replay buffer coverage')
    samples = [preflop_rows(buffer) for buffer in replay['buffers']]
    del replay
    checksum = hashlib.sha256()
    for seat, sample in enumerate(samples):
        for key in ('features', 'advantages', 'iterations'):
            data = getattr(sample, key)
            checksum.update(f'{seat}/{key}/{data.shape}/{data.dtype.str}\n'.encode())
            checksum.update(data.tobytes())
    check(fit['selected_rows_per_seat'] == [len(s) for s in samples] and fit['replay_sha256'] == checksum.hexdigest()
          and fit['sampler_method'] == 'stratum-scaled-feature-conditional', 'Changed sampler or replay rows')
    sampler = ScaledRegretSampler(samples, base_strata, 'stratum-scaled-feature-conditional')
    rng_seed = 20261119 + seed
    check((fit['sampler_seed'], fit['batch_size'], fit['learning_rate']) == (rng_seed, BATCH, .001),
          f'{seed}: incorrect fitting hyperparameters')
    uniforms = np.random.RandomState(rng_seed).random_sample(BATCH * max(STEPS))
    ids = np.asarray(np.searchsorted(sampler.cdf, uniforms, side='right'), dtype='<u8')
    order = {steps: hashlib.sha256(ids[:BATCH*steps].tobytes()).hexdigest() for steps in STEPS}
    proof = {}
    with np.load(parent['preflop']['path'], allow_pickle=False) as base:
        mutable = set(ARRAY_MAP.values())
        check(mutable < set(base.files) and 'opening_scale' in base and 'main_scales' in base,
              'Incomplete specialist export')
        for label, steps in zip(LABELS, STEPS):
            evidence = fit['arms'][label]
            check(evidence['optimizer_steps'] == steps and evidence['training_draws'] == BATCH*steps
                  and evidence['sample_order_sha256'] == order[steps]
                  and evidence['prefix64_sha256'] == (order[64] if steps else None), f'{seed}/{label}: sampling proof differs')
            policy = read(ref[label]['path'])
            check(policy['format'] == 'preflop-street-policy-v1' and policy['postflop'] == parent['postflop'],
                  'Postflop policy changed')
            verify_files({asset['path']: asset['sha256'] for asset in (ref[label], policy['preflop'], policy['postflop'])})
            with np.load(policy['preflop']['path'], allow_pickle=False) as model:
                check(set(model.files) == set(base.files)
                      and all(np.isfinite(model[k]).all() for k in base.files if base[k].dtype.kind in 'fc'),
                      'Different architecture or nonfinite weights')
                changed = {k for k in base.files if not np.array_equal(base[k], model[k])}
                check(changed <= mutable and bool(changed) == (steps != 0), 'Unexpected frozen or updated tensors')
                # Reconstruct the training-state digest from the deployed NumPy export.
                state_arrays = {name: name for name in ('main_scales', 'opening_scale')}
                for expert, prefix in (('main', ''), ('opening', 'opening_')):
                    state_arrays.update({f'{expert}.{key}': prefix+value for key, value in ARRAY_MAP.items()})
                weights = hashlib.sha256()
                for key, array in state_arrays.items():
                    weights.update(key.encode())
                    weights.update(model[array].tobytes())
                check(weights.hexdigest() == evidence['weights_sha256'], 'Export differs from fitted training weights')
            proof[label] = dict(preflop_sha256=policy['preflop']['sha256'], postflop_sha256=policy['postflop']['sha256'],
                                sample_order_sha256=order[steps], optimizer_steps=steps, training_draws=BATCH*steps)
    check(len({info['preflop_sha256'] for info in proof.values()}) == len(STEPS), 'Duplicate dose weights')
    return dict(replay_sha256=checksum.hexdigest(), selected_rows_per_seat=fit['selected_rows_per_seat'], arms=proof)


def verify_design(plan, summary):
    warm = STUDIES/'warm-start'/'evaluation-plan.json'
    swap = STUDIES/'preflop-swap'
    verify_files({warm: SOURCE_PLAN, swap/'summary.json': SOURCE_SUMMARY, swap/'final-audit.json': SOURCE_AUDIT,
                  STUDY/'evaluation-plan.json': summary['evaluation_plan_sha256'],
                  STUDY/'fit-ready.json': summary['fit_ready_sha256']})
    frozen = plan['frozen_files']
    verify_files(frozen)
    for path in (swap/'summary.json', swap/'final-audit.json', STUDY/'design.json', STUDY/'fit-ready.json'):
        check(frozen[str(path)] == digest(path), 'Source not frozen before evaluation')
    previous = read(warm)
    ready, design = read(STUDY/'fit-ready.json'), read(STUDY/'design.json')
    pairs = len(SEEDS)*len(FIELDS)*DEALS*SEATS
    expected = {f'{field}-seed-{seed}-{label}': (label, seed, field)
                for label in LABELS for seed in SEEDS for field in FIELDS}
    check(len(plan['jobs']) == len(expected) and expected == {
        j['id']: (j['label'], j['seed'], j['field']) for j in plan['jobs']}, 'Unexpected evaluation jobs')
    check((plan['card_start'], plan['deals'], plan['planned_hands']) == (START, DEALS, pairs*len(LABELS))
          and plan['opponents'] == previous['opponents'] and plan['source'] == previous['source'],
          'Different game, card range or opponents')
    check(design['prior_card_ranges'] and all(first < end and (end <= START or first >= START+DEALS)
          for first, end in design['prior_card_ranges']), 'Overlap with recorded previous evaluations')
    check(design['steps'] == list(STEPS) and design['batch'] == BATCH and design['seeds'] == list(SEEDS)
          and set(design['fields']) == set(FIELDS) and design['contrasts'] == list(CONTRASTS)
          and (design['evaluation_deal_seed'], design['evaluation_deals']) == (START, DEALS)
          and design['shared_prefix'] and design['freeze_opening_and_postflop']
          and design['fixed_replay_iteration'] == 480 and design['new_selfplay_iterations'] == 0
          and design['model_promotion'] is False, 'Study design changed')
    for record in (summary, design):
        check((record['source_summary_sha256'], record['source_audit_sha256']) == (SOURCE_SUMMARY, SOURCE_AUDIT),
              'Different source experiment')
    check(summary['completed'] and (summary['completed_cells'], summary['new_cells'], summary['reused_cells']) == (27, 27, 0)
          and summary['new_hands'] == summary['verified_hands'] == pairs*len(LABELS)
          and (summary['independent_card_blocks'], summary['evaluation_deal_seed']) == (DEALS, START)
          and summary['model_promoted'] is False and summary['new_selfplay_iterations'] == 0
          and set(summary['result_sha256']) == set(expected)
          and set(summary['comparisons']) == set(summary['simultaneous_95']) == set(CONTRASTS),
          'Results incomplete or different from declared experiment')
    seed_keys = set(map(str, SEEDS))
    check(ready['fitting_complete'] and ready['policies'] == plan['policies'] and set(ready['policies']) == set(LABELS)
          and set(ready['seed_evidence_sha256']) == seed_keys and all(set(refs) == seed_keys for refs in ready['policies'].values()),
          'Evaluation weights differ from completed fit')
    sys.path[:0] = [previous['source'], str(Path(previous['source']).parent/'tools')]
    proofs = {}
    for seed in SEEDS:
        evidence = STUDY/f'seed-{seed}'/'fit.json'
        check(frozen[str(evidence)] == ready['seed_evidence_sha256'][str(seed)] == digest(evidence), 'Unverified fit evidence')
        proofs[str(seed)] = verify_fit(seed, plan, previous, read(evidence))
    return proofs


def checked_cell(plan, result_hashes, label, seed, field):
    name = f'{field}-seed-{seed}-{label}'
    folder = STUDY/'evaluation'/name
    verify_files({folder/'results.json': result_hashes[name]})
    result = read(folder/'results.json')
    raw = folder/'hands.jsonl.gz'
    verify_files({raw: result['raw_hands_sha256'], folder/'provenance.json': result['provenance_sha256'],
                  BRIDGE: result['engine_sha256']})
    provenance = read(folder/'provenance.json')
    ref = plan['policies'][label][str(seed)]
    for record in (result, provenance):
        check(record['candidate'] == ref['path'] and record['candidate_sha256'] == ref['sha256']
              and record['opponents'] == plan['opponents'][field] and record['candidate_type'] == 'preflop-street-policy-v1'
              and (record['field'], record['deals'], record['hands'], record['deal_seed'])
              == (field, DEALS, DEALS*SEATS, START), 'Cell assignment mismatch: '+name)
    verification = result['verification']
    check(result['complete'] and result['raw_hands_verified'] and result['source_provenance_verified']
          and result['terminal_payoffs_replayed'] and result['big_blind_chips'] == 100
          and verification['terminal_payoffs_replayed'] and verification['aggregate_recomputed']
          and verification['chip_conservation_verified'] and verification['hands'] == DEALS*SEATS
          and verification['card_blocks'] == DEALS, 'Unverified cell: '+name)
    deps = dict(result['dependency_hashes'])
    check(deps == provenance['dependency_hashes'], 'Cell provenance mismatch')
    verify_files(deps)
    spec = read(ref['path'])
    for asset in (ref, spec['preflop'], spec['postflop']):
        check(deps.pop(asset['path']) == asset['sha256'], 'Incorrect evaluation weights')
    check(all(plan['frozen_files'].get(path) == sha for path, sha in deps.items())
          and deps[str(ROOT/'scripts/evaluate_average_baseline.py')] == result['evaluator_sha256']
          and deps[str(ROOT/'scripts/poker_average_runtime.py')] == result['runtime_sha256']
          and deps[str(Path(plan['source'])/'bot/features.py')] == result['source_features_sha256'],
          'Unbound evaluator, features or opponent assets')
    protocol = (deps, result['evaluator_sha256'], result['runtime_sha256'],
                result['source_features_sha256'], provenance['semantic_rng'])
    return raw, result, protocol


def match_hands(cells, traces, engine):
    check(cells[0][2] == cells[1][2] == cells[2][2], 'Different evaluation protocols')
    chips = np.empty((len(LABELS), DEALS, SEATS), dtype=np.int64)
    count = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, 'rb')) for path, _, _ in cells]
        for index, lines in enumerate(zip_longest(*streams)):
            deal, hero = divmod(index, SEATS)
            check(deal < DEALS and None not in lines, 'Unmatched card-seat triplet')
            card = START+deal
            seed = int.from_bytes(hashlib.sha256(f'rich-evaluation-hand:{card}:{hero}'.encode()).digest()[:8], 'big')
            rows = [json.loads(line) for line in lines]
            for i, row in enumerate(rows):
                check((row['deal'], row['hero'], row['seed'], row['hand_seed']) == (deal, hero, card, seed),
                      'Different deal, hero seat or hand RNG')
                utilities = row['utilities']
                check(len(utilities) == SEATS and all(type(v) is int for v in utilities) and sum(utilities) == 0
                      and type(row['chips']) is int and row['chips'] == utilities[hero]
                      and isinstance(row['actions'], list) and 0 < len(row['actions']) == row['steps'] <= 512,
                      'Invalid raw action or payoff')
                if deal % REPLAY_STRIDE == 0:
                    state = engine.call(dict(op='reset', seed=card, button=0))
                    for action in row['actions']:
                        check(not state['terminal'] and state['seat_to_act'] == action['seat'], 'Illegal recorded action order')
                        state = engine.call(dict(op='step', action=action))
                    check(state['terminal'] and state['utilities'] == utilities, 'Terminal spot replay disagrees with log')
                chips[i, deal, hero] = row['chips']
            for key, (before, after) in CONTRASTS.items():
                traces[key].add(rows[before], rows[after])
            count += 1
    check(count == DEALS*SEATS, 'Incomplete raw ledgers')
    means = chips.mean(axis=2)
    for blocks, (_, result, _) in zip(means, cells, strict=True):
        np.testing.assert_allclose(blocks, result['deal_blocks'], rtol=0, atol=1e-8)
        np.testing.assert_allclose(blocks.mean(), result['bb_per_100'], rtol=0, atol=1e-8)
    return means


def audit():
    plan, summary = (read(STUDY/name) for name in ('evaluation-plan.json', 'summary.json'))
    fits = verify_design(plan, summary)
    z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)*len(CONTRASTS)))
    estimates = {key: {} for key in CONTRASTS}
    groups = {key: {} for key in CONTRASTS}
    moves = {key: {} for key in CONTRASTS}
    with closing(Bridge()) as engine:
        for field in FIELDS:
            traces = {key: FirstActionDifferences(engine) for key in CONTRASTS}
            scores = []
            for seed in SEEDS:
                cells = [checked_cell(plan, summary['result_sha256'], label, seed, field) for label in LABELS]
                scores.append(match_hands(cells, traces, engine))
            scores = np.asarray(scores)  # fixed training seed, policy, original card block
            for i, label in enumerate(LABELS):
                check_estimate(scores[:, i].mean(axis=0), summary['absolute_scores'][label][field])
            deltas = {key: scores[:, after]-scores[:, before] for key, (before, after) in CONTRASTS.items()}
            np.testing.assert_allclose(deltas['64_minus_0']+deltas['256_minus_64'], deltas['256_minus_0'], rtol=0, atol=1e-8)
            for key, by_seed in deltas.items():
                paired = by_seed.mean(axis=0)
                estimate = summary['comparisons'][key][field]
                check_estimate(paired, estimate)
                check([r['seed'] for r in estimate['per_seed']] == list(SEEDS), 'Training seeds differ')
                per_seed = by_seed.mean(axis=1)
                np.testing.assert_allclose(per_seed, [r['mean'] for r in estimate['per_seed']], rtol=0, atol=1e-8)
                sim = summary['simultaneous_95'][key][field]
                se = float(paired.std(ddof=1)/math.sqrt(DEALS))
                mean = float(paired.mean())
                bounds = [mean-z*se, mean+z*se]
                np.testing.assert_allclose(bounds, [sim['lower'], sim['upper']], rtol=0, atol=1e-8)
                groups[key][field], moves[key][field] = traces[key].results(paired)
                check(all(groups[key][field][k]['paired_hands'] == 0 for k in ('opening','postflop')),
                      'First hero divergence occurred outside the nonopening preflop route')
                estimates[key][field] = dict(mean=mean, per_seed_mean=per_seed.tolist(),
                    nominal_95=[mean-1.96*se, mean+1.96*se], simultaneous_95=bounds)
    pairs = len(FIELDS)*len(SEEDS)*DEALS*SEATS
    return dict(checked_utc=utc(), summary_sha256=digest(STUDY/'summary.json'), source_summary_sha256=SOURCE_SUMMARY,
                source_audit_sha256=SOURCE_AUDIT, fit_ready_sha256=summary['fit_ready_sha256'], new_cells=27,
                reused_cells=0, raw_hand_records_reaggregated=len(LABELS)*pairs,
                matched_card_seat_seed_field_triplets=pairs, independent_card_blocks=DEALS,
                first_card=START, last_card=START+DEALS-1, fixed_training_seeds=list(SEEDS), seats_per_card=SEATS,
                new_selfplay_iterations=0, model_promoted=False, frozen_opening_scales_and_postflop_verified=True,
                sampling_order_reconstructed_from_480_replay=True, per_seed_fitting=fits,
                raw_hand_and_provenance_hashes_verified=True, terminal_action_replay_verified_by_source_cells=True,
                terminal_hands_independently_replayed=len(range(0, DEALS, REPLAY_STRIDE))*SEATS*27,
                terminal_spot_replay_card_stride=REPLAY_STRIDE,
                absolute_and_paired_intervals_recomputed=True, simultaneous_interval_family_size=9,
                contrast_additivity_per_card_per_seed_verified=True, paired_estimates=estimates,
                first_action_divergence=groups, nonopening_preflop_first_action_change=moves,
                first_action_scope='Payoff changes per 100 of all hands, grouped by first differing action. '
                                   'Opponent-first pairs are included.')


if __name__ == '__main__':
    atomic_json(STUDY/'final-audit.json', audit())
