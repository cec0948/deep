"""Describe the single 504/528 exports' errors on exactly the same final replay draws."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from audit_poker_flop_replay_predictions import SAMPLE, predict_errors
from audit_poker_opponent_replay_age import STAGE_HASH
from controlled_poker_refit import read, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from run_poker_preflop_pilot import CAMPAIGN, SOURCE
from train_poker_average_continuation import load_checkpoint
from train_poker_fixed_preflop_window import postflop_sampler


RELATIVE = Path('preflop-followup/480-v1/pool-extension-528-20261126')
STUDY = ROOT/'artifacts'/RELATIVE
PARENT = STUDY.parent/'opponent-diversity-20261123'
RNG_SEED = 20261126


def final_distribution(state):
    from compact_regret import ReplayView
    from regret_partition import base_strata

    views, features, origins = [], [], []
    for buf in state['buffers']:
        n = buf['count']
        view = ReplayView(buf['features'][:n], buf['advantages'][:n], buf['iterations'][:n])
        views.append(view)
        postflop = view.features[:, 17] == 0
        features.append(view.features[postflop])
        origins.append(view.iterations[postflop])
    sampler, counts = postflop_sampler(views)
    # Reproduce the Nov23 replay-age audit: stratification preserves the order of
    # every postflop row within its group. Check the resulting row probabilities
    # against both the actual sampler and the equal-group/iteration-weight formula.
    x = np.concatenate(features)
    order = np.argsort(base_strata(x), kind='stable')
    t = np.concatenate(origins)[order].astype(float)
    if (not np.array_equal(x[order], sampler.features) or not np.array_equal(t, np.floor(t))
            or t.min() < 1 or t.max() != 528):
        raise ValueError('Reconstructed replay row identity or generation range differs')
    groups = np.unique(sampler.groups)
    if not np.array_equal(groups, np.arange(10, 22)):
        raise ValueError('Unexpected postflop stratum')
    for group in groups:
        mask = sampler.groups == group
        expected = t[mask] / t[mask].sum() / len(groups)
        if not np.allclose(expected, sampler.probabilities[mask], rtol=1e-13, atol=0):
            raise ValueError('Restored fitting probabilities differ from replay-age audit formula')
    return sampler, t, counts


def compare(seed, stage, parent):
    key = str(seed)
    folder = STUDY/'pool528'/f'seed-{seed}'
    ready = stage['arms']['pool528'][key]
    if read(folder/'current.json')['files'] != ready['files']:
        raise ValueError('Checkpoint pointer differs from completed stage')
    state, _ = load_checkpoint(folder)
    if (state['seed'], state['iteration']) != (seed, 528):
        raise ValueError('Wrong seed/iteration')
    sampler, origins, counts = final_distribution(state)
    del state
    verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256']})
    last = json.loads((folder/'diagnostics.jsonl').read_bytes().splitlines()[-1])
    if last['iteration'] != 528 or counts != last['fit_rows_by_street']:
        raise ValueError('Street counts differ from the last fit')

    rng = np.random.default_rng([RNG_SEED, seed])
    ids = np.searchsorted(sampler.cdf, rng.random(SAMPLE), side='right')
    groups = sampler.groups[ids]
    recent = origins > 504
    masks = {'all':np.ones(SAMPLE, dtype=bool), 'up_to_504':~recent[ids], '505_to_528':recent[ids],
             **{street:sampler.features[ids, i] == 1 for i, street in enumerate(('flop','turn','river'), 18)}}
    comparisons = {k:{'draws':int(mask.sum())} for k,mask in masks.items()}
    model_hashes = {}
    endpoints = {'current504':parent['policies']['pool504'][key], 'current528':stage['policies']['current528'][key]}
    for label, ref in endpoints.items():
        verify_files({ref['path']:ref['sha256']})
        model = read(ref['path'])['postflop']
        model_hashes[label] = model['sha256']
        squared, scales = predict_errors(model, sampler, ids)
        if label == 'current528' and not np.array_equal(scales[groups], sampler.scales[groups]):
            raise ValueError('Final export scales differ from fit')
        normalized = squared / np.square(sampler.scales[groups].astype(float))
        for name, mask in masks.items():
            comparisons[name][label] = {'mse_bb2':float(squared[mask].mean()),
                                        'mse_final_normalization':float(normalized[mask].mean())}
    return dict(seed=seed, replay_sha256=ready['files']['arrays.npz'], model_sha256=model_hashes,
                sample_index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                postflop_rows=len(origins), rows_from_505_to_528=int(recent.sum()),
                exact_draw_probability_505_to_528=float(sampler.probabilities[recent].sum()), comparisons=comparisons)


def main():
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN/'tools')]
    torch.set_num_threads(1)
    verify_files({PARENT/'stage-ready.json':STAGE_HASH})
    stage = read(STUDY/'stage-ready.json')
    if not stage['training_complete'] or stage['iteration'] != 528 or stage['parent_stage_sha256'] != STAGE_HASH:
        raise ValueError('Completed child stage lineage differs')
    parent = read(PARENT/'stage-ready.json')
    rows = [compare(seed, stage, parent) for seed in (342, 343, 344)]
    total = {label:float(np.mean([r['comparisons']['all'][label]['mse_final_normalization'] for r in rows]))
             for label in ('current504', 'current528')}
    report = dict(scope='Descriptive in-sample postflop regret prediction, with no new hands or model fitting. '
                        'This diagnostic does not estimate deployment strength or evaluate the four-snapshot mixture.',
                  source_stage_sha256=STAGE_HASH, stage_sha256=digest(STUDY/'stage-ready.json'),
                  script_sha256=digest(__file__), prediction_helper_sha256=digest(ROOT/'scripts/audit_poker_flop_replay_predictions.py'),
                  sampling='Same row IDs for both exports, drawn with replacement from the final 528 fitting CDF.',
                  generator='NumPy default_rng([rng_seed, training_seed])', rng_seed=RNG_SEED, draws_per_seed=SAMPLE,
                  units='Each export uses its own BB scales. Both squared errors use common 528 stratum scales '
                        'for normalization; masked illegal outputs count as zero in the five-output mean.',
                  realized_training_draws_by_generation_logged=False, seeds=rows, pooled_mse_final_normalization=total)
    atomic_json(STUDY/'replay-fit.json', report)
    atomic_json(ROOT/'reports'/RELATIVE/'replay-fit.json', report)


if __name__ == '__main__':
    main()
