"""Compare 480 and 504 raw regret predictions on identical rows of final training replay."""
import hashlib
import json
import sys

import numpy as np
import torch

from controlled_poker_refit import read, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from run_poker_preflop_pilot import CAMPAIGN, SOURCE
from train_poker_average_continuation import load_checkpoint
from train_poker_fixed_preflop_window import postflop_sampler


STUDY = ROOT/'artifacts/preflop-followup/480-v1/opponent-diversity-20261123'
REPORT = ROOT/'reports/preflop-followup/480-v1/flop-rollback-20261123/replay-fit.json'
STAGE_SHA = '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9'
SAMPLE = 32768
RNG_SEED = 20261123


def restore_sampler(state):
    from compact_regret import ReplayView
    from regret_partition import base_strata

    views = []
    for buf in state['buffers']:
        n = buf['count']
        views.append(ReplayView(buf['features'][:n], buf['advantages'][:n], buf['iterations'][:n]))
    sampler, counts = postflop_sampler(views)
    # The actual fitting sampler concatenates seats, then groups stably by
    # stratum. Postflop groups (10–21) preserve each individual observation.
    masks = [v.features[:,17] == 0 for v in views]
    x = np.concatenate([v.features[m] for v,m in zip(views,masks,strict=True)])
    t = np.concatenate([v.iterations[m] for v,m in zip(views,masks,strict=True)])
    groups = base_strata(x)
    order = np.argsort(groups, kind='stable')
    if not (np.array_equal(x[order], sampler.features)
            and np.array_equal(groups[order], sampler.groups)
            and np.all(t == np.floor(t)) and t.min() >= 1 and t.max() == 504):
        raise ValueError('Row identity, provenance or group ordering differs')
    return sampler, t[order], counts


def predict_errors(model_ref, sampler, ids):
    from compact_regret import ARRAY_MAP
    from training.network import DuelingAdvantageNet

    verify_files({model_ref['path']: model_ref['sha256']})
    net = DuelingAdvantageNet().eval()
    with np.load(model_ref['path'], allow_pickle=False) as weights:
        net.load_state_dict({name:torch.from_numpy(weights[key].copy()) for name,key in ARRAY_MAP.items()})
        scales = weights['main_scales'].copy()
    squared_errors = []
    with torch.inference_mode():
        for batch in np.array_split(ids, 32):
            x = torch.from_numpy(sampler.features[batch])
            legal = x[:,46:51]
            output = net(x, legal).numpy().astype(float)
            bb_output = output * scales[sampler.groups[batch],None]
            residual = (bb_output - sampler.raw_targets[batch]) * legal.numpy()
            # As in fitting, illegal outputs are zero and the MSE denominator is five actions.
            squared_errors.append(np.square(residual).mean(axis=1))
    errors = np.concatenate(squared_errors)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite replay prediction error')
    return errors, scales


def audit_seed(seed, stage):
    key = str(seed)
    ready = stage['arms']['pool504'][key]
    folder = STUDY/'pool504'/f'seed-{seed}'
    if read(folder/'current.json')['files'] != ready['files']:
        raise ValueError('Final replay differs from the registered stage')
    state, _ = load_checkpoint(folder)
    if (state['seed'],state['iteration']) != (seed,504):
        raise ValueError('Wrong seed or final iteration')
    sampler, row_iteration, counts = restore_sampler(state)
    del state
    verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256']})
    last = json.loads((folder/'diagnostics.jsonl').read_text().splitlines()[-1])
    if last['iteration'] != 504 or counts != last['fit_rows_by_street']:
        raise ValueError('Restored postflop row counts differ from the final fitting log')
    rng = np.random.default_rng([RNG_SEED,seed])
    ids = np.searchsorted(sampler.cdf, rng.random(SAMPLE), side='right')
    groups = sampler.groups[ids]
    origins = row_iteration[ids]
    new = row_iteration > 480
    masks = {'all':np.ones(SAMPLE,dtype=bool),'up_to_480':origins <= 480,'481_to_504':origins > 480,
             'flop':sampler.features[ids,18] == 1,'turn':sampler.features[ids,19] == 1,
             'river':sampler.features[ids,20] == 1}
    metrics = {name:{'draws':int(mask.sum())} for name,mask in masks.items()}
    refs = {}
    for label, arm in [('endpoint480','specialist480'),('endpoint504','pool504')]:
        policy = stage['policies'][arm][key]
        verify_files({policy['path']:policy['sha256']})
        ref = read(policy['path'])['postflop']
        refs[label] = ref['sha256']
        errors, scales = predict_errors(ref, sampler, ids)
        if arm == 'pool504' and not np.array_equal(scales[groups], sampler.scales[groups]):
            raise ValueError('Stored final scales do not match the reconstructed final sampler')
        norm_errors = errors / np.square(sampler.scales[groups].astype(float))
        for name, mask in masks.items():
            metrics[name][label] = dict(mse_bb2=float(errors[mask].mean()),
                                        mse_final_normalization=float(norm_errors[mask].mean()))
    return dict(seed=seed, replay_sha256=ready['files']['arrays.npz'], model_sha256=refs,
                postflop_rows=len(new), postflop_rows_after_480=int(new.sum()),
                exact_draw_probability_after_480=float(sampler.probabilities[new].sum()),
                index_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(), comparisons=metrics)


def main():
    verify_files({STUDY/'stage-ready.json':STAGE_SHA})
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN/'tools')]
    torch.set_num_threads(1)
    stage = read(STUDY/'stage-ready.json')
    rows = [audit_seed(seed,stage) for seed in (342,343,344)]
    pooled = {}
    for name in rows[0]['comparisons']:
        by_seed = [row['comparisons'][name] for row in rows]
        weights = [item['draws'] for item in by_seed]
        pooled[name] = {'draws':sum(weights)}
        for endpoint in ('endpoint480','endpoint504'):
            pooled[name][endpoint] = {
                metric:float(np.average([item[endpoint][metric] for item in by_seed],weights=weights))
                for metric in ('mse_bb2','mse_final_normalization')}
    report = dict(source_stage_sha256=STAGE_SHA, audit_script_sha256=digest(__file__),
                  scope='Descriptive in-sample regret prediction. No new hands, fitting or strength estimate.',
                  policies='Single postflop endpoints at 480 and 504. Four-snapshot deployed mixture excluded.',
                  sampling='With replacement from the final 504 fitting CDF; identical row indices for both endpoints.',
                  pseudorandom_generator='NumPy default_rng([rng_seed, training_seed])', rng_seed=RNG_SEED,
                  draws_per_seed=SAMPLE, items='Zero illegal errors; mean over five outputs, then over rows.',
                  units='Raw BB squared using each export scale, and common final-stratum normalized squared error.',
                  realized_training_draws_by_generation_logged=False, seeds=rows, pooled=pooled)
    atomic_json(REPORT, report)


if __name__ == '__main__':
    main()
