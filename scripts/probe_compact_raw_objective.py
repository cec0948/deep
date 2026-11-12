"""Bounded importance-corrected main refit of immutable self-play replay.

All policy diagnostics below use the Torch fitting graph and the replay
training distribution. They do not measure deployment behavior or held-out
poker strength. Preserving an empirical loss objective does not guarantee poker strength.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from fullhouse_v3_common import atomic_json, digest


def tensor_state_digest(module):
    """Hash names, shapes, dtypes and exact CPU tensor bytes, not torch.save."""
    result = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        metadata = json.dumps([name, list(array.shape), str(array.dtype)],
                              separators=(',', ':')).encode()
        result.update(len(metadata).to_bytes(8, 'big'))
        result.update(metadata)
        result.update(array.tobytes(order='C'))
    return result.hexdigest()


def run(parent, output, seed, extra_steps):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Refit output must be outside the immutable parent run')
    if not 1 <= extra_steps <= 16384:
        raise ValueError('Offline refit budget must be 1..16384 steps')
    cfg = json.loads((parent / 'config.json').read_text())
    manifest = json.loads((parent / 'source-manifest.json').read_text())
    if seed not in cfg['seeds']:
        raise ValueError('Seed is absent from parent campaign')
    status = json.loads((parent / f'seed-{seed}/status.json').read_text())
    if status['status'] != 'complete' or status['iteration'] != cfg['iterations']:
        raise ValueError('Parent seed must be complete')
    for name, expected in manifest['files'].items():
        if digest(parent / name) != expected:
            raise ValueError('Parent frozen file changed: ' + name)
    checkpoint = parent / f'seed-{seed}/latest.pt'
    checkpoint_sha = digest(checkpoint)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(parent / 'tools'), str(parent / 'source')]
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import torch
    from training.buffer import ReservoirBuffer
    import compact_regret
    import regret_fitting
    import scaled_regret_fitting
    import regret_partition
    for module in (compact_regret, regret_fitting, scaled_regret_fitting,
                   regret_partition):
        if Path(module.__file__).resolve().parent != parent / 'tools':
            raise ValueError('Helper did not load from the frozen parent: ' + module.__name__)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (ck['config'] != cfg or ck['source_manifest'] != manifest
            or ck['iteration'] != cfg['iterations'] or ck['seed'] != seed):
        raise ValueError('Parent checkpoint provenance mismatch')
    buffers = []
    for state in ck['buffers']:
        buffer = ReservoirBuffer(state['count'], 51)
        buffer.load_state_dict(state)
        buffers.append(buffer)
    net = compact_regret.CompactRegretNet()
    net.load_state_dict(ck['net'])
    net.eval()
    teacher = copy.deepcopy(net.main).eval()
    teacher.requires_grad_(False)
    teacher_sha = tensor_state_digest(teacher)
    opening_before = {k: v.clone() for k, v in net.opening.state_dict().items()}
    main_scales_before = net.main_scales.clone()
    opening_scale_before = net.opening_scale.clone()
    other, _ = compact_regret.split_replay(buffers)
    base_sampler = scaled_regret_fitting.ScaledRegretSampler(
        other, regret_partition.base_strata, 'stratum-scaled-feature-conditional')
    from raw_objective_fitting import RawObjectiveRegretSampler
    sampler = RawObjectiveRegretSampler(other, regret_partition.base_strata, base_sampler, net.main_scales.numpy())
    helper_path = Path(__file__).with_name('raw_objective_fitting.py')
    helper_sha = digest(helper_path)
    fit_seed = 2026111200 + seed
    diagnosis_rng = np.random.default_rng(fit_seed + 1000000)
    fixed_ids = np.searchsorted(sampler.cdf, diagnosis_rng.random(16384), side='right')

    def policy(scores, legal):
        positive = np.maximum(scores, 0) * legal
        total = positive.sum(axis=1, keepdims=True)
        return np.divide(positive, total, out=legal / legal.sum(axis=1, keepdims=True),
                         where=total > 0)

    def predict(module, x):
        with torch.no_grad():
            return np.concatenate([module(torch.from_numpy(block),
                                          torch.from_numpy(block[:, 46:51])).numpy()
                                   for block in np.array_split(x, max(1, len(x) // 2048))])

    fixed_x = sampler.features[fixed_ids]
    fixed_teacher_scores = predict(teacher, fixed_x)
    fixed_legal = fixed_x[:, 46:51]
    fixed_teacher_policy = policy(fixed_teacher_scores, fixed_legal)
    fixed_teacher_zero = (np.maximum(fixed_teacher_scores, 0) * fixed_legal).sum(1) <= 0

    def exact_objective_metrics():
        raw_mse = 0.; normalized_mse = 0.
        for lo in range(0, len(sampler.features), 4096):
            hi = min(lo+4096, len(sampler.features))
            x = sampler.features[lo:hi]
            scores = predict(net.main, x).astype(np.float64)
            legal = x[:,46:51]
            scales = sampler.scales[sampler.groups[lo:hi]].astype(np.float64)
            raw_errors = np.square(scores*scales[:,None]-sampler.raw_targets[lo:hi]*legal).mean(1)
            raw_mse += float(np.dot(sampler.original_probabilities[lo:hi],raw_errors))
            normalized_errors = np.square(scores-sampler.targets[lo:hi]*legal).mean(1)
            normalized_mse += float(np.dot(sampler.target_probabilities[lo:hi],normalized_errors))
        return dict(raw_iteration_weighted_training_mse=raw_mse+sampler.conditional_variance_constant,
                    raw_iteration_weighted_conditional_mse=raw_mse,
                    normalized_importance_training_objective=normalized_mse,
                    raw_conditional_objective_over_z=raw_mse/sampler.normalizer_z,
                    normalized_objective_rounding_gap=normalized_mse-raw_mse/sampler.normalizer_z,
                    conditional_variance_constant=sampler.conditional_variance_constant,
                    objective_normalizer_z=sampler.normalizer_z)

    def measure():
        x, y = fixed_x, sampler.targets[fixed_ids]
        scores = predict(net.main, x)
        legal = x[:, 46:51]
        zero_mse = float(np.square(y * legal).mean())
        mse = float(np.square(scores - y * legal).mean())
        teacher_mse = float(np.square(scores - fixed_teacher_scores).mean())
        probs = policy(scores, legal)
        policy_tv = np.abs(probs - fixed_teacher_policy).sum(1) / 2
        zero = (np.maximum(scores, 0) * legal).sum(1) <= 0
        target_tv = np.abs(probs - policy(y, legal)).sum(1) / 2

        def drift(mask):
            return dict(inputs=int(mask.sum()),
                        teacher_policy_tv=float(policy_tv[mask].mean()),
                        teacher_policy_tv_p95=float(np.quantile(policy_tv[mask], .95)),
                        teacher_policy_tv_over_half_fraction=float((policy_tv[mask] > .5).mean()),
                        conditional_target_tv=float(target_tv[mask].mean()),
                        teacher_positive_regret_absent_fraction=float(fixed_teacher_zero[mask].mean()),
                        positive_regret_absent_fraction=float(zero[mask].mean()),
                        newly_absent_fraction=float((~fixed_teacher_zero & zero)[mask].mean()),
                        resolved_absent_fraction=float((fixed_teacher_zero & ~zero)[mask].mean()))

        pre = sampler.groups < 6
        px, py = sampler.features[pre], sampler.targets[pre]
        pred = predict(net.main, px)
        tv = np.abs(policy(pred, px[:, 46:51]) - policy(py, px[:, 46:51])).sum(1) / 2
        groups, weights = sampler.groups[pre], sampler.probabilities[pre]
        per_group = {}
        for group in np.unique(groups):
            ids = groups == group
            per_group[str(group)] = dict(inputs=int(ids.sum()),
                conditional_target_tv=float(np.average(tv[ids], weights=weights[ids])))
        street = np.argmax(x[:, 17:21], axis=1)
        fixed_groups = sampler.groups[fixed_ids]
        return dict(fixed_training_sample_size=len(fixed_ids), normalized_training_mse=mse,
                    normalized_teacher_score_mse=teacher_mse,
                    **exact_objective_metrics(),
                    zero_predictor_mse=zero_mse,
                    training_mse_over_zero=mse / zero_mse if zero_mse else None,
                    unraised_preflop_conditional_target_tv=float(np.average(tv, weights=weights)),
                    unraised_preflop_by_group=per_group,
                    positive_regret_absent_fraction=float(zero.mean()),
                    policy_drift=drift(np.ones(len(x), dtype=bool)),
                    policy_drift_by_street={str(int(s)): drift(street == s) for s in np.unique(street)},
                    policy_drift_by_stratum={str(int(g)): drift(fixed_groups == g) for g in np.unique(fixed_groups)})

    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'source-manifest.json', manifest)
    report = dict(format='compact-raw-objective-offline-refit-v1', complete=False,
                  artifact_kind='derived-offline-raw-objective-refit', resumable_selfplay=False,
                  parent=str(parent), parent_iteration=ck['iteration'], seed=seed,
                  parent_checkpoint_sha256=checkpoint_sha,
                  parent_config=cfg, parent_config_sha256=digest(parent / 'config.json'),
                  parent_source_manifest_sha256=digest(parent / 'source-manifest.json'),
                  script_sha256=digest(Path(__file__)), extra_steps=extra_steps,
                  sampler_method='importance-corrected-balanced',
                  objective='E_oldq[(p_i*s_i^2/(Z*oldq_i))*mean_actions((f(x_i)-conditional_raw_target_i/s_i)^2)]',
                  objective_scope='Retained main replay with original iteration weights, raw BB squared error; conditional variance is parameter independent',
                  helper_sha256=helper_sha,
                  sampler=sampler.diagnostics(),
                  teacher_main_tensor_state_sha256=teacher_sha,
                  teacher_source='Diagnostic reference only: exact frozen deepcopy of parent120 main; not a training target',
                  fit_seed=fit_seed, diagnostic_seed=fit_seed + 1000000,
                  diagnostic_policy_implementation='Torch training graph; not NumPy deployment inference',
                  batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'],
                  replay_rows=[len(b) for b in buffers], new_selfplay_samples=0,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  limitations=['Diagnostics are training-fit measurements on frozen replay, not held-out poker strength',
                               'Torch fitting-graph policy diagnostics are not measurements of deployment behavior',
                               'Objective conservation does not guarantee lower exploitability or improved six-player poker strength',
                               'Existing float32 targets/scales are preserved and fixed importance weights are cast to float32; conservation is numerical rather than bitwise',
                               'Candidate is an offline main-expert refit; it cannot resume the original self-play trajectory',
                               'No default-model promotion and no change to continuation criteria'])
    started = time.monotonic()
    try:
        report['before'] = measure()
        atomic_json(output / 'results.json', report)
        print(json.dumps(dict(seed=seed, phase='before',
                              normalized_training_mse=report['before']['normalized_training_mse'],
                              sampler_method=report['sampler_method'])), flush=True)
        np.random.seed(fit_seed)
        torch.manual_seed(fit_seed)
        optimizer = torch.optim.Adam(net.main.parameters(), lr=cfg['learning_rate'])
        net.main.train()
        objectives = []
        schedule_reference = np.random.RandomState(fit_seed)
        sampled_ids_digest = hashlib.sha256()
        for _ in range(extra_steps):
            expected_ids = np.searchsorted(sampler.original_sampler_cdf, schedule_reference.random_sample(cfg['batch_size']), side='right')
            x, y, weights = sampler.sample(cfg['batch_size'])
            if not np.array_equal(sampler.last_sample_ids,expected_ids) or not np.array_equal(x,sampler.features[expected_ids]) or not np.array_equal(y,sampler.targets[expected_ids]) or not np.array_equal(weights,sampler.loss_weights[expected_ids].astype(np.float32)):
                raise AssertionError('Input schedule differs from the matched unweighted control')
            sampled_ids_digest.update(expected_ids.astype('<i8').tobytes())
            x, y, weights = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights)
            legal = x[:,46:51]
            loss = (weights*(net.main(x,legal)-y*legal).square().mean(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite raw-objective fitting loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.main.parameters(), 1.)
            optimizer.step()
            objectives.append(float(loss.detach()))
        report['refit_mean_loss'] = float(np.mean(objectives))
        report['matched_control_input_schedule_verified'] = True
        report['sampled_record_ids_sha256'] = sampled_ids_digest.hexdigest()
        report['sampler_after'] = sampler.diagnostics()
        net.eval()
        report['after'] = measure()
        if not all(torch.equal(opening_before[k], v) for k, v in net.opening.state_dict().items()):
            raise AssertionError('Opening expert changed during main-only refit')
        if not torch.equal(main_scales_before, net.main_scales) or not torch.equal(opening_scale_before, net.opening_scale):
            raise AssertionError('Frozen regret scales changed')
        if tensor_state_digest(teacher) != teacher_sha or any(p.grad is not None for p in teacher.parameters()):
            raise AssertionError('Fixed teacher changed or received gradients')
        if digest(helper_path) != helper_sha:
            raise AssertionError('Raw objective helper changed')
        if digest(checkpoint) != checkpoint_sha:
            raise AssertionError('Parent checkpoint changed')
        for name, expected in manifest['files'].items():
            if digest(parent / name) != expected:
                raise AssertionError('Parent frozen file changed: ' + name)
        report.update(opening_unchanged=True, scales_unchanged=True,
                      parent_checkpoint_unchanged=True, parent_frozen_files_unchanged=True,
                      teacher_unchanged=True)
        candidate = output / 'candidate.npz'
        compact_regret.export_compact(net, candidate)
        with np.load(candidate) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays.update(artifact_kind=np.asarray(report['artifact_kind']),
                      resumable_selfplay=np.asarray(False),
                      parent_checkpoint_sha256=np.asarray(checkpoint_sha),
                      offline_refit_steps=np.asarray(extra_steps),
                      sampler_method=np.asarray(report['sampler_method']),
                      objective_normalizer_z=np.asarray(sampler.normalizer_z),
                      teacher_main_tensor_state_sha256=np.asarray(teacher_sha))
        np.savez(candidate, **arrays)
        state_path = output / 'candidate.pt'
        torch.save(dict(format=report['format'], artifact_kind=report['artifact_kind'],
                        resumable_selfplay=False, net=net.state_dict(),
                        provenance={k: report[k] for k in ('parent', 'parent_iteration', 'seed',
                            'parent_checkpoint_sha256', 'parent_config_sha256',
                            'parent_source_manifest_sha256', 'extra_steps', 'fit_seed',
                            'sampler_method', 'helper_sha256', 'teacher_main_tensor_state_sha256', 'objective')}), state_path)
        report['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                           for p in (candidate, state_path, output / 'source-manifest.json')}
        report.update(complete=True, seconds=time.monotonic() - started,
                      completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(output / 'results.json', report)
        print(json.dumps(dict(seed=seed, phase='after',
                              normalized_training_mse=report['after']['normalized_training_mse'],
                              normalized_teacher_score_mse=report['after']['normalized_teacher_score_mse'],
                              teacher_policy_tv=report['after']['policy_drift']['teacher_policy_tv'],
                              seconds=report['seconds'], complete=True)), flush=True)
    except BaseException as exc:
        report.update(error=str(exc), seconds=time.monotonic() - started)
        atomic_json(output / 'results.json', report)
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--extra-steps', type=int, default=3072)
    args = parser.parse_args()
    run(args.parent, args.output, args.seed, args.extra_steps)
