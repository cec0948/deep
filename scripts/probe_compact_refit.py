"""Bounded main-expert refit of frozen replay; never a self-play resume.

Fit diagnostics use the replay training distribution. They are not held-out
estimates of poker strength; independent card evaluation is still required.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

from fullhouse_v3_common import atomic_json, digest


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
    # Avoid adding bytecode files to the frozen source tree during this probe.
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
    # Reject pre-imported helpers from another run rather than silently mixing
    # current workspace helpers with the parent's frozen experiment.
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
    opening_before = {k: v.clone() for k, v in net.opening.state_dict().items()}
    main_scales_before = net.main_scales.clone()
    opening_scale_before = net.opening_scale.clone()
    # The opening aggregate is unnecessary: only the non-opening replay is used.
    other, _ = compact_regret.split_replay(buffers)
    sampler = scaled_regret_fitting.ScaledRegretSampler(
        other, regret_partition.base_strata, 'stratum-scaled-feature-conditional')
    np.testing.assert_array_equal(sampler.scales, net.main_scales.numpy())
    fit_seed = 2026111200 + seed
    diagnosis_rng = np.random.default_rng(fit_seed + 1000000)
    fixed_ids = np.searchsorted(sampler.cdf, diagnosis_rng.random(16384), side='right')

    def policy(scores, legal):
        positive = np.maximum(scores, 0) * legal
        total = positive.sum(axis=1, keepdims=True)
        return np.divide(positive, total, out=legal / legal.sum(axis=1, keepdims=True),
                         where=total > 0)

    def predict(x):
        with torch.no_grad():
            return np.concatenate([net.main(torch.from_numpy(block),
                                           torch.from_numpy(block[:, 46:51])).numpy()
                                   for block in np.array_split(x, max(1, len(x) // 2048))])

    def measure():
        x, y = sampler.features[fixed_ids], sampler.targets[fixed_ids]
        scores = predict(x)
        legal = x[:, 46:51]
        zero_mse = float(np.square(y * legal).mean())
        mse = float(np.square(scores - y * legal).mean())
        pre = sampler.groups < 6
        px, py = sampler.features[pre], sampler.targets[pre]
        pred = predict(px)
        tv = np.abs(policy(pred, px[:, 46:51]) - policy(py, px[:, 46:51])).sum(1) / 2
        per_group = {}
        groups, weights = sampler.groups[pre], sampler.probabilities[pre]
        for group in np.unique(groups):
            ids = groups == group
            per_group[str(group)] = dict(inputs=int(ids.sum()),
                conditional_target_tv=float(np.average(tv[ids], weights=weights[ids])))
        return dict(fixed_training_sample_size=len(fixed_ids), normalized_training_mse=mse,
                    zero_predictor_mse=zero_mse,
                    training_mse_over_zero=mse / zero_mse if zero_mse else None,
                    unraised_preflop_conditional_target_tv=float(np.average(tv, weights=weights)),
                    unraised_preflop_by_group=per_group,
                    positive_regret_absent_fraction=float(
                        ((np.maximum(scores, 0) * legal).sum(1) <= 0).mean()))

    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'source-manifest.json', manifest)
    report = dict(format='compact-offline-refit-v1', complete=False,
                  artifact_kind='derived-offline-refit', resumable_selfplay=False,
                  parent=str(parent), parent_iteration=ck['iteration'], seed=seed,
                  parent_checkpoint_sha256=checkpoint_sha,
                  parent_config=cfg, parent_config_sha256=digest(parent / 'config.json'),
                  parent_source_manifest_sha256=digest(parent / 'source-manifest.json'),
                  script_sha256=digest(Path(__file__)), extra_steps=extra_steps,
                  fit_seed=fit_seed, diagnostic_seed=fit_seed + 1000000,
                  batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'],
                  replay_rows=[len(b) for b in buffers], new_selfplay_samples=0,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  limitations=['Diagnostics are training-fit measurements on frozen replay, not held-out poker strength',
                               'Candidate is an offline main-expert refit; it cannot resume the original self-play trajectory',
                               'No default-model promotion and no change to continuation criteria'])
    started = time.monotonic()
    try:
        report['before'] = measure()
        atomic_json(output / 'results.json', report)
        print(json.dumps(dict(seed=seed, before=report['before'])), flush=True)
        np.random.seed(fit_seed)
        torch.manual_seed(fit_seed)
        optimizer = torch.optim.Adam(net.main.parameters(), lr=cfg['learning_rate'])
        report['refit_mean_loss'] = regret_fitting.train_conditional(
            net.main, sampler, optimizer, extra_steps, cfg['batch_size'])
        net.eval()
        report['after'] = measure()
        if not all(torch.equal(opening_before[k], v) for k, v in net.opening.state_dict().items()):
            raise AssertionError('Opening expert changed during main-only refit')
        if not torch.equal(main_scales_before, net.main_scales) or not torch.equal(opening_scale_before, net.opening_scale):
            raise AssertionError('Frozen regret scales changed')
        if digest(checkpoint) != checkpoint_sha:
            raise AssertionError('Parent checkpoint changed')
        report.update(opening_unchanged=True, scales_unchanged=True, parent_checkpoint_unchanged=True)
        candidate = output / 'candidate.npz'
        compact_regret.export_compact(net, candidate)
        with np.load(candidate) as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays.update(artifact_kind=np.asarray('derived-offline-refit'),
                      resumable_selfplay=np.asarray(False),
                      parent_checkpoint_sha256=np.asarray(checkpoint_sha),
                      offline_refit_steps=np.asarray(extra_steps))
        np.savez(candidate, **arrays)
        state_path = output / 'candidate.pt'
        torch.save(dict(format=report['format'], artifact_kind=report['artifact_kind'],
                        resumable_selfplay=False, net=net.state_dict(),
                        provenance={k: report[k] for k in ('parent', 'parent_iteration', 'seed',
                            'parent_checkpoint_sha256', 'parent_config_sha256',
                            'parent_source_manifest_sha256', 'extra_steps', 'fit_seed')}), state_path)
        report['files'] = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p))
                           for p in (candidate, state_path, output / 'source-manifest.json')}
        report.update(complete=True, seconds=time.monotonic() - started,
                      completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(output / 'results.json', report)
        print(json.dumps(dict(seed=seed, after=report['after'], seconds=report['seconds'],
                              complete=True)), flush=True)
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
