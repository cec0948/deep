"""One frozen-replay, one-factor poker diagnostic. Never collects selfplay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from fullhouse_v3_common import atomic_json, digest, process_start

SEEDS = [342, 343, 344]
LEGACY_ARMS = {
    'legacy_control': dict(capacity=100000, steps=1024, batch=512, objective='balanced'),
    'legacy_small_replay': dict(capacity=5000, steps=1024, batch=512, objective='balanced'),
    'legacy_fewer_steps': dict(capacity=100000, steps=256, batch=512, objective='balanced'),
    'legacy_smaller_batch': dict(capacity=100000, steps=1024, batch=128, objective='balanced'),
    'legacy_raw_objective': dict(capacity=100000, steps=1024, batch=512, objective='raw_BB_importance_corrected'),
}
RICH_ARMS = ['rich_masked', 'rich_full']
ARMS = list(LEGACY_ARMS) + RICH_ARMS
BASELINES = ['original120', 'parent360', 'previous_rich96']
FIELDS = ['public', 'mixed', 'noregrets']
PAIRS = [('legacy_control', name) for name in LEGACY_ARMS if name != 'legacy_control']
PAIRS += [('rich_masked', 'rich_full')]


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_files(files):
    if not files or any(not Path(p).is_file() or digest(p) != sha for p, sha in files.items()):
        raise ValueError('Frozen diagnostic source or parent input changed')


def selected_indices(length, capacity, seed, seat):
    """Independent, deterministic nested subset; never consumes fitting RNG."""
    import numpy as np
    if not 0 < capacity <= length:
        raise ValueError('Invalid replay subset capacity')
    if capacity == length:
        return np.arange(length)
    return np.sort(np.random.default_rng(9160000 + seed * 6 + seat).choice(
        length, size=capacity, replace=False))


def views_from_states(states, capacity=None, mask_extra=False, seed=342):
    import numpy as np
    class View:
        def __init__(self, x, y, w):
            self.features, self.advantages, self.iterations = x, y, w
        def __len__(self):
            return len(self.features)
    result, selection = [], []
    for seat, state in enumerate(states):
        n = state['count']
        ids = (np.arange(n) if capacity is None else
               selected_indices(n, min(capacity, n), seed, seat))
        x = state['features'][ids]
        if mask_extra:
            x = x.copy(); x[:, 51:] = 0
        result.append(View(x, state['advantages'][ids], state['iterations'][ids]))
        selection.append(hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest())
    return result, selection


def assert_finite_arrays(model):
    import torch
    if any(not torch.isfinite(x).all() for x in model.state_dict().values()):
        raise FloatingPointError('Nonfinite fitted model')


def fit_legacy(plan, seed, arm, output):
    import numpy as np
    import torch
    from compact_lightweight_backup import load_state
    from compact_regret import CompactRegretNet, fit_compact, fit_opening, export_compact, split_replay
    from root_regret_replay import RootReplay
    from training import r1_support as r1
    spec = plan['parents']['legacy'][str(seed)]
    state = load_state(spec['checkpoint'])
    if state['seed'] != seed or state['iteration'] != 360:
        raise ValueError('Actual full360 parent replay required')
    cfg = state['config']
    expected = dict(traversals=256, buffer_capacity=100000, main_steps=1024,
                    batch_size=512, opening_steps=2048, opening_max_steps=8192)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Legacy reference training conditions changed')
    # Re-export the actual restored parent before refitting; no empty-parent shortcut.
    parent_net = CompactRegretNet(); parent_net.load_state_dict(state['net']); parent_net.eval()
    warm_path = output / 'restored-parent.npz'; export_compact(parent_net, warm_path)
    with np.load(warm_path, allow_pickle=False) as a, np.load(spec['model'], allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('Restored360 numerical export differs from parent')
    warm_path.unlink()
    del parent_net
    settings = LEGACY_ARMS[arm]
    buffers, selection = views_from_states(state['buffers'], settings['capacity'], seed=seed)
    root = RootReplay(); root.load_state_dict(state['root_replay'])
    rng = np.random.default_rng(); r1.restore_rng(state['rng'], rng)
    r1.WEIGHT_HORIZON = cfg['iterations']; r1.AVERAGE_SAMPLER = 'importance'
    # This is exactly the parent's reset-and-fit kernel, without a new traversal.
    net = CompactRegretNet()
    init_sha = hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in net.state_dict().values())).hexdigest()
    from scaled_regret_fitting import ScaledRegretSampler
    from regret_partition import base_strata
    other, _ = split_replay(buffers, root)
    base = ScaledRegretSampler(other, base_strata, 'stratum-scaled-feature-conditional')
    cdf_sha = hashlib.sha256(base.cdf.tobytes()).hexdigest()
    reference_rng = np.random.RandomState(); reference_rng.set_state(np.random.get_state())
    uniforms = reference_rng.random(settings['steps'] * settings['batch'])
    draw_sha = hashlib.sha256(uniforms.tobytes()).hexdigest(); del uniforms
    if settings['objective'] == 'balanced':
        del base, other
        fitted = fit_compact(net, buffers, settings['steps'], settings['batch'], root_replay=root,
                             opening_steps=cfg['opening_steps'], lr=cfg['learning_rate'])
    else:
        # Same conditional records, scale, sampling CDF and initial parameters.
        # Only fixed loss weights change; the existing objective/gradient-checked
        # helper supplies q*/q rather than introducing another learner here.
        from raw_objective_fitting import RawObjectiveRegretSampler
        sampler = RawObjectiveRegretSampler(other, base_strata, base, base.scales)
        net.main_scales.copy_(torch.from_numpy(base.scales))
        optimizer = torch.optim.Adam(net.main.parameters(), lr=cfg['learning_rate'])
        net.main.train()
        for _ in range(settings['steps']):
            xx, yy, ww = sampler.sample(settings['batch'])
            tx = torch.from_numpy(xx); ty = torch.from_numpy(yy); tw = torch.from_numpy(ww)
            loss = (tw * (net.main(tx, tx[:, 46:51])-ty).square().mean(1)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite corrected-objective fit')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.main.parameters(),1.)
            optimizer.step()
        net.main.eval()
        opening = fit_opening(net, root.means(), cfg['opening_steps'], cfg['learning_rate'])
        fitted = dict(main_loss=float(loss.detach()), opening=opening, objective=sampler.diagnostics())
    actual_rng = np.random.get_state(); expected_rng = reference_rng.get_state()
    if not (np.array_equal(actual_rng[1], expected_rng[1]) and actual_rng[2:] == expected_rng[2:]):
        raise ValueError('Actual fitting draws differ from the declared common random stream')
    x, y, mass = root.means(); opening_steps = cfg['opening_steps']
    legal = x[:, 46:51].astype(float); expected = np.maximum(y, 0) * legal
    totals = expected.sum(1, keepdims=True)
    expected = np.divide(expected, totals, out=legal / legal.sum(1, keepdims=True), where=totals > 0)
    def quality():
        actual = net.get_strategy(torch.from_numpy(x), torch.from_numpy(x[:, 46:51])).numpy()
        return float(np.average(np.abs(actual - expected).sum(1) / 2, weights=mass))
    tv = quality()
    while tv > cfg['max_opening_tv'] and opening_steps < cfg['opening_max_steps']:
        extra = min(1024, cfg['opening_max_steps'] - opening_steps)
        fit_opening(net, (x, y, mass), extra, cfg['learning_rate'])
        opening_steps += extra; tv = quality()
    if tv > cfg['max_opening_tv']:
        raise ValueError('Original opening-quality criterion failed; no automatic retry')
    assert_finite_arrays(net)
    model_path = output / 'policy.npz'; export_compact(net, model_path)
    with np.load(model_path, allow_pickle=False) as saved:
        opening_sha = hashlib.sha256(b''.join(saved[k].tobytes() for k in sorted(saved.files)
                                              if k.startswith('opening_'))).hexdigest()
    return dict(model=str(model_path), model_sha256=digest(model_path),
                parent_numerical_export_verified=True, settings=settings,
                initial_parameter_sha256=init_sha, selected_row_sha256=selection,
                sampling_cdf_sha256=cdf_sha, fitting_random_stream_sha256=draw_sha,
                replay_rows=[len(b) for b in buffers], fitting=fitted,
                opening_steps=opening_steps, opening_tv=tv, opening_sha256=opening_sha,
                exact_online_resume=False, new_selfplay_iterations=0)


def fit_rich(plan, seed, arm, output):
    import numpy as np
    import torch
    from compact_lightweight_backup import load_state
    from rich_regret_model import RichRegretModel
    from rich_poker_observation import RICH_DIM, SCHEMA
    from train_rich_selfplay import validate_fitting_report
    spec = plan['parents']['rich'][str(seed)]
    state = load_state(spec['checkpoint'])
    if state['iteration'] != 96 or state['seed'] != seed:
        raise ValueError('Actual rich96 frozen replay required')
    masked = arm == 'rich_masked'
    buffers, selection = views_from_states(state['buffers'], mask_extra=masked, seed=seed)
    fit_seed = 916000 + seed
    random.seed(fit_seed); np.random.seed(fit_seed); torch.manual_seed(fit_seed)
    model = RichRegretModel(plan['parents']['legacy'][str(seed)]['model'], RICH_DIM,
                            plan['source'], width=128)
    # Both arms have the same architecture/initialization/rows/targets/batches.
    init_sha = hashlib.sha256(b''.join(v.detach().numpy().tobytes()
                                     for v in model.residual.state_dict().values())).hexdigest()
    report = model.fit(buffers, steps=1024, batch_size=512, learning_rate=.001)
    validate_fitting_report(report, dict(fitting_max_prior_mse_ratio=1.05))
    before = [model.strategy(b.features[i], b.features[i, 46:51])
              for b in buffers for i in range(min(4, len(b)))]
    if masked:
        # Fold the feature mask into exported weights so the EXISTING runtime is exact.
        with torch.no_grad():
            model.residual.layers[0].weight[:, 51:] = 0
        model._sync_numpy()
    after = [model.strategy(b.features[i], b.features[i, 46:51])
             for b in buffers for i in range(min(4, len(b)))]
    if any(not np.array_equal(a, b) for a, b in zip(before, after)):
        raise ValueError('Folding the input mask changed fitted predictions')
    # Probe full, unmasked retained observations through the exported architecture.
    if masked:
        for original in state['buffers']:
            for x in original['features'][:4]:
                if not np.array_equal(model.strategy(x, x[46:51]),
                                      model.strategy(np.r_[x[:51], np.zeros(RICH_DIM-51, np.float32)], x[46:51])):
                    raise ValueError('Extra input mask was not preserved at inference')
    model_path = output / 'residual.npz'; model.export(model_path)
    descriptor = dict(format='rich-policy-export-v1', model=str(model_path),
        model_sha256=digest(model_path), teacher=str(model.teacher_path),
        teacher_sha256=model.teacher_sha256, source=plan['source'], iteration=96,
        seed=seed, feature_schema=SCHEMA, parent_iteration=360, average_policy_available=False)
    path = output / 'policy.json'; atomic_json(path, descriptor)
    # State fingerprints establish equal random draws for the feature-only pair.
    rng_sha = hashlib.sha256(np.random.get_state()[1].tobytes()
                            + str(np.random.get_state()[2:]).encode()
                            + torch.get_rng_state().numpy().tobytes()).hexdigest()
    return dict(model=str(path), model_sha256=digest(path),
                settings=dict(steps=1024, batch=512, capacity=5000, mask_extra=masked),
                initial_parameter_sha256=init_sha, final_rng_sha256=rng_sha,
                selected_row_sha256=selection, replay_rows=[len(b) for b in buffers],
                fitting=report, mask_folded_losslessly=True,
                old51_replay_transferred=False, new_selfplay_iterations=0,
                data_origin='Completed rich96 replay, shared within this feature pair only')


def run(study, arm, seed):
    study = Path(study).resolve(); plan = read(study / 'plan.json')
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Undeclared diagnostic arm/seed')
    verify_files(plan['frozen_files'])
    source = plan['source']; sys.path[:0] = [source, plan['legacy_tools']]
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output = study / 'fitting' / f'{arm}-seed-{seed}'
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    atomic_json(output / 'status.json', dict(status='running', phase='fitting',
        observed_utc=utc(), pid=os.getpid(), process_start=process_start(os.getpid()),
        arm=arm, seed=seed, new_selfplay_iterations=0))
    try:
        result = (fit_legacy if arm in LEGACY_ARMS else fit_rich)(plan, seed, arm, output)
        verify_files(plan['frozen_files'])
        result.update(complete=True, completed_utc=utc(), arm=arm, seed=seed,
                      seconds=time.monotonic()-began, kind='derived-policy-probe',
                      resumable_selfplay=False, model_promoted=False, automatic_retry=False)
        atomic_json(output / 'result.json', result)
        atomic_json(output / 'status.json', dict(status='complete', observed_utc=utc(),
            arm=arm, seed=seed, seconds=result['seconds'], new_selfplay_iterations=0))
        return result
    except BaseException as error:
        atomic_json(output / 'status.json', dict(status='failed', observed_utc=utc(),
            arm=arm, seed=seed, error=type(error).__name__+': '+str(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    a = p.parse_args(); print(json.dumps(run(a.study, a.arm, a.seed), allow_nan=False))
