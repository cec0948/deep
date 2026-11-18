"""Fit only non-opening preflop regrets and compose the policy with frozen postflop play."""
from pathlib import Path
import sys

import numpy as np

from controlled_poker_refit import assert_finite_arrays, read
from fullhouse_v3_common import atomic_json, digest
from poker_average_runtime import policy_for as original_policy


FORMAT = 'preflop-street-policy-v1'


def file_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write_policy(path, before_flop, after_flop):
    atomic_json(path, dict(format=FORMAT, preflop=file_ref(before_flop), postflop=file_ref(after_flop)))
    return path


class StreetPolicy:
    def __init__(self, model, source):
        self.model = Path(model).resolve()
        self.model_type = FORMAT
        self.dependency_hashes = {}
        spec = read(self.model)
        if spec.get('format') != FORMAT:
            raise ValueError('Unexpected street-policy format')
        self.policies = {}
        for street in ('preflop', 'postflop'):
            ref = spec[street]
            if digest(ref['path']) != ref['sha256']:
                raise ValueError(f'{street} weights changed')
            policy = original_policy(ref['path'], source)
            if policy.model_type != 'compact_advantage':
                raise ValueError('Expected a frozen compact regret policy')
            self.policies[street] = policy
            self.dependency_hashes.update(policy.dependency_hashes)

    def decide(self, state, seed):
        street = state['street']
        if street not in ('preflop', 'flop', 'turn', 'river'):
            raise ValueError('Invalid betting street')
        policy = self.policies['preflop' if street == 'preflop' else 'postflop']
        return policy.decide(state, seed)


def evaluator_policy(model, source):
    policy = StreetPolicy(model, source) if Path(model).suffix == '.json' else original_policy(model, source)
    code = Path(__file__).resolve()
    policy.dependency_hashes[str(code)] = digest(code)
    return policy


def preflop_rows(buffer):
    from compact_regret import ReplayView
    from regret_partition import is_opening
    count = int(buffer['count'])
    x = buffer['features'][:count]
    selected = (x[:, 17] == 1) & ~is_opening(x)
    if not selected.any():
        raise ValueError('Missing non-opening preflop samples')
    return ReplayView(x[selected], buffer['advantages'][:count][selected], buffer['iterations'][:count][selected])


def fit(campaign, seed, folder, steps):
    import torch
    # Use this campaign's frozen training implementation and feature encoder.
    sys.path[:0] = [str(campaign/'source'), str(campaign/'tools')]
    from training import r1_support as r1
    from compact_regret import CompactRegretNet, export_compact
    from regret_fitting import train_conditional
    from regret_partition import base_strata
    from scaled_regret_fitting import ScaledRegretSampler
    from train_poker_average_continuation import load_checkpoint

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parent, checkpoint = load_checkpoint(campaign/f'seed-{seed}')
    if (parent['seed'], parent['iteration']) != (seed, 480):
        raise ValueError('Expected seed-matched 480 checkpoint')
    buffers = parent['buffers']
    if len(buffers) != 6 or any(b['count'] != 100000 for b in buffers):
        raise ValueError('Unexpected parent replay coverage')
    replay = [preflop_rows(b) for b in buffers]
    config = parent['config']
    if (config['main_steps'], config['batch_size'], config['learning_rate']) != (1024, 512, .001):
        raise ValueError('Frozen parent fitting budget differs')
    net = CompactRegretNet()
    net.load_state_dict(parent['net'])
    rng = np.random.default_rng()
    r1.restore_rng(parent['rng'], rng)
    del buffers, parent
    parent_path = checkpoint/'current.npz'
    restored = folder/'restored.npz'
    export_compact(net, restored)
    with np.load(restored, allow_pickle=False) as a, np.load(parent_path, allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('480 state differs from the independently exported policy')
    restored.unlink()
    sampler = ScaledRegretSampler(replay, base_strata, 'stratum-scaled-feature-conditional')
    loss = train_conditional(net.main, sampler, torch.optim.Adam(net.main.parameters(), lr=config['learning_rate']),
                             steps, config['batch_size'])
    groups = np.unique(sampler.groups)
    net.main_scales[groups] = torch.from_numpy(sampler.scales[groups])
    assert_finite_arrays(net)
    path = folder/'specialist.npz'
    export_compact(net, path)
    with np.load(path, allow_pickle=False) as a, np.load(parent_path, allow_pickle=False) as b:
        if (not np.isfinite(loss) or sampler.draws.sum() != steps*config['batch_size']
                or any(not np.array_equal(a[k], b[k]) for k in a.files if k.startswith('opening_'))
                or np.array_equal(a['trunk_w0'], b['trunk_w0'])):
            raise ValueError('Invalid fit, unchanged main weights or changed opening expert')
    return dict(seed=seed, parent_iteration=480, new_selfplay_iterations=0,
                selected_rows_per_seat=[len(v) for v in replay], steps=steps, batch=config['batch_size'],
                training_draws=int(sampler.draws.sum()), mean_training_loss=loss,
                restored_parent_verified=True, opening_expert_unchanged=True, model=file_ref(path))
