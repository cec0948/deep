"""Fit nested 0/64/256-step doses from the confirmed 480 preflop specialist."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from controlled_poker_refit import assert_finite_arrays, read, verify_files
from fullhouse_v3_common import atomic_json
from poker_preflop_specialist import file_ref, preflop_rows, write_policy
from run_poker_preflop_pilot import CAMPAIGN, SEEDS
from snapshot_poker_preflop_online import verify_export
from train_poker_warm_preflop import OUTPUT as WARM_START, connect, fingerprint


SOURCE_PLAN = WARM_START / 'evaluation-plan.json'
SOURCE_SHA = 'ff9bf107066817372f1529962d6743a69e056a98b35b97e6e87199f312be7399'
METHOD = 'stratum-scaled-feature-conditional'
SEED_OFFSET = 20261119
BATCH = 512
LEARNING_RATE = .001


class AuditedBatches:
    """Observe upstream batches and verify their IDs against an independent MT19937."""

    def __init__(self, sampler, seed):
        self.sampler = sampler
        self.rng = np.random.RandomState(seed)
        self.order = hashlib.sha256()
        self.counts = np.zeros_like(sampler.draws)

    def sample(self, size):
        ids = np.searchsorted(self.sampler.cdf, self.rng.random_sample(size), side='right')
        x, y = self.sampler.sample(size)
        if not np.array_equal(x, self.sampler.features[ids]) or not np.array_equal(y, self.sampler.targets[ids]):
            raise ValueError('Actual fit batch differs from the committed draw sequence')
        np.add.at(self.counts, ids, 1)
        self.order.update(ids.astype('<u8').tobytes())
        return x, y

    def verify(self, updates):
        actual = np.random.get_state()
        expected = self.rng.get_state()
        if (actual[0] != expected[0] or not np.array_equal(actual[1], expected[1])
                or actual[2:] != expected[2:] or not np.array_equal(self.sampler.draws, self.counts)
                or self.counts.sum() != updates * BATCH):
            raise ValueError('Actual fit RNG state or sample counts differ')


def replay_digest(replay):
    checksum = hashlib.sha256()
    for seat, view in enumerate(replay):
        for key in ('features', 'advantages', 'iterations'):
            data = getattr(view, key)
            checksum.update(f'{seat}/{key}/{data.shape}/{data.dtype.str}\n'.encode())
            checksum.update(data.tobytes())
    return checksum.hexdigest()


def verify_exported_dose(path, net, parent, mapping):
    verify_export(path, net.state_dict(), mapping)
    with np.load(path, allow_pickle=False) as fitted, np.load(parent, allow_pickle=False) as initial:
        if set(fitted.files) != set(initial.files):
            raise ValueError('Changed compact file layout')
        changed = {k for k in initial.files if not np.array_equal(initial[k], fitted[k])}
        if not changed or not changed.issubset(mapping.values()):
            raise ValueError('Expected only non-opening main parameters to change')


def fit(seed, folder, ref):
    """Create a new seed folder; return 3 deployable policy references after verification."""
    verify_files({SOURCE_PLAN: SOURCE_SHA})
    plan = read(SOURCE_PLAN)
    if seed not in SEEDS or ref != plan['policies']['specialist480'][str(seed)]:
        raise ValueError('Expected the independently pinned specialist480 for this seed')
    frozen = plan['frozen_files']
    verify_files(frozen)
    stage_path = CAMPAIGN.parent / 'stage-ready.json'
    stage = read(stage_path)['seeds'][str(seed)]
    if (str(stage_path) not in frozen or stage['iteration'] != 480 or
            Path(stage['checkpoint']) != CAMPAIGN / f'seed-{seed}' / 'checkpoints/iter-00480'):
        raise ValueError('Unexpected source replay checkpoint')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    networks, state, connection = connect(seed, ref)
    # connect() binds the frozen campaign modules and verifies both real policy routes.
    from compact_regret import ARRAY_MAP, export_compact
    from regret_fitting import train_conditional
    from regret_partition import base_strata
    from scaled_regret_fitting import ScaledRegretSampler

    buffers = state['buffers']
    cfg = state['config']
    if (connection['checkpoint_files'] != stage['files'] or len(buffers) != 6
            or any(b['count'] != 100000 for b in buffers)
            or (cfg['batch_size'], cfg['learning_rate']) != (BATCH, LEARNING_RATE)):
        raise ValueError('Changed retained replay or optimizer configuration')
    replay = [preflop_rows(b) for b in buffers]
    selected_sha = replay_digest(replay)
    selected_rows = [len(v) for v in replay]
    del state, buffers
    net = networks.specialist
    spec = read(ref['path'])
    sampler = ScaledRegretSampler(replay, base_strata, METHOD)
    groups = np.unique(sampler.groups)
    if not np.array_equal(net.main_scales.numpy()[groups], sampler.scales[groups]):
        raise ValueError('Parent regret units differ from scales recomputed on its frozen replay')
    sampler_seed = SEED_OFFSET + seed
    np.random.seed(sampler_seed)
    torch.manual_seed(sampler_seed)
    batches = AuditedBatches(sampler, sampler_seed)
    optimizer = torch.optim.Adam(net.main.parameters(), lr=LEARNING_RATE)
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    policies = {'steps0': ref}
    arms = {'steps0': dict(optimizer_steps=0, training_draws=0, prefix64_sha256=None,
                           sample_order_sha256=batches.order.hexdigest(), weights_sha256=fingerprint(net))}
    if arms['steps0']['weights_sha256'] != connection['weights']['preflop']:
        raise ValueError('The first fit update would not use the deployed specialist weights')
    previous = 0
    prefix64 = None
    for updates in (64, 256):
        loss = train_conditional(net.main, batches, optimizer, updates - previous, BATCH)
        batches.verify(updates)
        if (not np.isfinite(loss) or any(int(optimizer.state[p]['step']) != updates
                                        for p in net.main.parameters())):
            raise ValueError('Nonfinite fit or interrupted Adam trajectory')
        net.eval()
        assert_finite_arrays(net)
        label = f'steps{updates}'
        model = folder / f'{label}.npz'
        export_compact(net, model)
        verify_exported_dose(model, net, spec['preflop']['path'], ARRAY_MAP)
        policy = write_policy(folder / f'{label}.json', model, spec['postflop']['path'])
        if read(policy)['postflop'] != spec['postflop']:
            raise ValueError('Postflop policy changed')
        policies[label] = file_ref(policy)
        if updates == 64:
            prefix64 = batches.order.hexdigest()
        arms[label] = dict(optimizer_steps=updates, training_draws=updates * BATCH,
                           sample_order_sha256=batches.order.hexdigest(), prefix64_sha256=prefix64,
                           weights_sha256=fingerprint(net), segment_mean_loss=loss)
        previous = updates
    if len({a['weights_sha256'] for a in arms.values()}) != len(arms):
        raise ValueError('One or more dose snapshots did not change the specialist')
    verify_files({p['path']: p['sha256'] for p in (ref, spec['postflop'], spec['preflop'])})
    atomic_json(folder / 'fit.json', dict(seed=seed, iteration=480, new_selfplay_iterations=0,
        source_plan=file_ref(SOURCE_PLAN), source_stage=file_ref(stage_path), connection=connection,
        replay_sha256=selected_sha, selected_rows_per_seat=selected_rows, sampler_method=METHOD,
        sampler_seed=sampler_seed, batch_size=BATCH, learning_rate=LEARNING_RATE, policies=policies,
        arms=arms, opening_unchanged=True, postflop_unchanged=True, scales_unchanged=True))
    return policies
