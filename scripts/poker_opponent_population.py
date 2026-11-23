"""External-sampling self-play with a fixed postflop opponent assignment per seat and tree."""
from collections import Counter
from copy import deepcopy
from functools import partial
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from controlled_poker_refit import read, verify_files
from fullhouse_v3_common import ROOT, digest
from snapshot_poker_preflop_online import verify_export

STAGES = (300, 360, 420)
BOUNDED = ROOT/'artifacts/bounded-continuation/20261113T014811152621Z'
CONTINUATION = ROOT/'artifacts/average-continuation/20261117T032100Z'
EXTENSION = ROOT/'artifacts/average-extension/20261117T063900Z'


def historical_inputs(seeds):
    """Link each seed's immutable exports and the next campaign's restored parent."""
    owners = [path/'stage-ready.json' for path in [BOUNDED, CONTINUATION, EXTENSION]]
    bounded, continuation, extension = map(read, owners)
    frozen = {str(p):digest(p) for p in owners}
    policies = {}
    for s in seeds:
        seed = str(s)
        seat_folder = Path('campaign')/f'seed-{seed}'
        refs = {}
        for stage in STAGES:
            folder = BOUNDED if stage < 420 else CONTINUATION
            exported = folder/seat_folder/f'checkpoints/iter-{stage:05d}/current.npz'
            expected = (bounded['checkpoints'][str(stage)]['models'][seed] if stage < 420
                        else continuation['seeds'][seed]['files']['current.npz'])
            refs[str(stage)] = {'path':str(exported),'sha256':expected}
            frozen[str(exported)] = expected
        frozen[str(CONTINUATION/seat_folder/'parent360.npz')] = refs['360']['sha256']
        frozen[str(EXTENSION/seat_folder/'parent420.npz')] = refs['420']['sha256']
        frozen[str(EXTENSION/seat_folder/'checkpoints/iter-00480/current.npz')] = extension['seeds'][seed]['files']['current.npz']
        if len(set(r['sha256'] for r in refs.values())) != len(STAGES):
            raise ValueError('Repeated historical postflop network')
        policies[seed] = refs
    verify_files(frozen)
    return policies, frozen


def load_historical(networks, refs):
    from compact_regret import CompactNumpyPolicy, ARRAY_MAP

    if set(refs) != set(map(str, STAGES)):
        raise ValueError('Expected all three registered same-seed history entries')
    verify_files({r['path']:r['sha256'] for r in refs.values()})
    result = {}
    for key, ref in refs.items():
        # Copying the initialized network does not consume the live training RNG.
        model = deepcopy(networks.general)
        with np.load(ref['path'], allow_pickle=False) as data:
            for head, prefix in [(model.main,''),(model.opening,'opening_')]:
                head.load_state_dict({param: torch.from_numpy(data[prefix+name].copy())
                                      for param, name in ARRAY_MAP.items()})
            model.main_scales.copy_(torch.from_numpy(data['main_scales'].copy()))
            model.opening_scale.copy_(torch.from_numpy(data['opening_scale'].copy()))
        verify_export(ref['path'], model.state_dict(), ARRAY_MAP)
        result[int(key)] = CompactNumpyPolicy(model.eval())
    return result


def assignments(seed, iteration, count, traverser):
    """Codes in seat order: -1 traverser, 0 current, or the frozen iteration."""
    if traverser != (iteration-1)%6:
        raise ValueError('Unexpected traverser')
    rng = np.random.default_rng(np.random.SeedSequence([0x504f4f4c, seed, iteration]))
    slots = np.array([0, 0, 0, *STAGES], dtype='<i2')[rng.integers(6, size=(count,6))]
    slots[:,traverser] = -1
    return slots


class TreePolicy:
    def __init__(self, current, historical, visits):
        self.current = current
        self.historical = historical
        self.visits = visits

    def strategy(self, features, legal):
        if features[17] == 1:
            return self.current.strategy(features, legal)
        # The training encoder exposes position relative to the dealer at 21:27.
        position = int(np.argmax(features[21:27]))
        slot = int(self.by_position[position])
        self.visits[slot] += 1
        if slot <= 0:
            return self.current.strategy(features, legal)
        return self.historical[slot].strategy(features, legal)


def pool_traversals(seed, historical, current_only, evidence, n_traversals, n_players, traverser,
                    net, buffer, iteration, rng, equity_tables, *, opponent_exploration,
                    strategy_buffer, randomize_stacks):
    from training import traverse_fast as sampler

    if (n_players != 6 or randomize_stacks or strategy_buffer is not None
            or opponent_exploration != 0. or equity_tables is None):
        raise ValueError('Opponent pool requires the registered six-seat traversal settings')
    sampler._EQUITY_TABLES = equity_tables
    slots = assignments(seed, iteration, n_traversals, traverser)
    if current_only:
        slots[slots > 0] = 0
    visits = Counter()
    router = TreePolicy(sampler.NumpyPolicy(net), historical, visits)
    features, advantages = [], []
    total = 0.
    for assignment in slots:
        dealer = int(rng.integers(0, n_players))
        state, deck, holes = sampler.new_hand(n_players, dealer, rng, None)
        # This permutation is fixed for every descendant of the current root hand.
        router.by_position = np.roll(assignment, -dealer)
        total += sampler.traverse_external(state, deck, holes, traverser, router, features, advantages,
                                          [], [], [], rng, opponent_exploration)
    if features:
        buffer.add_batch(np.array(features), np.array(advantages), iteration)
    evidence.update(assignment_sha256=sha256(slots.tobytes()).hexdigest(),
                    opponent_seat_assignments={str(k):int(np.count_nonzero(slots==k)) for k in (0,*STAGES)},
                    postflop_node_visits={str(k):visits[k] for k in (-1,0,*STAGES)})
    return total/n_traversals


def pool_update(networks, buffers, root, rng, iteration, cfg, equity, seed, historical, *, current_only=False):
    from training import traverse_fast
    from train_poker_fixed_preflop_window import update_postflop

    evidence = {}
    collect = partial(pool_traversals, seed, historical, current_only, evidence)
    # The archived updater imports this function on entry. Bind just the collector
    # in this single-threaded worker so replay, root collection and fitting are identical.
    with patch.object(traverse_fast, 'run_traversals_fast', collect):
        row = update_postflop(networks, buffers, root, rng, iteration, cfg, equity)
    return dict(row, opponent_population=evidence)
