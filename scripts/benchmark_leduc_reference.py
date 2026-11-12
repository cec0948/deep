"""Bounded OpenSpiel CFR/Deep-CFR diagnostics with exact two-player evaluation.

The neural traversal, reservoir sampling, regret matching and fitting are the
installed OpenSpiel implementation. This wrapper exposes its alternating loop
to save predeclared learning curves. Checkpoint-only strategy fits preserve RNG
and do not influence subsequent advantage training. This is a reduced-budget
library benchmark, not a reproduction of the original paper's full experiment.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_config(cfg, method, seed):
    if cfg['game'] != 'leduc_poker':
        raise ValueError('Only fixed two-player Leduc is authorized by this benchmark')
    if method not in ('tabular', 'deep'):
        raise ValueError('Unknown method')
    if method == 'deep' and seed not in cfg['seeds']:
        raise ValueError('Seed is absent from the predeclared configuration')
    for kind in ('tabular', 'deep'):
        count, checkpoints = cfg[kind + '_iterations'], cfg[kind + '_checkpoints']
        if (type(count) is not int or not 1 <= count <= 5000
                or checkpoints != sorted(set(checkpoints))
                or checkpoints[0] != 0 or checkpoints[-1] != count
                or any(type(t) is not int or not 0 <= t <= count for t in checkpoints)):
            raise ValueError('Invalid bounded iterations/checkpoints')
    limits = {'traversals_per_player': 512, 'advantage_steps': 4096,
              'strategy_steps': 8192, 'batch_size': 4096,
              'memory_capacity': 1000000, 'max_seconds_per_process': 1800}
    for key, maximum in limits.items():
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError('Invalid bounded setting: ' + key)
    if cfg['batch_size'] > cfg['memory_capacity']:
        raise ValueError('Batch exceeds memory capacity')
    if not 0 < cfg['learning_rate'] <= .1:
        raise ValueError('Invalid learning rate')
    if (not cfg['network_layers'] or len(cfg['network_layers']) > 4
            or any(type(n) is not int or not 1 <= n <= 512 for n in cfg['network_layers'])):
        raise ValueError('Invalid bounded network shape')


def normalized_legal(raw, legal):
    raw, legal = np.asarray(raw, dtype=np.float64), np.asarray(legal, dtype=np.float64)
    if raw.shape != legal.shape or not np.isfinite(raw).all() or np.any(raw < -1e-12):
        raise ValueError('Invalid policy probabilities')
    if np.any(legal.sum(1) == 0):
        raise ValueError('Information state without legal actions')
    values = np.maximum(raw, 0) * legal
    masses = values.sum(1, keepdims=True)
    return np.divide(values, masses, out=legal / legal.sum(1, keepdims=True), where=masses > 0)


def check_policy(values, legal):
    if (values.shape != legal.shape or not np.isfinite(values).all()
            or np.any(values < -1e-12) or np.max(np.abs(values.sum(1) - 1)) > 1e-8
            or np.any(np.abs(values[legal == 0]) > 1e-12)):
        raise ValueError('Policy is not a legal normalized distribution')


class ExactPolicySpace:
    """Full infosets and perfect-recall own-action paths; no poker compression."""
    def __init__(self, game):
        from open_spiel.python import policy
        self.game = game
        self.template = policy.TabularPolicy(game)
        self.legal = self.template.legal_actions_mask.astype(np.float64)
        self.inputs = self.template.state_in.astype(np.float32)
        self.players = np.array([state.current_player() for state in self.template.states])
        self.keys = [None] * len(self.players)
        for key, index in self.template.state_lookup.items():
            self.keys[index] = key
        self.own_paths = [None] * len(self.players)
        def visit(state, paths):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    visit(state.child(action), paths)
                return
            player = state.current_player()
            row = self.template.state_lookup[state.information_state_string(player)]
            if self.own_paths[row] is None:
                self.own_paths[row] = paths[player]
            elif self.own_paths[row] != paths[player]:
                raise ValueError('Information-state encoding violates perfect recall')
            for action in state.legal_actions():
                updated = list(paths)
                updated[player] = paths[player] + ((row, action),)
                visit(state.child(action), tuple(updated))
        visit(game.new_initial_state(), ((), ()))
        if any(path is None for path in self.own_paths):
            raise ValueError('Incomplete information-state enumeration')
        self.numerator = np.zeros_like(self.legal, dtype=np.float64)
        self.mass = np.zeros(len(self.players), dtype=np.float64)

    def as_policy(self, values):
        check_policy(values, self.legal)
        result = copy.copy(self.template)
        result.action_probability_array = np.asarray(values, dtype=np.float64).copy()
        return result

    def own_reach(self, values):
        return np.array([np.prod([values[row, action] for row, action in path])
                         for path in self.own_paths], dtype=np.float64)

    def accumulate(self, values, weight):
        check_policy(values, self.legal)
        if not weight > 0:
            raise ValueError('Average weight must be positive')
        reach = self.own_reach(values) * weight
        self.numerator += reach[:, None] * values
        self.mass += reach

    def average(self):
        return np.divide(self.numerator, self.mass[:, None],
                         out=self.legal / self.legal.sum(1, keepdims=True),
                         where=self.mass[:, None] > 0)

    def metrics(self, values):
        from open_spiel.python.algorithms import exploitability, expected_game_score
        profile = self.as_policy(values)
        result = exploitability.nash_conv(self.game, profile, return_only_nash_conv=False)
        returns = expected_game_score.policy_value(self.game.new_initial_state(), [profile, profile])
        if abs(float(sum(returns))) > 1e-9:
            raise ValueError('Non-zero-sum exact policy value')
        return dict(exploitability=float(result.nash_conv / 2),
                    nash_conv=float(result.nash_conv),
                    player_improvements=np.asarray(result.player_improvements).tolist(),
                    policy_value=np.asarray(returns).tolist(),
                    evaluation='OpenSpiel exact best response over all chance outcomes and information sets')


def current_policy(solver, space):
    """Call the official regret matcher at every full information state."""
    result = np.zeros_like(space.legal)
    for i, state in enumerate(space.template.states):
        _, result[i] = solver._sample_action_from_advantage(state, state.current_player())
    # Correct only floating-point row-sum drift, not the official fallback rule.
    result = normalized_legal(result, space.legal)
    check_policy(result, space.legal)
    return result


def fit_average_for_evaluation(solver, space):
    """A fresh fixed-budget strategy fit, isolated from future training RNG."""
    import torch
    numpy_rng, torch_rng, python_rng = np.random.get_state(), torch.get_rng_state(), random.getstate()
    state = copy.deepcopy(solver._policy_network.state_dict())
    optimizer = copy.deepcopy(solver._optimizer_policy.state_dict())
    try:
        solver._reinitialize_policy_network()
        loss = solver._learn_strategy_network()
        with torch.inference_mode():
            raw = solver._policy_network(torch.from_numpy(space.inputs)).numpy().astype(np.float64)
        legal_mass = (raw * space.legal).sum(1)
        values = normalized_legal(raw, space.legal)
        saved = copy.deepcopy(solver._policy_network.state_dict())
        details = dict(loss=float(loss) if loss is not None else None,
                       raw_legal_mass_min=float(legal_mass.min()),
                       raw_legal_mass_mean=float(legal_mass.mean()),
                       adapter='Mask illegal softmax outputs and renormalize; OpenSpiel action_probabilities does not renormalize legal mass')
        return values, details, saved
    finally:
        solver._policy_network.load_state_dict(state)
        solver._optimizer_policy.load_state_dict(optimizer)
        np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng); random.setstate(python_rng)


def run(config, method, seed, output):
    import pyspiel
    import torch
    from open_spiel.python import policy
    from open_spiel.python.algorithms import cfr, exploitability, best_response, expected_game_score
    from open_spiel.python.pytorch import deep_cfr
    cfg = json.loads(Path(config).read_text()); validate_config(cfg, method, seed)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing benchmark output is immutable')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    game = pyspiel.load_game(cfg['game'], {'players': 2})
    space = ExactPolicySpace(game)
    output.mkdir(parents=True)
    sources = {str(Path(__file__).resolve()): digest(__file__),
               str(Path(config).resolve()): digest(config)}
    for module in (policy, cfr, exploitability, best_response, expected_game_score, deep_cfr, pyspiel):
        path = Path(inspect.getfile(module)).resolve(); sources[str(path)] = digest(path)
    report = dict(format='leduc-reference-benchmark-v1', complete=False, method=method,
        seed=seed, config=cfg, started_utc=utc(), source_hashes=sources,
        versions={name: importlib.metadata.version(name) for name in ('open_spiel', 'torch', 'numpy', 'dm-tree')},
        game=str(game), num_players=2, information_states=len(space.players),
        input_dimensions=space.inputs.shape[1], actions=game.num_distinct_actions(),
        initial_uniform=space.metrics(space.template.action_probability_array), curve=[],
        limitations=['Reduced-budget OpenSpiel reference implementation, not full paper reproduction',
          'Exact two-player Leduc exploitability does not validate the six-player hold-em model or its compressed features',
          'Neural advantage/value networks are not the compact-v4 training architecture',
          'Realized average uses exact own-reach weights and linear iteration weights; it avoids a strategy-network fitting error',
          'Checkpoint strategy fits use full current sampled memory but preserve advantage-training RNG; each fit starts fresh',
          'The latest/current policy need not improve monotonically'])
    write_json(output/'config.json', cfg)
    write_json(output/'information-states.json', space.keys)
    began = time.monotonic(); final_state = None
    def status(state, iteration, **extra):
        write_json(output/'status.json', dict(status=state, method=method, seed=seed,
            iteration=iteration, planned_iterations=cfg[method+'_iterations'], pid=os.getpid(),
            updated_utc=utc(), seconds=time.monotonic()-began, **extra))
    def checkpoint(iteration, profiles, extra=None):
        if time.monotonic()-began > cfg['max_seconds_per_process']:
            raise TimeoutError('Predeclared benchmark time budget exhausted')
        row = dict(iteration=iteration, seconds=time.monotonic()-began,
                   metrics={name:space.metrics(values) for name,values in profiles.items()})
        row.update(extra or {})
        saved = output/f'policies-{iteration:05d}.npz'
        np.savez_compressed(saved, **profiles)
        row['policy_file'] = dict(name=saved.name, bytes=saved.stat().st_size, sha256=digest(saved))
        report['curve'].append(row)
        write_json(output/'results.json', report)
        print(json.dumps({k:v for k,v in row.items() if k != 'policy_file'}), flush=True)
    iteration = 0
    try:
        status('running', 0)
        if method == 'tabular':
            solver = cfr.CFRSolver(game)
            def profiles():
                return {name:np.asarray(policy.action_probability_array).copy()
                        for name,policy in [('current',solver.current_policy()),('average',solver.average_policy())]}
            checkpoint(0, profiles())
            for iteration in range(1, cfg['tabular_iterations']+1):
                if time.monotonic()-began > cfg['max_seconds_per_process']:
                    raise TimeoutError('Predeclared benchmark time budget exhausted')
                solver.evaluate_and_update_policy()
                status('running', iteration)
                if iteration in cfg['tabular_checkpoints']:
                    checkpoint(iteration, profiles())
        else:
            solver = deep_cfr.DeepCFRSolver(game,
                policy_network_layers=tuple(cfg['network_layers']),
                advantage_network_layers=tuple(cfg['network_layers']),
                num_iterations=cfg['deep_iterations'], num_traversals=cfg['traversals_per_player'],
                learning_rate=cfg['learning_rate'], batch_size_advantage=cfg['batch_size'],
                batch_size_strategy=cfg['batch_size'], memory_capacity=cfg['memory_capacity'],
                policy_network_train_steps=cfg['strategy_steps'],
                advantage_network_train_steps=cfg['advantage_steps'],
                reinitialize_advantage_networks=True, device='cpu', seed=seed, print_nash_convs=False)
            checkpoint(0, dict(current=current_policy(solver, space),
                               realized_average=space.average(),
                               uniform=space.template.action_probability_array.copy()))
            for iteration in range(1, cfg['deep_iterations']+1):
                if time.monotonic()-began > cfg['max_seconds_per_process']:
                    raise TimeoutError('Predeclared benchmark time budget exhausted')
                if solver._iteration != iteration:
                    raise AssertionError('Official iteration numbering changed')
                losses=[]
                for player in range(2):
                    for _ in range(cfg['traversals_per_player']):
                        solver._traverse_game_tree(solver._root_node, player)
                    solver._reinitialize_advantage_network(player)
                    loss=solver._learn_advantage_network(player)
                    losses.append(float(loss) if loss is not None else None)
                    if player == 0:
                        # p1 pre-update is sampled while traversing p0; p0 post-update
                        # is sampled while traversing p1. This is the realized profile
                        # represented by the official strategy memory for iteration t.
                        space.accumulate(current_policy(solver, space), iteration)
                solver._iteration += 1
                status('running', iteration)
                if iteration in cfg['deep_checkpoints']:
                    fitted, details, policy_state = fit_average_for_evaluation(solver, space)
                    checkpoint(iteration, dict(current=current_policy(solver, space),
                        realized_average=space.average(), fitted_average=fitted),
                        dict(advantage_losses=losses, strategy_fit=details,
                            advantage_rows=[len(b) for b in solver.advantage_buffers],
                            strategy_rows=len(solver.strategy_buffer)))
                    final_state=dict(advantage=[n.state_dict() for n in solver._advantage_networks],
                        fitted_average=policy_state, iteration=iteration, seed=seed, config=cfg,
                        kind='two-player-leduc-benchmark', resumable_selfplay=False)
            torch.save(final_state, output/'models.pt')
        for name, sha in sources.items():
            if digest(name) != sha:
                raise ValueError('Benchmark source/config changed during execution')
        report.update(complete=True, completed_utc=utc(), seconds=time.monotonic()-began)
        write_json(output/'results.json', report); status('complete', iteration)
        return report
    except BaseException as error:
        report.update(error=str(error), stopped_utc=utc(), seconds=time.monotonic()-began)
        write_json(output/'results.json', report)
        status('failed', iteration, error=str(error)); raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--method', choices=['tabular','deep'], required=True)
    parser.add_argument('--seed', type=int, default=71)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    run(args.config,args.method,args.seed,args.output)
