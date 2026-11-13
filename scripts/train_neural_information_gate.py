"""Bounded shared-neural, full-information CFR candidate gate.

This trains a new small-game neural regret approximator. It does not extend,
replace or certify the old six-player model. The exact average-policy table is
used only to evaluate the neural training trajectory with an exact best response.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from regret_fitting import train_conditional
from scaled_regret_fitting import ScaledRegretSampler
from validate_strength_training import (Replay, adapter_strata, component_checks,
    exact_iteration, exploitability_of, game_and_infosets,
    independent_average_contributions, kuhn_features, regret_match)


@dataclass(frozen=True)
class NeuralBudget:
    seeds: tuple[int, ...] = (71, 72, 73)
    iterations: int = 32
    fit_steps: int = 256
    batch_size: int = 128
    hidden: int = 128
    learning_rate: float = .003
    max_seconds: float = 480.

    def validate(self):
        if self.seeds != (71, 72, 73):
            raise ValueError('Declared neural gate uses fixed seeds71,72,73')
        for key, maximum in [('iterations', 64), ('fit_steps', 512), ('batch_size', 512), ('hidden', 256)]:
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Invalid bounded neural budget: '+key)
        if not 0 < self.learning_rate <= .01 or not 0 < self.max_seconds <= 600:
            raise ValueError('Invalid learning-rate or wall-clock budget')


class SharedRegretNet(torch.nn.Module):
    """One shared function over observations; no infoset-ID lookup parameters."""
    def __init__(self, input_dim=51, hidden=128):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 5))

    def forward(self, x, legal):
        return self.body(x) * legal


class GlobalTargetUnits:
    """Change only the global positive target unit, preserving sampling law.

    The production sampler clamps its per-stratum scale at1. Full-tree tiny-game
    counterfactual regrets can be much smaller than1, so this explicit global
    unit keeps neural optimization well-conditioned. It does not alter relative
    loss weights or regret-matching probabilities.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        legal = sampler.features[:, 46:51]
        second = np.average(np.square(sampler.targets * legal).mean(1),
                            weights=sampler.probabilities)
        self.scale = max(float(np.sqrt(second)), 1e-8)

    def sample(self, size):
        x, y = self.sampler.sample(size)
        return x, y / self.scale


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def process_start(pid):
    raw = Path(f'/proc/{pid}/stat').read_text()
    return raw[raw.rfind(')')+2:].split()[19]


def neural_iteration(features, sampler, seed, budget):
    # Reinitialization is declared before results and matches the Deep CFR
    # regression design. Samples collected earlier still remain in replay.
    torch.manual_seed(seed)
    model = SharedRegretNet(features.shape[1], budget.hidden)
    scaled = GlobalTargetUnits(sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=budget.learning_rate)
    last_loss = train_conditional(model, scaled, optimizer, budget.fit_steps, budget.batch_size)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features)
        scores = model(x, x[:, 46:51]).numpy()
        sx = torch.from_numpy(sampler.features)
        prediction = model(sx, sx[:, 46:51]).numpy()
    target = sampler.targets / scaled.scale * sampler.features[:, 46:51]
    weighted_mse = float(np.average(np.square(prediction-target).mean(1), weights=sampler.probabilities))
    if not np.isfinite(scores).all() or not np.isfinite(weighted_mse):
        raise FloatingPointError('Nonfinite neural fitting result')
    return model, scores, dict(loss=float(last_loss), global_target_unit=scaled.scale,
                              full_replay_normalized_weighted_mse=weighted_mse,
                              shared_parameters=sum(p.numel() for p in model.parameters()))


def train_seed(game_name, seed, budget, deadline, progress, checkpoint):
    np.random.seed(seed)
    game, keys, infosets = game_and_infosets(game_name)
    features = kuhn_features(keys, infosets)
    checks, current = component_checks(game, keys, infosets, features, game_name)
    if not checks['passed']:
        raise ValueError('Mathematical component prerequisite failed')
    initial = exploitability_of(game, current)
    curve = [dict(iteration=0, average_exploitability=initial, current_exploitability=initial)]
    history_y = []
    average_sum = {k: np.zeros(5) for k in keys}
    average_mass = {k: 0. for k in keys}
    reference_sum = {k: np.zeros(5) for k in keys}
    reference_mass = {k: 0. for k in keys}
    started = time.monotonic()
    for iteration in range(1, budget.iterations+1):
        if time.monotonic() >= deadline:
            raise TimeoutError('Fixed neural gate budget exhausted; no retry')
        regrets, numerator, mass = exact_iteration(game, current, infosets)
        own = independent_average_contributions(game, current)
        for key in keys:
            average_sum[key] += iteration*numerator[key]
            average_mass[key] += iteration*mass[key]
            reference_sum[key] += iteration*own[key]*current[key]
            reference_mass[key] += iteration*own[key]
        history_y.append(np.stack([regrets[k] for k in keys]).astype(np.float32))
        replay = Replay(np.tile(features, (iteration, 1)), np.concatenate(history_y),
                        np.repeat(np.arange(1, iteration+1, dtype=np.float32), len(keys)))
        sampler = ScaledRegretSampler([replay], adapter_strata, 'stratum-scaled-feature-conditional')
        model, scores, fit = neural_iteration(features, sampler, seed*100000+iteration, budget)
        current = {key: regret_match(scores[i], features[i, 46:51]) for i, key in enumerate(keys)}
        average = {key: average_sum[key]/average_mass[key] for key in keys}
        if iteration in {1, 4, 12, 24, budget.iterations}:
            curve.append(dict(iteration=iteration, average_exploitability=exploitability_of(game, average),
                              current_exploitability=exploitability_of(game, current)))
        progress(seed=seed, iteration=iteration, planned_iterations=budget.iterations,
                 elapsed_seed_seconds=time.monotonic()-started, latest_curve=curve[-1], fit=fit)
        if iteration == budget.iterations:
            arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
            arrays['metadata_json'] = np.array(json.dumps(dict(kind='small_game_neural_gate',
                game=game_name, seed=seed, input_dim=features.shape[1], hidden=budget.hidden,
                iteration=iteration, global_target_unit=fit['global_target_unit'],
                six_player_policy=False)))
            temporary = checkpoint.with_suffix('.partial.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(checkpoint)
    average_error = max(float(np.max(np.abs(average[k]-reference_sum[k]/reference_mass[k]))) for k in keys)
    passed = bool(checks['passed'] and average_error < 1e-10
                  and curve[-1]['average_exploitability'] < initial*.5
                  and fit['full_replay_normalized_weighted_mse'] < 1.)
    return dict(seed=seed, game=game_name, shared_neural_approximator=True,
        tabular_approximator_used=False, information_state_tensor_size=game.information_state_tensor_size(),
        padded_input_dimension=features.shape[1], information_sets=len(keys),
        component_checks=checks, learning_curve=curve, final_fit=fit,
        independent_average_accumulation_max_error=average_error,
        neural_curve_gate_passed=passed, neural_checkpoint_sha256=digest(checkpoint),
        elapsed_seconds=time.monotonic()-started)


def run(output, budget=NeuralBudget(), game_name='leduc_poker'):
    budget.validate()
    if game_name not in ('kuhn_poker', 'leduc_poker'):
        raise ValueError('Only declared exact small games are accepted')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Neural gate output must be new; no overwrite or automatic retry')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = [Path(__file__), Path(__file__).with_name('validate_strength_training.py'),
               Path(__file__).with_name('regret_fitting.py'), Path(__file__).with_name('scaled_regret_fitting.py')]
    hashes = {str(path.resolve()):digest(path) for path in sources}
    began = time.monotonic()
    deadline = began+budget.max_seconds
    pid = os.getpid()
    token = process_start(pid)
    plan = dict(format='full-information-neural-gate-v1', created_utc=utc(), game=game_name,
        budget=asdict(budget), architecture=f'shared51→{budget.hidden}→{budget.hidden}→5ReLU MLP; reset each iteration',
        collector='Own exact full-tree counterfactual regret collector',
        fitter='Existing ScaledRegretSampler and train_conditional; explicit global target-unit normalization',
        strategy_average='Exact reach-weighted average of neural-generated policies, not a trained average-policy network',
        gate='All3 seeds: average exploitability below50percent uniform; weighted normalized regression MSE<1; mathematical checks pass',
        source_sha256=hashes, six_player_policy=False, selfplay_extension=False,
        automatic_retry=False, model_promotion=False)
    atomic_json(output/'plan.json', plan)
    results = []

    def progress(**fields):
        atomic_json(output/'status.json', dict(status='running', phase='neural_training', observed_utc=utc(),
            pid=pid, process_start=token, elapsed_seconds=time.monotonic()-began,
            finished_seeds=[r['seed'] for r in results], max_seconds=budget.max_seconds, **fields))

    def alarm(*args):
        raise TimeoutError('Fixed neural gate wall-clock budget exhausted; no retry')

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget.max_seconds)
    progress(seed=None, iteration=0, planned_iterations=budget.iterations)
    try:
        for seed in budget.seeds:
            result = train_seed(game_name, seed, budget, deadline, progress, output/f'seed-{seed}-neural.npz')
            atomic_json(output/f'seed-{seed}.json', result)
            results.append(result)
        if any(digest(path) != sha for path, sha in hashes.items()):
            raise ValueError('Frozen neural gate inputs changed')
        passed = all(r['neural_curve_gate_passed'] for r in results)
        report = dict(format='full-information-neural-gate-v1', status='complete', completed_utc=utc(),
            elapsed_seconds=time.monotonic()-began, seed_results=results,
            all_three_neural_gates_passed=passed, shared_neural_training_executed=True,
            six_player_policy_trained=False, production_six_player_path_validated=False,
            exact_full_tree_sampling=True, fixed_six_player_action_abstraction_validated=False,
            source_sha256=hashes, plan_sha256=digest(output/'plan.json'),
            automatic_retry=False, model_promoted=False,
            next_gate='Richer six-player observation collector and external-sampling neural path; small-game improvement is not six-player strength')
        atomic_json(output/'report.json', report)
        atomic_json(output/'status.json', dict(status='complete', phase='complete', observed_utc=utc(),
            pid=pid, process_start=token, elapsed_seconds=time.monotonic()-began,
            all_three_neural_gates_passed=passed, finished_seeds=list(budget.seeds),
            shared_neural_training_executed=True, six_player_policy_trained=False))
        return report
    except BaseException as error:
        failure = dict(format='full-information-neural-gate-v1', status='failed', observed_utc=utc(),
            pid=pid, process_start=token, elapsed_seconds=time.monotonic()-began,
            error_type=type(error).__name__, error=str(error), finished_seeds=[r['seed'] for r in results],
            all_three_neural_gates_passed=False, automatic_retry=False,
            source_sha256=hashes, six_player_policy_trained=False)
        atomic_json(output/'report.json', failure)
        atomic_json(output/'status.json', failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=32)
    parser.add_argument('--fit-steps', type=int, default=256)
    parser.add_argument('--max-seconds', type=float, default=480.)
    args = parser.parse_args()
    result = run(args.output, NeuralBudget(iterations=args.iterations, fit_steps=args.fit_steps,
                                         max_seconds=args.max_seconds))
    print(json.dumps(dict(status=result['status'], all_three_neural_gates_passed=result['all_three_neural_gates_passed'])))


if __name__ == '__main__':
    main()
