"""Bounded component validation of the current regret sampler/fitting path.

This is NOT a new six-player policy, nor validation of the production51-feature
abstraction. A full-information Kuhn adapter exercises the actual existing
Conditional/ScaledRegretSampler and train_conditional function. Its tabular
function approximator isolates mathematical plumbing from network capacity.
The incompatible production strata and missing production average-policy path
remain explicit blocking gates, even if every component check passes.
"""
import argparse
from dataclasses import dataclass
import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from regret_fitting import ConditionalRegretSampler, train_conditional
from scaled_regret_fitting import ScaledRegretSampler


@dataclass(frozen=True)
class Budget:
    seed: int = 20261113
    iterations: int = 48
    fit_steps: int = 96
    batch_size: int = 128
    max_seconds: float = 30.

    def validate(self):
        for key, maximum in [('iterations', 128), ('fit_steps', 512), ('batch_size', 512)]:
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Invalid bounded component budget: ' + key)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('Invalid deterministic seed')
        if not 0 < self.max_seconds <= 120:
            raise ValueError('Component validation must finish within120seconds')


def regret_match(values, legal=None):
    values = np.asarray(values, dtype=float)
    mask = np.ones_like(values, dtype=bool) if legal is None else np.asarray(legal, dtype=bool)
    if values.ndim != 1 or values.shape != mask.shape or not mask.any() or not np.isfinite(values).all():
        raise ValueError('Invalid finite legal regret vector')
    positive = np.maximum(values, 0.) * mask
    # The production NumPy policy uses the maximum legal score when none is
    # positive. Validation initial strategies are supplied separately as uniform.
    if positive.sum() > 0:
        return positive / positive.sum()
    result = np.zeros_like(values); result[np.argmax(np.where(mask, values, -np.inf))] = 1.
    return result


def reach_weighted_average(strategies, own_reaches, iteration_weights):
    strategies = np.asarray(strategies, dtype=float)
    mass = np.asarray(own_reaches, dtype=float) * np.asarray(iteration_weights, dtype=float)
    if (strategies.ndim != 2 or mass.shape != (len(strategies),) or not np.isfinite(strategies).all()
            or not np.isfinite(mass).all() or np.any(mass < 0) or not mass.sum() > 0
            or np.any(strategies < 0) or not np.allclose(strategies.sum(1), 1.)):
        raise ValueError('Invalid reach-weighted strategy samples')
    return (strategies * mass[:, None]).sum(0) / mass.sum()


def game_and_infosets(game_name='kuhn_poker'):
    import pyspiel
    if game_name not in ('kuhn_poker', 'leduc_poker'):
        raise ValueError('Only declared small perfect-recall poker games are supported')
    game = pyspiel.load_game(game_name)
    found = {}

    def visit(state):
        if state.is_terminal():
            return
        if not state.is_chance_node():
            key = state.information_state_string(); player = state.current_player()
            tensor = np.asarray(state.information_state_tensor(), dtype=np.float32)
            legal = state.legal_actions()
            if key in found:
                assert np.array_equal(found[key]['tensor'], tensor) and found[key]['legal'] == legal
            else:
                found[key] = dict(player=player, tensor=tensor, legal=legal)
        actions = [a for a, _ in state.chance_outcomes()] if state.is_chance_node() else state.legal_actions()
        for action in actions:
            visit(state.child(action))

    visit(game.new_initial_state()); keys = sorted(found)
    return game, keys, found


def exact_iteration(game, strategies, infosets):
    """Full-tree counterfactual regrets; independent of OpenSpiel CFR trainer.

    Chance and opponent reach weight instantaneous regrets. Own reach weights
    strategy accumulation. Chance mass is constant for each Kuhn infoset over
    iterations and therefore cancels in its normalized average.
    """
    regrets = {k: np.zeros(5) for k in infosets}
    average_numerator = {k: np.zeros(5) for k in infosets}
    average_mass = {k: 0. for k in infosets}

    def walk(state, target, reaches, chance):
        if state.is_terminal():
            return state.returns()[target]
        if state.is_chance_node():
            return sum(prob * walk(state.child(action), target, reaches, chance * prob)
                       for action, prob in state.chance_outcomes())
        player = state.current_player(); key = state.information_state_string()
        sigma = strategies[key]; legal = state.legal_actions(); values = np.zeros(5)
        for action in legal:
            updated = reaches.copy(); updated[player] *= sigma[action]
            values[action] = walk(state.child(action), target, updated, chance)
        value = float(np.dot(sigma, values))
        if player == target:
            regrets[key][legal] += chance * reaches[1-player] * (values[legal] - value)
        if target == 0:
            mass = chance * reaches[player]
            average_numerator[key] += mass * sigma; average_mass[key] += mass
        return value

    for player in (0, 1):
        walk(game.new_initial_state(), player, np.ones(2), 1.)
    return regrets, average_numerator, average_mass


class Replay:
    def __init__(self, features, advantages, iterations):
        self.features = features; self.advantages = advantages; self.iterations = iterations

    def __len__(self):
        return len(self.features)


def kuhn_features(keys, infosets):
    """Full information state, losslessly padded; NOT production encoding."""
    rows = []
    for key in keys:
        x = np.zeros(51, np.float32); tensor = infosets[key]['tensor']
        if len(tensor) > 44:
            raise ValueError('Full information state exceeds declared lossless padding')
        x[:len(tensor)] = tensor; x[44] = infosets[key]['player']
        x[46 + np.asarray(infosets[key]['legal'])] = 1.
        rows.append(x)
    result = np.stack(rows)
    if len(np.unique(result, axis=0)) != len(keys):
        raise ValueError('Full-information adapter merged distinct infosets')
    return result


def adapter_strata(x):
    # Production base_strata interprets preflop/street/position coordinates.
    # Kuhn does not have those semantics, so this declared adapter uses player.
    return np.asarray(x[:, 44], dtype=np.int64)


class ExactFeatureTable(torch.nn.Module):
    """Independent parameter row per full infoset; no neural generalization claim."""
    def __init__(self, features):
        super().__init__(); self.register_buffer('features', torch.as_tensor(features))
        self.lookup = {row.tobytes(): i for i, row in enumerate(np.asarray(features, np.float32))}
        self.values = torch.nn.Parameter(torch.zeros((len(features), 5)))

    def forward(self, x, legal):
        try:
            indices = [self.lookup[row.tobytes()] for row in x.detach().cpu().numpy()]
        except KeyError as error:
            raise ValueError('Fitting input is not a unique full infoset') from error
        return self.values[torch.tensor(indices, dtype=torch.int64)] * legal


def exploitability_of(game, strategies):
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability
    result = policy.TabularPolicy(game)
    for key, index in result.state_lookup.items():
        result.action_probability_array[index] = strategies[key][:game.num_distinct_actions()]
    return float(exploitability.exploitability(game, result))


def reference_counterfactual_regret_error(game, expected):
    """Independent first-step oracle only; this solver never trains our candidate."""
    from open_spiel.python.algorithms import cfr
    solver = cfr._CFRSolver(game, alternating_updates=False, linear_averaging=False, regret_matching_plus=False)
    solver.evaluate_and_update_policy()
    return max(abs(expected[key][action] - node.cumulative_regret[action])
               for key, node in solver._info_state_nodes.items() for action in node.legal_actions)


def independent_average_contributions(game, strategies):
    """One own-reach record per infoset; independent of chance-history summation."""
    result = {}

    def visit(state, reach):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action), reach)
            return
        player = state.current_player(); key = state.information_state_string(); own = reach[player]
        if key in result:
            if not np.isclose(result[key], own, rtol=0, atol=1e-12):
                raise ValueError('Own reach differs inside a supposed perfect-recall infoset')
        else:
            result[key] = own
        for action in state.legal_actions():
            updated = reach.copy(); updated[player] *= strategies[key][action]
            visit(state.child(action), updated)

    visit(game.new_initial_state(), np.ones(2))
    return result


def component_checks(game, keys, infosets, features, game_name='kuhn_poker'):
    uniform = {}
    for key in keys:
        row = np.zeros(5); row[infosets[key]['legal']] = 1. / len(infosets[key]['legal']); uniform[key] = row
    regrets, _, _ = exact_iteration(game, uniform, infosets)
    # Independently hand-derived under uniform play: root pass/bet utilities
    # differ by3/4 chip; chance mass for a private card is1/3.
    root_errors = ([np.max(np.abs(regrets[str(card)][:2] - [-.125, .125])) for card in range(3)]
                   if game_name == 'kuhn_poker' else [])
    oracle_error = reference_counterfactual_regret_error(game, regrets)
    unscaled = np.stack([regrets[k] for k in keys])
    # A game-independent synthetic coverage fixture, separate from training:
    # its RMS is37 so positive scaling is exercised even when full-tree
    # counterfactual reach makes Leduc's real instantaneous regrets tiny.
    fixture_multiplier = 37. / max(float(np.sqrt(np.square(unscaled).mean())), 1e-12)
    x = np.concatenate([features, features]); y1 = fixture_multiplier*unscaled; y2 = -2*y1
    y = np.concatenate([y1, y2]).astype(np.float32); w = np.r_[np.ones(len(keys)), np.full(len(keys), 3)].astype(np.float32)
    replay = Replay(x, y, w); conditional = ConditionalRegretSampler([replay], adapter_strata)
    scaled = ScaledRegretSampler([replay], adapter_strata, 'stratum-scaled-feature-conditional')
    errors = []; relative_errors = []; invariant = []
    for row, target in zip(conditional.features, conditional.targets):
        matches = np.all(x == row, axis=1); expected = np.average(y[matches], axis=0, weights=w[matches])
        errors.append(float(np.max(np.abs(target - expected))))
        relative_errors.append(float(np.max(np.abs(target - expected))) / max(1.,float(np.max(np.abs(expected)))))
    for target, group, normalized in zip(scaled.raw_targets, scaled.groups, scaled.targets):
        legal = scaled.features[len(invariant), 46:51]
        invariant.append(float(np.max(np.abs(regret_match(target, legal) - regret_match(normalized, legal)))))
    strategies = np.array([[.9, .1], [.1, .9]]); own = [.2, .8]; weights = [1., 3.]
    expected = np.array([.16153846153846155, .8384615384615385])
    average = reach_weighted_average(strategies, own, weights)
    negative = (strategies * np.asarray(weights)[:, None]).sum(0) / sum(weights)
    checks = dict(hand_derived_counterfactual_root_target_max_error=float(max(root_errors)) if root_errors else None,
        independent_reference_first_iteration_regret_max_error=float(oracle_error),
        actual_conditional_sampler_iteration_weighted_target_max_error=max(errors),
        actual_conditional_sampler_iteration_weighted_target_max_relative_error=max(relative_errors),
        synthetic_scaling_fixture_multiplier=fixture_multiplier,
        actual_scaled_sampler_regret_matching_max_error=max(invariant),
        nontrivial_positive_scaling_exercised=bool(np.any(scaled.scales > 1.)),
        reach_weighted_average_max_error=float(np.max(np.abs(average - expected))),
        missing_own_reach_negative_control_detected=bool(np.max(np.abs(negative - expected)) > .1),
        full_information_adapter_unique_infosets=len(keys))
    checks['passed'] = bool((not root_errors or max(root_errors) < 1e-12) and oracle_error < 1e-10
        and max(relative_errors) < 1e-6 and max(invariant) < 1e-6 and checks['nontrivial_positive_scaling_exercised']
        and checks['reach_weighted_average_max_error'] < 1e-12 and checks['missing_own_reach_negative_control_detected'])
    return checks, uniform


def validate(budget, game_name='kuhn_poker'):
    budget.validate(); began = time.monotonic(); np.random.seed(budget.seed); torch.manual_seed(budget.seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    game, keys, infosets = game_and_infosets(game_name); features = kuhn_features(keys, infosets)
    checks, current = component_checks(game, keys, infosets, features, game_name)
    history_x = []; history_y = []; history_w = []
    average_sum = {k: np.zeros(5) for k in keys}; average_mass = {k: 0. for k in keys}
    reference_sum = {k: np.zeros(5) for k in keys}; reference_mass = {k: 0. for k in keys}
    curve = [dict(iteration=0, current_exploitability=exploitability_of(game, current),
                  average_exploitability=exploitability_of(game, current))]
    last_error = None; last_policy_tv = None
    for iteration in range(1, budget.iterations + 1):
        if time.monotonic() - began > budget.max_seconds:
            raise TimeoutError('Bounded component validation exceeded declared seconds; no retry')
        regrets, numerator, mass = exact_iteration(game, current, infosets)
        own_reaches = independent_average_contributions(game, current)
        for key in keys:
            average_sum[key] += iteration * numerator[key]; average_mass[key] += iteration * mass[key]
            reference_sum[key] += iteration * own_reaches[key] * current[key]
            reference_mass[key] += iteration * own_reaches[key]
        history_x.append(features); history_y.append(np.stack([regrets[k] for k in keys]).astype(np.float32))
        history_w.append(np.full(len(keys), iteration, np.float32))
        sampler = ScaledRegretSampler([Replay(np.concatenate(history_x), np.concatenate(history_y), np.concatenate(history_w))],
                                     adapter_strata, 'stratum-scaled-feature-conditional')
        net = ExactFeatureTable(features)
        train_conditional(net, sampler, torch.optim.SGD(net.parameters(), lr=6.), budget.fit_steps, budget.batch_size)
        with torch.no_grad():
            predicted = net(torch.from_numpy(sampler.features), torch.from_numpy(sampler.features[:, 46:51])).numpy()
            last_error = float(np.max(np.abs(predicted - sampler.targets * sampler.features[:, 46:51])))
            last_policy_tv = max(float(np.abs(regret_match(p, legal)-regret_match(t,legal)).sum()/2)
                for p,t,legal in zip(predicted,sampler.targets,sampler.features[:,46:51]))
            scores = net(torch.from_numpy(features), torch.from_numpy(features[:, 46:51])).numpy()
        current = {k: regret_match(scores[i], features[i, 46:51]) for i, k in enumerate(keys)}
        average = {k: average_sum[k] / average_mass[k] for k in keys}
        if iteration in {1, 4, 12, 24, budget.iterations}:
            curve.append(dict(iteration=iteration, current_exploitability=exploitability_of(game, current),
                              average_exploitability=exploitability_of(game, average)))
    average_error = max(float(np.max(np.abs(average[k] - reference_sum[k]/reference_mass[k]))) for k in keys)
    improved = curve[-1]['average_exploitability'] < curve[0]['average_exploitability'] * .5
    passed=bool(checks['passed'] and improved and average_error < 1e-10 and np.isfinite(last_error) and last_policy_tv < 1e-5)
    return dict(format='strength-training-component-validation-v1', status='complete',
        seed=budget.seed, game=game_name, information_state_tensor_size=game.information_state_tensor_size(),
        full_information_infosets=len(keys), iterations=budget.iterations, fit_steps=budget.fit_steps,
        batch_size=budget.batch_size, max_seconds=budget.max_seconds, elapsed_seconds=time.monotonic()-began,
        component_checks=checks, final_fit_max_absolute_error=last_error, final_fit_max_target_policy_tv=last_policy_tv,learning_curve=curve,
        bounded_small_game_component_curve_gate_passed=bool(improved),
        independent_average_accumulation_max_error=average_error,
        component_gates_passed=passed,
        full_information_tabular_candidate_pipeline_gate_passed=passed,
        full_information_replacement_component_pipeline_measured=True,
        end_to_end_custom_leduc_proven=False, production_six_player_path_validated=False,
        next_six_player_training_authorized=False, six_player_policies_trained=0,
        exercised_existing_components=['ConditionalRegretSampler','ScaledRegretSampler','train_conditional'],
        declared_adaptations=['Full information-state tensor padded to51; no production feature abstraction',
            'Strata are small-game player IDs, not the incompatible production street/position strata',
            'ExactFeatureTable replaces production DuelingAdvantageNet to isolate sampler/fitting mathematics',
            'Full-tree exact counterfactual collector replaces production external-sampling six-player collector',
            'Reach-weighted average is a validation-only path; production currently has no average-policy learner'],
        blocked_gates=['Production51-feature representation loses known public-history/suit information',
            'The actual production collector, fixed5-action abstraction and shared neural generalization are not validated by this test',
            'No faithful full-information Leduc integration of the complete production learner has been established',
            'No independent six-player strength gain or5–10second online-solving safety is established'])


def run(output, budget, game_name='kuhn_poker'):
    budget.validate(); output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Validation output must be a new directory; existing reports are immutable')
    output.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name('regret_fitting.py'), Path(__file__).with_name('scaled_regret_fitting.py')]
    source_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        result = validate(budget, game_name)
        if any(hashlib.sha256(p.read_bytes()).hexdigest() != source_sha256[p.name] for p in sources):
            raise ValueError('Validation source changed while running; result is not attributable')
    except Exception as error:
        result = dict(format='strength-training-component-validation-v1', status='failed', error=str(error),
            component_gates_passed=False, end_to_end_custom_leduc_proven=False,
            production_six_player_path_validated=False, next_six_player_training_authorized=False)
    result['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['source_sha256'] = source_sha256
    (output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def run_leduc_suite(output, budget, seeds=(71,72,73)):
    budget.validate(); output = Path(output).resolve()
    if tuple(seeds) != (71,72,73):
        raise ValueError('The declared Leduc replacement-candidate suite has three fixed seeds71,72,73')
    if output.exists():
        raise FileExistsError('Validation output must be a new directory')
    output.mkdir(parents=True); began=time.monotonic(); results=[]
    for seed in seeds:
        candidate=Budget(seed,budget.iterations,budget.fit_steps,budget.batch_size,budget.max_seconds)
        results.append(run(output/f'seed-{seed}',candidate,'leduc_poker'))
    passed=all(x.get('component_gates_passed') for x in results)
    result=dict(format='strength-training-full-information-leduc-suite-v1',status='complete',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=list(seeds),
        game='leduc_poker',iterations=budget.iterations,fit_steps=budget.fit_steps,batch_size=budget.batch_size,
        elapsed_seconds=time.monotonic()-began,declared_max_seconds_per_seed=budget.max_seconds,
        component_gates_passed=passed,full_information_replacement_component_pipeline_measured=True,
        full_information_tabular_candidate_pipeline_gate_passed=passed,
        all_three_seed_curve_gates_passed=all(x.get('bounded_small_game_component_curve_gate_passed') for x in results),
        end_to_end_custom_leduc_proven=False,production_six_player_path_validated=False,
        six_player_policies_trained=0,next_six_player_training_authorized=False,
        curve_gate='Final exact average-policy exploitability must be below50percent of uniform for each fixed seed',
        candidate_scope='Own full-tree collector + existing sampler/fitting function + tabular function approximator + own reach-weighted averaging. Production external sampler, fixed feature semantics and neural architecture remain unvalidated.',
        seed_results=[dict(seed=x.get('seed',seed),status=x.get('status'),component_gates_passed=x.get('component_gates_passed'),
            initial_exploitability=x.get('learning_curve',[{}])[0].get('average_exploitability'),
            final_exploitability=x.get('learning_curve',[{}])[-1].get('average_exploitability'),
            error=x.get('error')) for seed,x in zip(seeds,results)])
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=48); parser.add_argument('--fit-steps', type=int, default=96)
    parser.add_argument('--batch-size', type=int, default=128); parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--max-seconds', type=float, default=30.)
    parser.add_argument('--leduc-suite', action='store_true', help='Declared three-seed full-information replacement-candidate experiment')
    args = parser.parse_args(); budget=Budget(args.seed,args.iterations,args.fit_steps,args.batch_size,args.max_seconds)
    result = run_leduc_suite(args.output,budget) if args.leduc_suite else run(args.output,budget)
    print(json.dumps({k:result.get(k) for k in ['status','component_gates_passed','bounded_small_game_component_curve_gate_passed',
        'production_six_player_path_validated','end_to_end_custom_leduc_proven','elapsed_seconds']}))
    return 0 if result.get('component_gates_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
