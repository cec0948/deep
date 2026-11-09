"""End-to-end learning gate using real V3 components and exact Kuhn BRs."""
import argparse
import json
from pathlib import Path
import sys
import time

from fullhouse_v3_common import ROOT, atomic_json, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/fullhouse-learning-validation.json')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--method', choices=['budget', 'stratified-conditional'], default='budget')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); settings = cfg['kuhn']
    if args.seed not in settings['seeds']: p.error('Seed not in declared experiment')
    source = ROOT/cfg['source_run']/'source'
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from training import r1_support as r1
    from training.network import DuelingAdvantageNet, AverageStrategyNet
    from training.buffer import ReservoirBuffer
    from training.train import train_network, train_strategy_network
    from training.traverse_fast import NumpyPolicy
    from regret_fitting import ConditionalRegretSampler, train_conditional
    from kuhn_validation import (features, own_reach, game_adapter, collect_regrets,
                                 exact_metrics, equilibrium, buffer_average, tabular_reference, traversal_oracle)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    r1.WEIGHT_HORIZON = settings['iterations']; r1.AVERAGE_SAMPLER = 'importance'
    r1.STRATEGY_BATCH_SIZE = 512
    # Use the real 256-wide network and actual 51-input/5-action interfaces.
    net = DuelingAdvantageNet()
    with torch.no_grad():
        for parameter in net.parameters(): parameter.zero_()
    net.eval()
    x = torch.from_numpy(features()); legal = x[:, 46:51]
    buffers = [ReservoirBuffer(settings['capacity'], 51) for _ in (0, 1)]
    strategy_buffer = ReservoirBuffer(settings['capacity'], 51, store_sample_weights=True)
    numer = np.zeros((12, 2)); mass = np.zeros(12)
    reference = tabular_reference(settings['tabular_reference_iterations'])
    traversal_check = traversal_oracle()
    known = exact_metrics(equilibrium())
    if abs(known['value_player0']+1/18) > 1e-12 or abs(known['exploitability']) > 1e-12:
        raise AssertionError('Independent oracle does not reproduce known Kuhn equilibrium')
    report = dict(complete=False, seed=args.seed, method=args.method, config=cfg,
                  config_sha256=digest(args.config), script_sha256=digest(Path(__file__)),
                  oracle_sha256=digest(ROOT/'scripts/kuhn_validation.py'),
                  fitter_sha256=digest(ROOT/'scripts/regret_fitting.py'),
                  source_hashes={str(f.relative_to(source)):digest(f) for f in sorted(source.rglob('*.py'))},
                  known_equilibrium=known, tabular_reference=reference, traversal_oracle=traversal_check,
                  initial=exact_metrics(np.full((12, 2), .5)), curve=[], max_torch_numpy_policy_error=0.)
    start = time.monotonic()
    atomic_json(args.output/'results.json', report)
    with game_adapter() as new_hand:
        for iteration in range(1, settings['iterations']+1):
            if time.monotonic()-start > cfg['limits']['seconds_per_process']:
                raise TimeoutError('Kuhn experiment time budget')
            policy = NumpyPolicy(net)
            before = np.asarray([policy.strategy(row, row[46:51])[:2] for row in x.numpy()])
            parity = float(np.max(np.abs(before-net.get_strategy(x, legal).numpy()[:, :2])))
            report['max_torch_numpy_policy_error'] = max(report['max_torch_numpy_policy_error'], parity)
            if parity > 1e-4: raise AssertionError('Kuhn Torch/NumPy policy mismatch')
            reach = own_reach(before)
            numer += iteration*reach[:, None]*before; mass += iteration*reach
            traverser = (iteration-1) % 2
            collect_regrets(net, traverser, buffers[traverser], iteration,
                            settings['traversals'], rng, new_hand)
            r1.collect_average(None, rng, settings['traversals'], net, strategy_buffer, iteration)
            net = DuelingAdvantageNet()
            optimizer = torch.optim.Adam(net.parameters(), lr=.001)
            if args.method == 'budget':
                train_network(net, buffers, optimizer, settings['regret_steps'], settings['regret_batch'], torch.device('cpu'))
            else:
                sampler = ConditionalRegretSampler(buffers, r1.strategy_strata)
                train_conditional(net, sampler, optimizer, settings['regret_steps'], settings['regret_batch'])
            if iteration not in settings['milestones']: continue
            exact = numer/mass[:, None]
            empirical, empirical_mass = buffer_average(strategy_buffer)
            avg_net = AverageStrategyNet()
            optimizer = torch.optim.Adam(avg_net.parameters(), lr=.001)
            train_strategy_network(avg_net, strategy_buffer, optimizer, settings['strategy_steps'], 512, torch.device('cpu'))
            predicted = avg_net.get_strategy(x, legal).numpy()[:, :2]
            collector_tv = np.abs(exact-empirical).sum(1)/2
            network_tv = np.abs(predicted-empirical).sum(1)/2
            row = dict(iteration=iteration, seconds=time.monotonic()-start,
                       exact_average=exact_metrics(exact), replay_average=exact_metrics(empirical),
                       neural_average=exact_metrics(predicted), current_regret_policy=exact_metrics(net.get_strategy(x, legal).numpy()[:, :2]),
                       collector_max_tv=float(collector_tv.max()), network_max_tv=float(network_tv.max()),
                       all_infosets_observed=bool((empirical_mass > 0).all()),
                       exact_profile=exact.tolist(), replay_profile=empirical.tolist(), neural_profile=predicted.tolist(),
                       regret_rows=[len(b) for b in buffers], strategy_rows=len(strategy_buffer))
            report['curve'].append(row)
            atomic_json(args.output/'results.json', report)
            r1.atomic_save(dict(iteration=iteration, net=net.state_dict(), strategy_net=avg_net.state_dict(),
                                buffers=[b.state_dict() for b in buffers], strategy_buffer=strategy_buffer.state_dict(),
                                rng=r1.pack_rng(rng), exact_numerator=numer, exact_mass=mass, config=cfg, method=args.method),
                           args.output/'latest.pt')
            print(json.dumps({k:v for k,v in row.items() if 'profile' not in k}), flush=True)
    final = report['curve'][-1]
    checks = dict(tabular=reference['exploitability'] <= settings['max_tabular_exploitability_ante'],
                  exploitability=final['neural_average']['exploitability'] <= settings['max_final_exploitability_ante'],
                  improvement=final['neural_average']['exploitability'] <= report['initial']['exploitability']/2,
                  collector=final['collector_max_tv'] <= settings['max_average_collector_tv'],
                  average_fitting=final['network_max_tv'] <= settings['max_average_network_tv'],
                  all_infosets_observed=final['all_infosets_observed'])
    report['gate'] = dict(passed=all(checks.values()), checks=checks)
    report['complete'] = True; report['elapsed_seconds'] = time.monotonic()-start
    atomic_json(args.output/'results.json', report)
    print(json.dumps(report['gate']), flush=True)


if __name__ == '__main__': main()
