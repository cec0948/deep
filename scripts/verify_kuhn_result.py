"""Recompute reported policies and exact best responses from the final checkpoint.

Also recovers reports whose last JSON write failed after the full checkpoint was
saved. The original report/log is retained; verified-results.json is separate.
"""
import argparse
import json
from pathlib import Path
import sys

from fullhouse_v3_common import ROOT, atomic_json, digest


def kuhn_gate(report, settings):
    final = report['curve'][-1]
    checks = dict(tabular=report['tabular_reference']['exploitability'] <= settings['max_tabular_exploitability_ante'],
                  exploitability=final['neural_average']['exploitability'] <= settings['max_final_exploitability_ante'],
                  improvement=final['neural_average']['exploitability'] <= report['initial']['exploitability']/2,
                  collector=final['collector_max_tv'] <= settings['max_average_collector_tv'],
                  average_fitting=final['network_max_tv'] <= settings['max_average_network_tv'],
                  all_infosets_observed=final['all_infosets_observed'])
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(passed=all(checks.values()), checks=checks)


def main():
    p = argparse.ArgumentParser(); p.add_argument('folder', type=Path); args = p.parse_args()
    raw = args.folder/'results.json'
    report = json.loads(raw.read_text()); cfg = report['config']; settings = cfg['kuhn']
    source = ROOT/cfg['source_run']/'source'; sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from training.network import AverageStrategyNet
    from training.buffer import ReservoirBuffer
    from kuhn_validation import exact_metrics, features, buffer_average
    torch.set_num_threads(1)
    checkpoint = args.folder/'latest.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    if ck['iteration'] != settings['iterations'] or ck['config'] != cfg or ck['method'] != report['method']:
        raise ValueError('Incomplete or incompatible checkpoint')
    for name, expected in report['source_hashes'].items():
        if digest(source/name) != expected: raise ValueError('Source changed: '+name)
    final = report['curve'][-1]
    if final['iteration'] != settings['iterations']: raise ValueError('Final measurement missing')
    net = AverageStrategyNet(); net.load_state_dict(ck['strategy_net']); net.eval()
    x = torch.from_numpy(features())
    predicted = net.get_strategy(x, x[:, 46:51]).numpy()[:, :2]
    exact = ck['exact_numerator']/ck['exact_mass'][:, None]
    state = ck['strategy_buffer']; b = ReservoirBuffer(state['count'], 51, store_sample_weights=True); b.load_state_dict(state)
    empirical, mass = buffer_average(b)
    for name, profile in [('exact', exact), ('replay', empirical), ('neural', predicted)]:
        np.testing.assert_allclose(profile, final[name+'_profile'], atol=1e-7, rtol=1e-7)
        metrics = exact_metrics(profile)
        for key in ('value_player0', 'nash_conv', 'exploitability'):
            if abs(metrics[key]-final[name+'_average'][key]) > 1e-7:
                raise AssertionError('Reported exact evaluation mismatch')
        final[name+'_average'] = metrics
    final['collector_max_tv'] = float(np.max(np.abs(exact-empirical).sum(1)/2))
    final['network_max_tv'] = float(np.max(np.abs(predicted-empirical).sum(1)/2))
    final['all_infosets_observed'] = bool((mass > 0).all())
    report['verification'] = dict(raw_report_sha256=digest(raw), checkpoint_sha256=digest(checkpoint),
                                 verifier_sha256=digest(Path(__file__)), oracle_sha256=digest(ROOT/'scripts/kuhn_validation.py'),
                                 recovered_completion_metadata=not report['complete'],
                                 reason='Final model, replay and exact averages independently reconstructed; original logs retained')
    report['gate'] = kuhn_gate(report, settings)
    report['complete'] = True
    atomic_json(args.folder/'verified-results.json', report)
    print(json.dumps(dict(seed=report['seed'], method=report['method'], gate=report['gate'],
                         neural_exploitability=final['neural_average']['exploitability'])), flush=True)


if __name__ == '__main__': main()
