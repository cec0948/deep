"""Fit normalized regret candidates against unchanged replay/strength gates."""
import argparse
import json
from pathlib import Path
import sys
import time

from fullhouse_v3_common import ROOT, atomic_json, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/fullhouse-scaled-validation.json')
    p.add_argument('--replay-seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    pairs = dict(cfg['frozen_pairs'])
    if args.replay_seed not in pairs: p.error('Undeclared replay seed')
    source = ROOT/cfg['source_run']/'source'; sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import validate_frozen_regret as audit
    from training.network import DuelingAdvantageNet
    from training.buffer import ReservoirBuffer
    from training.train import _export_advantage_npz
    from training import r1_support as r1
    from regret_fitting import train_conditional
    from scaled_regret_fitting import ScaledRegretSampler, BBPredictions
    audit.np = np; audit.torch = torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/cfg['source_run']/f'seed-{args.replay_seed}/latest.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    if ck['iteration'] != 120 or ck['r1_config']['seed'] != args.replay_seed: raise ValueError('Wrong replay')
    previous = ROOT/cfg['control_run']/f'frozen-{args.replay_seed}/results.json'
    old = json.loads(previous.read_text())
    if old['checkpoint_sha256'] != digest(checkpoint) or old['fit_seed'] != pairs[args.replay_seed]:
        raise ValueError('Reused control does not match')
    controls = [r for r in old['refits'] if r['mode'] == 'control' and r['steps'] == cfg['baseline_steps']]
    if len(controls) != 1: raise ValueError('Missing control')
    control = controls[0]
    if digest(ROOT/control['model']) != control['model_sha256']: raise ValueError('Control changed')
    buffers = []
    for state in ck['buffers']:
        b = ReservoirBuffer(state['count'], 51); b.load_state_dict(state); buffers.append(b)
    root = audit.root_oracle(buffers)
    report = dict(complete=False, replay_seed=args.replay_seed, fit_seed=pairs[args.replay_seed],
                  config=cfg, config_sha256=digest(args.config), checkpoint_sha256=digest(checkpoint),
                  control_report_sha256=digest(previous),
                  validation_script='validate_scaled_regret.py', script_sha256=digest(Path(__file__)),
                  fitter_sha256=digest(ROOT/'scripts/regret_fitting.py'),
                  scaled_fitter_sha256=digest(ROOT/'scripts/scaled_regret_fitting.py'),
                  source_hashes=old['source_hashes'], original=old['original'], refits=[control])
    del ck, old
    atomic_json(args.output/'results.json', report)
    started = time.monotonic()
    for method in cfg['candidate_modes']:
        np.random.seed(pairs[args.replay_seed]); torch.manual_seed(pairs[args.replay_seed])
        net = DuelingAdvantageNet(); optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        sampler = ScaledRegretSampler(buffers, r1.strategy_strata, method)
        previous_steps = 0; start = time.monotonic()
        for steps in cfg['candidate_steps']:
            if time.monotonic()-started > cfg['limits']['seconds_per_process']: raise TimeoutError('Fit time budget')
            loss = train_conditional(net, sampler, optimizer, steps-previous_steps, cfg['candidate_batch'])
            metrics = audit.measure(BBPredictions(net, sampler).eval(), buffers, root)
            model = args.output/f'{method}-{steps}.npz'
            _export_advantage_npz(net, model)
            with np.load(model) as data: arrays = dict(data)
            np.savez(model, **arrays, regret_fit_method=np.asarray(method), regret_scale_by_stratum=sampler.scales)
            row = dict(mode=method, steps=steps, batch=cfg['candidate_batch'], seconds=time.monotonic()-start,
                       loss=loss, metrics=metrics, sampling=sampler.diagnostics(), model=str(model.relative_to(ROOT)), model_sha256=digest(model))
            row['gate'] = audit.gate(row, control, cfg['fixed_buffer_gates'])
            report['refits'].append(row); atomic_json(args.output/'results.json', report)
            print(json.dumps(dict(mode=method,steps=steps,seconds=row['seconds'],gate=row['gate'],root=metrics['root'],global_fit=metrics['global_fit'])),flush=True)
            previous_steps = steps
    report['complete'] = True; report['elapsed_seconds'] = time.monotonic()-started
    atomic_json(args.output/'results.json', report)


if __name__ == '__main__': main()
