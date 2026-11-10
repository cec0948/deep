"""Fail-closed decision on whether a validated method may enter six-player work.

No training is launched here. --require-six-player exits 2 until the SAME
method passes every frozen-replay pair and every verified Kuhn seed.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil

from fullhouse_v3_common import ROOT, atomic_json, digest
from validate_frozen_regret import gate as frozen_gate
from verify_kuhn_result import kuhn_gate


def assess(cfg, frozen, kuhn):
    methods = {}
    for method in cfg['candidate_modes']:
        fitting = []
        for replay_seed, fit_seed in cfg['frozen_pairs']:
            report = frozen.get(replay_seed)
            if not report or not report.get('complete'):
                fitting.append(dict(replay_seed=replay_seed, passed=False, reason='Missing or incomplete fitting report'))
                continue
            controls = [r for r in report['refits'] if r['mode'] == 'control' and r['steps'] == cfg['baseline_steps']]
            rows = [r for r in report['refits'] if r['mode'] == method and r['steps'] == cfg['fixed_buffer_gates']['eligible_steps']]
            if len(controls) != 1 or len(rows) != 1 or report['fit_seed'] != fit_seed:
                raise ValueError('Wrong or duplicate fitting arm')
            result = frozen_gate(rows[0], controls[0], cfg['fixed_buffer_gates'])
            fitting.append(dict(replay_seed=replay_seed, **result,
                                root_tv=rows[0]['metrics']['root']['weighted_tv'],
                                aa_fold=rows[0]['metrics']['root']['aa']['policy'][0],
                                global_mse_ratio=rows[0]['metrics']['global_fit']['mse']/controls[0]['metrics']['global_fit']['mse']))
        small = []
        for seed in cfg['kuhn']['seeds']:
            report = kuhn.get((method, seed))
            if not report or not report.get('complete') or not report.get('verification'):
                small.append(dict(seed=seed, passed=False, reason='Missing verified final Kuhn checkpoint'))
                continue
            small.append(dict(seed=seed, **kuhn_gate(report, cfg['kuhn']),
                              exploitability=report['curve'][-1]['neural_average']['exploitability']))
        methods[method] = dict(frozen=fitting, kuhn=small,
                               passed=all(r['passed'] for r in fitting+small))
    eligible = [name for name, row in methods.items() if row['passed']]
    return dict(eligible_methods=eligible, six_player_ready=bool(eligible), methods=methods,
                decision='eligible for a bounded six-player experiment' if eligible else 'blocked: no method passed both gates')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', nargs='?', type=Path)
    p.add_argument('--require-six-player', action='store_true')
    args = p.parse_args()
    folder = args.folder or Path(json.loads((ROOT/'artifacts/learning-validation/current.json').read_text())['folder'])
    cfg = json.loads((folder/'config.json').read_text()); config_hash = digest(folder/'config.json')
    report_dir = ROOT/'reports/learning-validation'/folder.name
    report_dir.mkdir(parents=True, exist_ok=True)
    frozen, kuhn, files = {}, {}, {}
    archived_tools = list((folder/'execution-sources').glob('*.py'))
    def check_tool(name, expected):
        candidates = [ROOT/'scripts'/name]+archived_tools
        if not any(f.exists() and digest(f) == expected for f in candidates):
            raise ValueError('Executed tool version unavailable: '+name)
    def load(path):
        report = json.loads(path.read_text())
        if report['config'] != cfg or report['config_sha256'] != config_hash:
            raise ValueError('Config changed: '+str(path))
        source = ROOT/cfg['source_run']/'source'
        for name, expected in report['source_hashes'].items():
            if digest(source/name) != expected: raise ValueError('Source changed: '+name)
        files[str(path.relative_to(folder))] = digest(path)
        return report
    for replay_seed, _ in cfg['frozen_pairs']:
        path = folder/f'frozen-{replay_seed}/results.json'
        if not path.exists(): continue
        r = load(path)
        if r['replay_seed'] != replay_seed: raise ValueError('Replay seed mismatch')
        checkpoint = ROOT/cfg['source_run']/f'seed-{replay_seed}/latest.pt'
        if digest(checkpoint) != r['checkpoint_sha256']: raise ValueError('Replay changed')
        check_tool(r.get('validation_script','validate_frozen_regret.py'), r['script_sha256'])
        check_tool('regret_fitting.py', r['fitter_sha256'])
        if r.get('scaled_fitter_sha256'): check_tool('scaled_regret_fitting.py', r['scaled_fitter_sha256'])
        for row in r['refits']:
            if digest(ROOT/row['model']) != row['model_sha256']: raise ValueError('Model changed')
        frozen[replay_seed] = r
    for method in cfg['candidate_modes']:
        for seed in cfg['kuhn']['seeds']:
            path = folder/f'kuhn-{method}-{seed}/verified-results.json'
            if not path.exists(): continue
            r = load(path)
            if r['seed'] != seed or r['method'] != method: raise ValueError('Kuhn arm mismatch')
            verification = r['verification']
            if digest(path.parent/'results.json') != verification['raw_report_sha256']:
                raise ValueError('Raw Kuhn report changed after verification')
            if digest(path.parent/'latest.pt') != verification['checkpoint_sha256']:
                raise ValueError('Kuhn checkpoint changed after verification')
            for name, field in [('validate_kuhn_learning.py','script_sha256'), ('kuhn_validation.py','oracle_sha256'), ('regret_fitting.py','fitter_sha256')]:
                check_tool(name, r[field])
            check_tool('verify_kuhn_result.py', verification['verifier_sha256'])
            if r.get('scaled_fitter_sha256'): check_tool('scaled_regret_fitting.py', r['scaled_fitter_sha256'])
            kuhn[(method, seed)] = r
    result = assess(cfg, frozen, kuhn)
    result.update(config_sha256=config_hash, verifier_sha256=digest(Path(__file__)), input_hashes=files)
    atomic_json(folder/'gates.json', result)
    atomic_json(report_dir/'gates.json', result)
    # Small evidence is committed. Model/replay files remain in artifacts.
    evidence = ['config.json', 'target-diagnostics.json', 'tabular-matched-iterations.json']+list(files)
    for relative in files:
        if relative.endswith('/verified-results.json'):
            parent = str(Path(relative).parent)
            evidence += [parent+'/results.json', parent+'.log']
    for relative in evidence:
        src = folder/relative
        if not src.exists(): continue
        dest = report_dir/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        if relative.startswith('frozen-') and relative.endswith('/results.json'):
            # Preserve every root/input diagnostic without adding 220k lines of
            # pretty-printed replay measurements to the code review.
            with gzip.open(str(dest)+'.gz', 'wb') as stream: stream.write(src.read_bytes())
        else:
            shutil.copyfile(src, dest)
    for path in archived_tools:
        dest = report_dir/'execution-sources'/path.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    print(json.dumps(result, indent=2), flush=True)
    if args.require_six_player and not result['six_player_ready']: raise SystemExit(2)


if __name__ == '__main__': main()
