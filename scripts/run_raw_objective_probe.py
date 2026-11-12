"""Gate an objective correction, then fit and evaluate one bounded candidate.

The exact Leduc objective/gradient gate is mandatory and precedes all poker
fitting. Original four performance conditions apply only to raw-objective
minus current120 on the original opponent fields. All other comparisons are
diagnostic, and nothing in this runner promotes or retries a model.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import (
    check_card_range, interval, previous_ranges, summarize, utc, verify_result,
)
from run_poker_foundation_study import terminate_children, verify_reference


VARIANTS = ['current', 'refit', 'raw-objective']
FIELDS = ['public', 'mixed', 'noregrets']
REFERENCES = ['calling-station', 'tight-value', 'pressure']
KIND = 'derived-offline-raw-objective-refit'
GATE_CHECKS = {
    'all_exact_gradient_identities', 'direct_sampling_equivalence',
    'conditional_loss_identity', 'missing_scale_squared_negative_control_detected',
    'importance_sampling_monte_carlo', 'fixed_weights_have_full_support',
    'compression_actually_exercised',
}


def validate_config(cfg, parent_cfg):
    if cfg['seeds'] != [342, 343, 344] or cfg['seeds'] != parent_cfg['seeds']:
        raise ValueError('All three declared parent seeds are required')
    if (cfg['variants'] != VARIANTS or cfg['evaluation_fields'] != FIELDS
            or cfg['reference_opponents'] != REFERENCES):
        raise ValueError('All predeclared policies and six opponent fields are required')
    if parent_cfg['iterations'] != 120:
        raise ValueError('The verified120 parent is required')
    if cfg['extra_steps'] != 3072 or cfg['batch_size'] != 512 or cfg['learning_rate'] != .001:
        raise ValueError('The extra fitting budget must remain fixed')
    if any(cfg[key] != parent_cfg[key] for key in ('batch_size', 'learning_rate')):
        raise ValueError('Parent batch size and learning rate must remain fixed')
    for key, maximum in (('parallelism', 3), ('gate_seconds', 120), ('fit_seconds', 1200),
                         ('evaluation_seconds', 600), ('max_seconds', 1800)):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError('Invalid bounded execution budget: ' + key)
    if cfg['promotion'] != 'none':
        raise ValueError('The probe cannot promote a model')
    if (cfg.get('method') != 'importance-corrected-balanced'
            or cfg.get('fit_seed_base') != 2026111200
            or cfg['gate_config']['method'] != cfg['method']):
        raise ValueError('The fixed objective method and original input RNG are required')
    if cfg['gate_config']['max_seconds'] > cfg['gate_seconds']:
        raise ValueError('Gate internal time budget exceeds its process bound')
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], {})


def verify_gate(path, cfg):
    """Require actual gate evidence; a completed process alone is insufficient."""
    path = Path(path)
    row = json.loads((path/'results.json').read_text())
    if (row.get('complete') is not True or row.get('config') != cfg
            or row.get('method') != cfg['method']):
        raise ValueError('Incomplete or mismatched sampling-objective gate')
    sources = row.get('source_hashes', {})
    required = {str((ROOT/'scripts'/name).resolve()) for name in
                ('validate_sampling_objective.py', 'raw_objective_fitting.py')}
    if not required.issubset(sources):
        raise ValueError('Gate is missing objective helper provenance')
    for name, sha in sources.items():
        if digest(name) != sha:
            raise ValueError('Gate source changed: ' + name)
    replay = row['replay_file']; artifact = (path/replay['name']).resolve()
    if (artifact.parent != path.resolve() or digest(artifact) != replay['sha256']
            or artifact.stat().st_size != replay['bytes']):
        raise ValueError('Gate replay evidence changed')
    checks = row.get('gate', {}).get('checks', {})
    if set(checks) != GATE_CHECKS or any(type(value) is not bool for value in checks.values()):
        raise ValueError('Gate has no complete boolean checks')
    passed = all(checks.values())
    if row.get('passed') is not passed or row.get('gate', {}).get('passed') is not passed:
        raise ValueError('Gate pass flag disagrees with its checks')
    if not passed:
        raise ValueError('Sampling-objective gate failed; no poker fitting is permitted')
    # Recompute from archived rows rather than trusting the producer's booleans.
    import numpy as np
    from validate_sampling_objective import verify_objective
    with np.load(artifact, allow_pickle=False) as data:
        actual, _ = verify_objective({key:data[key] for key in data.files}, cfg)
    if actual['gate'] != row['gate'] or actual['passed'] is not True:
        raise ValueError('Archived sampling-objective gate does not reproduce')
    if len(actual['gradient_checks']) != len(cfg['probe_seeds']) or len(row['gradient_checks']) != len(cfg['probe_seeds']):
        raise ValueError('Gate did not check every predeclared gradient probe')
    for observed, recomputed in zip(row['gradient_checks'], actual['gradient_checks']):
        for key in ('gradient_relative_error', 'direct_sampling_gradient_relative_error',
                    'missing_scale_squared_relative_gradient_error', 'loss_relative_error'):
            if abs(observed[key]-recomputed[key]) > 1e-12:
                raise ValueError('Gate gradient measurements do not reproduce')
    return row


def verify_fit(path, seed, cfg, parent, checkpoint_sha):
    path = Path(path); row = json.loads((path/'results.json').read_text())
    for name in ('complete', 'opening_unchanged', 'scales_unchanged', 'parent_checkpoint_unchanged',
                 'parent_frozen_files_unchanged', 'teacher_unchanged', 'matched_control_input_schedule_verified'):
        if row.get(name) is not True:
            raise ValueError('Raw-objective fitting preservation check failed: ' + name)
    if (row.get('seed') != seed or row.get('extra_steps') != cfg['extra_steps']
            or row.get('parent') != str(parent) or row.get('parent_iteration') != 120
            or row.get('parent_checkpoint_sha256') != checkpoint_sha
            or row.get('resumable_selfplay') is not False or row.get('artifact_kind') != KIND
            or row.get('new_selfplay_samples') != 0
            or row.get('sampler_method') != cfg['method']
            or row.get('fit_seed') != cfg['fit_seed_base'] + seed
            or row.get('helper_sha256') != digest(ROOT/'scripts/raw_objective_fitting.py')
            or row.get('script_sha256') != digest(ROOT/'scripts/probe_compact_raw_objective.py')
            or row.get('batch_size') != cfg['batch_size'] or row.get('learning_rate') != cfg['learning_rate']):
        raise ValueError('Raw-objective candidate provenance mismatch')
    for name in ('candidate.npz', 'candidate.pt'):
        entry = row['files'][name]; artifact = path/name
        if artifact.stat().st_size != entry['bytes'] or digest(artifact) != entry['sha256']:
            raise ValueError('Raw-objective candidate bytes changed')
    return row


def verify_prior_refit_record(row, seed, cfg, parent, checkpoint_sha):
    """The frozen comparator must be the matched earlier unanchored experiment."""
    for key in ('complete', 'opening_unchanged', 'scales_unchanged', 'parent_checkpoint_unchanged'):
        if row.get(key) is not True:
            raise ValueError('Prior unanchored refit preservation evidence is incomplete: ' + key)
    expected = dict(format='compact-offline-refit-v1', artifact_kind='derived-offline-refit',
        seed=seed, parent=str(parent), parent_iteration=120, parent_checkpoint_sha256=checkpoint_sha,
        extra_steps=cfg['extra_steps'], fit_seed=cfg['fit_seed_base']+seed,
        batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'], new_selfplay_samples=0)
    if row.get('resumable_selfplay') is not False or any(row.get(key) != value for key,value in expected.items()):
        raise ValueError('Prior refit provenance differs from the declared matched comparison')


def summarize_all(rows, cfg):
    """Apply original gates once; keep extra opponents and comparator diagnostic."""
    absolute, paired = {}, {}
    all_fields = cfg['evaluation_fields'] + cfg['reference_opponents']
    for field in all_fields:
        for variant in cfg['variants']:
            per_seed = [rows[f'{field}-seed-{seed}-{variant}']['deal_blocks'] for seed in cfg['seeds']]
            absolute[f'{field}/{variant}'] = dict(
                **interval([statistics.mean(block) for block in zip(*per_seed)]),
                per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], per_seed)])
        for before, after in (('current', 'refit'), ('current', 'raw-objective'), ('refit', 'raw-objective')):
            changes = [[b-a for a,b in zip(rows[f'{field}-seed-{seed}-{before}']['deal_blocks'],
                                           rows[f'{field}-seed-{seed}-{after}']['deal_blocks'])]
                       for seed in cfg['seeds']]
            paired[f'{field}/{after}-minus-{before}'] = dict(
                **interval([statistics.mean(block) for block in zip(*changes)]),
                per_seed=[dict(seed=seed, **interval(values)) for seed, values in zip(cfg['seeds'], changes)])
    gate_cfg = dict(cfg, variants=['current', 'raw-objective'])
    gate_rows = {f'{field}-seed-{seed}-{variant}':rows[f'{field}-seed-{seed}-{variant}']
                 for field in cfg['evaluation_fields'] for seed in cfg['seeds']
                 for variant in gate_cfg['variants']}
    original = summarize(gate_rows, gate_cfg)
    return dict(complete=True, completed_evaluations=len(rows), total_hands=sum(row['hands'] for row in rows.values()),
        absolute_scores=absolute, paired_changes=paired, decisions=original['decisions'],
        performance_gate_scope='raw-objective minus current120; public, mixed and noregrets only',
        diagnostic_comparisons=['Prior unanchored refit on the same fresh cards',
                                'All three transparent heuristic opponent families'],
        limitations=['One predeclared correction candidate; nominal intervals are not adjusted for repeated research',
            'Shared cards form one block across training seeds and hero seats',
            'Leduc objective equivalence does not establish poker strength or six-player convergence',
            'Raw replay may still have inadequate coverage and the51-feature representation remains unchanged',
            'Heuristic opponents are independent behavior families, not validated strong poker agents',
            'No automatic self-play extension, model promotion or retry'])


def run(parent, output, config):
    parent, output, config = Path(parent).resolve(), Path(output).resolve(), Path(config).resolve()
    cfg = json.loads(config.read_text()); parent_cfg = json.loads((parent/'config.json').read_text())
    validate_config(cfg, parent_cfg)
    prior_refit = (ROOT/cfg['prior_refit']).resolve()
    if (output.parent != ROOT/'artifacts/refit-probe' or output == parent or parent in output.parents
            or output == prior_refit or prior_refit in output.parents):
        raise ValueError('Probe output must be a new isolated refit-probe run')
    if (json.loads((parent/'campaign-status.json').read_text())['status'] != 'complete'
            or json.loads((parent/'verification.json').read_text()).get('passed') is not True):
        raise ValueError('Parent checkpoint is not verified complete')
    previous_summary = json.loads((prior_refit/'summary.json').read_text())
    if (json.loads((prior_refit/'status.json').read_text())['status'] != 'complete'
            or previous_summary.get('complete') is not True
            or previous_summary.get('parent') != str(parent)
            or previous_summary.get('raw_hands_verified') is not True
            or previous_summary.get('source_provenance_verified') is not True):
        raise ValueError('Prior unanchored comparator is not a verified completed probe')
    prior_ranges = previous_ranges(output)
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], prior_ranges)
    frozen = {}
    def freeze(path, expected=None):
        path = Path(path).resolve(); sha = digest(path)
        if expected is not None and expected != sha:
            raise ValueError('Frozen input changed before execution: ' + str(path))
        frozen[str(path)] = sha
        return sha
    manifest = json.loads((parent/'source-manifest.json').read_text())
    for name, sha in manifest['files'].items():
        freeze(parent/name, sha)
    for name in ('config.json', 'source-manifest.json', 'verification.json'):
        freeze(parent/name)
    for name in ('config.json', 'plan.json', 'status.json', 'summary.json'):
        freeze(prior_refit/name)
    prior_candidates = {}
    for seed in cfg['seeds']:
        folder = parent/f'seed-{seed}'
        for name in ('latest.pt', 'current_00120.npz'):
            freeze(folder/name)
        prior_result_path = prior_refit/f'seed-{seed}/results.json'
        prior_result = json.loads(prior_result_path.read_text())
        verify_prior_refit_record(prior_result, seed, cfg, parent,
                                 frozen[str(parent/f'seed-{seed}/latest.pt')])
        freeze(prior_result_path)
        candidate = prior_refit/f'seed-{seed}/candidate.npz'
        entry = prior_result['files']['candidate.npz']
        if candidate.stat().st_size != entry['bytes']:
            raise ValueError('Prior comparator size changed')
        freeze(candidate, entry['sha256']); prior_candidates[str(seed)] = str(candidate)
    script_names = ('run_raw_objective_probe.py', 'probe_compact_raw_objective.py', 'raw_objective_fitting.py',
        'validate_sampling_objective.py', 'benchmark_leduc_reference.py', 'evaluate_compact.py',
        'evaluate_poker_reference.py', 'poker_reference_opponents.py', 'run_compact_refit_probe.py',
        'run_poker_foundation_study.py', 'fullhouse_v3_common.py', 'fullhouse_v4_runtime.py',
        'fullhouse_v3_runtime.py', 'fullhouse_runtime.py', 'regret_partition.py')
    for name in script_names:
        freeze(ROOT/'scripts'/name)
    freeze(config); freeze(ROOT/'configs/fullhouse-model.json')
    registry_path = ROOT/'artifacts/selfplay/current.json'
    registry = json.loads(registry_path.read_text()); freeze(registry_path)
    freeze(ROOT/registry['model'], registry['sha256'])
    from fullhouse_runtime import BRIDGE, SOURCE, Fullhouse
    public = Fullhouse(); freeze(public.model, public.digest); freeze(BRIDGE)
    for path in SOURCE.rglob('*.py'):
        freeze(path)
    for path in (SOURCE/'data').glob('*.npz'):
        freeze(path)
    output.mkdir(parents=True, exist_ok=False)
    lock = (output/'probe.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output/'logs').mkdir(); (output/'evaluation').mkdir(); (output/'reference-refit').mkdir()
    atomic_json(output/'config.json', cfg); atomic_json(output/'gate-config.json', cfg['gate_config'])
    freeze(output/'config.json'); freeze(output/'gate-config.json')
    comparator_copies = {}
    for seed in cfg['seeds']:
        destination = output/'reference-refit'/f'seed-{seed}'; destination.mkdir()
        for name in ('candidate.npz', 'results.json'):
            original = prior_refit/f'seed-{seed}'/name
            shutil.copy2(original, destination/name); freeze(destination/name, frozen[str(original)])
        comparator_copies[str(seed)] = dict(original=prior_candidates[str(seed)],
            copied=str(destination/'candidate.npz'), sha256=digest(destination/'candidate.npz'))
    jobs = []
    for variant in cfg['variants']:
        for seed in cfg['seeds']:
            model = (parent/f'seed-{seed}/current_00120.npz' if variant == 'current' else
                     output/'reference-refit'/f'seed-{seed}/candidate.npz' if variant == 'refit' else
                     output/f'seed-{seed}/candidate.npz')
            for field in cfg['evaluation_fields'] + cfg['reference_opponents']:
                reference = field in cfg['reference_opponents']; name = f'{field}-seed-{seed}-{variant}'
                command = [str(PYTHON), str(ROOT/'scripts'/('evaluate_poker_reference.py' if reference else 'evaluate_compact.py')),
                    '--model', str(model), '--source', str(parent/'source'), '--opponent' if reference else '--field', field,
                    '--deals', str(cfg['evaluation_deals']), '--deal-seed', str(cfg['evaluation_deal_seed']),
                    '--timeout', str(cfg['evaluation_seconds']), '--output', str(output/'evaluation'/name)]
                if reference:
                    command += ['--trace-hands', '12']
                jobs.append(dict(id='eval/'+name, name=name, field=field, opponent=field if reference else None,
                    reference=reference, variant=variant, seed=seed, model=str(model),
                    timeout=cfg['evaluation_seconds'], command=command))
    plan = dict(created_utc=utc(), parent=str(parent), parent_iteration=120, prior_refit=str(prior_refit),
        config=cfg, config_source_sha256=digest(config), frozen_files=frozen,
        prior_card_ranges=prior_ranges,
        card_range=[cfg['evaluation_deal_seed'], cfg['evaluation_deal_seed']+cfg['evaluation_deals']],
        comparator_copies=comparator_copies, evaluation_jobs=jobs, artifact_kind=KIND,
        note='Mandatory objective gate, one correction trial, no original campaign or model promotion changes')
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    began = time.monotonic(); active = {}; finished = []; phase = 'objective_gate'; gate_passed = False
    events = (output/'probe.log').open('a', buffering=1)
    def remaining_budget():
        if time.monotonic()-began > cfg['max_seconds']:
            raise TimeoutError('Total raw-objective probe time budget exhausted')
    def status(state, pending=(), **extra):
        atomic_json(output/'status.json', dict(status=state, phase=phase, updated_utc=utc(),
            seconds=time.monotonic()-began, supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
            plan_sha256=plan_sha, gate_passed=gate_passed,
            finished=finished[:], pending=list(pending), running={name:row['process'].pid for name,row in active.items()},
            completed_evaluations=sum(name.startswith('eval/') for name in finished), planned_evaluations=len(jobs),
            artifact_kind=KIND, **extra))
    def terminate(*unused):
        raise InterruptedError('Raw-objective probe interrupted')
    handlers = {sig:signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def execute(work):
        pending = list(work)
        while pending or active:
            remaining_budget()
            while pending and len(active) < cfg['parallelism']:
                job = pending.pop(0); name = job['id']
                if name.startswith(('fit/', 'eval/')) and not gate_passed:
                    raise ValueError('Objective gate must pass before fitting or evaluation')
                log = (output/'logs'/(name.replace('/', '-')+'.log')).open('w')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = dict(process=child, log=log, started=time.monotonic(), timeout=job['timeout'])
                events.write(json.dumps(dict(event='started', job=name, pid=child.pid, utc=utc()))+'\n')
            for name, row in list(active.items()):
                result = row['process'].poll()
                if result is None:
                    if time.monotonic()-row['started'] > row['timeout']:
                        raise TimeoutError('Probe job exceeded budget: ' + name)
                    continue
                row['log'].close()
                if result:
                    raise RuntimeError(f'Probe job failed: {name}, exit {result}')
                del active[name]; finished.append(name)
                events.write(json.dumps(dict(event='finished', job=name, utc=utc()))+'\n')
            status('running', [job['id'] for job in pending])
            if pending or active:
                time.sleep(2)
    def verify_frozen():
        for name, sha in frozen.items():
            if digest(name) != sha:
                raise ValueError('Frozen probe input changed: ' + name)
        if digest(output/'plan.json') != plan_sha:
            raise ValueError('Predeclared raw-objective plan changed')
    try:
        execute([dict(id='gate/objective', timeout=cfg['gate_seconds'], command=[str(PYTHON),
            str(ROOT/'scripts/validate_sampling_objective.py'), '--config', str(output/'gate-config.json'),
            '--output', str(output/'objective-gate')])])
        gate = verify_gate(output/'objective-gate', cfg['gate_config']); gate_sha = digest(output/'objective-gate/results.json')
        verify_frozen(); gate_passed = True
        phase = 'fitting'; status('running')
        execute([dict(id=f'fit/{seed}', timeout=cfg['fit_seconds'], command=[str(PYTHON),
            str(ROOT/'scripts/probe_compact_raw_objective.py'), '--parent', str(parent),
            '--output', str(output/f'seed-{seed}'), '--seed', str(seed), '--extra-steps', str(cfg['extra_steps'])])
            for seed in cfg['seeds']])
        fit_reports = {str(seed):verify_fit(output/f'seed-{seed}', seed, cfg, parent,
            frozen[str(parent/f'seed-{seed}/latest.pt')]) for seed in cfg['seeds']}
        expected = {}
        for job in jobs:
            reference = job['reference']
            row = dict(candidate=digest(job['model']), runtime_sha256=frozen[str(ROOT/'scripts/fullhouse_v4_runtime.py')],
                engine_sha256=frozen[str(BRIDGE)], source_features_sha256=digest(parent/'source/bot/features.py'),
                evaluator_sha256=digest(ROOT/'scripts'/('evaluate_poker_reference.py' if reference else 'evaluate_compact.py')))
            if reference:
                row['opponent_source_sha256'] = frozen[str(ROOT/'scripts/poker_reference_opponents.py')]
            else:
                row['opponents'] = dict(noregrets=registry['sha256'], public=public.digest if job['field'] in ('public','mixed') else None)
            expected[job['name']] = row
        evaluation_plan = dict(jobs=jobs, expected=expected, objective_gate_passed=True, objective_gate_sha256=gate_sha,
            fit_result_sha256={str(seed):digest(output/f'seed-{seed}/results.json') for seed in cfg['seeds']})
        atomic_json(output/'evaluation-plan.json', evaluation_plan); evaluation_plan_sha = digest(output/'evaluation-plan.json')
        verify_frozen(); phase = 'evaluating'; execute(jobs)
        phase = 'verifying'; status('running'); remaining_budget()
        rows = {}
        for job in jobs:
            remaining_budget()
            verifier = verify_reference if job['reference'] else verify_result
            rows[job['name']] = verifier(output/'evaluation'/job['name'], job, cfg, expected[job['name']])
        verify_frozen()
        if (digest(output/'evaluation-plan.json') != evaluation_plan_sha
                or digest(output/'objective-gate/results.json') != gate_sha):
            raise ValueError('Gate or evaluation plan changed during evaluation')
        verify_gate(output/'objective-gate', cfg['gate_config'])
        for seed in cfg['seeds']:
            if digest(output/f'seed-{seed}/results.json') != evaluation_plan['fit_result_sha256'][str(seed)]:
                raise ValueError('Candidate fitting evidence changed')
            verify_fit(output/f'seed-{seed}', seed, cfg, parent, frozen[str(parent/f'seed-{seed}/latest.pt')])
        summary = summarize_all(rows, cfg)
        summary.update(plan_sha256=plan_sha, evaluation_plan_sha256=evaluation_plan_sha,
            config=cfg, parent=str(parent), prior_refit=str(prior_refit), artifact_kind=KIND,
            resumable_selfplay=False, parent_iteration=120, completed_utc=utc(), seconds=time.monotonic()-began,
            raw_hands_verified=True, source_provenance_verified=True,
            objective_gate=dict(passed=True, report_sha256=gate_sha, checks=gate['gate']['checks'],
                scope='Objective/gradient equivalence and sampling checks, not a poker strength test'),
            no_automatic_retry=True, no_model_promotion=True)
        remaining_budget(); atomic_json(output/'summary.json', summary); phase='complete'; status('complete')
        print(json.dumps(dict(output=str(output), complete=True, evaluations=len(rows), decisions=summary['decisions'])), flush=True)
        return summary
    except BaseException as error:
        failed_phase = phase
        phase = 'interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed'
        status(phase, failure_stage=failed_phase, error=str(error)); raise
    finally:
        terminate_children(active)
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        events.close(); lock.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args=parser.parse_args(); run(args.parent, args.output, args.config)
