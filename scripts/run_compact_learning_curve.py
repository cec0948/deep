"""A separately authorized, bounded120→180→240 self-play learning study.

This runner never writes the original compact campaign registry or controller.
The fixed research budget is distinct from unchanged candidate adoption gates.
It preserves the parent payload, stops on integrity/numerical/runtime failures,
and never retries or promotes a model.
"""
import argparse
import ast
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from extend_compact_run import extend
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_compact_refit_probe import check_card_range, previous_ranges


STAMP = re.compile(r'\d{8}T\d{6,12}Z')
CONFIG_METADATA = {'iterations', 'stages', 'evaluation_iterations', 'evaluation_deals',
    'evaluation_deal_seed', 'evaluation_all_stages_fields', 'evaluation_baselines',
    'seconds_per_training_worker', 'extension_from'}
PROTECTED = ('artifacts/compact-v4/current.json', 'artifacts/retraining-loop.json')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(cfg, parent_cfg):
    if (cfg.get('seeds') != [342, 343, 344] or cfg['seeds'] != parent_cfg['seeds']
            or parent_cfg['iterations'] != 120 or parent_cfg.get('verification_fixture', False)):
        raise ValueError('The verified production120 parent and all three seeds are required')
    expected = dict(stages=[180, 240], evaluation_iterations=[120, 180, 240],
        evaluation_fields=['public', 'mixed', 'noregrets'], variants=['current', 'mixture'],
        evaluation_deals=20000, evaluation_deal_seed=2026160000,
        snapshot_stages=[180, 240], promotion='none')
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError('The predeclared bounded study/evaluation design changed')
    for key, maximum in (('training_seconds_per_stage', 5400),
                         ('evaluation_seconds_per_cell', 2400), ('max_seconds', 14400)):
        if type(cfg.get(key)) is not int or not 1 <= cfg[key] <= maximum:
            raise ValueError('Invalid bounded execution budget: ' + key)
    if parent_cfg['checkpoint_every'] != 6:
        raise ValueError('The original checkpoint cadence must be preserved')


def validate_audit(path, parent):
    audit = read(path)
    checks, sources = audit.get('checks'), audit.get('source_hashes')
    if (audit.get('complete') is not True or audit.get('no_blocking_implementation_faults') is not True
            or audit.get('parent') != str(Path(parent).resolve())
            or not isinstance(checks, dict) or not checks or not all(v is True for v in checks.values())
            or not isinstance(sources, dict) or not sources):
        raise ValueError('Completed, passing method audit is required before any training')
    for name, sha in sources.items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != sha:
            raise ValueError('Audited source or checkpoint changed: ' + name)
    return audit


def assert_same(left, right, name='payload'):
    """Exact recursive state equality, including types, array shapes and dtypes."""
    import numpy as np
    import torch
    if torch.is_tensor(left):
        same = (torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        same = (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and np.array_equal(left, right))
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('Preserved state keys differ: ' + name)
        for key in left:
            assert_same(left[key], right[key], name + '/' + str(key))
        return
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('Preserved state sequence differs: ' + name)
        for i, (a, b) in enumerate(zip(left, right)):
            assert_same(a, b, name + '/' + str(i))
        return
    else:
        same = type(left) is type(right) and left == right
    if not same:
        raise ValueError('Preserved state differs: ' + name)


def assert_finite(value, name='state'):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite learned state: ' + name)
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError('Nonfinite learned state: ' + name)
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, name + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            assert_finite(item, name + '/' + str(i))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError('Nonfinite learned state: ' + name)


def metadata_only_config(before, after):
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if changed - CONFIG_METADATA:
        raise ValueError('Training hyperparameters changed: ' + ', '.join(sorted(changed-CONFIG_METADATA)))
    return sorted(changed)


def replace_checkpoint_config(parent, campaign, cfg):
    """Rewrite only metadata, then verify every other checkpoint key exactly."""
    import torch
    parent_cfg, prior = read(parent/'config.json'), read(campaign/'config.json')
    metadata_only_config(parent_cfg, cfg)
    metadata_only_config(prior, cfg)
    atomic_json(campaign/'config.json', cfg)
    report = {}
    for seed in cfg['seeds']:
        original = torch.load(parent/f'seed-{seed}/latest.pt', map_location='cpu', weights_only=False)
        path = campaign/f'seed-{seed}/latest.pt'
        copied = torch.load(path, map_location='cpu', weights_only=False)
        if copied.get('config') != prior or original.get('config') != parent_cfg:
            raise ValueError('Fork checkpoint configuration mismatch')
        assert_same({k:v for k,v in original.items() if k != 'config'},
                    {k:v for k,v in copied.items() if k != 'config'})
        assert_finite(copied)
        copied['config'] = cfg
        temporary = path.with_suffix('.study.tmp')
        with temporary.open('wb') as stream:
            torch.save(copied, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        restored = torch.load(path, map_location='cpu', weights_only=False)
        assert_same(copied, restored)
        assert_same({k:v for k,v in original.items() if k != 'config'},
                    {k:v for k,v in restored.items() if k != 'config'})
        report[str(seed)] = dict(parent_checkpoint_sha256=digest(parent/f'seed-{seed}/latest.pt'),
            initial_checkpoint_sha256=digest(path), complete_payload_except_config_identical=True,
            original_iteration=original['iteration'], original_seed=original['seed'])
    return report


class DiagnosticGuard:
    """Read only newly completed JSON lines, rejecting NaNs and missing rounds."""
    def __init__(self, campaign, seeds, initial=120):
        self.state = {}
        for seed in seeds:
            path = campaign/f'seed-{seed}/diagnostics.jsonl'
            self.state[seed] = dict(path=path, offset=path.stat().st_size, partial='', iteration=initial)

    def check(self):
        for seed, row in self.state.items():
            with row['path'].open() as stream:
                stream.seek(row['offset']); data = row['partial'] + stream.read(); row['offset'] = stream.tell()
            lines = data.split('\n'); row['partial'] = lines.pop()
            for line in lines:
                if not line:
                    continue
                record = json.loads(line)
                if record.get('seed') != seed or record.get('iteration') != row['iteration'] + 1:
                    raise ValueError('Training diagnostics lost a round or changed seed')
                assert_finite(record, 'training-diagnostic')
                if not isinstance(record.get('main_loss'), (int, float)):
                    raise ValueError('Missing main training loss')
                row['iteration'] = record['iteration']

    def finished(self, stage):
        self.check()
        if any(row['iteration'] != stage or row['partial'] for row in self.state.values()):
            raise ValueError('Stage diagnostics are incomplete')


def verify_stage(campaign, stage, cfg):
    import numpy as np
    import torch
    manifest = read(campaign/'source-manifest.json'); reports = []
    tree = ast.parse((campaign/'tools/compact_regret.py').read_text())
    maps = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'ARRAY_MAP' for target in node.targets)]
    if len(maps) != 1 or not isinstance(maps[0], dict) or not maps[0]:
        raise ValueError('Frozen export mapping is missing')
    array_map = maps[0]
    for name, sha in manifest['files'].items():
        if digest(campaign/name) != sha:
            raise ValueError('Frozen training source changed')
    for seed in cfg['seeds']:
        path = campaign/f'seed-{seed}'; saved = read(path/'status.json')
        if (saved.get('status') != 'complete' or saved.get('iteration') != stage
                or saved.get('through') != stage):
            raise ValueError('All seed workers must finish the exact stage')
        ck = torch.load(path/'latest.pt', map_location='cpu', weights_only=False)
        if (ck.get('iteration') != stage or ck.get('seed') != seed or ck.get('config') != cfg
                or ck.get('source_manifest') != manifest):
            raise ValueError('Stage checkpoint provenance mismatch')
        assert_finite(ck); assert_finite(saved)
        if saved['quality']['weighted_tv'] > cfg['max_opening_tv']:
            raise ValueError('Stage opening fit quality failed')
        history = ck['snapshots']
        if (len(history) != stage or [r['weight'] for r in history] != list(range(1, stage+1))
                or [r['iteration'] for r in history] != list(range(1, stage+1))):
            raise ValueError('Missing pre-update mixture history')
        for item in history:
            artifact = path/item['file']
            if artifact.parent != path or digest(artifact) != item['sha256']:
                raise ValueError('Policy history changed')
        for iteration in range(cfg['checkpoint_every'], stage, cfg['checkpoint_every']):
            if digest(path/f'current_{iteration:05d}.npz') != digest(path/f'policy_{iteration+1:05d}.npz'):
                raise ValueError('Policy snapshot is not the pre-update policy')
        mixture = read(path/f'mixture_{stage:05d}.json')
        if mixture['models'] != history or mixture['iteration'] != stage:
            raise ValueError('Current mixture and checkpoint differ')
        root = ck['root_replay']; count = cfg['root_repetitions_first'] + (stage-1)*cfg['root_repetitions']
        if (len(root['features']) != 169 or not np.all(root['count'] == count)
                or not np.allclose(root['mass'], stage*(stage+1)/2, atol=1e-10, rtol=0)):
            raise ValueError('Root replay time weighting or coverage changed')
        with np.load(path/f'current_{stage:05d}.npz', allow_pickle=False) as data:
            for key in data.files:
                assert_finite(data[key], 'export/'+key)
            if str(data['training_revision']) != 'r1-cpu-v4' or str(data['regret_fit_method']) != cfg['method']:
                raise ValueError('Exported model method changed')
            for state_key, export_key in array_map.items():
                for prefix, export_prefix in [('main.', ''), ('opening.', 'opening_')]:
                    assert_same(ck['net'][prefix+state_key].numpy(), data[export_prefix+export_key],
                                'export-parity/'+prefix+state_key)
            for key in ('main_scales', 'opening_scale'):
                assert_same(ck['net'][key].numpy(), data[key], 'export-parity/'+key)
                if not np.all(data[key] > 0):
                    raise ValueError('Nonpositive regret scale')
        reports.append(dict(seed=seed, iteration=stage, checkpoint_sha256=digest(path/'latest.pt'),
            current_sha256=digest(path/f'current_{stage:05d}.npz'),
            mixture_sha256=digest(path/f'mixture_{stage:05d}.json'), snapshots=len(history),
            finite_state=True, root_weighting_verified=True, pre_update_history_verified=True,
            current_export_checkpoint_parity=True))
    return dict(complete=True, passed=True, stage=stage, verified_utc=utc(), seeds=reports)


def verify_evaluation_completion(output, cfg, study_plan_sha):
    summary = read(output/'evaluation-summary.json'); plan = read(output/'evaluation-plan.json')
    if (summary.get('complete') is not True or summary.get('completed_evaluations') != 54
            or summary.get('total_hands') != 6480000 or summary.get('raw_hands_verified') is not True
            or summary.get('source_provenance_verified') is not True
            or summary.get('config') != cfg or plan.get('config') != cfg
            or plan.get('study_plan_sha256') != study_plan_sha
            or summary.get('evaluation_plan_sha256') != digest(output/'evaluation-plan.json')):
        raise ValueError('Complete verified evaluation and matching plan evidence are required')
    return summary


def terminate_owned(active, evaluator_grace=30):
    """Signal all owned groups first; let the evaluator reap its cell groups.

    Evaluation cells use separate sessions and their helper has a bounded8s
    cleanup per cell. Killing that helper after5s could strand later cells.
    Popen ownership and the recorded start token prevent acting on reused PIDs.
    """
    signaled = []
    for name, row in active.items():
        child = row['process']
        if child.poll() is None and process_start(child.pid) == row['start']:
            try:
                os.killpg(child.pid, signal.SIGTERM); signaled.append((name, row))
            except ProcessLookupError:
                pass
    for name, row in signaled:
        child = row['process']
        try:
            child.wait(timeout=evaluator_grace if name == 'evaluation' else 5)
        except subprocess.TimeoutExpired:
            if process_start(child.pid) == row['start']:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.wait(timeout=5)


def run(parent, output, config, audit_path):
    parent, output, config, audit_path = [Path(p).resolve() for p in (parent, output, config, audit_path)]
    cfg, parent_cfg = read(config), read(parent/'config.json')
    validate_config(cfg, parent_cfg); audit = validate_audit(audit_path, parent)
    if output.parent != ROOT/'artifacts/continuation-study' or not STAMP.fullmatch(output.name):
        raise ValueError('Study output must be a new isolated timestamp directory')
    if (read(parent/'campaign-status.json').get('status') != 'complete'
            or read(parent/'verification.json').get('passed') is not True):
        raise ValueError('Parent must be a verified completed campaign')
    output.parent.mkdir(parents=True, exist_ok=True)
    study_lock = (output.parent/'runner.lock').open('a+')
    fcntl.flock(study_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ranges = previous_ranges(output); check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], ranges)
    frozen = {}
    def freeze(path, expected=None):
        path = Path(path).resolve(); sha = digest(path)
        if expected is not None and expected != sha:
            raise ValueError('Input changed before study: ' + str(path))
        frozen[str(path)] = sha
        return sha
    protected = {str(ROOT/name):freeze(ROOT/name) for name in PROTECTED}
    for name, sha in audit['source_hashes'].items():
        freeze(ROOT/name, sha)
    for name in ('config.json', 'source-manifest.json', 'launch.json', 'campaign-status.json', 'verification.json'):
        freeze(parent/name)
    for seed in cfg['seeds']:
        for path in (parent/f'seed-{seed}').iterdir():
            if path.is_file() and path.suffix in ('.pt', '.npz', '.json', '.jsonl'):
                freeze(path)
    helper_names = ('run_compact_learning_curve.py', 'evaluate_compact_learning_curve.py',
        'snapshot_learning_curve.py', 'train_compact_selfplay.py', 'extend_compact_run.py',
        'fullhouse_v3_common.py', 'run_compact_refit_probe.py', 'run_poker_foundation_study.py')
    for name in helper_names:
        freeze(ROOT/'scripts'/name)
    freeze(config); freeze(audit_path)
    output.mkdir(exist_ok=False); (output/'campaign').mkdir(); (output/'logs').mkdir()
    atomic_json(output/'config.json', cfg); atomic_json(output/'method-audit.json', audit)
    campaign = output/'campaign'/output.name
    extend(parent, campaign, cfg['stages'][-1], cfg['evaluation_deal_seed'])
    campaign_cfg = dict(read(campaign/'config.json'), stages=cfg['stages'],
        evaluation_iterations=cfg['evaluation_iterations'], evaluation_deals=cfg['evaluation_deals'],
        evaluation_all_stages_fields=True, evaluation_baselines=[],
        seconds_per_training_worker=cfg['training_seconds_per_stage'])
    preserved = replace_checkpoint_config(parent, campaign, campaign_cfg)
    preparation = dict(complete=True, prepared_utc=utc(), parent=str(parent), campaign=str(campaign),
        preserved_payloads=preserved, changed_config_keys=metadata_only_config(parent_cfg, campaign_cfg),
        original_registries_unchanged=True, method_audit_sha256=digest(output/'method-audit.json'))
    atomic_json(output/'preparation.json', preparation)
    for name in ('config.json', 'method-audit.json', 'preparation.json'):
        freeze(output/name)
    for name in ('config.json', 'source-manifest.json', 'extension.json'):
        freeze(campaign/name)
    plan = dict(created_utc=utc(), parent=str(parent), parent_iteration=120, campaign=str(campaign),
        config=cfg, frozen_files=frozen, protected_registry_files=protected,
        prior_card_ranges=ranges, card_range=[cfg['evaluation_deal_seed'], cfg['evaluation_deal_seed']+cfg['evaluation_deals']],
        method_audit_sha256=digest(output/'method-audit.json'), preparation_sha256=digest(output/'preparation.json'),
        planned_evaluations=54, planned_hands=6480000,
        continuation_rule='Preauthorized fixed budget through240; original adoption conditions unchanged',
        no_model_promotion=True, no_automatic_retry=True)
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    began = time.monotonic(); active = {}; stage = 120; phase = 'prepared'; completed_stages = []; backups = {}
    stage_evidence = {}
    guard = DiagnosticGuard(campaign, cfg['seeds'])
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    ready = dict(updated_utc=utc(), completed_stages=[], checkpoints={'120':dict(iteration=120, parent_verified=True)})
    atomic_json(output/'stage-ready.json', ready)
    def check_frozen(all_files=False):
        for name, sha in (frozen if all_files else protected).items():
            if digest(name) != sha:
                raise ValueError('Protected study input changed: ' + name)
        if digest(output/'plan.json') != plan_sha:
            raise ValueError('Predeclared study plan changed')
    def status(state, **extra):
        atomic_json(output/'status.json', dict(status=state, phase=phase, stage=stage,
            updated_utc=utc(), seconds=time.monotonic()-began, supervisor_pid=os.getpid(),
            supervisor_start=process_start(os.getpid()), plan_sha256=plan_sha, parent_iteration=120,
            planned_iterations=240, completed_stages=completed_stages[:], campaign=str(campaign),
            running={name:dict(pid=row['process'].pid, process_start=row['start']) for name,row in active.items()
                     if row['process'].poll() is None},
            checkpoint_iterations={str(seed):read(campaign/f'seed-{seed}/status.json')['iteration'] for seed in cfg['seeds']},
            method_audit_passed=True, backups=backups.copy(), original_registries_unchanged=True, **extra))
    def launch(name, command):
        log = (output/'logs'/(name.replace('/', '-')+'.log')).open('w')
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[name] = dict(process=proc, start=process_start(proc.pid), log=log, started=time.monotonic())
        return proc
    def consume(name):
        row = active.pop(name); row['log'].close()
        if row['process'].returncode:
            raise RuntimeError('Study subprocess failed: ' + name)
    def monitor():
        if time.monotonic()-began > cfg['max_seconds']:
            raise TimeoutError('Fixed learning study time budget exhausted')
        check_frozen(); guard.check()
        for name,row in active.items():
            if row['process'].poll() not in (None, 0):
                raise RuntimeError('Study subprocess failed: ' + name)
        status('running')
    def stop(*unused):
        raise InterruptedError('Learning study interrupted')
    handlers = {sig:signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        check_frozen(True)
        atomic_json(campaign/'launch.json', dict(status='running', study_supervised=True, pids={}, process_starts={}))
        status('running')
        evaluator = launch('evaluation', [str(PYTHON), str(ROOT/'scripts/evaluate_compact_learning_curve.py'),
            '--study', str(output), '--campaign', str(campaign), '--config', str(output/'config.json')])
        for stage in cfg['stages']:
            phase = 'training'; stage_began = time.monotonic()
            workers = {seed:launch(f'train/{seed}', [str(PYTHON), str(ROOT/'scripts/train_compact_selfplay.py'),
                '--run-dir', str(campaign), '--seed', str(seed), '--through', str(stage), '--resume']) for seed in cfg['seeds']}
            atomic_json(campaign/'launch.json', dict(status='running', through=stage, study_supervised=True,
                pids={str(seed):proc.pid for seed,proc in workers.items()},
                process_starts={str(seed):process_start(proc.pid) for seed,proc in workers.items()}))
            while any(proc.poll() is None for proc in workers.values()):
                if time.monotonic()-stage_began > cfg['training_seconds_per_stage'] + 120:
                    raise TimeoutError('Training stage time budget exhausted')
                monitor(); time.sleep(3)
            for seed in cfg['seeds']:
                consume(f'train/{seed}')
            guard.finished(stage); phase = 'verifying_stage'; status('running'); check_frozen(True)
            verification = verify_stage(campaign, stage, campaign_cfg)
            atomic_json(output/f'stage-{stage}-verification.json', verification)
            stage_evidence[str(output/f'stage-{stage}-verification.json')] = digest(output/f'stage-{stage}-verification.json')
            completed_stages.append(stage); ready['completed_stages'] = completed_stages[:]
            ready['checkpoints'][str(stage)] = dict(iteration=stage, verification_sha256=digest(output/f'stage-{stage}-verification.json'))
            ready['updated_utc'] = utc(); atomic_json(output/'stage-ready.json', ready)
            phase = 'snapshotting'; status('running')
            snapshot = launch('snapshot', [str(PYTHON), str(ROOT/'scripts/snapshot_learning_curve.py'),
                '--study', str(output), '--campaign', str(campaign), '--expected-stage', str(stage)])
            while snapshot.poll() is None:
                monitor(); time.sleep(3)
            consume('snapshot'); backup = read(output/f'stage-{stage}-backup.json')
            if (backup.get('iterations') != {str(s):stage for s in cfg['seeds']}
                    or type(backup.get('verified_files')) is not int or backup['verified_files'] <= 0
                    or Path(backup['archive']).stat().st_size != backup['bytes']
                    or digest(backup['archive']) != backup['sha256']):
                raise ValueError('Stage snapshot verification failed')
            backups[str(stage)] = backup; status('running')
            stage_evidence[str(output/f'stage-{stage}-backup.json')] = digest(output/f'stage-{stage}-backup.json')
        atomic_json(campaign/'launch.json', dict(status='complete', study_supervised=True,
            through=240, exits={str(seed):0 for seed in cfg['seeds']}))
        phase = 'evaluating'; status('running')
        while evaluator.poll() is None:
            monitor(); time.sleep(3)
        consume('evaluation'); phase = 'verifying'; status('running'); check_frozen(True)
        evaluation = verify_evaluation_completion(output, cfg, plan_sha)
        for name, sha in stage_evidence.items():
            if digest(name) != sha:
                raise ValueError('Completed stage verification or backup record changed')
        for backup in backups.values():
            if Path(backup['archive']).stat().st_size != backup['bytes'] or digest(backup['archive']) != backup['sha256']:
                raise ValueError('Completed stage archive changed')
        summary = dict(complete=True, completed_utc=utc(), seconds=time.monotonic()-began,
            plan_sha256=plan_sha, config=cfg, parent=str(parent), campaign=str(campaign),
            completed_stages=completed_stages, selfplay_iteration=240, parent_iteration=120,
            additional_selfplay_iterations=120, original_registries_unchanged=True,
            method_audit_passed=True, raw_hands_verified=True, source_provenance_verified=True,
            completed_evaluations=54, total_hands=6480000, evaluation_summary_sha256=digest(output/'evaluation-summary.json'),
            stage_evidence_sha256=stage_evidence, backups=backups, no_model_promotion=True, no_automatic_retry=True)
        atomic_json(output/'summary.json', summary); phase = 'complete'; status('complete')
        print(json.dumps(dict(complete=True, output=str(output), selfplay_iteration=240)), flush=True)
        return summary
    except BaseException as error:
        failed_phase = phase; phase = 'interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'failed'
        status(phase, error=str(error), failure_stage=failed_phase)
        atomic_json(campaign/'launch.json', dict(status=phase, study_supervised=True, error=str(error)))
        raise
    finally:
        terminate_owned(active)
        for row in active.values():
            row['log'].close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        study_lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args(); run(args.parent, args.output, args.config, args.audit)
