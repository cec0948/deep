"""Verify and snapshot a stopped stage of the isolated self-play study.

This produces a local resumable training-state archive, never a Drive upload.
It does not resume the old controller or claim orchestration/evaluation resume.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile

from fullhouse_v3_common import ROOT, atomic_json, digest


STAMP = re.compile(r'\d{8}T\d{6,12}Z')
STUDY_FILES = ('config.json', 'plan.json', 'method-audit.json', 'preparation.json',
               'stage-ready.json')


def validate_checkpoint(ck, cfg, manifest, seed, expected_stage):
    """Check actual saved tensors, replay and RNG without changing global RNG."""
    import numpy as np
    import torch
    if (ck.get('seed') != seed or ck.get('iteration') != expected_stage
            or ck.get('config') != cfg or ck.get('source_manifest') != manifest):
        raise ValueError('Checkpoint stage/config/source mismatch')
    net = ck.get('net', {})
    if not net or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in net.values()):
        raise ValueError('Invalid or nonfinite model state')
    buffers = ck.get('buffers', [])
    if len(buffers) != 6:
        raise ValueError('Six seat replay buffers are required')
    def array(value, shape, label):
        value = np.asarray(value)
        if value.shape != shape or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError('Invalid replay array: ' + label)
        return value
    def legal(features):
        mask = features[:, 46:51]
        if not np.isin(mask, [0, 1]).all() or np.any(mask.sum(1) < 1):
            raise ValueError('Invalid replay legal-action mask')
    for b in buffers:
        n = b.get('count')
        if (type(n) is not int or not 0 < n <= cfg['buffer_capacity']
                or type(b.get('total_seen')) is not int or b['total_seen'] < n):
            raise ValueError('Invalid reservoir replay counters')
        x = array(b['features'], (n, 51), 'features'); legal(x)
        y = array(b['advantages'], (n, 5), 'advantages')
        t = array(b['iterations'], (n,), 'iterations')
        if np.any(t < 1) or np.any(t > expected_stage) or np.any(t != np.floor(t)):
            raise ValueError('Invalid raw replay iteration weights')
        if np.any(y[x[:, 46:51] == 0] != 0):
            raise ValueError('Illegal actions have nonzero replay regret')
        if 'sample_weights' in b and np.any(array(b['sample_weights'], (n,), 'sample_weights') <= 0):
            raise ValueError('Invalid replay sample weights')
    replay = ck.get('root_replay', {}); n = len(replay.get('features', []))
    if not n:
        raise ValueError('Root replay is empty')
    x = array(replay['features'], (n, 51), 'root features'); legal(x)
    if len(np.unique(x, axis=0)) != n:
        raise ValueError('Duplicate root replay features')
    for name in ('numerator', 'second_numerator'):
        array(replay[name], (n, 5), name)
    for name in ('mass', 'squared_mass', 'count'):
        if np.any(array(replay[name], (n,), name) <= 0):
            raise ValueError('Nonpositive root replay mass/count')
    if np.any(replay['count'] != np.floor(replay['count'])) or np.any(replay['second_numerator'] < 0):
        raise ValueError('Invalid root replay second moment/count')
    rng = ck.get('rng', {})
    if set(rng) != {'python', 'numpy', 'generator', 'torch'}:
        raise ValueError('Incomplete saved RNG state')
    try:
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        generator = np.random.default_rng(); generator.bit_generator.state = rng['generator']
        torch.Generator(device='cpu').set_state(rng['torch'])
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        raise ValueError('Invalid saved RNG state') from error
    snapshots = ck.get('snapshots', [])
    if (len(snapshots) != expected_stage
            or [s.get('weight') for s in snapshots] != list(range(1, expected_stage + 1))
            or [s.get('iteration') for s in snapshots] != list(range(1, expected_stage + 1))):
        raise ValueError('Incomplete pre-update policy history')
    for number, item in enumerate(snapshots, 1):
        if item.get('file') != f'policy_{number:05d}.npz' or not re.fullmatch('[0-9a-f]{64}', item.get('sha256', '')):
            raise ValueError('Unsafe or malformed policy history entry')


def verify_archive(path, manifest):
    """Reject missing, duplicate, extra, non-file, hash or byte-count mismatch."""
    checked = set(); manifest_seen = False
    with tarfile.open(path, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                raise ValueError('Unsafe archive member')
            if member.name == 'checkpoint-backup-manifest.json':
                if manifest_seen or json.load(tar.extractfile(member)) != manifest:
                    raise ValueError('Restore manifest mismatch')
                manifest_seen = True
                continue
            expected = manifest['files'].get(member.name)
            if expected is None or member.name in checked or member.size != expected['bytes']:
                raise ValueError('Unexpected, duplicate or wrong-sized archive member')
            stream = tar.extractfile(member); sha = hashlib.sha256(); size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block); size += len(block)
            if sha.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError('Archive content mismatch')
            checked.add(member.name)
    if not manifest_seen or checked != set(manifest['files']):
        raise ValueError('Incomplete checkpoint archive')
    return len(checked)


def validate_current_export(path, cfg):
    """Numeric tensors must be finite; declared Unicode metadata is not numeric."""
    import numpy as np
    expected_strings = {'training_revision': 'r1-cpu-v4', 'regret_fit_method': cfg['method']}
    with np.load(path, allow_pickle=False) as saved:
        if not saved.files:
            raise ValueError('Empty current policy export')
        for key in saved.files:
            value = saved[key]
            if key in expected_strings:
                if value.shape != () or value.item() != expected_strings[key]:
                    raise ValueError('Current policy metadata mismatch: ' + key)
            elif value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
                raise ValueError('Invalid current policy tensor: ' + key)
        if not set(expected_strings).issubset(saved.files):
            raise ValueError('Current policy metadata missing')
        for key in ('main_scales', 'opening_scale'):
            if key not in saved.files or np.any(saved[key] <= 0):
                raise ValueError('Current policy scales must be positive')


def snapshot(study, campaign, expected_stage, root=ROOT):
    import torch
    root, study, campaign = Path(root).resolve(), Path(study).resolve(), Path(campaign).resolve()
    if (study.parent != root/'artifacts/continuation-study' or not STAMP.fullmatch(study.name)
            or campaign != study/'campaign'/study.name or expected_stage not in (180, 240)):
        raise ValueError('Expected an isolated study and declared completed stage')
    cfg = json.loads((campaign/'config.json').read_text())
    source_manifest = json.loads((campaign/'source-manifest.json').read_text())
    plan = json.loads((study/'plan.json').read_text())
    study_status = json.loads((study/'status.json').read_text())
    if study_status.get('plan_sha256') != digest(study/'plan.json'):
        raise ValueError('Study status does not identify the current frozen plan')
    if cfg.get('seeds') != [342, 343, 344] or cfg.get('iterations') != 240:
        raise ValueError('Expected all three seeds and bounded240 campaign')
    if source_manifest['driver_sha256'] != digest(root/'scripts/train_compact_selfplay.py'):
        raise ValueError('Frozen training driver changed')
    def check_frozen():
        for name, sha in plan['frozen_files'].items():
            if digest(name) != sha:
                raise ValueError('Frozen study input changed: ' + name)
        for name, sha in plan.get('protected_registry_files', {}).items():
            if digest(name) != sha:
                raise ValueError('Original campaign registry changed')
    check_frozen()
    destination = root/'artifacts/exports'; destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = destination/f'compact-v4-{campaign.name}-snapshot-{stamp}.tar.gz'
    partial = archive.with_suffix('.partial'); locks = []; iterations = {}
    if archive.exists() or partial.exists():
        raise FileExistsError('Existing checkpoint export is immutable')
    try:
        for seed in cfg['seeds']:
            folder = campaign/f'seed-{seed}'
            lock = (folder/'worker.lock').open('r'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = json.loads((folder/'status.json').read_text())
            if status.get('status') != 'complete' or status.get('iteration') != expected_stage:
                raise ValueError('All workers must have completed the same stage')
        with tempfile.TemporaryDirectory(dir=destination, prefix='.curve-snapshot-') as temporary:
            stage = Path(temporary); restored = stage/campaign.relative_to(root)
            restored_study = stage/study.relative_to(root); restored.mkdir(parents=True)
            for name, sha in source_manifest['files'].items():
                source = (campaign/name).resolve()
                if not source.is_relative_to(campaign) or digest(source) != sha:
                    raise ValueError('Frozen campaign source changed or escapes root')
                target = restored/name; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if digest(target) != sha:
                    raise ValueError('Copied frozen source differs from its manifest')
            for name in ('config.json', 'source-manifest.json', 'extension.json', 'supervisor-manifest.json'):
                if (campaign/name).is_file():
                    shutil.copyfile(campaign/name, restored/name)
            for name in STUDY_FILES + (f'stage-{expected_stage}-verification.json',):
                source = study/name
                if not source.is_file():
                    raise ValueError('Missing study restore provenance: ' + name)
                shutil.copyfile(source, restored_study/name)
            provenance = restored_study/'snapshot-provenance'; provenance.mkdir()
            for name in ('status.json', 'evaluation-status.json'):
                if (study/name).exists():
                    shutil.copyfile(study/name, provenance/name)
            for seed in cfg['seeds']:
                source = campaign/f'seed-{seed}'; target = restored/f'seed-{seed}'; target.mkdir()
                with (source/'latest.pt').open('rb') as reader, (target/'latest.pt').open('wb') as writer:
                    shutil.copyfileobj(reader, writer, 1024 * 1024)
                ck = torch.load(target/'latest.pt', map_location='cpu', weights_only=False)
                validate_checkpoint(ck, cfg, source_manifest, seed, expected_stage)
                if digest(source/'latest.pt') != digest(target/'latest.pt'):
                    raise ValueError('Checkpoint changed despite worker locks')
                iterations[str(seed)] = ck['iteration']
                for item in ck['snapshots']:
                    path = source/item['file']
                    if digest(path) != item['sha256']:
                        raise ValueError('Policy history hash changed')
                    shutil.copyfile(path, target/path.name)
                    if digest(target/path.name) != item['sha256']:
                        raise ValueError('Copied policy history hash differs')
                for pattern in ('current_*.npz', 'mixture_*.json'):
                    for path in source.glob(pattern):
                        if int(path.stem.rsplit('_', 1)[1]) <= expected_stage:
                            shutil.copyfile(path, target/path.name)
                current = target/f'current_{expected_stage:05d}.npz'
                validate_current_export(current, cfg)
                mixture = json.loads((target/f'mixture_{expected_stage:05d}.json').read_text())
                if (mixture.get('iteration') != expected_stage or mixture.get('models') != ck['snapshots']
                        or mixture.get('model_type') != 'compact_snapshot_mixture'):
                    raise ValueError('Current mixture does not match saved history')
                rows = [json.loads(line) for line in (source/'diagnostics.jsonl').read_text().splitlines() if line]
                if ([row.get('iteration') for row in rows] != list(range(1, expected_stage + 1))
                        or any(row.get('seed') != seed for row in rows)):
                    raise ValueError('Training diagnostics history mismatch')
                (target/'diagnostics.jsonl').write_text('\n'.join(json.dumps(row, allow_nan=False) for row in rows)+'\n')
                atomic_json(target/'status.json', dict(iteration=expected_stage, through=expected_stage,
                    status='complete', restored_from_snapshot=True, process_running=False))
                (target/'worker.lock').touch(); del ck
            gate_source = (root/cfg['gate_run']).resolve()
            if not gate_source.is_relative_to(root):
                raise ValueError('Gate evidence escapes repository')
            gate = stage/gate_source.relative_to(root); gate.mkdir(parents=True, exist_ok=True)
            for name in ('gates.json', 'deployment-verification.json'):
                shutil.copyfile(gate_source/name, gate/name)
            for directory in ('scripts', 'configs', 'native', 'patches'):
                if (root/directory).exists():
                    shutil.copytree(root/directory, stage/directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbi', '*.nbc'))
            study_doc = root/'docs/retraining-learning-curve-20261112.md'
            if not study_doc.is_file():
                raise ValueError('Study preregistration document is required')
            for name in ('THIRD_PARTY_NOTICES.md', 'README.md', 'docs/retraining-learning-curve-20261112.md'):
                if (root/name).is_file():
                    target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(root/name, target)
            stopped = dict(status='interrupted', phase='restored_snapshot', process_running=False,
                supervisor_pid=None, supervisor_start=None, expected_stage=expected_stage,
                reason='Restored checkpoint state; no training, evaluator or controller process is running')
            atomic_json(restored_study/'status.json', stopped)
            atomic_json(restored/'campaign-status.json', stopped)
            atomic_json(restored/'launch.json', dict(status='interrupted', snapshot_iterations=iterations))
            # No registry is restored: the original compact registry/controller must remain untouched.
            commands = [] if expected_stage == 240 else [
                ['python', 'scripts/train_compact_selfplay.py', '--run-dir', str(campaign.relative_to(root)),
                 '--seed', str(seed), '--resume', '--through', '240'] for seed in cfg['seeds']]
            note = dict(format='isolated-learning-curve-checkpoint-v1', run=str(campaign.relative_to(root)),
                study=str(study.relative_to(root)), iterations=iterations, resumable_selfplay=True,
                orchestration_resume_supported=False, evaluation_resume_supported=False,
                restore='Extract at a repository root and install pinned dependencies. Verify all manifest hashes. Confirm no original or restored workers are alive before deliberate recovery; each worker command acquires its own saved worker lock. At180, the listed frozen worker commands can resume training state through240; coordinate their supervision separately. At240 the authorized training budget is complete. Do not run the original run_compact_retraining.py or continue_compact_learning.py on this study.',
                resume_worker_commands=commands,
                evaluation_restore='Evaluation raw logs and external opponent assets are not in this checkpoint archive. Restore the separately verified evaluation assets and re-establish an explicit evaluation plan; automatic evaluator/orchestration resume is unsupported.',
                limitations=['This is a completed-stage training-state backup, not proof of policy quality',
                    'No process liveness or original registry/controller state is restored',
                    'A training stage may finish before its large evaluation completes'])
            files = sorted(path for path in stage.rglob('*') if path.is_file())
            note['files'] = {str(path.relative_to(stage)):dict(bytes=path.stat().st_size, sha256=digest(path)) for path in files}
            atomic_json(stage/'checkpoint-backup-manifest.json', note)
            check_frozen()
            with tarfile.open(partial, 'w:gz', compresslevel=3) as tar:
                for path in files + [stage/'checkpoint-backup-manifest.json']:
                    tar.add(path, arcname=str(path.relative_to(stage)), recursive=False)
            checked = verify_archive(partial, note)
            partial.replace(archive)
        result = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=digest(archive),
            iterations=iterations, verified_files=checked, external_backup=False,
            study=study.name, run=campaign.name, expected_stage=expected_stage,
            resumable_selfplay=True, orchestration_resume_supported=False, evaluation_resume_supported=False)
        atomic_json(archive.with_suffix('.json'), result)
        atomic_json(campaign/'latest-backup.json', result)
        atomic_json(study/f'stage-{expected_stage}-backup.json', result)
        return result
    finally:
        for lock in locks:
            lock.close()
        if partial.exists():
            partial.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--expected-stage', type=int, choices=(180, 240), required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.study, args.campaign, args.expected_stage)))
