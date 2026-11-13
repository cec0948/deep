"""Bounded, lossless latest-policy/replay recovery; no historical policy mixture.

This tool never starts training or contacts Drive. Its restore is deliberately
incompatible with legacy --resume: it creates arrays.npz/state.json, not latest.pt.
"""
import argparse
from collections.abc import Mapping
from collections import deque
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import tarfile
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'compact-bounded-replay-recovery-v1'
SEEDS = (342, 343, 344)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_state(state, folder):
    """Encode numerical state without pickle, dtype conversion or row removal."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    arrays = {}

    def encode(x):
        if torch.is_tensor(x) or isinstance(x, (np.ndarray, np.generic)):
            kind = 'torch' if torch.is_tensor(x) else 'scalar' if isinstance(x, np.generic) else 'numpy'
            a = x.detach().cpu().numpy() if kind == 'torch' else np.asarray(x)
            if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
                raise ValueError('Non-numeric or non-finite checkpoint array')
            key = f'a{len(arrays):04d}'; arrays[key] = a.copy()
            return {'t': kind, 'key': key}
        if isinstance(x, Mapping):
            if any(not isinstance(k, str) for k in x):
                raise ValueError('Only string checkpoint keys are supported')
            return {'t': 'dict', 'items': [[k, encode(v)] for k, v in x.items()]}
        if isinstance(x, (tuple, list)):
            return {'t': 'tuple' if isinstance(x, tuple) else 'list', 'items': [encode(v) for v in x]}
        if x is None or type(x) in (bool, int, float, str):
            return {'t': 'value', 'value': x}
        raise TypeError('Unsupported checkpoint type: ' + type(x).__name__)

    metadata = encode(state)
    (folder/'state.json').write_text(json.dumps(metadata, allow_nan=False, separators=(',', ':'))+'\n')
    np.savez_compressed(folder/'arrays.npz', **arrays)


def load_state(folder):
    folder = Path(folder); metadata = json.loads((folder/'state.json').read_text()); used = set()
    with np.load(folder/'arrays.npz', allow_pickle=False) as saved:
        def decode(node):
            t = node['t']
            if t in ('torch', 'numpy', 'scalar'):
                key = node['key']
                if key not in saved.files or key in used:
                    raise ValueError('Missing or repeated array reference')
                used.add(key); a = saved[key].copy()
                if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
                    raise ValueError('Invalid recovery array')
                return torch.from_numpy(a) if t == 'torch' else a[()] if t == 'scalar' else a
            if t == 'dict':
                items = node['items']
                if any(not isinstance(k, str) for k, _ in items) or len({k for k, _ in items}) != len(items):
                    raise ValueError('Invalid recovery dictionary')
                return {k: decode(v) for k, v in items}
            if t in ('tuple', 'list'):
                values = [decode(v) for v in node['items']]
                return tuple(values) if t == 'tuple' else values
            if t == 'value' and (node['value'] is None or type(node['value']) in (bool, int, float, str)):
                return node['value']
            raise ValueError('Invalid state encoding')
        state = decode(metadata)
        if used != set(saved.files):
            raise ValueError('Unreferenced recovery arrays')
    return state


def equal_state(before, after):
    if torch.is_tensor(before):
        return torch.is_tensor(after) and before.dtype == after.dtype and torch.equal(before, after)
    if isinstance(before, np.ndarray):
        return isinstance(after, np.ndarray) and before.dtype == after.dtype and before.shape == after.shape and before.tobytes() == after.tobytes()
    if isinstance(before, np.generic):
        return isinstance(after, np.generic) and before.dtype == after.dtype and before.tobytes() == after.tobytes()
    if isinstance(before, Mapping):
        return isinstance(after, Mapping) and before.keys() == after.keys() and all(equal_state(v, after[k]) for k, v in before.items())
    if isinstance(before, (list, tuple)):
        return type(before) == type(after) and len(before) == len(after) and all(equal_state(x, y) for x, y in zip(before, after))
    return type(before) == type(after) and before == after


def validate_state(state, seed, iteration):
    if state['seed'] != seed or state['iteration'] != iteration or 'snapshots' in state:
        raise ValueError('Recovery seed/iteration/history mismatch')
    capacity = state['config']['buffer_capacity']
    if not 0 < capacity <= 100000 or len(state['buffers']) != 6 or not state['net']:
        raise ValueError('Recovery exceeds fixed replay capacity')
    rows = []
    for b in state['buffers']:
        n = b['count']; rows.append(n)
        if type(n) is not int or not 0 < n <= capacity or b['total_seen'] < n:
            raise ValueError('Invalid replay counters')
        if b['features'].shape != (n, 51) or b['advantages'].shape != (n, 5) or b['iterations'].shape != (n,):
            raise ValueError('Invalid replay shape')
        mask = b['features'][:, 46:51]
        if not np.isin(mask, [0, 1]).all() or np.any(mask.sum(1) < 1) or np.any(b['advantages'][mask == 0] != 0):
            raise ValueError('Invalid replay legality')
        if np.any(b['iterations'] < 1) or np.any(b['iterations'] > iteration) or np.any(b['iterations'] != np.floor(b['iterations'])):
            raise ValueError('Invalid replay iteration weights')
        if 'sample_weights' in b and (b['sample_weights'].shape != (n,) or np.any(b['sample_weights'] <= 0)):
            raise ValueError('Invalid replay sample weights')
    root = state['root_replay']; n = len(root['features'])
    if not 0 < n <= 169:
        raise ValueError('Root sufficient statistics exceed the fixed169 classes')
    shapes = dict(features=(n,51),numerator=(n,5),second_numerator=(n,5),
                  mass=(n,),squared_mass=(n,),count=(n,))
    if any(root[k].shape != shape for k,shape in shapes.items()):
        raise ValueError('Invalid root sufficient statistics')
    if (len(np.unique(root['features'],axis=0)) != n or np.any(root['second_numerator'] < 0)
            or any(np.any(root[k] <= 0) for k in ['mass','squared_mass','count'])
            or np.any(root['count'] != np.floor(root['count']))):
        raise ValueError('Invalid root counters or moments')
    if set(state['rng']) != {'python', 'numpy', 'generator', 'torch'}:
        raise ValueError('Missing RNG state')
    rng = state['rng']
    random.Random().setstate(rng['python']); np.random.RandomState().set_state(rng['numpy'])
    generator = np.random.default_rng(); generator.bit_generator.state = rng['generator']
    torch.Generator(device='cpu').set_state(rng['torch'])
    return rows


def safe_member(name):
    p = Path(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and str(p) == name


def verify_archive(archive, destination=None):
    """Extract to a new directory, then verify every byte and bounded state."""
    archive = Path(archive)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or not safe_member(m.name) for m in members):
            raise ValueError('Unsafe or duplicate recovery archive member')
        if names.count('recovery-manifest.json') != 1:
            raise ValueError('Recovery manifest missing')
        manifest = json.load(tar.extractfile('recovery-manifest.json'))
        if (manifest.get('format') != FORMAT or manifest.get('exact_legacy_resume') is not False
                or manifest.get('historical_mixture_available') is not False
                or set(manifest.get('iterations', {})) != {str(s) for s in SEEDS}
                or set(names) != set(manifest['files']) | {'recovery-manifest.json'}):
            raise ValueError('Recovery manifest coverage/mode mismatch')
        if destination is None:
            with tempfile.TemporaryDirectory(prefix='compact-recovery-verify-') as temporary:
                return _extract_verified(tar, members, manifest, Path(temporary))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        return _extract_verified(tar, members, manifest, destination)


def _extract_verified(tar, members, manifest, destination):
    for m in members:
        target = destination/m.name; target.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(m) as src, target.open('wb') as out:
            shutil.copyfileobj(src, out, 1024*1024)
        if m.name != 'recovery-manifest.json':
            row = manifest['files'][m.name]
            if target.stat().st_size != row['bytes'] or sha256(target) != row['sha256']:
                raise ValueError('Recovery file size/hash mismatch')
    rows = {str(seed): validate_state(load_state(destination/f'seed-{seed}'), seed, manifest['iterations'][str(seed)]) for seed in SEEDS}
    return dict(passed=True, verified_files=len(manifest['files']), replay_rows=rows,
                exact_legacy_resume=False, historical_mixture_available=False)


def create(study, original_archive, original_sha256, output_root=ROOT/'artifacts/exports', expected_iteration=240):
    study = Path(study).resolve(); campaign = study/'campaign'/study.name
    status = json.loads((study/'status.json').read_text())
    if (type(expected_iteration) is not int or expected_iteration <= 0
            or status.get('status') != 'complete' or status.get('stage') != expected_iteration or status.get('running')):
        raise ValueError('Only the explicitly identified completed study stage may be repackaged')
    if sha256(original_archive) != original_sha256:
        raise ValueError('Original full archive hash mismatch')
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output_root/f'compact-v4-{study.name}-snapshot-{stamp}.tar.gz'
    locks = []
    try:
        for seed in SEEDS:
            lock = (campaign/f'seed-{seed}'/'worker.lock').open('r'); locks.append(lock)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='.bounded-recovery-', dir=output_root) as tmp:
            stage = Path(tmp); iterations = {}; parity = {}
            # Only small frozen source/config/provenance files are copied. Each
            # latest model and replay is serialized separately below.
            with tarfile.open(original_archive, 'r|gz') as tar:
                for m in tar:
                    if not m.isfile() or not safe_member(m.name):
                        raise ValueError('Unsafe source archive')
                    n = Path(m.name).name
                    if (n == 'latest.pt' or n == 'checkpoint-backup-manifest.json'
                            or (n.endswith('.npz') and n.startswith(('policy_', 'current_')))
                            or (n.endswith('.json') and n.startswith('mixture_'))):
                        continue
                    target = stage/'frozen'/m.name; target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(m) as src, target.open('wb') as out:
                        if n == 'diagnostics.jsonl':
                            out.write(b''.join(deque(src, maxlen=60)))
                        else:
                            shutil.copyfileobj(src, out)
            for seed in SEEDS:
                src = campaign/f'seed-{seed}'
                before_sha = sha256(src/'latest.pt')
                state = torch.load(src/'latest.pt', map_location='cpu', weights_only=False)
                iteration = state['iteration']; iterations[str(seed)] = iteration
                if iteration != expected_iteration:
                    raise ValueError('Source seed does not match the declared completed stage')
                omitted = state.pop('snapshots', [])
                if not isinstance(omitted, list):
                    raise ValueError('Invalid optional source policy history metadata')
                target = stage/f'seed-{seed}'; save_state(state, target)
                restored = load_state(target); validate_state(restored, seed, iteration)
                if not equal_state(state, restored) or sha256(src/'latest.pt') != before_sha:
                    raise ValueError('Lossless state restoration failed or source changed')
                shutil.copyfile(src/f'current_{iteration:05d}.npz', target/'current.npz')
                if sha256(target/'current.npz') != sha256(src/f'current_{iteration:05d}.npz'):
                    raise ValueError('Latest policy changed')
                parity[str(seed)] = dict(all_state_fields_bitwise_equal=True, current_policy_byte_identical=True,
                                         replay_rows=[b['count'] for b in restored['buffers']], omitted_history_models=len(omitted))
                del state, restored, omitted
            shutil.copyfile(Path(__file__), stage/'restore_lightweight.py')
            (stage/'RESTORE.txt').write_text(
                'Latest policy and bounded replay recovery. No training is started by restore.\n'
                'Install the pinned numpy/torch dependencies and run:\n'
                'python restore_lightweight.py restore ARCHIVE --destination NEW_EMPTY_DIRECTORY\n'
                'load_state(DIRECTORY/seed-N) returns the losslessly restored network, replay, root statistics and RNG.\n'
                'This is material for a separately authorized warm-start training run. It does not include historical mixture models,\n'
                'does not create latest.pt, and must not be passed to the old --resume worker or controller.\n'
                'The source fixed-budget study has completed. This backup does not authorize further training.\n')
            files = {str(p.relative_to(stage)):dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in stage.rglob('*') if p.is_file()}
            manifest = dict(format=FORMAT, run=study.name, iterations=iterations,
                original_full_archive_sha256=original_sha256, replay_capacity_rows_per_seat=100000,
                latest_policy_available=True, bounded_replay_lossless=True, exact_legacy_resume=False,
                historical_mixture_available=False, training_automatically_started=False,
                parity=parity, files=files)
            (stage/'recovery-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            with tarfile.open(archive.with_suffix('.partial'), 'w:gz', compresslevel=3) as tar:
                for p in sorted(stage.rglob('*')):
                    if p.is_file():tar.add(p, arcname=str(p.relative_to(stage)), recursive=False)
            archive.with_suffix('.partial').replace(archive)
        proof = verify_archive(archive)
        result = dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha256(archive),
            iterations=iterations, checkpoint_mode=FORMAT, resumable_selfplay=False,
            exact_legacy_resume=False, warm_start_state_verified=True, historical_mixture_available=False,
            source_archive_sha256=original_sha256, verification=proof, parity=parity)
        archive.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        for lock in locks:lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('create'); pack.add_argument('--study', type=Path, required=True)
    pack.add_argument('--original-archive', type=Path, required=True); pack.add_argument('--original-sha256', required=True)
    pack.add_argument('--expected-iteration', type=int, default=240)
    verify = commands.add_parser('verify'); verify.add_argument('archive', type=Path)
    restore = commands.add_parser('restore'); restore.add_argument('archive', type=Path); restore.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':result = create(args.study, args.original_archive, args.original_sha256, expected_iteration=args.expected_iteration)
    else:result = verify_archive(args.archive, args.destination if args.command == 'restore' else None)
    print(json.dumps(result, indent=2))
