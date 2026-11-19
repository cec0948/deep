"""Capture scheduled postflop inference exports from a fixed-preflop worker's committed checkpoints.

Register before the first scheduled checkpoint. Invoke capture after each target's
current.json and matching diagnostics.jsonl row have been published. Finalize
only after every scheduled slot has been captured. All writes are local.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from fullhouse_v3_common import atomic_json, digest


FORMAT = 'fixed-preflop-scheduled-postflop-mixture-v1'


def read(path):
    return json.loads(Path(path).read_bytes())


@contextmanager
def locked(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
        os.fsync(fd)
    finally:
        os.close(fd)


def copy_model(src, dest, expected):
    if src.is_symlink() or not src.is_file():
        raise ValueError('Expected a regular source export')
    with src.open('rb') as source, dest.open('xb') as sink:
        shutil.copyfileobj(source, sink)
        sink.flush()
        os.fsync(sink.fileno())
    if digest(dest) != expected:
        raise ValueError('Inference export differs from its committed SHA-256')
    with np.load(dest, allow_pickle=False) as model:
        if (str(model['training_revision']) != 'r1-cpu-v4' or 'opening_trunk_w0' not in model
                or not all(np.isfinite(model[k]).all() for k in model if model[k].dtype.kind in 'biufc')):
            raise ValueError('Expected a finite compact V4 export')


def schedule(iterations):
    if (not 2 <= len(iterations) <= 4 or any(type(t) is not int or not 0 < t <= 99999 for t in iterations)
            or iterations != sorted(set(iterations))):
        raise ValueError('Register two to four distinct increasing checkpoint iterations')


def register(dest, worker, seed, arm, parent_policy, iterations, source):
    schedule(iterations)
    dest, worker = Path(dest).absolute(), Path(worker).resolve()
    if dest.resolve().is_relative_to(worker):
        raise ValueError('Capture directory must be separate from the training worker')
    encoder_sha = digest(Path(source)/'bot/features.py')
    parent_bytes = Path(parent_policy).read_bytes()
    parent = json.loads(parent_bytes)
    if parent.get('format') != 'preflop-street-policy-v1':
        raise ValueError('Expected a fixed street-policy parent')
    preflop = parent['preflop']
    plan = dict(format=FORMAT, worker=str(worker), seed=seed, arm=arm, iterations=iterations,
                source_features_sha256=encoder_sha,
                parent_policy_sha256=hashlib.sha256(parent_bytes).hexdigest(), preflop_sha256=preflop['sha256'])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with locked(dest.parent):
        if dest.exists():
            if read(dest/'plan.json') != plan or digest(dest/'preflop.npz') != plan['preflop_sha256']:
                raise ValueError('Capture directory already has a different plan or preflop export')
            return plan
        for name in ('current.json', 'status.json'):
            record = worker/name
            if record.exists() and read(record)['iteration'] >= iterations[0]:
                raise ValueError('Register the schedule before its first checkpoint is reached')
        with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.capture-') as tmp:
            pending = Path(tmp)/'new'
            pending.mkdir()
            copy_model(Path(preflop['path']), pending/'preflop.npz', plan['preflop_sha256'])
            atomic_json(pending/'plan.json', plan)
            pending.rename(dest)
    return plan


def slot_receipt(dest, plan, iteration):
    slot = dest/f'iter-{iteration:05d}'
    row = read(slot/'receipt.json')
    if (row['iteration'] != iteration or row['plan_sha256'] != digest(dest/'plan.json')
            or row['export_sha256'] != digest(slot/'current.npz')
            or row['export_bytes'] != (slot/'current.npz').stat().st_size or iteration not in plan['iterations']):
        raise ValueError('Captured slot differs from the registered plan or digest')
    return row


def capture(dest, iteration):
    dest = Path(dest).resolve()
    with locked(dest):
        plan = read(dest/'plan.json')
        if iteration not in plan['iterations']:
            raise ValueError('Unscheduled checkpoint')
        slot = dest/f'iter-{iteration:05d}'
        if slot.exists():
            return slot_receipt(dest, plan, iteration)
        worker = Path(plan['worker'])
        ptr_bytes = (worker/'current.json').read_bytes()
        ptr = json.loads(ptr_bytes)
        status = read(worker/'status.json')
        if (ptr['iteration'] != iteration or ptr['directory'] != slot.name
                or status['seed'] != plan['seed'] or status['arm'] != plan['arm']
                or status['iteration'] < iteration or status['status'] == 'failed'
                or not status['process_start'] or type(status['pid']) is not int or status['pid'] < 1):
            raise ValueError('Wrong current checkpoint or worker status for this slot')
        start_bytes = (worker/'start-verification.json').read_bytes()
        start = json.loads(start_bytes)
        if (start['seed'] != plan['seed'] or start['iteration'] >= plan['iterations'][0]
                or start['deployed_policy_sha256'] != plan['parent_policy_sha256']
                or start['street_weight_sha256']['preflop'] != plan['preflop_sha256']):
            raise ValueError('Wrong worker parent or frozen preflop identity')
        ckpt = worker/'checkpoints'/slot.name
        if ckpt.is_symlink() or digest(ckpt/'state.json') != ptr['files']['state.json']:
            raise ValueError('Changed checkpoint state metadata')
        state = dict(read(ckpt/'state.json')['items'])
        if (state['seed'] != {'t':'value', 'value':plan['seed']}
                or state['iteration'] != {'t':'value', 'value':iteration}):
            raise ValueError('Checkpoint state identifies another seed or iteration')
        # Publication order is pointer first, then a newline-terminated diagnostics row.
        # A partially appended next line does not invalidate a committed earlier record.
        with (worker/'diagnostics.jsonl').open('rb') as source:
            rows = [json.loads(line) for line in source if line.endswith(b'\n')]
        matches = [r for r in rows if r['iteration'] == iteration]
        if len(matches) != 1:
            raise ValueError('Exactly one completed diagnostics row is required')
        row = matches[0]
        if (row['seed'] != plan['seed'] or row['checkpoint_iteration'] != iteration
                or row['input_weights']['preflop'] != start['weights']['preflop']
                or row['output_weights']['preflop'] != start['weights']['preflop']
                or row['preflop_steps'] != 0 or row['opening_steps'] != 0
                or row['frozen_preflop_probes'] < 1 or row['frozen_preflop_max_probability_error'] != 0):
            raise ValueError('Checkpoint diagnostics do not verify the registered fixed preflop')
        with tempfile.TemporaryDirectory(dir=dest, prefix='.capture-') as tmp:
            pending = Path(tmp)/'new'
            pending.mkdir()
            sha = ptr['files']['current.npz']
            copy_model(ckpt/'current.npz', pending/'current.npz', sha)
            receipt = dict(iteration=iteration, export_sha256=sha, export_bytes=(pending/'current.npz').stat().st_size,
                           plan_sha256=digest(dest/'plan.json'), start_sha256=hashlib.sha256(start_bytes).hexdigest(),
                           worker_pid=status['pid'], worker_start_token=status['process_start'],
                           source_current_sha256=hashlib.sha256(ptr_bytes).hexdigest(),
                           source_state_metadata_sha256=ptr['files']['state.json'])
            atomic_json(pending/'receipt.json', receipt)
            if (worker/'current.json').read_bytes() != ptr_bytes or (worker/'start-verification.json').read_bytes() != start_bytes:
                raise ValueError('Worker changed while exporting this slot')
            pending.rename(slot)
        return receipt


def manifest(dest):
    plan = read(dest/'plan.json')
    schedule(plan['iterations'])
    if plan['format'] != FORMAT or digest(dest/'preflop.npz') != plan['preflop_sha256']:
        raise ValueError('Fixed preflop or capture format changed')
    slots = [slot_receipt(dest, plan, t) for t in plan['iterations']]
    if len({(s['start_sha256'], s['worker_pid'], s['worker_start_token']) for s in slots}) != 1:
        raise ValueError('Slots belong to different worker runs')
    return dict(format=FORMAT, seed=plan['seed'], arm=plan['arm'], plan_sha256=digest(dest/'plan.json'),
                source_features_sha256=plan['source_features_sha256'],
                preflop=dict(file='preflop.npz', sha256=plan['preflop_sha256']),
                postflop=[dict(iteration=s['iteration'], file=f"iter-{s['iteration']:05d}/current.npz",
                               sha256=s['export_sha256']) for s in slots],
                selection='Equal probability, one private draw per hand', resumable_selfplay=False)


def finalize(dest):
    dest = Path(dest).resolve()
    with locked(dest):
        spec = manifest(dest)
        path = dest/'mixture.json'
        if path.exists():
            if read(path) != spec:
                raise ValueError('Existing final manifest differs')
        else:
            atomic_json(path, spec)
        return spec


class ScheduledPostflopMixture:
    """Load a portable capture directory using the pinned V4 source feature encoder."""
    def __init__(self, model, source):
        from poker_preflop_specialist import evaluator_policy

        self.model = Path(model).resolve()
        dest = self.model.parent
        spec = read(self.model)
        if self.model.name != 'mixture.json' or manifest(dest) != spec:
            raise ValueError('Final manifest differs from the verified capture slots')
        if digest(Path(source)/'bot/features.py') != spec['source_features_sha256']:
            raise ValueError('Runtime feature encoder differs from the registered source')
        refs = [spec['preflop'], *spec['postflop']]
        self.dependency_hashes = {str(dest/ref['file']): ref['sha256'] for ref in refs}
        self.dependency_hashes[str(Path(__file__).resolve())] = digest(__file__)
        policies = [evaluator_policy(dest/ref['file'], source) for ref in refs]
        if any(p.model_type != 'compact_advantage' for p in policies):
            raise ValueError('Expected compact V4 advantage policies')
        for policy in policies:
            self.dependency_hashes.update(policy.dependency_hashes)
        self.preflop, self.postflop = policies[0], policies[1:]
        self.model_type = FORMAT
        self.active = None

    def begin_hand(self, hand_seed):
        if type(hand_seed) is not int or not 0 <= hand_seed < 2**64:
            raise ValueError('Expected a separate 64-bit hand seed')
        rng = random.Random(b'scheduled-postflop-choice-v1\0' + hand_seed.to_bytes(8, 'big'))
        self.active = self.postflop[rng.randrange(len(self.postflop))]

    def decide(self, state, seed):
        if self.active is None or state['street'] not in ('preflop', 'flop', 'turn', 'river'):
            raise ValueError('Expected begin_hand and a valid betting street')
        return (self.preflop if state['street'] == 'preflop' else self.active).decide(state, seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    setup = modes.add_parser('register')
    setup.add_argument('dest', type=Path)
    setup.add_argument('worker', type=Path)
    setup.add_argument('seed', type=int)
    setup.add_argument('arm')
    setup.add_argument('parent_policy', type=Path)
    setup.add_argument('iterations', type=int, nargs='+')
    setup.add_argument('--source', type=Path, required=True, help='Frozen V4 source tree with bot/features.py')
    take = modes.add_parser('capture')
    take.add_argument('dest', type=Path)
    take.add_argument('iteration', type=int)
    modes.add_parser('finalize').add_argument('dest', type=Path)
    args = vars(parser.parse_args())
    mode = args.pop('mode')
    action = {'register': register, 'capture': capture, 'finalize': finalize}[mode]
    print(json.dumps(action(**args), indent=2, allow_nan=False))
