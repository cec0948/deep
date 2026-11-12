"""Fork a completed campaign without discarding learned state or its history."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from fullhouse_v3_common import ROOT, atomic_json, digest


def validate_card_ranges(previous_seed, previous_deals, new_seed, new_deals):
    if min(previous_deals, new_deals) < 2:
        raise ValueError('At least two evaluation deals required')
    # evaluate_compact resets the engine with starting seed + deal index.
    if max(previous_seed, new_seed) < min(previous_seed + previous_deals, new_seed + new_deals):
        raise ValueError('Evaluation card seed ranges overlap')


def extend(parent, folder, through, deal_seed):
    import torch

    parent = Path(parent).resolve()
    folder = Path(folder).resolve()
    cfg = json.loads((parent / 'config.json').read_text())
    manifest = json.loads((parent / 'source-manifest.json').read_text())
    if json.loads((parent / 'launch.json').read_text())['status'] != 'complete':
        raise ValueError('Parent training must be complete')
    if through <= cfg['iterations'] or through % cfg['checkpoint_every']:
        raise ValueError('Extend to a later checkpoint boundary')
    validate_card_ranges(cfg['evaluation_deal_seed'], cfg['evaluation_deals'],
                         deal_seed, cfg['evaluation_deals'])
    if folder.exists():
        raise FileExistsError(folder)
    if manifest['driver_sha256'] != digest(ROOT / 'scripts/train_compact_selfplay.py'):
        raise ValueError('Training driver changed; cannot continue this experiment')
    for name, sha in manifest['files'].items():
        if digest(parent / name) != sha:
            raise ValueError('Parent source changed: ' + name)
    cfg = dict(cfg)
    previous = cfg['iterations']
    provenance = dict(parent=str(parent), iteration=previous,
                      config_sha256=digest(parent / 'config.json'), checkpoints={})
    cfg.update(iterations=through, stages=[through], evaluation_iterations=[previous, through],
               evaluation_all_stages_fields=True, evaluation_deal_seed=deal_seed,
               extension_from=provenance)
    locks = []
    try:
        for seed in cfg['seeds']:
            path = parent / f'seed-{seed}'
            lock = (path / 'worker.lock').open('r')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
            status = json.loads((path / 'status.json').read_text())
            if status['iteration'] != previous or status['status'] != 'complete':
                raise ValueError('Parent seed is incomplete')
            provenance['checkpoints'][str(seed)] = digest(path / 'latest.pt')
        folder.mkdir(parents=True)
        for name in ('source', 'tools'):
            shutil.copytree(parent / name, folder / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbi', '*.nbc'))
        atomic_json(folder / 'config.json', cfg)
        atomic_json(folder / 'source-manifest.json', manifest)
        for seed in cfg['seeds']:
            source = parent / f'seed-{seed}'
            target = folder / f'seed-{seed}'
            target.mkdir()
            ck = torch.load(source / 'latest.pt', map_location='cpu', weights_only=False)
            old_cfg = json.loads((parent / 'config.json').read_text())
            if ck['iteration'] != previous or ck['config'] != old_cfg or ck['source_manifest'] != manifest or ck['seed'] != seed:
                raise ValueError('Parent checkpoint metadata mismatch')
            if len(ck['snapshots']) != previous or [s['weight'] for s in ck['snapshots']] != list(range(1, previous + 1)):
                raise ValueError('Missing pre-update history')
            for snapshot in ck['snapshots']:
                if digest(source / snapshot['file']) != snapshot['sha256']:
                    raise ValueError('Changed parent policy')
            for pattern in ('policy_*.npz', 'current_*.npz', 'mixture_*.json', 'diagnostics.jsonl', 'status.json'):
                for path in source.glob(pattern):
                    shutil.copyfile(path, target / path.name)
            # Only the declared stopping point/evaluation metadata changes.
            # Models, raw iteration weights, replay, RNG and snapshot list do not.
            ck['config'] = cfg
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location('_extension_atomic', folder / 'source/training/r1_support.py')
            support = module_from_spec(spec)
            spec.loader.exec_module(support)
            support.atomic_save(ck, target / 'latest.pt')
            del ck
        atomic_json(folder / 'extension.json', dict(**provenance, through=through,
                    preserved=['net', 'buffers', 'root_replay', 'rng', 'snapshots'],
                    note='Forked metadata; original run is unchanged. Compact fitting uses raw iteration weights, independent of WEIGHT_HORIZON.'))
        atomic_json(folder / 'launch.json', dict(status='prepared'))
    finally:
        for lock in locks:
            lock.close()
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('parent', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--through', type=int, required=True)
    p.add_argument('--deal-seed', type=int, required=True)
    a = p.parse_args()
    print(extend(a.parent, a.output, a.through, a.deal_seed))


if __name__ == '__main__':
    main()
