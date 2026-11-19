"""Archive the verified 504-round training state and replay its deployed policy after extraction."""
import ast
from contextlib import closing
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from compact_lightweight_backup import equal_state, validate_state
from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import StreetPolicy, write_policy
from poker_strategy_average import StrategyReplay
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files, source_archives
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import load_checkpoint
from train_poker_fixed_preflop_window import ARM, FINAL, INITIAL, OUTPUT


MODE = 'fixed-preflop-window-recovery-v1'
ARCHIVE = ROOT / 'artifacts/exports/poker-fixed-preflop-20261119-stage-00504.tar.gz'
DEPENDENCIES = [
    dict(parent_checkpoint_sha256=FULL, scope='Original 480-round training state and baseline postflop'),
    dict(parent_backup_sha256=SPECIALIST, scope='Independently confirmed baseline and frozen preflop weights'),
]


def source_files(stage, plan):
    if (stage['training_complete'] is not True or stage['iteration'] != FINAL
            or set(stage['arms']) != {ARM} or set(stage['arms'][ARM]) != set(map(str, SEEDS))
            or (plan['parent_archive_sha256'], plan['parent_policy_bundle_sha256']) != (FULL, SPECIALIST)):
        raise ValueError('Stage or confirmed baseline does not match this recovery profile')
    files = {n: OUTPUT/n for n in ('stage-ready.json', 'plan.json', 'config.json')}
    for group in ('source', 'tools'):
        for path in (CAMPAIGN/group).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.nbc', '.nbi'):
                files[str(path.relative_to(CAMPAIGN))] = path
    for name in plan['frozen_files']:
        path = Path(name)
        if path.parent in (ROOT/'scripts', ROOT/'configs'):
            files[str(path.relative_to(ROOT))] = path
    for path in (Path(__file__).resolve(), ROOT/'scripts/snapshot_poker_preflop_dose.py'):
        files[str(path.relative_to(ROOT))] = path
    for seed in SEEDS:
        worker = OUTPUT/ARM/f'seed-{seed}'
        saved = read(worker/'current.json')
        checkpoint = worker/'checkpoints'/f'iter-{FINAL:05d}'
        entry = stage['arms'][ARM][str(seed)]
        if (saved != dict(iteration=FINAL, directory=checkpoint.name, files=entry['files'])
                or Path(entry['checkpoint']) != checkpoint
                or set(entry['files']) != {'state.json','arrays.npz','current.npz','average.npz'}):
            raise ValueError('Checkpoint pointer does not identify the completed four-file worker checkpoint')
        verify_files({checkpoint/name: sha for name, sha in entry['files'].items()})
        if (Path(entry['preflop_export']['path']) != worker/'preflop.npz'
                or Path(entry['policy']['path']) != worker/'policy.json'):
            raise ValueError('Final policy or preflop weights point outside the registered worker')
        for path in [worker/name for name in ('current.json','preflop.npz','policy.json','diagnostics.jsonl',
                                               'start-verification.json')] + [checkpoint/name for name in saved['files']]:
            files[str(path.relative_to(OUTPUT))] = path
    if any(p.is_symlink() or not p.is_file() for p in files.values()):
        raise ValueError('A recovery source is missing or symbolic')
    return files


def check_decisions(engine, restored, candidate, parent):
    counts = dict(preflop=0, flop=0, turn=0, river=0)
    for deal in range(2037839900, 2037839904):
        state = engine.call(dict(op='reset', seed=deal, button=0))
        while not state['terminal']:
            actor = state['seat_to_act']
            key = semantic_seed(deal, actor, state)
            actual = restored.decide(state, key)
            comparisons = [candidate.decide(state, key)]
            if state['street'] == 'preflop':
                comparisons.append(parent.decide(state, key))
            for ref in comparisons:
                if (actual['distribution'], actual['action']) != (ref['distribution'], ref['action']):
                    raise ValueError('Restored policy probabilities or action differ')
            counts[state['street']] += 1
            move = dict(seat=actor, kind='check' if state['can_check'] else 'call')
            state = engine.call(dict(op='step', action=move))
    if any(v < 4 for v in counts.values()):
        raise ValueError('Restore exercise missed a street')
    return counts


def verify_restored(folder, stage, plan):
    tree = ast.parse((folder/'tools/compact_regret.py').read_text())
    mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ARRAY_MAP' for t in n.targets))
    proofs = {}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            key = str(seed)
            live_dir = OUTPUT/ARM/f'seed-{seed}'
            moved = folder/ARM/f'seed-{seed}'
            state, checkpoint = load_checkpoint(moved)
            live, _ = load_checkpoint(live_dir)
            counts = validate_state(state, seed, FINAL)
            strategy = StrategyReplay.from_state(state['strategy_replay'])
            if (not equal_state(state, live) or counts != [100000]*6 or strategy.size != 100000
                    or strategy.capacity != 100000 or state['config']['training_arm'] != ARM
                    or state['averaging_window'] != [361, FINAL]):
                raise ValueError('Restored state or bounded reservoirs differ from final checkpoint')
            del live
            parent_ref = plan['parent_policies'][key]
            parent_spec = read(parent_ref['path'])
            candidate_ref = stage['arms'][ARM][key]['policy']
            preflop = moved/'preflop.npz'
            verify_files({parent_ref['path']: parent_ref['sha256'], preflop: parent_spec['preflop']['sha256'],
                          candidate_ref['path']: candidate_ref['sha256']})
            verify_export(preflop, state['specialist_net'], mapping)
            verify_export(checkpoint/'current.npz', state['net'], mapping)
            if digest(checkpoint/'current.npz') == parent_spec['postflop']['sha256']:
                raise ValueError('Final trained postflop export equals its source')
            with np.load(checkpoint/'average.npz', allow_pickle=False) as average:
                metadata = json.loads(str(average['metadata_json']))
                expected = state['average_state']
                if (metadata['seed'] != seed or metadata['historical_mixture_available'] is not False
                        or metadata['averaging_window'] != [361, FINAL]
                        or set(average.files) != set(expected) | {'format','metadata_json'}
                        or any(not np.array_equal(v.numpy(), average[k]) for k,v in expected.items())):
                    raise ValueError('Restored average differs from final state')
            policy = write_policy(moved/'relocated-policy.json', preflop, checkpoint/'current.npz')
            decisions = check_decisions(engine, StreetPolicy(policy, folder/'source'),
                                        StreetPolicy(candidate_ref['path'], SOURCE), StreetPolicy(parent_ref['path'], SOURCE))
            proofs[key] = dict(regret_rows=counts, strategy_rows=strategy.size, decisions=decisions,
                               restored_checkpoint_statistics_and_rng=True, frozen_preflop_sha256=digest(preflop),
                               trained_postflop_sha256=digest(checkpoint/'current.npz'))
    return proofs


def snapshot():
    from run_poker_fixed_preflop_window import verify_training

    stage, plan = read(OUTPUT/'stage-ready.json'), read(OUTPUT/'plan.json')
    verify_files(plan['frozen_files'])
    checked = verify_training(plan)
    if (stage['arms'] != checked['arms'] or stage['seed_verification'] != checked['seed_verification']):
        raise ValueError('Persisted training stage changed')
    source_archives(read(ROOT/'artifacts/offsite-backups.json'))
    files = source_files(stage, plan)
    manifest = dict(kind='training-checkpoint', checkpoint_mode=MODE, arm=ARM, iteration=FINAL, seeds=SEEDS,
                    checkpoint_directory=f'{ARM}/seed-<seed>/checkpoints/iter-{FINAL:05d}',
                    restore_policy='Use each seed directory with load_checkpoint and pair its preflop.npz with the checkpoint current.npz through write_policy.',
                    restore_dependencies=DEPENDENCIES, archive_includes_final_evaluation_summary=False,
                    files={name: file_meta(path) for name,path in sorted(files.items())})
    with tempfile.TemporaryDirectory(prefix='.fixed-preflop-restore-', dir=ARCHIVE.parent) as temp:
        root = Path(temp)
        archive = ARCHIVE if ARCHIVE.exists() else root/ARCHIVE.name
        if not archive.exists():
            with tarfile.open(archive, 'x:gz', compresslevel=3) as tar:
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
                data = json.dumps(manifest,indent=2,allow_nan=False).encode()
                item = tarfile.TarInfo('recovery-manifest.json')
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
        folder = root/'restored'
        verified = restore_files(archive, folder, 'recovery-manifest.json', lambda _: True, 1024**3, manifest)
        proofs = verify_restored(folder, stage, plan)
        if archive != ARCHIVE:
            os.link(archive, ARCHIVE)
    record = dict(kind='training-checkpoint', checkpoint_mode=MODE, training_family='fixed-preflop-window',
                  run='20261119-fixed-preflop-504', arms=[ARM], iterations={str(s): FINAL for s in SEEDS},
                  new_selfplay_iterations_per_seed=FINAL-INITIAL, training_stage_completed=True,
                  averaging_window=[361,FINAL], historical_mixture_available=False, exact_legacy_resume=False,
                  resumable_selfplay=False, warm_start_state_verified=True, restoration_verified=True,
                  archive=str(ARCHIVE), **file_meta(ARCHIVE), external_backup=False,
                  archive_includes_final_evaluation_summary=False, restore_dependencies=DEPENDENCIES,
                  verification=dict(actual_extraction_verified=True, lossless_state_verified=True,
                                    both_street_weights_verified=True, frozen_preflop_verified=True,
                                    verified_files=verified, seeds=proofs))
    atomic_json(OUTPUT/'backup-ready.json', record)
    return record


if __name__ == '__main__':
    ready = snapshot()
    print(json.dumps({k: ready[k] for k in ('archive','bytes','sha256','restoration_verified')}))
