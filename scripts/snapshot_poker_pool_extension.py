"""Create, restore and privately register the 528-round diverse-opponent recovery."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path

from controlled_poker_refit import read, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_rich_backup_retention import _dependencies
from poker_average_runtime import WindowAveragePolicy
from poker_opponent_population import STAGES, historical_inputs
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from run_poker_pool528 import (ARM, CAPTURES, CONFIG, FINAL, INITIAL, PARENT_ARCHIVE, PARENT_STAGE, SEEDS,
                              SOURCE, STUDY, read_bound, verify_stage)
from snapshot_poker_fixed_preflop_window import DEPENDENCIES as BASELINE_DEPENDENCIES, check_decisions
from snapshot_poker_opponent_diversity import restore_opponents
from snapshot_poker_preflop_dose import file_meta, source_archives
from snapshot_poker_scheduled_window import MODE, collect_files, pack_and_restore, restore_exercise

ARCHIVE = ROOT/'artifacts/exports/poker-scheduled-postflop-20261126-pool-extension-528.tar.gz'
TARGET = BackupTarget(STUDY, ARCHIVE, MODE)
DEPENDENCIES = [*BASELINE_DEPENDENCIES,
                dict(parent_checkpoint_sha256=PARENT_ARCHIVE, scope="November 23 complete 504-round source and mixture")]


def verify_parents():
    index = read(INDEX)
    source_archives(index)
    parents = [row for row in index['backups'] if row.get('sha256') == PARENT_ARCHIVE and not row.get('deleted_remote')]
    if len(parents) != 1:
        raise ValueError("Missing or ambiguous source 504 backup")
    parent = parents[0]
    if (parent['status'] != 'complete' or not all(parent[key] for key in
            ('restoration_verified', 'verified_remote_size', 'verified_owner_only', 'external_backup'))
            or file_meta(Path(parent['local_archive'])) != {key:parent[key] for key in ('bytes', 'sha256')}):
        raise ValueError("Source 504 backup integrity differs from its private index")


def archive_inputs():
    verify_parents()
    plan = read_bound()
    stage = read(STUDY/'stage-ready.json')
    audit = read(STUDY/'training-audit.json')
    if (audit['training_complete'] is not True or audit['stage_sha256'] != digest(STUDY/'stage-ready.json')
            or audit['training_plan_sha256'] != digest(STUDY/'training-plan.json')
            or audit['source_audit_sha256'] != digest(STUDY/'source-state-audit.json')
            or audit['actual_round_updates'] != len(SEEDS) * (FINAL-INITIAL)):
        raise ValueError("The stage differs from the independent training audit")
    verified = json.loads(json.dumps(verify_stage(plan), allow_nan=False))
    for key in ('training_complete', 'warm_start_weight_chain_verified', 'parent_stage_sha256',
                'iteration', 'arms', 'policies', 'seed_verification'):
        if stage[key] != verified[key]:
            raise ValueError(f"Final stage verification differs for {key}")
    if plan['parent_stage_sha256'] != PARENT_STAGE or plan['parent_archive_sha256'] != PARENT_ARCHIVE:
        raise ValueError("Incorrect 504 source identity")
    files = collect_files(stage, plan, output=STUDY, arm=ARM, mixture_label='mixture528', slots=CAPTURES,
                          stage_files=['stage-ready.json','training-audit.json','source-state-audit.json','training-plan.json',
                                       'training-bound.json','registration.json'])
    for path in (Path(__file__).resolve(), CONFIG, ROOT/'scripts/audit_poker_pool528.py',
                 ROOT/'scripts/snapshot_poker_opponent_diversity.py', ROOT/'scripts/poker_average_runtime.py',
                 ROOT/'scripts/poker_average_inference.py'):
        files[str(path.relative_to(ROOT))] = path
    live, frozen = historical_inputs(SEEDS)
    verify_files(frozen)
    if live != plan['historical_policies']:
        raise ValueError("Archived opponents differ from the ones used for training")
    pool = {}
    for seed, rows in live.items():
        pool[seed] = {}
        for iteration, ref in rows.items():
            name = f'opponents/seed-{seed}/iter-{int(iteration):05d}.npz'
            pool[seed][iteration] = dict(file=name, sha256=ref['sha256'])
            files[name] = Path(ref['path'])
    return stage, files, pool, live


def pack():
    stage, files, pool, live = archive_inputs()
    manifest = dict(kind='training-checkpoint', checkpoint_mode=MODE, arm=ARM, iteration=FINAL,
                    seeds=SEEDS, slots=CAPTURES, auxiliary_average_window=[361, FINAL],
                    restore_dependencies=DEPENDENCIES, training_opponents=pool,
                    archive_includes_final_evaluation_summary=False,
                    restore_state='load_checkpoint(<archive>/pool528/seed-<seed>)',
                    restore_mixture='ScheduledPostflopMixture(<archive>/snapshots/seed-<seed>/mixture.json, <archive>/source)',
                    restore_opponents='Use each training_opponents relative file path and sha256 with load_historical.',
                    files={name:file_meta(path) for name,path in sorted(files.items())})

    def restore(tree):
        proofs = restore_exercise(tree, stage, output=STUDY, arm=ARM, mixture_label='mixture528',
                                  slots=CAPTURES, current_label='current528')
        counts = restore_opponents(tree, pool, live, arm=ARM)
        with closing(Bridge()) as engine:
            for seed, proof in proofs.items():
                local = tree/ARM/f'seed-{seed}'/'checkpoints'/f'iter-{FINAL:05d}'/'average.npz'
                actual = WindowAveragePolicy(local, tree/'source')
                ref = stage['policies']['average528'][seed]
                verify_files({local:ref['sha256'], ref['path']:ref['sha256']})
                source = WindowAveragePolicy(ref['path'], SOURCE)
                if actual.metadata['averaging_window'] != [361, FINAL]:
                    raise ValueError("Restored auxiliary average has an incorrect window")
                proof['restored_window_average_decisions'] = check_decisions(engine, actual, source, source)
                proof['restored_opponent_decisions'] = counts[seed]
        return proofs

    count, seeds = pack_and_restore(ARCHIVE, files, manifest, restore)
    row = dict(kind='training-checkpoint', checkpoint_mode=MODE, training_family='opponent-diversity',
               run='20261126-pool-extension-528', arms=[ARM], iterations={str(s):FINAL for s in SEEDS},
               new_selfplay_iterations_per_seed=FINAL-INITIAL, training_stage_completed=True,
               scheduled_snapshot_iterations=CAPTURES, auxiliary_average_window=[361, FINAL],
               historical_mixture_available=False, exact_legacy_resume=False, resumable_selfplay=False,
               warm_start_state_verified=True, restoration_verified=True, archive=str(ARCHIVE),
               **file_meta(ARCHIVE), external_backup=False, training_opponent_iterations=list(STAGES),
               archive_includes_final_evaluation_summary=False, restore_dependencies=DEPENDENCIES,
               verification=dict(actual_extraction_verified=True, lossless_state_verified=True,
                                 current_and_scheduled_policies_verified=True, training_opponents=pool,
                                 verified_files=count, seeds=seeds))
    validate_profile(row)
    atomic_json(STUDY/'backup-ready.json', row)
    return row


def validate_profile(row):
    ids = set(map(str, SEEDS))
    proof = row['verification']
    if (row['archive'] != str(ARCHIVE) or file_meta(ARCHIVE) != {k:row[k] for k in ('bytes','sha256')}
            or (row['kind'],row['checkpoint_mode'],row['arms']) != ('training-checkpoint',MODE,[ARM])
            or row['iterations'] != {s:FINAL for s in ids}
            or row['new_selfplay_iterations_per_seed'] != FINAL-INITIAL
            or row['scheduled_snapshot_iterations'] != CAPTURES or row['auxiliary_average_window'] != [361,FINAL]
            or row['training_opponent_iterations'] != list(STAGES)
            or row['archive_includes_final_evaluation_summary'] is not False
            or not all(row[k] is True for k in ('training_stage_completed','restoration_verified','warm_start_state_verified'))
            or _dependencies(row['restore_dependencies']) != _dependencies(DEPENDENCIES)
            or set(proof['seeds']) != ids or set(proof['training_opponents']) != ids
            or not all(proof[k] is True for k in ('actual_extraction_verified','lossless_state_verified',
                                                  'current_and_scheduled_policies_verified'))):
        raise ValueError("Invalid verified 528 archive")
    for seed, state in proof['seeds'].items():
        if (state['regret_rows'] != [100000]*6 or state['strategy_rows'] != 100000
                or state['slots'] != CAPTURES or min(state['restored_window_average_decisions'].values()) < 4
                or set(state['restored_opponent_decisions']) != set(map(str, STAGES))
                or min(state['restored_opponent_decisions'].values()) < 3
                or set(proof['training_opponents'][seed]) != set(map(str, STAGES))):
            raise ValueError("Incomplete restored state, snapshots, window average or historical opponent pool")


def ready():
    row = read(STUDY/'backup-ready.json')
    validate_profile(row)
    return row


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('pack','prepare','manifest','check','commit'))
    cli.add_argument('--owner', default=os.environ.get('POKER_BACKUP_OWNER'))
    arg = cli.parse_args()
    if arg.action == 'pack':
        result = {k:v for k,v in pack().items() if k in ('bytes','sha256','restoration_verified')}
    elif arg.action == 'prepare':
        result = prepare(TARGET, ready)
    elif arg.owner:
        result = register(arg.action, arg.owner, TARGET, ready, plan_retention)
        if arg.action == 'commit':
            result.update(auxiliary_average_window=[361,FINAL], restored_training_opponents=len(SEEDS)*len(STAGES),
                          restored_scheduled_snapshots=len(SEEDS)*len(CAPTURES))
            atomic_json(STUDY/'backup-verification.json', result)
    else:
        cli.error('Provide the approved account owner')
    print(json.dumps(result))
