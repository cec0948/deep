"""Pack and restore the 504-round diverse-opponent study and its fixed training opponents."""
import argparse
from contextlib import closing
import json
import os
import sys
from pathlib import Path

import numpy as np

from controlled_poker_refit import read, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_rich_backup_retention import _dependencies
from poker_opponent_population import STAGES, historical_inputs, load_historical
from poker_preflop_specialist import evaluator_policy
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from run_poker_preflop_online import StreetNetworks
from run_poker_preflop_pilot import SEEDS, SOURCE
from snapshot_poker_fixed_preflop_window import DEPENDENCIES
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, source_archives
from snapshot_poker_scheduled_window import MODE, SLOTS, collect_files, pack_and_restore, restore_exercise
from train_poker_average_continuation import load_checkpoint

OUTPUT = ROOT/'artifacts/preflop-followup/480-v1/opponent-diversity-20261123'
ARM = 'pool504'
MIXTURE = 'poolMixture504'
CONFIG = ROOT/'configs/poker-opponent-diversity-20261123.json'
ARCHIVE = ROOT/'artifacts/exports/poker-scheduled-postflop-20261123-opponent-diversity-504.tar.gz'
TARGET = BackupTarget(OUTPUT, ARCHIVE, MODE)


def archive_inputs():
    from run_poker_opponent_diversity_train import read_bound, verify_stage

    stage = read(OUTPUT/'stage-ready.json')
    plan = read_bound()
    reg = read(OUTPUT/'registration.json')
    verify_files({CONFIG:reg['config_sha256'], **reg['frozen_files'], **plan['frozen_files']})
    # JSON persists iteration numbers used as dictionary keys as strings.
    recomputed = json.loads(json.dumps(verify_stage(plan), allow_nan=False))
    if ((plan['parent_archive_sha256'],plan['parent_policy_bundle_sha256']) != (FULL,SPECIALIST)
            or stage.get('warm_start_weight_chain_verified') is not True
            or any(stage[key] != recomputed[key] for key in ('arms', 'policies', 'seed_verification'))):
        raise ValueError('Live treatment workers differ from the completed stage')
    historical, frozen = historical_inputs(SEEDS)
    verify_files(frozen)
    if (plan['historical_policies'] != historical or
        any(reg['historical_policies'][s][t] != ref for s, rows in historical.items() for t, ref in rows.items())):
        raise ValueError('The retained historical opponents differ from the training registration')
    evidence = ['control-lineage-audit.json', 'control-round-audit.json', 'dispatch-audit.json']
    audit = read(OUTPUT/'training-audit.json')
    if (audit['training_complete'] is not True or audit['stage_sha256'] != digest(OUTPUT/'stage-ready.json')
            or audit['training_plan_sha256'] != digest(OUTPUT/'training-plan.json')
            or audit['independent_proofs'] != {name: digest(OUTPUT/name) for name in evidence}):
        raise ValueError('The independent training audit does not match the completed stage')
    stage_files = ['stage-ready.json', 'training-plan.json', 'training-bound.json', 'registration.json',
                   'training-audit.json', *evidence]
    files = collect_files(stage, plan, output=OUTPUT, arm=ARM, mixture_label=MIXTURE, stage_files=stage_files)
    files[str(CONFIG.relative_to(ROOT))] = CONFIG
    for script in [Path(__file__).resolve(), ROOT/'scripts/poker_opponent_population.py',
                   ROOT/'scripts/run_poker_opponent_diversity_train.py', ROOT/'scripts/audit_poker_opponent_diversity_training.py']:
        files[str(script.relative_to(ROOT))] = script
    pool = {}
    for seed, rows in historical.items():
        pool[seed] = {}
        for iteration, reference in rows.items():
            name = f'opponents/seed-{seed}/iter-{int(iteration):05d}.npz'
            files[name] = Path(reference['path'])
            pool[seed][iteration] = dict(file=name, sha256=reference['sha256'])
    return stage, files, pool


def restore_opponents(tree, pool, live):
    """Load the relative-path pool using the training router and check its postflop distributions."""
    old_path = sys.path.copy()
    sys.path[:0] = [str(tree/'source'), str(tree/'tools')]
    try:
        from compact_regret import CompactRegretNet
        with closing(Bridge()) as engine:
            decisions = {}
            for seed, records in pool.items():
                state, _ = load_checkpoint(tree/ARM/f'seed-{seed}')
                general = CompactRegretNet().eval()
                general.load_state_dict(state['net'])
                del state
                refs = {t:dict(path=str(tree/row['file']), sha256=row['sha256']) for t,row in records.items()}
                saved = load_historical(StreetNetworks(general, None), refs)
                original = {t:evaluator_policy(row['path'],SOURCE) for t,row in live[seed].items()}
                counts = {t:0 for t in refs}
                observation = engine.call(dict(op='reset',seed=2038460100+int(seed),button=0))
                while not observation['terminal']:
                    if observation['street'] != 'preflop':
                        for t, deploy in original.items():
                            features, mask = deploy.features.encode_state_dict(observation, deploy.lookup._equity_tables)
                            actual = saved[int(t)].strategy(features,mask.astype(np.float32))
                            expected = deploy.lookup.get_strategy(observation)
                            if not np.allclose(actual,expected,rtol=0,atol=1e-6):
                                raise ValueError('Restored training opponent differs from the frozen inference export')
                            counts[t] += 1
                    observation = engine.call(dict(op='step',action=dict(seat=observation['seat_to_act'],
                                              kind='check' if observation['can_check'] else 'call')))
                if min(counts.values())<3:
                    raise ValueError('Missing historical opponent postflop validation')
                decisions[seed] = counts
    finally:
        sys.path[:] = old_path
    return decisions


def pack():
    source_archives(read(INDEX))
    stage, files, pool = archive_inputs()
    manifest = dict(kind='training-checkpoint', checkpoint_mode=MODE, arm=ARM, iteration=504,
                    seeds=SEEDS, slots=SLOTS, restore_dependencies=DEPENDENCIES,
                    training_opponents=pool, archive_includes_final_evaluation_summary=False,
                    restore_state='load_checkpoint(<archive>/pool504/seed-<seed>)',
                    restore_mixture='ScheduledPostflopMixture(<archive>/snapshots/seed-<seed>/mixture.json, <archive>/source)',
                    restore_opponents='Build load_historical() refs from training_opponents using the extracted file paths and SHA-256.',
                    files={name:file_meta(path) for name,path in sorted(files.items())})

    def restore(tree):
        seeds = restore_exercise(tree,stage,output=OUTPUT,arm=ARM,mixture_label=MIXTURE)
        counts = restore_opponents(tree,pool,read(OUTPUT/'training-plan.json')['historical_policies'])
        for seed in seeds:
            seeds[seed]['restored_opponent_decisions'] = counts[seed]
        return seeds

    count, seeds = pack_and_restore(ARCHIVE,files,manifest,restore)
    record = dict(kind='training-checkpoint',checkpoint_mode=MODE,training_family='opponent-diversity',
                  run='20261123-opponent-diversity-504',arms=[ARM],iterations={str(s):504 for s in SEEDS},
                  new_selfplay_iterations_per_seed=24,training_stage_completed=True,
                  scheduled_snapshot_iterations=SLOTS,historical_mixture_available=False,exact_legacy_resume=False,
                  resumable_selfplay=False,warm_start_state_verified=True,restoration_verified=True,
                  archive=str(ARCHIVE),**file_meta(ARCHIVE),external_backup=False,
                  training_opponent_iterations=list(STAGES),archive_includes_final_evaluation_summary=False,
                  restore_dependencies=DEPENDENCIES,
                  verification=dict(actual_extraction_verified=True,lossless_state_verified=True,
                                    current_and_scheduled_policies_verified=True,training_opponents=pool,
                                    verified_files=count,seeds=seeds))
    validate_profile(record)
    atomic_json(OUTPUT/'backup-ready.json',record)
    return record


def validate_profile(record):
    proof = record['verification']
    seeds = set(map(str,SEEDS))
    pool = proof['training_opponents']
    if (record['checkpoint_mode']!=MODE or record['kind']!='training-checkpoint'
            or record['training_family']!='opponent-diversity' or record['arms']!=[ARM]
            or record['iterations']!={s:504 for s in seeds} or record['scheduled_snapshot_iterations']!=SLOTS
            or record['training_opponent_iterations']!=list(STAGES) or record['new_selfplay_iterations_per_seed']!=24
            or record['archive_includes_final_evaluation_summary'] is not False
            or not all(record[k] is True for k in ('training_stage_completed','restoration_verified','warm_start_state_verified'))
            or not all(proof[k] is True for k in ('actual_extraction_verified','lossless_state_verified',
                                                  'current_and_scheduled_policies_verified'))
            or _dependencies(record['restore_dependencies'])[1]!={FULL,SPECIALIST}
            or set(pool)!=seeds or set(proof['seeds'])!=seeds
            or any(set(proof['seeds'][s]['restored_opponent_decisions'])!=set(pool[s]) or
                   min(proof['seeds'][s]['restored_opponent_decisions'].values())<3 for s in seeds)
            or any(set(refs)!=set(map(str,STAGES)) for refs in pool.values())
            or record['archive']!=str(ARCHIVE) or file_meta(ARCHIVE)!={k:record[k] for k in ('bytes','sha256')}):
        raise ValueError('Unverified opponent-diversity recovery')


def ready():
    record = read(OUTPUT/'backup-ready.json')
    validate_profile(record)
    return record


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['pack','prepare','manifest','check','commit'])
    parser.add_argument('--owner',default=os.environ.get('POKER_BACKUP_OWNER'))
    args = parser.parse_args()
    if args.action=='pack':
        record=pack()
        print(json.dumps({k:record[k] for k in ('bytes','sha256','restoration_verified')}))
    elif args.action=='prepare':
        print(json.dumps(prepare(TARGET,ready)))
    elif args.owner:
        result = register(args.action,args.owner,TARGET,ready,plan_retention)
        if args.action=='commit':
            result.update(training_opponent_iterations=list(STAGES), restored_training_opponents=9,
                          restored_training_opponent_decisions={s:v['restored_opponent_decisions']
                                for s,v in ready()['verification']['seeds'].items()})
            atomic_json(OUTPUT/'backup-verification.json',result)
        print(json.dumps(result))
    else:
        parser.error('Supply the approved Drive owner')
