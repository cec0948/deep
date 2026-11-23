"""Pack the final slow504 training state and four registered postflop policies per seed."""
import argparse
import ast
import copy
from contextlib import closing
import hashlib
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
from plan_poker_backup_retention import validate_record
from plan_rich_backup_retention import _dependencies
from poker_capture_postflop_snapshots import ScheduledPostflopMixture, manifest as capture_manifest
from poker_preflop_specialist import StreetPolicy, write_policy
from poker_strategy_average import StrategyReplay
import register_poker_fixed_preflop_backup as drive
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from snapshot_poker_fixed_preflop_window import DEPENDENCIES, check_decisions
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files, source_archives
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import load_checkpoint


OUTPUT = ROOT/'artifacts/preflop-followup/480-v1/postflop-averaging-20261120'
ARCHIVE = ROOT/'artifacts/exports/poker-scheduled-postflop-20261120-slow504.tar.gz'
MODE = 'scheduled-postflop-window-recovery-v1'
TARGET = drive.BackupTarget(OUTPUT, ARCHIVE, MODE)
ARM = 'slow504'
SLOTS = [486, 492, 498, 504]


def collect_files(stage, plan, *, output=OUTPUT, arm=ARM, mixture_label='mixture504',
                  stage_files=('stage-ready.json', 'plan.json', 'registration.json')):
    if (not stage['training_complete'] or stage['iteration'] != 504 or set(stage['arms']) != {arm}
            or set(stage['arms'][arm]) != set(map(str, SEEDS))):
        raise ValueError('Expected all three completed scheduled training workers')
    files = {name: output/name for name in stage_files}
    for name in plan['frozen_files']:
        path = Path(name)
        if path.parent in (ROOT/'scripts', ROOT/'configs'):
            files[str(path.relative_to(ROOT))] = path
    for path in (Path(__file__).resolve(), ROOT/'scripts/poker_capture_postflop_snapshots.py'):
        files[str(path.relative_to(ROOT))] = path
    for group in ('source', 'tools'):
        for path in (CAMPAIGN/group).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.nbc', '.nbi'):
                files[str(path.relative_to(CAMPAIGN))] = path
    for seed in SEEDS:
        entry = stage['arms'][arm][str(seed)]
        worker = output/arm/f'seed-{seed}'
        pointer = read(worker/'current.json')
        final = worker/'checkpoints'/'iter-00504'
        if (Path(entry['checkpoint']) != final or pointer != dict(iteration=504, directory=final.name, files=entry['files'])
                or set(pointer['files']) != {'arrays.npz', 'current.npz', 'state.json', 'average.npz'}):
            raise ValueError('Final checkpoint pointer does not match the verified stage')
        verify_files({final/name: sha for name,sha in entry['files'].items()})
        verify_files({worker/'diagnostics.jsonl':entry['diagnostics_sha256'],
                      worker/'start-verification.json':entry['start_sha256'],
                      worker/'preflop.npz':entry['preflop_export']['sha256'],
                      worker/'policy.json':entry['policy']['sha256']})
        capture = output/'snapshots'/f'seed-{seed}'
        reference = stage['policies'][mixture_label][str(seed)]
        mixture = read(capture/'mixture.json')
        if (Path(reference['path']) != capture/'mixture.json' or digest(capture/'mixture.json') != reference['sha256']
                or mixture != capture_manifest(capture) or mixture['arm'] != arm or mixture['seed'] != seed
                or [r['iteration'] for r in mixture['postflop']] != SLOTS
                or mixture['postflop'][-1]['sha256'] != pointer['files']['current.npz']
                or mixture['preflop']['sha256'] != entry['preflop_export']['sha256']):
            raise ValueError('The four captured policies differ from the frozen schedule or the trained worker')
        members = [worker/name for name in ('current.json','ready.json','diagnostics.jsonl','start-verification.json',
                                             'preflop.npz','policy.json')]
        members += [final/name for name in pointer['files']]
        members += [capture/name for name in ('mixture.json','plan.json','preflop.npz')]
        for slot in SLOTS:
            members.extend(capture/f'iter-{slot:05d}'/name for name in ('current.npz','receipt.json'))
        for path in members:
            files[str(path.relative_to(output))] = path
    if any(not p.is_file() or p.is_symlink() for p in files.values()):
        raise ValueError('A registered recovery member is missing or symbolic')
    return files


def check_mixture(engine, restored, live):
    """Exercise every private per-hand choice and each street from the extracted tree."""
    seen = set()
    counts = dict(preflop=0, flop=0, turn=0, river=0)
    for nonce in range(100):
        restored.begin_hand(nonce)
        live.begin_hand(nonce)
        choice = next(i for i,p in enumerate(restored.postflop) if p is restored.active)
        if live.active is not live.postflop[choice]:
            raise ValueError('Restored mixture chose a different postflop component')
        if choice in seen:
            continue
        seen.add(choice)
        card = 2038460000 + nonce
        state = engine.call(dict(op='reset', seed=card, button=0))
        while not state['terminal']:
            actor = state['seat_to_act']
            rng = semantic_seed(card, actor, state)
            a, b = restored.decide(state, rng), live.decide(state, rng)
            if (a['distribution'], a['action']) != (b['distribution'], b['action']):
                raise ValueError('Relocated snapshot mixture has different action probabilities')
            counts[state['street']] += 1
            state = engine.call(dict(op='step',action=dict(kind='check' if state['can_check'] else 'call',seat=actor)))
        if len(seen) == len(SLOTS):
            break
    if len(seen) != len(SLOTS) or min(counts.values()) < 4:
        raise ValueError('Mixture restore verification missed a checkpoint or street')
    return counts


def restore_exercise(tree, stage, *, output=OUTPUT, arm=ARM, mixture_label='mixture504'):
    syntax = ast.parse((tree/'tools/compact_regret.py').read_text())
    mapping = next(ast.literal_eval(n.value) for n in syntax.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ARRAY_MAP' for t in n.targets))
    proofs = {}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            key = str(seed)
            moved = tree/arm/f'seed-{seed}'
            state, checkpoint = load_checkpoint(moved)
            original, _ = load_checkpoint(output/arm/f'seed-{seed}')
            counts = validate_state(state, seed, 504)
            strategy = StrategyReplay.from_state(state['strategy_replay'])
            if (not equal_state(state, original) or counts != [100000]*6 or strategy.size != 100000
                    or strategy.capacity != 100000 or state['config']['training_arm'] != arm):
                raise ValueError('Recovered models, replay reservoirs, statistics or RNG differ')
            del original
            pre = moved/'preflop.npz'
            verify_export(pre, state['specialist_net'], mapping)
            verify_export(checkpoint/'current.npz', state['net'], mapping)
            with np.load(checkpoint/'average.npz',allow_pickle=False) as average:
                stored = state['average_state']
                if (set(average.files) != set(stored)|{'format','metadata_json'}
                        or any(not np.array_equal(v.numpy(),average[name]) for name,v in stored.items())):
                    raise ValueError('Restored average export differs from the training state')
            p = write_policy(moved/'restored-policy.json', pre, checkpoint/'current.npz')
            policies = stage['policies']
            parent_ref = policies['specialist480'][key]
            refs = [parent_ref, policies[arm][key], policies[mixture_label][key]]
            verify_files({r['path']:r['sha256'] for r in refs})
            current_counts = check_decisions(engine, StreetPolicy(p,tree/'source'),
                                       StreetPolicy(policies[arm][key]['path'],SOURCE), StreetPolicy(parent_ref['path'],SOURCE))
            mixture_counts = check_mixture(engine,
                    ScheduledPostflopMixture(tree/'snapshots'/f'seed-{seed}'/'mixture.json',tree/'source'),
                    ScheduledPostflopMixture(policies[mixture_label][key]['path'],SOURCE))
            proofs[key] = dict(regret_rows=counts,strategy_rows=strategy.size,slots=SLOTS,
                               final_policy_decisions=current_counts,mixture_decisions=mixture_counts)
    return proofs


def pack_and_restore(target, files, manifest, restore):
    with tempfile.TemporaryDirectory(prefix='.scheduled-restore-',dir=target.parent) as temporary:
        tmp = Path(temporary)
        archive = target if target.exists() else tmp/target.name
        if not archive.exists():
            with tarfile.open(archive,'x:gz',compresslevel=3,dereference=True) as tar:
                for name,path in sorted(files.items()):
                    tar.add(path,arcname=name,recursive=False)
                data=json.dumps(manifest,indent=2,allow_nan=False).encode()
                item=tarfile.TarInfo('recovery-manifest.json')
                item.size=len(data)
                tar.addfile(item,io.BytesIO(data))
        extracted = tmp/'restored'
        count = restore_files(archive,extracted,'recovery-manifest.json',lambda _:True,1024**3,manifest)
        proofs = restore(extracted)
        if archive != target:
            os.link(archive,target)
    return count, proofs


def pack():
    source_archives(read(drive.INDEX))
    stage = read(OUTPUT/'stage-ready.json')
    plan = read(OUTPUT/'plan.json')
    from run_poker_postflop_averaging import verify_stage

    verify_files({**plan['frozen_files'], OUTPUT/'plan.json':read(OUTPUT/'registration.json')['plan_sha256']})
    if (plan['learning_rate'] != .00025 or plan['capture_iterations'] != SLOTS
            or (plan['parent_archive_sha256'],plan['parent_policy_bundle_sha256']) != (FULL,SPECIALIST)):
        raise ValueError('Registered learning rate, snapshot schedule or parents differ')
    recomputed = verify_stage(plan)
    for key in ('arms','policies','seed_verification','warm_start_weight_chain_verified'):
        if stage[key] != recomputed[key]:
            raise ValueError('Published stage differs from actual completed training')
    files = collect_files(stage,plan)
    members = {name:file_meta(p) for name,p in sorted(files.items())}
    if ARCHIVE.is_file() and (OUTPUT/'backup-ready.json').is_file():
        # A previously verified archive contains the packaging code from its own
        # creation. Rerun the full state/strategy restore with that recorded member.
        ready()
        script = str(Path(__file__).resolve().relative_to(ROOT))
        with tarfile.open(ARCHIVE, 'r:gz') as saved:
            members[script] = json.load(saved.extractfile('recovery-manifest.json'))['files'][script]
    manifest = dict(kind='training-checkpoint',checkpoint_mode=MODE,arm=ARM,iteration=504,seeds=SEEDS,
                    slots=SLOTS,restore_dependencies=DEPENDENCIES,archive_includes_final_evaluation_summary=False,
                    restore_state='load_checkpoint(<archive>/slow504/seed-<seed>)',
                    restore_mixture='ScheduledPostflopMixture(<archive>/snapshots/seed-<seed>/mixture.json, <archive>/source)',
                    files=members)
    count, proofs = pack_and_restore(ARCHIVE, files, manifest, lambda tree: restore_exercise(tree, stage))
    record = dict(kind='training-checkpoint',checkpoint_mode=MODE,training_family='scheduled-postflop-window',
                  run='20261120-scheduled-postflop-504',arms=[ARM],iterations={str(s):504 for s in SEEDS},
                  new_selfplay_iterations_per_seed=24,training_stage_completed=True,
                  scheduled_snapshot_iterations=SLOTS,historical_mixture_available=False,exact_legacy_resume=False,
                  resumable_selfplay=False,warm_start_state_verified=True,restoration_verified=True,
                  archive=str(ARCHIVE),**file_meta(ARCHIVE),external_backup=False,
                  archive_includes_final_evaluation_summary=False,restore_dependencies=DEPENDENCIES,
                  verification=dict(actual_extraction_verified=True,lossless_state_verified=True,
                                    current_and_scheduled_policies_verified=True,verified_files=count,seeds=proofs))
    atomic_json(OUTPUT/'backup-ready.json',record)
    return record


def validate_profile(row):
    if row.get('arms') == ['pool504']:
        from snapshot_poker_opponent_diversity import validate_profile as validate_pool
        return validate_pool(row)
    if (row['checkpoint_mode']!=MODE or row['kind']!='training-checkpoint' or row['arms']!=[ARM]
            or row['iterations']!={str(s):504 for s in SEEDS} or row['scheduled_snapshot_iterations']!=SLOTS
            or row['new_selfplay_iterations_per_seed']!=24 or row['archive_includes_final_evaluation_summary'] is not False
            or not all(row[k] is True for k in ('training_stage_completed','restoration_verified','warm_start_state_verified'))
            or not all(row['verification'][k] is True for k in ('actual_extraction_verified','lossless_state_verified',
                                                                 'current_and_scheduled_policies_verified'))
            or _dependencies(row['restore_dependencies'])[1] != {FULL,SPECIALIST}
            or set(row['verification']['seeds']) != set(map(str,SEEDS))):
        raise ValueError('Incomplete or mismatched snapshot training recovery')
    if row['archive'] != str(ARCHIVE) or file_meta(ARCHIVE) != {k:row[k] for k in ('bytes','sha256')}:
        raise ValueError('Archive identity changed since the restore exercise')


def ready():
    row = read(OUTPUT/'backup-ready.json')
    validate_profile(row)
    return row


def retention(index):
    shadow=copy.deepcopy(index)
    protected=set()
    for n,row in enumerate(index['backups']):
        if row.get('deleted_remote') or row.get('checkpoint_mode') != MODE:
            continue
        validate_profile(row)
        if row['status']!='complete' or row['local_archive']!=row['archive']:
            raise ValueError('Incomplete indexed recovery')
        shadow['backups'][n]['kind']='evaluation-assets'
        validate_record(shadow['backups'][n],ROOT)
        protected.add(n)
    plan=drive.plan_retention(shadow)
    kept={item['backup_index']:item for item in plan['retained']}
    if not protected<=set(kept):
        raise ValueError('Latest scheduled snapshot recovery excluded')
    for n in protected:
        kept[n].update(kind='training-checkpoint',checkpoint_mode=MODE,reasons=['completed_scheduled_postflop_window'])
    plan.update(format='private-scheduled-postflop-retention-plan-v1',
                index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['pack','prepare','manifest','check','commit'])
    parser.add_argument('--owner',default=os.environ.get('POKER_BACKUP_OWNER'))
    args=parser.parse_args()
    if args.action=='pack':
        result=pack()
        print(json.dumps({k:result[k] for k in ('bytes','sha256','restoration_verified')}))
    elif args.action=='prepare':
        print(json.dumps(drive.prepare(TARGET,ready)))
    elif args.owner:
        print(json.dumps(drive.register(args.action,args.owner,TARGET,ready,retention)))
    else:
        parser.error('Supply the approved Drive account')
