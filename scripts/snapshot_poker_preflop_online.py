"""Archive and restore both completed 492-round training arms in one bounded bundle."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np

from compact_lightweight_backup import equal_state, load_state, safe_member, validate_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy, write_policy
from poker_strategy_average import StrategyReplay
from run_poker_preflop_online import ARMS, CFG, OUTPUT, SEEDS
from run_poker_preflop_pilot import CAMPAIGN, SOURCE


MODE = 'preflop-online-comparison-recovery-v1'
ARCHIVE = ROOT/'artifacts/exports/poker-preflop-online-20261118-stage-00492.tar.gz'


def verify_export(path, net, mapping):
    with np.load(path, allow_pickle=False) as stored:
        for part, prefix in (('main', ''), ('opening', 'opening_')):
            for key, name in mapping.items():
                if not np.array_equal(net[f'{part}.{key}'].numpy(), stored[prefix+name]):
                    raise ValueError('Exported policy differs from restored training weights')
        for name in ('main_scales', 'opening_scale'):
            if not np.array_equal(net[name].numpy(), stored[name]):
                raise ValueError('Regret units differ from restored training state')


def verify_restored(folder, stage):
    tree = ast.parse((folder/'tools/compact_regret.py').read_text())
    mapping = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'ARRAY_MAP' for t in node.targets))
    result = {}
    engine = Bridge()
    try:
        for arm in ARMS:
            for seed in SEEDS:
                directory = folder/arm/f'seed-{seed}'
                saved = load_state(directory)
                replay_rows = validate_state(saved, seed, CFG['final_iteration'])
                replay = StrategyReplay.from_state(saved['strategy_replay'])
                if (saved['config']['training_arm'] != arm or replay.capacity != 100000
                        or saved['averaging_window'] != [361, CFG['final_iteration']]):
                    raise ValueError('Wrong restored arm, strategy capacity or window')
                original = stage['arms'][arm][str(seed)]
                if not equal_state(saved, load_state(Path(original['checkpoint']))):
                    raise ValueError('Restored recovery state is not lossless')
                restored_model = directory/'current.npz'
                verify_export(restored_model, saved['net'], mapping)
                if arm == 'specialist492':
                    verify_export(directory/'preflop.npz', saved['specialist_net'], mapping)
                    restored_model = write_policy(directory/'restored-policy.json', directory/'preflop.npz', restored_model)
                with np.load(directory/'average.npz', allow_pickle=False) as arrays:
                    if any(not np.array_equal(array.numpy(), arrays[name]) for name, array in saved['average_state'].items()):
                        raise ValueError('Restored average weights differ from the final average export')
                restored = evaluator_policy(restored_model, folder/'source')
                live = evaluator_policy(original['policy']['path'], SOURCE)
                decisions = {street: 0 for street in ('preflop', 'flop', 'turn', 'river')}
                for deal in range(CFG['online_smoke_deal_seed'], CFG['online_smoke_deal_seed']+4):
                    state = engine.call(dict(op='reset', seed=deal, button=0))
                    while not state['terminal']:
                        seed_rng = deal + sum(decisions.values())
                        if restored.decide(state, seed_rng) != live.decide(state, seed_rng):
                            raise ValueError('Restored decision differs on a legal public state')
                        decisions[state['street']] += 1
                        action = dict(seat=state['seat_to_act'], kind='check' if state['can_check'] else 'call')
                        state = engine.call(dict(op='step', action=action))
                if min(decisions.values()) < 1:
                    raise ValueError('Restore probes missed a betting street')
                result[f'{arm}/{seed}'] = dict(decisions=decisions, regret_rows=replay_rows, strategy_rows=replay.size)
    finally:
        engine.close()
    return result


def verify_archive(archive, stage):
    with tempfile.TemporaryDirectory(prefix='poker-492-restore-') as temp:
        folder = Path(temp)
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            names = {m.name for m in members}
            if (len(names) != len(members) or 'recovery-manifest.json' not in names
                    or sum(m.size for m in members) > 1024**3
                    or any(not m.isfile() or not safe_member(m.name) for m in members)):
                raise ValueError('Unsafe recovery archive membership')
            manifest = json.load(tar.extractfile('recovery-manifest.json'))
            if manifest['checkpoint_mode'] != MODE or names != manifest['files'].keys() | {'recovery-manifest.json'}:
                raise ValueError('Recovery manifest does not cover every file')
            tar.extractall(folder, members=members, filter='data')
        for name, ref in manifest['files'].items():
            path = folder/name
            if path.stat().st_size != ref['bytes'] or digest(path) != ref['sha256']:
                raise ValueError('Restored archive file differs in size or SHA-256')
        probes = verify_restored(folder, stage)
        return dict(passed=True, verified_files=len(manifest['files']), actual_extraction_verified=True,
                    lossless_state_verified=True, probes=probes)


def create_archive():
    stage = read(OUTPUT/'stage-ready.json')
    plan = read(OUTPUT/'plan.json')
    if (not stage['training_complete'] or stage['iteration'] != 492 or not stage['first_collection_parity']
            or set(stage['arms']) != set(ARMS)
            or any(set(records) != set(map(str, SEEDS)) for records in stage['arms'].values())):
        raise ValueError('Verified checkpoints for all six learners must be complete')
    verify_files(plan['frozen_files'])
    with tempfile.TemporaryDirectory(prefix='poker-492-pack-') as temp:
        folder = Path(temp)
        for name in ('source', 'tools'):
            shutil.copytree(CAMPAIGN/name, folder/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbi', '*.nbc'))
        scripts = folder/'scripts'
        scripts.mkdir()
        for name in ('run_poker_preflop_online.py', 'run_poker_preflop_confirmation.py',
                     'poker_preflop_specialist.py', 'snapshot_poker_preflop_online.py'):
            shutil.copyfile(ROOT/'scripts'/name, scripts/name)
        for name in ('stage-ready.json', 'plan.json'):
            shutil.copyfile(OUTPUT/name, folder/name)
        for arm in ARMS:
            for seed in SEEDS:
                row = stage['arms'][arm][str(seed)]
                checkpoint = Path(row['checkpoint'])
                verify_files({checkpoint/name: sha for name, sha in row['files'].items()})
                target = folder/arm/f'seed-{seed}'
                target.mkdir(parents=True)
                for name in row['files']:
                    shutil.copyfile(checkpoint/name, target/name)
                if row['preflop_export']:
                    ref = row['preflop_export']
                    verify_files({ref['path']: ref['sha256']})
                    shutil.copyfile(ref['path'], target/'preflop.npz')
        manifest = dict(checkpoint_mode=MODE, stage_iteration=492, arms=ARMS, seeds=SEEDS,
                        created_utc=utc(), checkpoint_paths='ARM/seed-SEED',
                        policy_paths=dict(control492='current.npz', specialist492=['preflop.npz', 'current.npz']),
                        files={str(p.relative_to(folder)): dict(bytes=p.stat().st_size, sha256=digest(p))
                               for p in folder.rglob('*') if p.is_file()})
        atomic_json(folder/'recovery-manifest.json', manifest)
        with tarfile.open(ARCHIVE, 'x:gz', compresslevel=3) as tar:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(folder)))
    verification = verify_archive(ARCHIVE, stage)
    row = dict(kind='training-checkpoint', checkpoint_mode=MODE, training_family='paired-preflop-online',
               archive=str(ARCHIVE), bytes=ARCHIVE.stat().st_size, sha256=digest(ARCHIVE),
               run='20261118', iterations={str(s): 492 for s in SEEDS}, arms=ARMS,
               new_selfplay_iterations_per_arm_seed=12, training_stage_completed=True,
               averaging_window=[361, 492], archive_includes_final_evaluation_summary=False,
               restoration_verified=True, warm_start_state_verified=True, resumable_selfplay=False,
               exact_legacy_resume=False, historical_mixture_available=False, external_backup=False,
               restore_dependencies=[dict(parent_checkpoint_sha256=plan['parent_archive_sha256'],
                                          scope='Common source and fixed 480-round baseline')],
               verification=verification)
    atomic_json(OUTPUT/'backup-ready.json', row)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    result = verify_archive(args.verify, read(OUTPUT/'stage-ready.json')) if args.verify else create_archive()
    print(json.dumps(result, indent=2, allow_nan=False))
