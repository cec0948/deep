"""One fixed-size, verified latest recovery for the window-average study.

Contains numeric regret/strategy replay, latest current+average weights, all RNG
state and frozen code. No historical network sequence or raw evaluation logs.
Exact worker resume is not implemented/claimed. The old360parent stays safe.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np

from compact_lightweight_backup import load_state,validate_state,safe_member,equal_state
from fullhouse_v3_common import ROOT,atomic_json,digest
from controlled_poker_refit import utc,read,verify_files
from train_poker_average_extension import CHECKPOINT_FORMAT,load_checkpoint
from poker_strategy_average import StrategyReplay
from poker_average_inference import load_average,average_strategy

MODE=CHECKPOINT_FORMAT
SEEDS=[342,343,344]


def verify_checkpoint(path,seed):
    state=load_state(path);validate_state(state,seed,480)
    if state.get('checkpoint_format')!=MODE or state.get('averaging_window')!=[361,480] or state.get('average_state') is None:
        raise ValueError('Incomplete final current+average state')
    replay=StrategyReplay.from_state(state['strategy_replay'])
    if replay.capacity!=100000 or state['config']['buffer_capacity']!=100000:
        raise ValueError('The preserved fixed replay capacity changed')
    tree=ast.parse((ROOT/'scripts/compact_regret.py').read_text())
    mapping=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='ARRAY_MAP' for t in n.targets))
    with np.load(Path(path)/'current.npz',allow_pickle=False) as arrays:
        for key,export in mapping.items():
            for prefix,label in [('main.',''),('opening.','opening_')]:
                if not np.array_equal(state['net'][prefix+key].numpy(),arrays[label+export]):
                    raise ValueError('Current export differs from trained state')
        for name in ('main_scales','opening_scale'):
            if not np.array_equal(state['net'][name].numpy(),arrays[name]):raise ValueError('Regret units changed')
    layers,meta=load_average(Path(path)/'average.npz')
    if meta.get('averaging_window')!=[361,480]:raise ValueError('Average window changed')
    with np.load(Path(path)/'average.npz',allow_pickle=False) as arrays:
        for name,value in state['average_state'].items():
            if not np.array_equal(value.numpy(),arrays[name]):raise ValueError('Average export differs from learned state')
    # Restoring the reservoir includes the NEXT replacement draw, not just rows.
    other=StrategyReplay.from_state(state['strategy_replay'])
    next_equal=np.array_equal(replay.rng.integers(2**32,size=32),other.rng.integers(2**32,size=32))
    if not next_equal:raise ValueError('Restored strategy reservoir RNG changed')
    tested=0
    for x in other.x[:min(256,other.size)]:
        p=average_strategy(layers,x,x[46:51])
        if not np.isfinite(p).all() or not np.isclose(p.sum(),1.) or np.any(p[x[46:51]==0]):
            raise ValueError('Restored average inference failed')
        tested+=1
    return dict(seed=seed,iteration=480,regret_rows=[b['count'] for b in state['buffers']],
        strategy_rows=other.size,strategy_capacity=other.capacity,current_and_average_array_parity=True,
        strategy_reservoir_next_rng_parity=True,average_runtime_probes=tested,
        average_approximation_tv=state['average_fit']['weighted_replay_tv'],
        average_within_original_quality_reference=state['average_fit']['weighted_replay_tv']<=.15,
        all_four_legacy_rng_states_present=set(state['rng'])=={'python','numpy','generator','torch'},
        reference_rng_present=bool(state.get('reference_rng')),averaging_window=[361,480])


def verify_archive(path):
    with tempfile.TemporaryDirectory(prefix='average-recovery-restore-') as temp:
        folder=Path(temp)
        with tarfile.open(path,'r:gz') as tar:
            members=tar.getmembers();names=[m.name for m in members]
            if (len(names)!=len(set(names)) or len(names)>20000 or sum(m.size for m in members)>2*1024**3
                    or any(not m.isfile() or not safe_member(m.name) for m in members)
                    or names.count('recovery-manifest.json')!=1):
                raise ValueError('Unsafe average recovery archive')
            manifest=json.load(tar.extractfile('recovery-manifest.json'))
            if (manifest.get('checkpoint_mode')!=MODE or manifest.get('averaging_window')!=[361,480]
                    or set(names)!=set(manifest['files'])|{'recovery-manifest.json'}):
                raise ValueError('Average recovery manifest mismatch')
            for m in members:
                target=folder/m.name;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
                if m.name!='recovery-manifest.json':
                    r=manifest['files'][m.name]
                    if target.stat().st_size!=r['bytes'] or digest(target)!=r['sha256']:
                        raise ValueError('Extracted recovery size/hash mismatch')
        plan=read(folder/'plan.json')
        if set(plan['policies'])!={'original120','parent420'}:
            raise ValueError('Both baseline comparison policies must be preserved')
        for label,policies in plan['policies'].items():
            if set(policies)!=set(map(str,SEEDS)):raise ValueError('Missing comparison seed')
            for seed,policy in policies.items():
                path=folder/'comparisons'/seed/(label+'.npz')
                if not path.is_file() or digest(path)!=policy['sha256']:
                    raise ValueError('Restored parent comparison differs from the frozen policy')
        rows=[verify_checkpoint(folder/f'seed-{seed}',seed) for seed in SEEDS]
        return dict(passed=True,actual_extraction_verified=True,verified_files=len(manifest['files']),seeds=rows,
            warm_start_state_verified=True,resumable_selfplay=False,exact_legacy_resume=False,
            historical_mixture_available=False,average_policy_available=True,averaging_window=[361,480],
            parent_comparison_policies_in_archive=True)


def create_archive(study):
    study=Path(study);plan=read(study/'plan.json');verify_files(plan['frozen_files'])
    ready=read(study/'stage-ready.json')
    if not ready.get('complete') or ready.get('stage')!=480 or set(ready['seeds'])!=set(map(str,SEEDS)):
        raise ValueError('All3final480 checkpoints must finish first')
    output=ROOT/'artifacts/exports';output.mkdir(exist_ok=True)
    archive=output/f'poker-average-{study.name}-stage-00480.tar.gz'
    if archive.exists():raise FileExistsError('Immutable recovery archive already exists')
    campaign=Path(plan['campaign'])
    with tempfile.TemporaryDirectory(prefix='average-recovery-pack-') as temp:
        folder=Path(temp)
        for name in ('source','tools'):
            shutil.copytree(campaign/name,folder/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
        (folder/'scripts').mkdir()
        for name in ('poker_average_runtime.py','poker_average_inference.py','snapshot_poker_average_extension.py',
                'fullhouse_runtime.py','fullhouse_v4_runtime.py','fullhouse_v3_runtime.py','rich_poker_runtime.py',
                'rich_poker_observation.py','poker_public_state.py','regret_partition.py','evaluate_average_baseline.py'):
            shutil.copyfile(ROOT/'scripts'/name,folder/'scripts'/name)
        for seed in SEEDS:
            _,checkpoint=load_checkpoint(campaign/f'seed-{seed}')
            if str(checkpoint)!=ready['seeds'][str(seed)]['checkpoint']:raise ValueError('Snapshot stage moved')
            for name,sha in ready['seeds'][str(seed)]['files'].items():
                if digest(checkpoint/name)!=sha:raise ValueError('Completed checkpoint changed before snapshot')
            shutil.copytree(checkpoint,folder/f'seed-{seed}')
            verify_checkpoint(folder/f'seed-{seed}',seed)
        # Preserve the exact parent comparison weights without requiring its
        # whole replay archive in the restore chain. The newest replay is complete.
        for label,policies in plan['policies'].items():
            for seed,policy in policies.items():
                target=folder/'comparisons'/seed/(label+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
                if digest(policy['path'])!=policy['sha256']:raise ValueError('Parent comparison changed')
                shutil.copyfile(policy['path'],target)
        for name in ('plan.json','stage-ready.json'):shutil.copyfile(study/name,folder/name)
        shutil.copyfile(campaign/'config.json',folder/'config.json')
        manifest=dict(checkpoint_mode=MODE,run=study.name,kind='training-checkpoint',created_utc=utc(),
            iterations={str(s):480 for s in SEEDS},averaging_window=[361,480],
            resumable_selfplay=False,exact_legacy_resume=False,warm_start_state_verified=True,
            average_policy_available=True,historical_mixture_available=False,
            archive_includes_final_evaluation=False,
            average_quality_is_diagnostic=True,
            files={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in folder.rglob('*') if p.is_file()})
        atomic_json(folder/'recovery-manifest.json',manifest)
        with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
            for p in sorted(folder.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(folder)))
    verification=verify_archive(archive)
    row=dict(kind='training-checkpoint',checkpoint_mode=MODE,training_family='legacy-window-average',run=study.name,
        archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),
        iterations={str(s):480 for s in SEEDS},averaging_window=[361,480],
        new_selfplay_iterations_per_seed=plan['new_selfplay_iterations_per_seed'],resumable_selfplay=False,exact_legacy_resume=False,
        historical_mixture_available=False,average_policy_available=True,warm_start_state_verified=True,
        restoration_verified=True,training_stage_completed=True,archive_includes_final_evaluation=False,
        average_quality_is_diagnostic=True,parent_comparison_policies_in_archive=True,
        replaces_archive_sha256=plan['config']['source_recovery_sha256'],
        average_approximation_reference_passed=all(r['average_within_original_quality_reference'] for r in verification['seeds']),
        verification=verification,external_backup=False,remote_sha256_verified=False,
        restore_dependencies=[dict(parent_checkpoint_sha256=plan['config']['original360_recovery_sha256'],
            scope='Preserve original360baseline and inherited pre-window history')])
    atomic_json(study/'backup-ready.json',row)
    return row


def prepare_upload(archive):
    from snapshot_rich_selfplay import prepare_upload as split
    return split(archive)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path);p.add_argument('--verify',type=Path)
    a=p.parse_args();print(json.dumps(verify_archive(a.verify) if a.verify else create_archive(a.study),indent=2))
