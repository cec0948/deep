"""Package and restore the three warm-trained 492-round states."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from compact_lightweight_backup import equal_state, load_state, safe_member, validate_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy, write_policy
from poker_strategy_average import StrategyReplay
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from run_poker_warm_preflop import verify_training
from snapshot_poker_preflop_online import verify_export
from train_poker_warm_preflop import ARM, FINAL, INITIAL, OUTPUT


MODE = 'preflop-warm-start-recovery-v1'
ARCHIVE = ROOT/'artifacts/exports/poker-preflop-warm-20261119-stage-00492.tar.gz'


def files_for(stage, plan):
    files = {name:OUTPUT/name for name in ('stage-ready.json','plan.json','config.json')}
    for group in ('source','tools'):
        folder = CAMPAIGN/group
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.nbc','.nbi','.pyc'}:
                files[f'{group}/{path.relative_to(folder)}'] = path
    for name in plan['frozen_files']:
        path = Path(name)
        if path.parent in {ROOT/'scripts', ROOT/'configs'}:
            files[str(path.relative_to(ROOT))] = path
    files['scripts/snapshot_poker_warm_preflop.py'] = Path(__file__).resolve()
    for seed in SEEDS:
        entry = stage['arms'][ARM][str(seed)]
        prefix = f'{ARM}/seed-{seed}'
        for name in entry['files']:
            files[f'{prefix}/{name}'] = Path(entry['checkpoint'])/name
        files[f'{prefix}/preflop.npz'] = Path(entry['preflop_export']['path'])
        for name in ('diagnostics.jsonl','start-verification.json'):
            files[f'{prefix}/{name}'] = OUTPUT/prefix/name
    return files


def verify_archive(archive, stage):
    import ast
    with tempfile.TemporaryDirectory(prefix='warm-preflop-restore-') as tmp:
        folder = Path(tmp)
        with tarfile.open(archive,'r:gz') as tar:
            members = tar.getmembers()
            names = {m.name for m in members}
            if (len(members) != len(names) or 'recovery-manifest.json' not in names
                    or sum(m.size for m in members)>1024**3
                    or any(not m.isfile() or not safe_member(m.name) for m in members)):
                raise ValueError('Unsafe or incomplete recovery archive')
            manifest = json.load(tar.extractfile('recovery-manifest.json'))
            if manifest['checkpoint_mode'] != MODE or names != manifest['files'].keys() | {'recovery-manifest.json'}:
                raise ValueError('Recovery manifest membership differs')
            tar.extractall(folder,members=members,filter='data')
        for name, meta in manifest['files'].items():
            asset = folder/name
            if asset.stat().st_size != meta['bytes'] or digest(asset) != meta['sha256']:
                raise ValueError('Extracted member differs from manifest')
        if read(folder/'stage-ready.json') != stage:
            raise ValueError('Archived stage identity differs')
        tree = ast.parse((folder/'tools/compact_regret.py').read_text())
        mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                       and any(isinstance(t,ast.Name) and t.id=='ARRAY_MAP' for t in n.targets))
        results = {}
        engine = Bridge()
        try:
            for seed in SEEDS:
                directory = folder/ARM/f'seed-{seed}'
                ready = stage['arms'][ARM][str(seed)]
                state = load_state(directory)
                counts = validate_state(state,seed,FINAL)
                replay = StrategyReplay.from_state(state['strategy_replay'])
                if (not equal_state(state,load_state(Path(ready['checkpoint']))) or state['config']['training_arm'] != ARM
                        or counts != [100000]*6 or replay.size != 100000 or state['averaging_window'] != [361,FINAL]):
                    raise ValueError('Restored training state differs')
                verify_export(directory/'current.npz',state['net'],mapping)
                verify_export(directory/'preflop.npz',state['specialist_net'],mapping)
                with np.load(directory/'average.npz',allow_pickle=False) as average:
                    if any(not np.array_equal(v.numpy(),average[k]) for k,v in state['average_state'].items()):
                        raise ValueError('Average export differs')
                spec = write_policy(directory/'restored-policy.json',directory/'preflop.npz',directory/'current.npz')
                restored = evaluator_policy(spec,folder/'source')
                live = evaluator_policy(ready['policy']['path'],SOURCE)
                probes = {name:0 for name in ('preflop','flop','turn','river')}
                for deal in range(2037560150,2037560154):
                    node = engine.call(dict(op='reset',seed=deal,button=0))
                    while not node['terminal']:
                        key = deal+sum(probes.values())
                        if restored.decide(node,key) != live.decide(node,key):
                            raise ValueError('Restored probability or decision mismatch')
                        probes[node['street']] += 1
                        move = dict(seat=node['seat_to_act'],kind='check' if node['can_check'] else 'call')
                        node = engine.call(dict(op='step',action=move))
                if min(probes.values()) < 1:
                    raise ValueError('Restore audit missed a betting street')
                results[str(seed)] = dict(regret_rows=counts,reference_rows=replay.size,decisions=probes)
        finally:
            engine.close()
        return dict(actual_extraction_verified=True, lossless_state_verified=True, verified_files=len(manifest['files']),
                    both_street_weights_verified=True,seeds=results)


def snapshot():
    stage, plan = read(OUTPUT/'stage-ready.json'), read(OUTPUT/'plan.json')
    verify_files(plan['frozen_files'])
    checked = verify_training(plan)
    if checked['arms'] != stage['arms'] or checked['seed_verification'] != stage['seed_verification']:
        raise ValueError('Completed stage changed after verification')
    if ARCHIVE.is_file():
        verification = verify_archive(ARCHIVE,stage)
    else:
        files = files_for(stage,plan)
        manifest = dict(checkpoint_mode=MODE,iteration=FINAL,arms=[ARM],seeds=SEEDS,
            restore_policy='Per seed use preflop.npz before the flop and current.npz thereafter; both networks are in state.json and arrays.npz.',
            files={n:dict(bytes=p.stat().st_size,sha256=digest(p)) for n,p in files.items()})
        with tempfile.TemporaryDirectory(dir=ARCHIVE.parent,prefix='.warm-pack-') as tmp:
            pending = Path(tmp)/ARCHIVE.name
            with tarfile.open(pending,'x:gz',compresslevel=3) as tar:
                for name, path in sorted(files.items()):
                    tar.add(path,arcname=name,recursive=False)
                data = json.dumps(manifest,indent=2,allow_nan=False).encode()
                item = tarfile.TarInfo('recovery-manifest.json')
                item.size = len(data)
                tar.addfile(item,io.BytesIO(data))
            verification = verify_archive(pending,stage)
            os.link(pending,ARCHIVE)
    ready = dict(kind='training-checkpoint',checkpoint_mode=MODE,training_family='warm-preflop',
        run='20261119',iterations={str(s):FINAL for s in SEEDS},arms=[ARM],
        new_selfplay_iterations_per_seed=FINAL-INITIAL,training_stage_completed=True,
        averaging_window=[361,FINAL],historical_mixture_available=False,exact_legacy_resume=False,
        warm_start_state_verified=True,resumable_selfplay=False,restoration_verified=True,
        archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=digest(ARCHIVE),
        external_backup=False,archive_includes_final_evaluation_summary=False,verification=verification,
        restore_dependencies=[dict(parent_checkpoint_sha256=plan['parent_archive_sha256'],scope='480 source training state'),
                              dict(parent_backup_sha256=plan['parent_policy_bundle_sha256'],scope='Selected 480 preflop baseline')])
    atomic_json(OUTPUT/'backup-ready.json',ready)
    return ready


if __name__ == '__main__':
    print(json.dumps(snapshot(),indent=2,allow_nan=False))
