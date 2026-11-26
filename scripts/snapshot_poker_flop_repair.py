"""Back up a learned flop repair, and restore it using its hash-identified 480 parents."""
import argparse
from contextlib import closing
import gzip
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import DEFAULT_BUDGET
from poker_preflop_specialist import file_ref, write_policy
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files, source_archives


NAME = 'flop-repair-20261126'
STUDY = ROOT/'artifacts/preflop-followup/480-v1'/NAME
ARCHIVE = ROOT/'artifacts/exports'/(NAME+'-policies.tar.gz')
MODE = 'selfplay-flop-repair-policy-bundle-v1'
TARGET = BackupTarget(STUDY, ARCHIVE, MODE)
ASSETS = '3f9bab3f5419937b67e10e2464caa37f68d2dc90df344214f7d41c77953ee1ba'
SEEDS = (342, 343, 344)
DEPENDENCIES = [
    dict(parent_checkpoint_sha256=FULL, scope='Frozen 480 postflop weights and source/bot/features.py'),
    dict(parent_backup_sha256=SPECIALIST, scope='Frozen specialist480 preflop weights'),
    dict(parent_backup_sha256=ASSETS, scope='Standard engine bridge and runtime equity assets'),
]


def locate(ref, manifest):
    path = (manifest.parent/ref['path']).resolve()
    verify_files({path: ref['sha256']})
    return path


def bundle_files():
    design = read(STUDY/'training-design.json')['seed_sources']
    summary = read(STUDY/'fit-complete.json')
    design_sha = digest(STUDY/'training-design.json')
    if (not summary['training_complete'] or summary['training_design_sha256'] != design_sha
            or summary['kind'] != 'derived-policy-probe' or set(summary['seeds']) != set(map(str,SEEDS))
            or summary['performance_matches'] != 0):
        raise ValueError('Incomplete final training receipt')
    files = {name: STUDY/name for name in ('training-design.json','fit-complete.json',
              'action-label-audit.json','policy-integrity-audit.json','jsonl-smoke.json')}
    scripts = ('poker_flop_repair.py', 'train_poker_flop_repair.py', 'fit_poker_flop_repair.py',
               'serve_poker_flop_repair.py', 'run_poker.py', 'snapshot_poker_flop_repair.py')
    for name in scripts:
        files['scripts/'+name] = ROOT/'scripts'/name
    routes = {}
    for seed in SEEDS:
        sid = str(seed)
        subdir = f'seed-{seed}'
        folder = STUDY/subdir
        path = folder/'policy.json'
        spec = read(path)
        parent = read(locate(spec['base480'], path))
        critic = locate(spec['critic'], path)
        encoder = (folder/spec['feature_source']['path']/'bot/features.py').resolve()
        verify_files({encoder: spec['feature_source']['sha256']})
        source = design[sid]
        trained = read(folder/'training-receipt.json')
        if (trained != summary['seeds'][sid] or not trained['complete'] or trained['seed'] != seed
                or trained['training_design_sha256'] != design_sha or trained['policy_sha256'] != digest(path)
                or trained['critic_sha256'] != digest(critic) or trained['parent_sha256'] != spec['base480']['sha256']):
            raise ValueError('Training receipt or fitted weights differ')
        if (spec['seed'] != seed or spec['format'] != 'selfplay-flop-action-repair-v1'
                or spec['base480']['sha256'] != source['parent_policy_sha256'] or critic != folder/'critic.npz'
                or any(parent[street]['sha256'] != source[street+'_sha256'] for street in ('preflop','postflop'))):
            raise ValueError('Fitted policy references a different parent, seed or critic')
        files[f'{subdir}/source-policy.json'] = path
        for name in ('training-receipt.json', 'collection-complete.json', 'critic.npz'):
            files[f'{subdir}/{name}'] = folder/name
        routes[sid] = {k: source[k+'_sha256'] for k in ('preflop','postflop')}
    return files, routes


def restore(archive, full, specialist, dst):
    verify_files({full: FULL, specialist: SPECIALIST})
    restore_files(archive, dst, 'restore-manifest.json', lambda n: True, 100*1024**2)
    with tarfile.open(archive, 'r:gz') as package:
        manifest = json.load(package.extractfile('restore-manifest.json'))
    if manifest['checkpoint_mode'] != MODE or manifest['restore_dependencies'] != DEPENDENCIES:
        raise ValueError('Invalid recovery manifest')
    restore_files(full, dst/'full480', 'recovery-manifest.json',
                  lambda n: n.startswith('source/') or n.endswith('/current.npz'), 2*1024**3)
    restore_files(specialist, dst/'specialist480', 'restore-manifest.json',
                  lambda n: n.endswith('/specialist.npz'), 100*1024**2)
    for seed in SEEDS:
        folder = dst/f'seed-{seed}'
        weights = {k: dst/label/f'seed-{seed}'/name for k,label,name in
                   [('preflop','specialist480','specialist.npz'),('postflop','full480','current.npz')]}
        route = manifest['parent_routes'][str(seed)]
        verify_files({p: route[k] for k,p in weights.items()})
        spec = read(folder/'source-policy.json')
        source = dst/'full480/source'
        critic = folder/'critic.npz'
        verify_files({source/'bot/features.py': spec['feature_source']['sha256'], critic: spec['critic']['sha256']})
        parent = write_policy(folder/'bound-parent.json', weights['preflop'], weights['postflop'])
        spec.update(base480=file_ref(parent), critic=file_ref(critic),
                    feature_source=dict(path=str(source), sha256=spec['feature_source']['sha256']))
        atomic_json(folder/'restored-policy.json', spec)
    return manifest


def verify_decisions(restored):
    from poker_flop_repair import FlopRepairPolicy
    coverage = {}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            folder = f'seed-{seed}'
            from_fit = FlopRepairPolicy(STUDY/folder/'policy.json')
            from_tar = FlopRepairPolicy(restored/folder/'restored-policy.json')
            streets = dict(preflop=0, flop=0, turn=0, river=0)
            for deal in range(103, 107):
                view = engine.call(dict(op='reset', seed=deal, button=0))
                while not view['terminal']:
                    key = semantic_seed(deal, view['seat_to_act'], view)
                    live, recovered = from_fit.decide(view, key), from_tar.decide(view, key)
                    if any(live[k] != recovered[k] for k in ('action','distribution')):
                        raise ValueError('Restored action or probability distribution differs')
                    streets[view['street']] += 1
                    if view['street'] == 'river':
                        break
                    action = dict(seat=view['seat_to_act'], kind='check' if view['can_check'] else 'call')
                    view = engine.call(dict(op='step', action=action))
            if min(streets.values()) < 4:
                raise ValueError('Insufficient restore action probes')
            active = 0
            with gzip.open(STUDY/folder/'positions.jsonl.gz', 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    view = row['root_observation']
                    key = semantic_seed(row['deal_seed'], view['seat_to_act'], view)
                    live = from_fit.decide(view, key)
                    if live.get('flop_repair_mass', 0.) <= 0:
                        continue
                    recovered = from_tar.decide(view, key)
                    if any(live[k] != recovered[k] for k in ('action','distribution','flop_repair_mass')):
                        raise ValueError('Restored learned correction does not match its source output')
                    active += 1
                    if active == 4:
                        break
            if active == 0:
                raise ValueError('No functional restore probe exercised a learned probability update')
            coverage[str(seed)] = dict(streets=streets, active_corrections=active)
    return coverage


def pack():
    index = read(INDEX)
    parents = source_archives(index)
    assets = [r for r in index['backups'] if r.get('sha256') == ASSETS and not r.get('deleted_remote')]
    if (len(assets) != 1 or not assets[0]['verified_remote_size'] or not assets[0]['verified_owner_only']
            or digest(assets[0]['local_archive']) != ASSETS):
        raise ValueError('Unverified evaluation/runtime dependency')
    files, routes = bundle_files()
    manifest = dict(kind='derived-policy-probe', checkpoint_mode=MODE, restore_dependencies=DEPENDENCIES,
                    parent_routes=routes, parent_iteration=480, resumable_selfplay=False,
                    files={name: file_meta(path) for name,path in sorted(files.items())})
    with tempfile.TemporaryDirectory(dir=ARCHIVE.parent, prefix='.repair-') as tmp:
        where = Path(tmp)
        archive = ARCHIVE if ARCHIVE.exists() else where/ARCHIVE.name
        if not archive.exists():
            with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
                for name,path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
                raw = json.dumps(manifest, indent=2, allow_nan=False).encode()
                member = tarfile.TarInfo('restore-manifest.json')
                member.size = len(raw)
                tar.addfile(member, io.BytesIO(raw))
        restored = where/'restored'
        parsed = restore(archive, parents[FULL]['local_archive'], parents[SPECIALIST]['local_archive'], restored)
        if parsed != manifest:
            raise ValueError('Existing archive source or metadata changed')
        decisions = verify_decisions(restored)
        if archive != ARCHIVE:
            os.link(archive, ARCHIVE)
    ready = dict(kind='derived-policy-probe', checkpoint_mode=MODE, run=NAME, parent_iteration=480,
                 parent_manifest_file_id=parents[FULL]['manifest_file_id'], archive=str(ARCHIVE), **file_meta(ARCHIVE),
                 restore_dependencies=DEPENDENCIES, restoration_verified=True, fitting_complete=True,
                 resumable_selfplay=False, new_selfplay_iterations=0, external_backup=False,
                 fit_complete_sha256=digest(STUDY/'fit-complete.json'), archive_includes_final_evaluation_summary=False,
                 restoration=dict(actual_extraction=True, active_decision_probes=decisions, source_archives_opened=2))
    atomic_json(STUDY/'backup-ready.json', ready)
    return {k:ready[k] for k in ('bytes','sha256','restoration_verified')}


def load_ready():
    row = read(STUDY/'backup-ready.json')
    if (row['kind'] != 'derived-policy-probe' or row['checkpoint_mode'] != MODE or row['parent_iteration'] != 480
            or row['restore_dependencies'] != DEPENDENCIES or not row['restoration_verified'] or not row['fitting_complete']
            or row['resumable_selfplay'] is not False or row['archive'] != str(ARCHIVE)
            or file_meta(ARCHIVE) != {k:row[k] for k in ('bytes','sha256')}
            or row['fit_complete_sha256'] != digest(STUDY/'fit-complete.json')):
        raise ValueError('Incomplete or changed derived policy')
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('restore','pack','prepare','check','commit'))
    parser.add_argument('--owner', default=os.environ.get('POKER_BACKUP_OWNER'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--full480', type=Path)
    parser.add_argument('--specialist480', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.action == 'restore':
        if not all((args.archive,args.full480,args.specialist480,args.destination)):
            parser.error('Supply the bundle, full480 and specialist480 archives and an empty destination')
        dest = args.destination.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        if any(dest.iterdir()):
            parser.error('Destination must be empty')
        restore(args.archive,args.full480,args.specialist480,dest)
        result = dict(restored_seed_manifests=[str(dest/f'seed-{s}/restored-policy.json') for s in SEEDS])
    elif args.action == 'pack':
        result = pack()
    elif args.action == 'prepare':
        size = sum(r['bytes']+r.get('manifest_bytes',0) for r in read(INDEX)['backups'] if not r.get('deleted_remote'))
        if size + load_ready()['bytes'] > DEFAULT_BUDGET:
            raise ValueError('Verified new file would exceed the reserved Drive budget')
        result = prepare(TARGET, load_ready)
    elif args.owner:
        result = register(args.action,args.owner,TARGET,load_ready,plan_retention)
    else:
        parser.error('Supply the approved Drive owner')
    print(json.dumps(result, indent=2))
