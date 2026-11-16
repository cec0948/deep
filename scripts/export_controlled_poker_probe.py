"""Self-contained policy/results bundle for a completed controlled refit study.

No replay is duplicated. The two original recoveries remain the dependencies
for repeating the fits. This bundle supports policy restoration, not selfplay
resume, and never performs Drive writes.
"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from controlled_poker_refit import ARMS, SEEDS, read, utc
from fullhouse_v3_common import ROOT, atomic_json, digest
from compact_lightweight_backup import safe_member

MODE = 'controlled-refit-policy-bundle-v1'


def verify_probe(archive):
    import numpy as np
    from rich_poker_runtime import load_descriptor, policy_for
    with tempfile.TemporaryDirectory(prefix='controlled-policy-restore-') as temp:
        folder = Path(temp)
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers(); names = [m.name for m in members]
            if (len(names) != len(set(names)) or len(names)>20000
                    or any(not m.isfile() or not safe_member(m.name) for m in members)
                    or sum(m.size for m in members)>1024**3 or names.count('restore-manifest.json')!=1):
                raise ValueError('Unsafe controlled policy bundle')
            manifest = json.load(tar.extractfile('restore-manifest.json'))
            if (manifest.get('checkpoint_mode') != MODE
                    or manifest.get('resumable_selfplay') is not False
                    or set(names) != set(manifest['files']) | {'restore-manifest.json'}
                    or set(manifest['policies']) != {f'{a}-seed-{s}' for a in ARMS for s in SEEDS}):
                raise ValueError('Incomplete or incorrect controlled policy manifest')
            for m in members:
                p = folder / m.name; p.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as source, p.open('wb') as out:
                    shutil.copyfileobj(source, out)
                if m.name != 'restore-manifest.json':
                    row = manifest['files'][m.name]
                    if p.stat().st_size != row['bytes'] or digest(p) != row['sha256']:
                        raise ValueError('Restored policy bundle bytes differ')
        for row in manifest['policies'].values():
            p = folder / row['path']
            policy = policy_for(p, folder/'source')
            if not hasattr(policy, 'decide'):
                raise ValueError('Restored candidate cannot load through the actual runtime')
            if p.suffix == '.json':
                spec = load_descriptor(p)
                if not Path(spec['model']).is_relative_to(folder) or not Path(spec['teacher']).is_relative_to(folder):
                    raise ValueError('Policy restore depends on an outside model path')
                p = Path(spec['model'])
            with np.load(p, allow_pickle=False) as model:
                for key in model.files:
                    if np.issubdtype(model[key].dtype, np.number) and not np.isfinite(model[key]).all():
                        raise ValueError('Nonfinite restored candidate')
        return dict(passed=True, policies_verified=len(ARMS)*len(SEEDS), verified_files=len(manifest['files']),
            actual_extraction_verified=True, relative_policy_dependencies_resolved=True,
            actual_runtime_load_verified=True,
            resumable_selfplay=False, replay_included=False, automatic_training_started=False)


def export_probe(study, output_root=None):
    study=Path(study).resolve(); plan=read(study/'plan.json'); summary=read(study/'summary.json')
    if not summary.get('complete') or summary.get('completed_evaluations')!=90:
        raise ValueError('Only a fully completed diagnostic can be exported')
    ready=read(study/'stage-ready.json'); output_root=Path(output_root or ROOT/'artifacts/exports')
    output_root.mkdir(parents=True,exist_ok=True)
    name='controlled-refit-'+study.name+'-policies.tar.gz'; archive=output_root/name
    if archive.exists():
        raise FileExistsError('A diagnostic export is immutable; no automatic repeat')
    with tempfile.TemporaryDirectory(prefix='controlled-policy-export-') as temp:
        bundle=Path(temp); policies={}
        for dirname in ['source','legacy-tools']:
            shutil.copytree(study/dirname,bundle/dirname,
                ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
        (bundle/'baselines').mkdir(); (bundle/'scripts').mkdir()
        for seed in SEEDS:
            p=Path(plan['parents']['legacy'][str(seed)]['model'])
            shutil.copyfile(p,bundle/'baselines'/f'parent360-seed-{seed}.npz')
        for source in plan['frozen_files']:
            p=Path(source)
            if p.parent==ROOT/'scripts':
                shutil.copyfile(p,bundle/'scripts'/p.name)
        for name in ['plan.json','config.json','summary.json','evaluation-summary.json','stage-ready.json']:
            shutil.copyfile(study/name,bundle/name)
        for arm in ARMS:
            for seed in SEEDS:
                key=f'{arm}-seed-{seed}'; original=study/'fitting'/key
                target=bundle/'fitting'/key;target.mkdir(parents=True)
                shutil.copyfile(original/'result.json',target/'result.json')
                p=Path(ready['policies'][arm][str(seed)]['path'])
                if p.suffix=='.json':
                    spec=read(p);model=Path(spec['model'])
                    shutil.copyfile(model,target/model.name)
                    spec.update(model=model.name,teacher=f'../../baselines/parent360-seed-{seed}.npz',
                                source='../../source')
                    atomic_json(target/'policy.json',spec); modelpath=target/'policy.json'
                else:
                    modelpath=target/'policy.npz';shutil.copyfile(p,modelpath)
                policies[key]=dict(path=str(modelpath.relative_to(bundle)),
                    original_policy_sha256=digest(p),archived_policy_sha256=digest(modelpath))
        files={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=digest(p))
               for p in bundle.rglob('*') if p.is_file()}
        dependencies=[dict(parent_checkpoint_sha256=sha,scope='Repeat original frozen replay fitting') for sha in
            ['65ec65f90cc1c0f73700ec4683d421baaaeb4bb82bb10c14fa2c9249fc504be8',
             '2d97f7ffff17ee4a6146e5f083e13d6aec612b47fa7ae1704ff948994f0bfaf1']]
        manifest=dict(checkpoint_mode=MODE,kind='derived-policy-probe',created_utc=utc(),
            run=study.name,policies=policies,files=files,resumable_selfplay=False,
            new_selfplay_iterations=0,replay_included=False,restore_dependencies=dependencies)
        atomic_json(bundle/'restore-manifest.json',manifest)
        with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
            for p in sorted(bundle.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(bundle)))
    verification=verify_probe(archive)
    record=dict(kind='derived-policy-probe',checkpoint_mode=MODE,run=study.name,
        archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),
        resumable_selfplay=False,new_selfplay_iterations=0,replay_included=False,
        restoration_verified=True,verification=verification,restore_dependencies=dependencies,
        study_evaluation_complete=True,external_backup=False,remote_sha256_verified=False,
        backup_family='controlled-refit')
    atomic_json(study/'backup-ready.json',record)
    return record


def prepare_upload(archive):
    from snapshot_rich_selfplay import prepare_upload as split_archive
    result=split_archive(archive)
    if 'parts' in result:
        result.update(checkpoint_mode=MODE,format='split-controlled-refit-v1')
        atomic_json(result['manifest_path'],result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path);p.add_argument('--verify',type=Path)
    a=p.parse_args();print(json.dumps(verify_probe(a.verify) if a.verify else export_probe(a.study),indent=2))
