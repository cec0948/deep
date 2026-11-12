"""Create one private, verified archive for a completed foundation study."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from fullhouse_v3_common import ROOT, atomic_json, digest


def export(run):
    run = Path(run).resolve()
    if run.parent != ROOT/'artifacts/foundation-study':
        raise ValueError('Expected local foundation-study output')
    status = json.loads((run/'status.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    if not (status['status']=='complete' and summary['complete']
            and summary['source_provenance_verified']
            and summary['benchmarks']['saved_policy_metrics_recomputed']
            and summary['reference_evaluation']['raw_hands_verified']):
        raise ValueError('Study is not fully verified complete')
    index=json.loads((ROOT/'artifacts/offsite-backups.json').read_text())
    def verified(b):
        return (not b.get('deleted_remote') and b.get('verified_remote_size') is True
                and b.get('verified_owner_only') is True)
    dependencies=[]
    for iteration in (60,120):
        parent=Path(summary['config']['parent'+str(iteration)]).name
        entry=next(b for b in reversed(index['backups']) if b.get('kind')=='training-checkpoint'
            and b.get('run')==parent and set(b.get('iterations',{}).values())=={iteration} and verified(b))
        dependencies.append(entry)
    assets=next(b for b in index['backups'] if b['kind']=='evaluation-assets' and verified(b))
    plan=json.loads((run/'plan.json').read_text())
    if digest(run/'plan.json') != summary['plan_sha256']:
        raise ValueError('Study plan changed')
    for name,expected in plan['frozen_files'].items():
        if digest(name)!=expected:
            raise ValueError('Study frozen input changed: '+name)
    files={p for p in run.rglob('*') if p.is_file() and p.suffix!='.lock'}
    for name in plan['frozen_files']:
        p=Path(name)
        if p.parent in (ROOT/'scripts',ROOT/'configs'):
            files.add(p)
    for relative in ('scripts/export_foundation_study.py','scripts/plot_foundation_study.py',
                     'scripts/audit_poker_information.py','configs/poker-foundation-requirements.txt',
                     'docs/retraining-foundations-20261112.md',
                     'reports/retraining-analysis-20261112/information-audit.json'):
        files.add(ROOT/relative)
    public=ROOT/'reports/foundation-study'/run.name
    if public.exists():
        files.update(p for p in public.rglob('*') if p.is_file())
    files=sorted(files)
    manifest=dict(format='research-validation-backup-v1',run=run.name,resumable_selfplay=False,
        source_checkpoints=dependencies,evaluation_assets=assets,
        restore='Restore both original60 and120 full checkpoints and evaluation assets; extract this archive at the repository root and install the pinned validation requirements. Leduc models are two-player reference artifacts, not six-player training checkpoints. Read the saved plans and limitations.',
        files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files})
    exports=ROOT/'artifacts/exports'; archive=exports/f'foundation-study-{run.name}.tar.gz'
    if archive.exists():
        raise FileExistsError('Existing research export is immutable')
    manifest_path=exports/f'foundation-study-{run.name}.manifest.json';atomic_json(manifest_path,manifest)
    temporary=archive.with_suffix('.partial')
    with tarfile.open(temporary,'w:gz',compresslevel=3) as tar:
        for p in files:
            tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        tar.add(manifest_path,arcname='research-backup-manifest.json',recursive=False)
    checked=set()
    with tarfile.open(temporary,'r:gz') as tar:
        for member in tar:
            if member.name=='research-backup-manifest.json':
                if json.load(tar.extractfile(member)) != manifest:
                    raise ValueError('Restore manifest mismatch')
                continue
            expected=manifest['files'][member.name]
            if member.name in checked or not member.isfile():
                raise ValueError('Unexpected archive member')
            stream=tar.extractfile(member);h=hashlib.sha256();size=0
            for block in iter(lambda:stream.read(1024*1024),b''):
                h.update(block);size+=len(block)
            if h.hexdigest()!=expected['sha256'] or size!=expected['bytes']:
                raise ValueError('Archive member mismatch: '+member.name)
            checked.add(member.name)
    if checked!=set(manifest['files']):
        raise ValueError('Research archive is incomplete')
    temporary.replace(archive)
    record=dict(kind='research-validation',run=run.name,resumable_selfplay=False,
        local_archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),
        verified_files=len(checked),folder_id=index['folder_id'],
        restore_dependencies=[dict(parent_manifest_file_id=b['manifest_file_id']) for b in dependencies])
    atomic_json(exports/f'foundation-study-{run.name}.backup.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    print(json.dumps(export(p.parse_args().run)))
