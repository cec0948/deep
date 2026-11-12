"""Export a completed derived-policy probe with its full-checkpoint dependency.

No upload or index modification occurs here. Restore manifests contain private
Drive references and belong only in ignored artifacts and private backups.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from fullhouse_v3_common import ROOT, atomic_json, digest


def export(run):
    run = Path(run).resolve()
    if run.parent != ROOT / 'artifacts/refit-probe':
        raise ValueError('Expected an existing local refit-probe run')
    status = json.loads((run / 'status.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    if not (status['status'] == 'complete' and summary['complete']
            and summary['raw_hands_verified'] and summary['source_provenance_verified']):
        raise ValueError('Only fully verified completed probes can be exported')
    index = json.loads((ROOT / 'artifacts/offsite-backups.json').read_text())
    parent = Path(summary['parent']).name
    def verified(row):
        return (not row.get('deleted_remote') and row.get('verified_remote_size')
                and row.get('verified_owner_only'))
    base = next(b for b in reversed(index['backups'])
                if b['kind'] == 'training-checkpoint' and b.get('run') == parent
                and set(b.get('iterations', {}).values()) == {120} and verified(b))
    assets = next(b for b in index['backups'] if b['kind'] == 'evaluation-assets' and verified(b))
    plan = json.loads((run / 'plan.json').read_text())
    files = {p for p in run.rglob('*') if p.is_file() and p.suffix != '.lock'}
    for name, expected in plan['frozen_files'].items():
        p = Path(name)
        if p.parent in (ROOT / 'scripts', ROOT / 'configs'):
            if digest(p) != expected:
                raise ValueError('Frozen experiment source changed: ' + name)
            files.add(p)
    for name in ('scripts/export_refit_probe.py', 'docs/retraining-improvement-20261112.md'):
        files.add(ROOT / name)
    files = sorted(files)
    manifest = dict(format='derived-policy-probe-backup-v1', run=run.name,
        resumable_selfplay=False, parent_full_checkpoint=base, evaluation_assets=assets,
        restore='Restore the parent full120 checkpoint and evaluation assets, then extract this archive at the repository root. Candidate NPZ/PT files are derived policies, not resumable self-play checkpoints. Read the saved config, provenance and evaluation summary before use.',
        files={str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in files})
    exports = ROOT / 'artifacts/exports'; exports.mkdir(exist_ok=True)
    archive = exports / f'refit-probe-{run.name}.tar.gz'
    manifest_path = exports / f'refit-probe-{run.name}.manifest.json'
    if archive.exists():
        raise FileExistsError('Existing export is immutable: ' + str(archive))
    atomic_json(manifest_path, manifest)
    temporary = archive.with_suffix('.partial')
    with tarfile.open(temporary, 'w:gz', compresslevel=3) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        tar.add(manifest_path, arcname='probe-backup-manifest.json', recursive=False)
    checked = set()
    with tarfile.open(temporary, 'r:gz') as tar:
        for member in tar:
            if member.name == 'probe-backup-manifest.json':
                if json.load(tar.extractfile(member)) != manifest:
                    raise ValueError('Restore manifest mismatch')
                continue
            expected = manifest['files'][member.name]
            if member.name in checked or not member.isfile():
                raise ValueError('Unexpected archive member')
            stream = tar.extractfile(member); h = hashlib.sha256(); size = 0
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block); size += len(block)
            if h.hexdigest() != expected['sha256'] or size != expected['bytes']:
                raise ValueError('Archive member mismatch: ' + member.name)
            checked.add(member.name)
    if checked != set(manifest['files']):
        raise ValueError('Archive is incomplete')
    temporary.replace(archive)
    result = dict(kind='derived-policy-probe', run=run.name, parent_run=parent,
        resumable_selfplay=False, archive=str(archive), bytes=archive.stat().st_size,
        sha256=digest(archive), verified_files=len(checked), folder_id=index['folder_id'],
        parent_manifest_file_id=base['manifest_file_id'])
    atomic_json(exports / f'refit-probe-{run.name}.backup.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    print(json.dumps(export(parser.parse_args().run)))
