"""Copy atomic checkpoints while training, then verify a portable archive."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from fullhouse_v3_common import ROOT, atomic_json, digest


def snapshot(folder):
    import torch
    folder = Path(folder).resolve()
    cfg = json.loads((folder/'config.json').read_text())
    manifest = json.loads((folder/'source-manifest.json').read_text())
    destination = ROOT/'artifacts/exports'; destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = destination/f'compact-v4-{folder.name}-snapshot-{stamp}.tar.gz'
    iterations = {}
    with tempfile.TemporaryDirectory(dir=destination, prefix='.snapshot-') as temporary:
        stage = Path(temporary)
        restored = stage/folder.relative_to(ROOT); restored.mkdir(parents=True)
        for name, expected in manifest['files'].items():
            if digest(folder/name) != expected:
                raise ValueError('Frozen source changed: '+name)
            target = restored/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder/name, target)
        for name in ('config.json','source-manifest.json','extension.json','supervisor-manifest.json'):
            if (folder/name).exists():shutil.copyfile(folder/name, restored/name)
        for seed in cfg['seeds']:
            source = folder/f'seed-{seed}'; target = restored/f'seed-{seed}'; target.mkdir()
            # open() keeps the old inode valid if the learner atomically replaces latest.pt.
            with (source/'latest.pt').open('rb') as reader, (target/'latest.pt').open('wb') as writer:
                shutil.copyfileobj(reader, writer, 1024*1024)
            ck = torch.load(target/'latest.pt', map_location='cpu', weights_only=False)
            if ck['seed'] != seed or ck['config'] != cfg or ck['source_manifest'] != manifest:
                raise ValueError('Copied checkpoint has incompatible provenance')
            iteration = ck['iteration']; iterations[str(seed)] = iteration
            if len(ck['snapshots']) != iteration or [s['weight'] for s in ck['snapshots']] != list(range(1,iteration+1)):
                raise ValueError('Incomplete policy history')
            for item in ck['snapshots']:
                path = source/item['file']
                if digest(path) != item['sha256']:raise ValueError('Policy history changed')
                shutil.copyfile(path, target/path.name)
            for pattern in ('current_*.npz','mixture_*.json'):
                for path in source.glob(pattern):
                    if int(path.stem.rsplit('_',1)[1]) <= iteration:
                        shutil.copyfile(path, target/path.name)
            kept = []
            for line in (source/'diagnostics.jsonl').read_text().splitlines():
                try:row = json.loads(line)
                except json.JSONDecodeError:continue
                if row['iteration'] <= iteration:kept.append(line)
            (target/'diagnostics.jsonl').write_text('\n'.join(kept)+'\n')
            atomic_json(target/'status.json', dict(iteration=iteration, through=cfg['iterations'],
                        status='complete' if iteration == cfg['iterations'] else 'interrupted'))
            (target/'worker.lock').touch()
            del ck
        gate = stage/cfg['gate_run']; gate.mkdir(parents=True,exist_ok=True)
        for name in ('gates.json','deployment-verification.json'):
            shutil.copyfile(ROOT/cfg['gate_run']/name, gate/name)
        for directory in ('scripts','configs','native','patches'):
            shutil.copytree(ROOT/directory,stage/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('THIRD_PARTY_NOTICES.md','README.md'):
            if (ROOT/name).exists():shutil.copyfile(ROOT/name,stage/name)
        atomic_json(restored/'launch.json',dict(status='interrupted',snapshot_iterations=iterations))
        atomic_json(restored/'campaign-status.json',dict(status='interrupted',reason='Restored checkpoint snapshot; no process is running'))
        atomic_json(restored.parent/'current.json',dict(folder=str(folder),status='interrupted'))
        note = dict(format=1,run=str(folder.relative_to(ROOT)),iterations=iterations,
                    restore='Extract at the repository root, install pinned dependencies, then run scripts/run_compact_retraining.py RUN --resume',
                    limitations=['Evaluation logs and the separately rebuilt NoRegrets opponent are not included in this checkpoint snapshot',
                                 'Each seed is consistent independently; their checkpoint iterations may differ'])
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        note['files'] = {str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
        atomic_json(stage/'checkpoint-backup-manifest.json', note)
        temporary_archive = archive.with_suffix('.gz.tmp')
        with tarfile.open(temporary_archive,'w:gz',compresslevel=3) as tar:
            for path in files+[stage/'checkpoint-backup-manifest.json']:
                tar.add(path,arcname=str(path.relative_to(stage)),recursive=False)
        checked = 0
        with tarfile.open(temporary_archive,'r:gz') as tar:
            for member in tar:
                expected = note['files'].get(member.name)
                if expected is None:continue
                sha = hashlib.sha256(); stream = tar.extractfile(member)
                for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
                if sha.hexdigest() != expected['sha256']:raise ValueError('Archive verification failed')
                checked += 1
        if checked != len(files):raise ValueError('Incomplete snapshot archive')
        temporary_archive.replace(archive)
    result = dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),
                  iterations=iterations,verified_files=checked,external_backup=False)
    atomic_json(archive.with_suffix('.json'), result)
    atomic_json(folder/'latest-backup.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    print(json.dumps(snapshot(args.folder),indent=2))
