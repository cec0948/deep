"""Archive a completed campaign and verify every archived file by SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    cfg=json.loads((folder/'config.json').read_text());launch=json.loads((folder/'launch.json').read_text())
    evaluation=json.loads((folder/'evaluation-status.json').read_text())
    if launch['status']!='complete' or evaluation['status']!='complete' or evaluation['failed']:
        raise ValueError('Export final results only after training and evaluation have completed')
    for seed in cfg['seeds']:
        status=json.loads((folder/f'seed-{seed}/status.json').read_text())
        if status['iteration']!=cfg['iterations']:raise ValueError('Missing final checkpoint')
    extras=[ROOT/'artifacts/compact-v4/current.json',ROOT/cfg['gate_run']/'gates.json',ROOT/cfg['gate_run']/'deployment-verification.json']
    files=sorted(f for f in folder.rglob('*') if f.is_file() and not f.name.endswith('.lock'))+extras
    manifest=dict(format=1,files={str(f.relative_to(ROOT)):dict(bytes=f.stat().st_size,sha256=digest(f)) for f in files},
                  description='Three full model/replay/RNG checkpoints, every pre-update policy, fixed source/data/tools, results and gate records',
                  requirements='Repository scripts and the pinned setup dependencies. Original V3 backup is needed only for redoing old research and baseline evaluations.')
    archive=ROOT/'artifacts/exports'/f'compact-v4-{folder.name}.tar.gz';archive.parent.mkdir(exist_ok=True)
    manifest_file=archive.with_suffix('.manifest.json');atomic_json(manifest_file,manifest)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for f in files:tar.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
        tar.add(manifest_file,arcname='compact-backup-manifest.json',recursive=False)
    checked=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            if member.name not in manifest['files']:continue
            h=hashlib.sha256();stream=tar.extractfile(member)
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            if h.hexdigest()!=manifest['files'][member.name]['sha256']:raise ValueError('Backup verification failed: '+member.name)
            checked+=1
    if checked!=len(files):raise ValueError('Missing archived file')
    metadata=dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),verified_files=checked,
                  requirements=manifest['requirements'],iterations={str(s):cfg['iterations'] for s in cfg['seeds']})
    output=ROOT/'reports/compact-v4'/folder.name;output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'backup.json',metadata);atomic_json(archive.with_suffix('.json'),metadata)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
