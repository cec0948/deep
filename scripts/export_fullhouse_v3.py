"""Consistent per-seed checkpoint bundle, including evaluation dependencies."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from fullhouse_v3_common import ROOT,digest,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='?',type=Path);a=p.parse_args()
    folder=(a.folder or Path(json.loads((ROOT/'artifacts/fullhouse-v3/current.json').read_text())['folder'])).resolve()
    state=json.loads((folder/'status.json').read_text());cfg=state['config']
    sys.path.insert(0,str(folder/'source'))
    import torch
    from training.network import AverageStrategyNet,DuelingAdvantageNet
    from training.train import _export_strategy_npz,_export_advantage_npz
    torch.set_num_threads(1)
    exports=ROOT/'artifacts/exports';exports.mkdir(exist_ok=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=exports/f'fullhouse-v3-{stamp}.tar.gz'
    with tempfile.TemporaryDirectory(prefix='fullhouse-v3-export-',dir=ROOT/'artifacts') as temp:
        staging=Path(temp);target=staging/folder.relative_to(ROOT);target.mkdir(parents=True)
        shutil.copytree(folder/'source',target/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('config.json','verification-0.json','verification-1.json'):
            shutil.copyfile(folder/name,target/name)
        snapshots={}
        for seed in cfg['seeds']:
            origin=folder/f'seed-{seed}';dest=target/f'seed-{seed}';dest.mkdir()
            # One open inode survives atomic replacement by a live trainer.
            with (origin/'latest.pt').open('rb') as source,(dest/'latest.pt').open('wb') as output:
                shutil.copyfileobj(source,output,1024*1024)
            ck=torch.load(dest/'latest.pt',weights_only=False,map_location='cpu')
            if ck['r1_config']['revision']!='r1-cpu-v3' or ck['r1_config']['seed']!=seed:raise ValueError('Checkpoint identity')
            iteration=ck['iteration'];snapshots[str(seed)]=iteration
            for old in origin.glob('iter_*.*'):
                if int(old.stem.split('_')[1])<iteration:shutil.copyfile(old,dest/old.name)
            advdest=target/f'seed-{seed}_adv';advdest.mkdir()
            for old in (folder/f'seed-{seed}_adv').glob('iter_*.npz'):
                if int(old.stem.split('_')[1])<iteration:shutil.copyfile(old,advdest/old.name)
            # Regenerate the most recent npz rather than race its non-atomic write.
            net=AverageStrategyNet();net.load_state_dict(ck['strategy_net']);_export_strategy_npz(net,dest/f'iter_{iteration:05d}.npz')
            adv=DuelingAdvantageNet();adv.load_state_dict(ck['net']);_export_advantage_npz(adv,advdest/f'iter_{iteration:05d}.npz')
            torch.save({k:v for k,v in ck.items() if k not in ('buffers','strategy_buffer')},dest/f'iter_{iteration:05d}.pt')
            for name in ('diagnostics.jsonl','strategy-audit.jsonl'):
                rows=[]
                for line in (origin/name).read_text().splitlines():
                    try:
                        if json.loads(line)['iteration']<=iteration:rows.append(line)
                    except json.JSONDecodeError:pass
                (dest/name).write_text('\n'.join(rows)+'\n')
            state['seeds'][str(seed)]['iteration']=iteration
            state['seeds'][str(seed)].pop('progress',None)
        for name,expected in state['external_hashes'].items():
            source=ROOT/name
            if digest(source)!=expected:raise ValueError('External dependency changed: '+name)
            dest=staging/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        for name in ('configs/fullhouse-model.json','configs/fullhouse-training-requirements.txt','configs/fullhouse-inference-requirements.txt','THIRD_PARTY_NOTICES.md'):
            dest=staging/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        for name in state['source_hashes']:
            if digest(staging/folder.relative_to(ROOT)/name)!=state['source_hashes'][name]:raise ValueError('Source snapshot changed')
        completed=state['status']=='complete' and all(n==cfg['through'] for n in snapshots.values())
        state.update(supervisor_pid=None,supervisor_start=None,active={},error=None)
        if completed:
            shutil.copytree(folder/'evaluation',target/'evaluation')
            atomic_json(target/'results.json',state)
        else:
            state.update(status='interrupted',phase='training',evaluations={},paired={})
        atomic_json(target/'status.json',state)
        pointer=staging/'artifacts/fullhouse-v3/current.json';atomic_json(pointer,{'folder':str(folder)})
        manifest={str(p.relative_to(staging)):digest(p) for p in staging.rglob('*') if p.is_file() and not p.is_symlink()}
        atomic_json(staging/'fullhouse-v3-bundle-manifest.json',manifest)
        with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
            for path in sorted(staging.iterdir()):tar.add(path,arcname=path.name)
    # Verify every archived regular file after compression.
    with tarfile.open(archive,'r:gz') as tar:
        import hashlib
        verified=set()
        for member in tar.getmembers():
            if member.isfile() and member.name in manifest:
                stream=tar.extractfile(member);h=hashlib.sha256()
                for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
                if h.hexdigest()!=manifest[member.name]:raise ValueError('Archive hash mismatch: '+member.name)
                if member.name.endswith('/poker-bridge') and not member.mode & 0o111:
                    raise ValueError('Archived engine lost its executable permission')
                verified.add(member.name)
        if verified!=set(manifest):raise ValueError('Archive is missing manifest files')
    meta=dict(archive=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=digest(archive),seeds=snapshots,
              regular_files_verified=len(manifest),run=str(folder.relative_to(ROOT)),source_revision=cfg['revision'],completed=completed,
              restore='Fresh checkout of this project at /workspace/deep; prepare pinned Fullhouse upstream and Python dependencies first. Extract at project root, then status. Resume only an incomplete campaign. Binary compatibility: Debian 13 x86_64/glibc 2.41. A completed bundle retains evaluation results; an incomplete bundle resets evaluations.',
              limitation='Archive is local until downloaded. No full NoRegrets training checkpoint; its evaluation blueprint is included.')
    atomic_json(archive.with_suffix(archive.suffix+'.json'),meta)
    atomic_json(ROOT/'reports/fullhouse-v3-export.json',meta)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
