"""Create a local resume bundle without stopping or modifying live training.

Use the training venv. The bundle is deliberately outside Git and includes
locally trained models, the frozen training source, and evaluation dependencies.
Restore into /workspace/deep after installing the pinned Python environment.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def export(folder,output):
    import torch
    if ROOT!=Path('/workspace/deep'):
        raise ValueError('This resume format currently expects /workspace/deep')
    relative=folder.relative_to(ROOT)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix='medium-export-',dir=output.parent) as tmp:
        stage=Path(tmp);target=stage/relative
        (target/'training').mkdir(parents=True)
        # Opening once retains the old inode if atomic_save replaces latest.pt.
        with (folder/'training/latest.pt').open('rb') as src,(target/'training/latest.pt').open('wb') as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        ck=torch.load(target/'training/latest.pt',map_location='cpu',weights_only=False)
        iteration=int(ck['iteration'])
        assert ck['r1_config']['horizon']==600 and len(ck['buffers'])==6 and 'r1_rng' in ck
        state=json.loads((folder/'status.json').read_text())
        state.update(status='interrupted',phase='training',supervisor_pid=None,supervisor_process_start=None,
                     child_pid=None,child_process_start=None,completed_training_iteration=iteration,error=None)
        state.pop('active_command',None);state.pop('progress',None)
        state['training_stages']=[r for r in state['training_stages'] if r['checkpoint']['iteration']<=iteration]
        state['evaluations']=[]
        state['exported_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (target/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        shutil.copyfile(folder/'config.json',target/'config.json')
        shutil.copytree(folder/'source',target/'source',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        # Publish only closed checkpoint exports at or before the copied latest.pt.
        for path in (folder/'training').glob('iter_*'):
            if path.suffix in ('.pt','.npz') and int(path.stem.split('_')[1])<=iteration:
                shutil.copy2(path,target/'training'/path.name)
        good=[]
        for line in (folder/'training/diagnostics.jsonl').read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if row['iteration']<=iteration:good.append(line)
        (target/'training/diagnostics.jsonl').write_text('\n'.join(good)+'\n')
        pointer=stage/'artifacts/fullhouse-medium/current.json'
        pointer.write_text(json.dumps({'folder':str(folder)},indent=2)+'\n')
        entry=json.loads((ROOT/'artifacts/selfplay/current.json').read_text())
        deps=[Path('artifacts/selfplay/current.json'),Path(entry['model']),Path(state['config']['baseline_model']),
              Path('.cache/noregrets/target/release/poker-bridge'),Path('.cache/noregrets/target/release/poker-json'),
              Path('artifacts/noregrets-smoke/run-NK70RShx/resumed.bin'),Path('.cache/noregrets/LICENSE')]
        for rel in deps:
            dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        for rel,expected in state['source_hashes'].items():assert digest(target/rel)==expected
        for rel,expected in state['external_hashes'].items():
            p=stage/rel
            assert digest(p if p.exists() else ROOT/rel)==expected
        assert digest(stage/entry['model'])==entry['sha256']
        manifest=dict(format_version=1,root=str(ROOT),run=str(relative),iteration=iteration,horizon=600,
                      exported_utc=state['exported_utc'],files=[])
        for p in sorted(stage.rglob('*')):
            if p.is_file() and not p.is_symlink():
                manifest['files'].append(dict(path=str(p.relative_to(stage)),bytes=p.stat().st_size,sha256=digest(p)))
        (stage/'resume-bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        try:
            with tarfile.open(output,'w:gz',compresslevel=1,dereference=False) as archive:
                for p in sorted(stage.iterdir()):archive.add(p,arcname=p.name)
        except BaseException:
            output.unlink(missing_ok=True);raise
    summary=dict(archive=str(output.relative_to(ROOT)),bytes=output.stat().st_size,sha256=digest(output),
                 checkpoint_iteration=iteration,horizon=600,exported_utc=manifest['exported_utc'],
                 includes_frozen_training_source=True,includes_evaluation_dependencies=True)
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=(args.run or Path(json.loads((ROOT/'artifacts/fullhouse-medium/current.json').read_text())['folder'])).resolve()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output or ROOT/'artifacts/exports'/f'fullhouse-medium-{stamp}.tar.gz'
    export(folder,output.resolve())

if __name__=='__main__':main()
