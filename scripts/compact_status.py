"""Read-only campaign status without loading Torch or checkpoint buffers."""
import argparse
import json
from pathlib import Path

from fullhouse_v3_common import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',nargs='?',type=Path);a=p.parse_args()
    folder=a.folder or Path(json.loads((ROOT/'artifacts/compact-v4/current.json').read_text())['folder'])
    cfg=json.loads((folder/'config.json').read_text());launch=json.loads((folder/'launch.json').read_text())
    rows=[]
    for seed in cfg['seeds']:
        path=folder/f'seed-{seed}';status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
        latest={};log=path/'diagnostics.jsonl'
        if log.exists():
            for line in reversed(log.read_text().splitlines()):
                try:latest=json.loads(line);break
                except json.JSONDecodeError:continue
        pid=launch.get('pids',{}).get(str(seed));alive=False
        if pid:
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                alive='train_compact_selfplay.py' in command and str(folder) in command
            except (FileNotFoundError,PermissionError):pass
        rows.append(dict(seed=seed,alive=alive,latest_iteration=latest.get('iteration',0),checkpoint_iteration=status.get('iteration',0),
                         root=latest.get('root'),quality=latest.get('quality'),last_iteration_seconds=latest.get('seconds')))
    print(json.dumps(dict(folder=str(folder),launch=launch.get('status'),planned_iterations=cfg['iterations'],seeds=rows),indent=2))


if __name__=='__main__':main()
