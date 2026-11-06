"""Fetch the reviewed source and validate source/artifacts without copying weights."""
import json
from pathlib import Path
import subprocess
from fullhouse_runtime import validate_source

if __name__ == '__main__':
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/fullhouse-model.json').read_text())
    source=root/'.cache/fullhouse-bot'
    if not source.exists():
        subprocess.run(['git','clone','--no-checkout',cfg['repository'],str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',cfg['commit']],check=True)
    validate_source()
    print(source)
