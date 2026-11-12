"""Freeze a compact CPU campaign's engine, small data and training helpers."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from fullhouse_v3_common import ROOT,atomic_json,digest

TOOLS=['compact_regret.py','regret_fitting.py','scaled_regret_fitting.py','regret_partition.py','root_regret_replay.py','train_compact_selfplay.py']


def prepare(folder,cfg):
    folder=Path(folder)
    if not cfg.get('verification_fixture'):
        if cfg.get('recovery_validation'):
            from verify_compact_recovery import verify_recovery
            verify_recovery(cfg)
        else:
            subprocess.run([sys.executable,str(ROOT/'scripts/check_learning_gates.py'),str(ROOT/cfg['gate_run']),'--require-six-player'],check=True,stdout=subprocess.DEVNULL)
        verified=json.loads((ROOT/cfg['gate_run']/'deployment-verification.json').read_text())
        if not verified['passed']:raise ValueError('Deployment verification failed')
    folder.mkdir(parents=True,exist_ok=False)
    atomic_json(folder/'config.json',cfg)
    original=ROOT/cfg['source_run']/'source'
    for name in ('training','bot'):
        shutil.copytree(original/name,folder/'source'/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
    (folder/'source/data').mkdir();shutil.copyfile(original/'data/preflop_equity.npz',folder/'source/data/preflop_equity.npz')
    if (original/'LICENSE.upstream').exists():shutil.copyfile(original/'LICENSE.upstream',folder/'source/LICENSE.upstream')
    (folder/'tools').mkdir()
    for name in TOOLS:shutil.copyfile(ROOT/'scripts'/name,folder/'tools'/name)
    files={str(path.relative_to(folder)):digest(path) for parent in ('source','tools') for path in sorted((folder/parent).rglob('*')) if path.is_file()}
    manifest=dict(files=files,driver_sha256=digest(ROOT/'scripts/train_compact_selfplay.py'))
    atomic_json(folder/'source-manifest.json',manifest)
    return folder


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/fullhouse-compact-training.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(prepare(a.output,json.loads(a.config.read_text())))


if __name__=='__main__':main()
