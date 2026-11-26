#!/usr/bin/env python3
"""Start the current local research model's JSONL process."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import argparse
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['noregrets','fullhouse','flop-repair'],default='noregrets')
parser.add_argument('--policy',type=Path,help='Private flop-repair policy manifest')
args=parser.parse_args()
if (args.backend=='flop-repair') != (args.policy is not None):
    parser.error('--policy is required with --backend flop-repair and only supported by that backend')
if args.backend in ('fullhouse','flop-repair'):
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    python=ROOT.parent/'.venvs/fullhouse/bin/python'
    if args.backend=='flop-repair':
        os.execv(str(python),[str(python),str(ROOT/'scripts/serve_poker_flop_repair.py'),str(args.policy)])
    os.execv(str(python),[str(python),str(ROOT/'scripts/fullhouse_runtime.py')])
registry=ROOT/'artifacts/selfplay/current.json'
if not registry.exists():raise SystemExit('No current model registry; see docs/selfplay-training.md')
entry=json.loads(registry.read_text())
model=ROOT/entry['model']
h=hashlib.sha256()
with model.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
if h.hexdigest()!=entry['sha256']:raise SystemExit('Current model integrity mismatch')
binary=ROOT/'.cache/noregrets/target/release/poker-json'
os.execv(str(binary),[str(binary),str(model)])
