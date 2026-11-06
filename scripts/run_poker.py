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
parser.add_argument('--backend',choices=['noregrets','fullhouse'],default='noregrets')
args=parser.parse_args()
if args.backend=='fullhouse':
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    python=ROOT.parent/'.venvs/fullhouse/bin/python'
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
