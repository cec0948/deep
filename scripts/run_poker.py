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
policy_servers = {'flop-repair':'serve_poker_flop_repair.py', 'street-progress':'poker_street_progress.py',
                  'street-progress-river':'poker_river_dominance.py',
                  'river-call-learning':'poker_river_call_learning.py',
                  'river-call-continuation':'poker_river_call_diverse.py',
                  'river-call-recalibration':'poker_river_call_recalibrated.py'}
parser.add_argument('--backend',choices=['noregrets','fullhouse',*policy_servers],default='noregrets')
parser.add_argument('--policy',type=Path,help='Private policy manifest for the selected trained policy')
args=parser.parse_args()
if (args.backend in policy_servers) != (args.policy is not None):
    parser.error('--policy is required for these backends only: ' + ', '.join(policy_servers))
if args.backend == 'fullhouse' or args.backend in policy_servers:
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    python=ROOT.parent/'.venvs/fullhouse/bin/python'
    if args.backend in policy_servers:
        os.execv(str(python),[str(python),str(ROOT/'scripts'/policy_servers[args.backend]),str(args.policy)])
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
