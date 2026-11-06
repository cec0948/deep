#!/usr/bin/env python3
"""Probe the pinned public Deep CFR model at a six-player 100BB opening node.

This uses Fullhouse's native state format, not this project's JSONL API.
It calls the learned policy directly so inference errors cannot silently turn
into the upstream bot's fallback or short-stack heuristic.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/fullhouse-bot'
COMMIT = 'e504793d480b1b975f25258d25939b45c6dbd5a4'
FILES = {
    'deep_cfr_model.npz': '1102326b68da95564de147106612df71cb891b42f0726ba0212d3b9a5bcae295',
    'preflop_equity.npz': '6b3b74778854fcebcd3179753c0252392e5f4118e6b623375c98fafa372863b3',
}


def main():
    head = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True).strip()
    if head != COMMIT or dirty:
        raise SystemExit('Fullhouse source differs from the reviewed version')
    for name, expected in FILES.items():
        if hashlib.sha256((SOURCE / 'data' / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Artifact integrity mismatch: {name}')
    os.environ['BOT_DATA_DIR'] = str(SOURCE / 'data')
    sys.path.insert(0, str(SOURCE))
    import numpy as np
    from bot.deep_cfr_lookup import DeepCFRLookup

    state = {
        'your_cards': ['As', 'Kd'], 'community_cards': [],
        'dealer': 0, 'seat_to_act': 3, 'street': 'preflop',
        'pot': 150, 'your_stack': 10000, 'your_bet_this_street': 0,
        'amount_owed': 100, 'current_bet': 100, 'min_raise_to': 200,
        'can_check': False, 'hand_num': 1,
        'players': [
            {'seat': i, 'stack': 10000 - bet, 'bet_this_street': bet,
             'is_folded': False, 'is_all_in': False}
            for i, bet in enumerate([0, 50, 100, 0, 0, 0])
        ],
        'action_log': [
            {'seat': 1, 'action': 'small_blind', 'amount': 50},
            {'seat': 2, 'action': 'big_blind', 'amount': 100},
        ],
    }
    policy = DeepCFRLookup(SOURCE / 'data/deep_cfr_model.npz')
    probs = policy.get_strategy(state)
    if probs.shape != (5,) or not np.isfinite(probs).all() or (probs < 0).any() or not np.isclose(probs.sum(), 1):
        raise SystemExit('Invalid policy distribution')
    print(json.dumps({
        'source_commit': COMMIT, 'backend': 'fullhouse',
        'model_type': 'advantage', 'strength_verified': False,
        'state': state,
        'abstract_action_probabilities': dict(zip(
            ['fold', 'check_call', 'half_pot', 'pot', 'all_in'], map(float, probs))),
    }, indent=2))


if __name__ == '__main__':
    main()
