"""Check all four streets via the normal JSONL command for the selected seed models."""
from contextlib import closing
import json
from pathlib import Path
import subprocess
import sys

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_river_runtime import action_key
from poker_street_progress import StreetProgressPolicy
from score_poker_street_progress import PUBLIC, SEEDS, STUDY, require


def requests(seed):
    history = [dict(seat=3, kind='call'), *(dict(seat=n, kind='fold') for n in (4, 5, 0, 1)),
               dict(seat=2, kind='check')]
    checks = [dict(seat=n, kind='check') for n in (2, 3)]
    board = ['2c', '7d', 'Jh', '9c', 'Qs']
    fixture = [('preflop', [], [], 3), ('flop', board[:3], history, 2),
               ('turn', board[:4], history + checks, 2), ('river', board, history + checks * 2, 2)]
    return {street: dict(hero=hero, button=0, hole=['As', 'Kd'], board=cards,
                         actions=actions, seed=seed) for street, cards, actions, hero in fixture}


def check_seed(path, seed, engine):
    checks = requests(seed)
    lines = [json.dumps({'unsupported_field': True}), *(json.dumps(req) for req in checks.values())]
    proc = subprocess.run([sys.executable, str(ROOT/'scripts/run_poker.py'), '--backend', 'street-progress',
                           '--policy', str(path)], input='\n'.join(lines) + '\n', capture_output=True,
                          text=True, check=True, timeout=90)
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    require(len(replies) == 5 and replies[0].get('ok') is False, 'Invalid-input handling failed')
    model = StreetProgressPolicy(path)
    passed = []
    for (street, req), reply in zip(checks.items(), replies[1:], strict=True):
        state = engine.call(dict(req, op='replay'))
        require(state['street'] == street and state['seat_to_act'] == req['hero'], 'Invalid protocol fixture')
        _validate_reply(state, reply)
        direct = model.decide(state, req['seed'])
        require(reply['backend'] == 'street-progress' and reply['policy_manifest_sha256'] == model.digest
                and reply['action'] == direct['action'] and reply['distribution'] == direct['distribution'],
                'JSONL action or distribution differs from selected weights')
        probs = {action_key(e['action']): e['probability'] for e in reply['distribution']}
        require(len(probs) == len(reply['distribution']) and probs.get(action_key(reply['action']), 0.) > 0.,
                'JSONL action must have positive probability in the unique concrete menu')
        for (kind, size) in probs:
            if kind == 'raise':
                require(state['raise_bounds'] and state['raise_bounds'][0] <= size <= state['raise_bounds'][1],
                        'Illegal concrete bet size')
            else:
                require(size is None and kind in ({'check'} if state['can_check'] else {'fold', 'call'}),
                        'Illegal check, fold or call')
        passed.append(street)
    return dict(valid_streets=passed, malformed_request_rejected=True, policy_sha256=model.digest)


def smoke():
    ready = read(STUDY/'play-ready.json')
    selection = read(STUDY/'turn/selected.json')
    require(ready['complete'] and selection['complete'] and ready['turn']['selection_sha256']
            == digest(STUDY/'turn/selected.json'), 'Missing published local policy selection')
    seeds = {}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            sid = str(seed)
            path = STUDY/'play'/f'seed-{seed}'/'policy.json'
            require(path.resolve() == Path(selection['policies'][sid]['path']).resolve(),
                    'Entrypoint points away from selected model')
            verify_files({path: ready['policy_sha256'][sid]})
            seeds[sid] = check_seed(path, seed, engine)
    receipt = dict(complete=True, completed_utc=utc(), selected_stage=ready['selected_stage'],
                   play_ready_sha256=digest(STUDY/'play-ready.json'), smoke_script_sha256=digest(__file__),
                   seeds=seeds, valid_decisions=4*len(SEEDS), malformed_requests=len(SEEDS), match_games=0)
    for dest in (STUDY, PUBLIC):
        atomic_json(dest/'jsonl-smoke.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(smoke()))
