"""Finish an explicitly selected interrupted opponent-panel cell in a new directory."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import zlib

import evaluate_poker_diverse_panel as panel
from evaluate_average_baseline import _check_deadline, _verify_rows, check_dependencies, hand_seed, interval, semantic_seed, utc
from fullhouse_runtime import Bridge
from fullhouse_v3_common import alive, atomic_json, digest, process_start


def source_hashes(folder):
    if (folder/'results.json').exists():
        raise ValueError('Completed cell cannot be used as a partial source')
    status = json.loads((folder/'status.json').read_text())
    if (status.get('phase') not in ('evaluating', 'verifying_raw_hands', 'failed')
            or type(status.get('pid')) is not int or not status.get('process_start')):
        raise ValueError('Original cell lacks an identifiable interrupted evaluation')
    if alive(status['pid'], status['process_start']):
        raise RuntimeError('Original evaluator is still alive')
    return {name: digest(folder/name) for name in ('status.json', 'provenance.json', 'hands.jsonl.gz')}


def recover_lines(path, tail):
    """Read every full JSONL row emitted before a truncated deflate stream or gzip trailer."""
    decoder = zlib.decompressobj(zlib.MAX_WBITS + 16)
    pending = b''
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            decoded = decoder.decompress(chunk)
            if decoder.unused_data:
                raise ValueError('Unexpected trailing gzip member or bytes')
            *complete, pending = (pending+decoded).split(b'\n')
            for line in complete:
                yield json.loads(line)
    tail.update(partial_line=pending, complete_gzip=decoder.eof)


def check_position(row, index, start, deals):
    deal, hero = divmod(index, 6)
    if (deal >= deals or (row['deal'], row['hero'], row['seed'], row['hand_seed']) !=
            (deal, hero, deal+start, hand_seed(deal+start, hero))):
        raise ValueError('Incorrect source card, seat or hand RNG')


def replay_retained(engine, row, field, counts, deadline):
    card, hero = row['seed'], row['hero']
    seats = panel.lineup(field, card)
    state = engine.call(dict(op='reset', seed=card, button=0))
    actions = row['actions']
    if type(actions) is not list or not len(actions) == row['steps'] <= 512:
        raise ValueError('Invalid retained action sequence')
    for action in actions:
        _check_deadline(deadline)
        if state['terminal'] or action.get('seat') != state['seat_to_act']:
            raise ValueError('Retained acting seat mismatch')
        actor = state['seat_to_act']
        role = 'hero' if actor == hero else seats[(actor-hero) % 6 - 1]
        if role == 'noregrets':
            reply = engine.call(dict(op='blueprint', seed=semantic_seed(card, hero, state)))
            if dict(reply['action'], seat=actor) != action:
                raise ValueError('Retained NoRegrets action differs from frozen policy')
            counts['noregrets/decisions'] += 1
            counts['noregrets/fallback'] += int(reply.get('fallback', False))
        counts[f'{role}/{state["street"]}/{action["kind"]}'] += 1
        state = engine.call(dict(op='step', action=action))
    if (not state['terminal'] or state['utilities'] != row['utilities']
            or row['chips'] != state['utilities'][hero]):
        raise ValueError('Retained terminal payoff mismatch')


def as_line(row):
    return json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n'


def finish(original, output, label, seed, field, start, deals, timeout=7200):
    original, output = Path(original).resolve(), Path(output).resolve()
    if (original == output or original in output.parents or output in original.parents
            or label not in ('steps0','steps256') or seed not in (342,343,344) or field not in panel.FIELDS
            or not 2 <= deals <= 20000 or not 0 <= start < start+deals <= 2**32 or not 0 < timeout <= 7200):
        raise ValueError('Invalid recovery request')
    snapshot = source_hashes(original)
    prov = json.loads((original/'provenance.json').read_text())
    started = time.monotonic()
    deadline = started+timeout
    candidate, strategies, opponents, bridge_model, deps = panel.frozen_inputs(label, seed, field)
    expected = dict(label=label, training_seed=seed, field=field, deal_seed=start, deals=deals, hands=deals*6,
        dependency_hashes=deps, opponents=opponents, candidate=str(candidate.model), candidate_type=candidate.model_type,
        candidate_sha256=deps[str(candidate.model)], big_blind_chips=100, evaluator_sha256=digest(panel.__file__),
        varies_by_card=field=='varied', first_deal_clockwise_lineup=list(panel.lineup(field,start)),
        semantic_rng='SHA256 of card seed, hero seat, actor and public action log',
        statistical_unit='One original card block; hero seats and training seeds are repeated observations')
    if prov != expected:
        raise ValueError('Original provenance differs from requested frozen cell')
    output.mkdir(parents=True, exist_ok=False)
    pid = os.getpid()
    token = process_start(pid)
    def status(phase, **info):
        atomic_json(output/'status.json', dict(phase=phase, observed_utc=utc(), pid=pid, process_start=token,
                    elapsed_seconds=time.monotonic()-started, **info))
    try:
        shutil.copyfile(original/'provenance.json', output/'provenance.json')
        status('recovering_prefix')
        blocks = []
        pending = []
        tail = {}
        counts = Counter()
        raw_path = output/'hands.jsonl.gz'
        engine = Bridge(bridge_model)
        try:
            with gzip.open(raw_path, 'xt', compresslevel=1) as stream:
                for index, row in enumerate(recover_lines(original/'hands.jsonl.gz', tail)):
                    check_position(row, index, start, deals)
                    pending.append(row)
                    if len(pending) == 6:
                        for retained in pending:
                            replay_retained(engine, retained, field, counts, deadline)
                            stream.write(as_line(retained))
                        blocks.append(statistics.mean(r['chips'] for r in pending))
                        pending.clear()
                reused = len(blocks)
                partial = tail['partial_line']
                if reused == deals and (pending or partial):
                    raise ValueError('Unexpected source data after last planned hand')
                status('finishing', reused_card_blocks=reused, completed_card_blocks=reused)
                for deal in range(reused,deals):
                    card = start+deal
                    seats = panel.lineup(field,card)
                    chips = []
                    for hero in range(6):
                        row = panel.play(engine, candidate, strategies, seats, card, hero, counts, deadline)
                        row['deal'] = deal
                        line = as_line(row)
                        if deal == reused:
                            if hero < len(pending) and row != pending[hero]:
                                raise ValueError('Recovered partial deal differs from frozen-policy rerun')
                            if hero == len(pending) and not line.encode().startswith(partial):
                                raise ValueError('Trailing partial action record differs from frozen-policy rerun')
                        stream.write(line)
                        chips.append(row['chips'])
                    blocks.append(statistics.mean(chips))
                    if len(blocks) % 100 == 0:
                        status('finishing', reused_card_blocks=reused, completed_card_blocks=len(blocks))
        finally:
            engine.close()
        stats = interval(blocks)
        recovery = dict(original=str(original), source_sha256=snapshot, reused_card_blocks=reused,
                    generated_card_blocks=deals-reused, partial_deal_full_rows_checked=len(pending),
                    partial_row_bytes_checked=len(partial), source_gzip_complete=tail['complete_gzip'],
                    script_sha256=digest(__file__))
        result = dict(prov, deal_blocks=blocks, bb_per_100=stats['mean'], ci95_half_width=stats['ci95_half_width'],
                      behavior=dict(counts), recovery=recovery)
        status('verifying_raw_hands')
        verification = _verify_rows(result, raw_path, replay=True, deadline=deadline, progress=status)
        check_dependencies(deps)
        if snapshot != source_hashes(original) or digest(output/'provenance.json') != snapshot['provenance.json']:
            raise ValueError('Original changed during recovery')
        _check_deadline(deadline)
        result.update(complete=True, verification=verification, raw_hands_verified=True, terminal_payoffs_replayed=True,
                      raw_hands_sha256=digest(raw_path), provenance_sha256=snapshot['provenance.json'], completed_utc=utc(),
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(output/'results.json',result)
        status('complete', results_sha256=digest(output/'results.json'), verified_hands=deals*6)
        return result
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}', automatic_retry=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--label', required=True, choices=('steps0','steps256'))
    parser.add_argument('--field', required=True, choices=panel.FIELDS)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--deal-seed', required=True, type=int)
    parser.add_argument('--deals', required=True, type=int)
    parser.add_argument('--timeout', default=7200, type=float)
    args = parser.parse_args()
    finish(args.original,args.output,args.label,args.seed,args.field,args.deal_seed,args.deals,args.timeout)
