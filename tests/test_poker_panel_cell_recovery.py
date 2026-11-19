"""Replay-equivalence checks for an interrupted frozen-policy panel cell."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import zlib

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_poker_diverse_panel import DOSE_PLAN, run
from finish_poker_panel_cell import finish


JOB = ('steps0', 342, 'tight-value', 95, 2)


@pytest.fixture(scope='module')
def original_cell(tmp_path_factory):
    if not DOSE_PLAN.exists():
        pytest.skip('The frozen specialist is available only on the evaluation runner')
    label, seed, field, start, deals = JOB
    full = tmp_path_factory.mktemp('reference') / 'cell'
    run(label, seed, field, start, deals, full)
    return full


def make_interrupted(full, directory, raw):
    directory.mkdir()
    shutil.copyfile(full / 'provenance.json', directory / 'provenance.json')
    status = json.loads((full / 'status.json').read_text())
    status.pop('results_sha256')
    status.pop('verified_hands')
    status.update(phase='evaluating', pid=100_000_000, process_start='1', completed_card_blocks=0)
    (directory / 'status.json').write_text(json.dumps(status))
    (directory / 'hands.jsonl.gz').write_bytes(raw)


def snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('interruption', ('partial-line', 'partial-gzip'))
def test_resume_uses_complete_card_blocks_and_reproduces_every_action(original_cell, tmp_path, interruption):
    full_rows = gzip.decompress((original_cell / 'hands.jsonl.gz').read_bytes()).splitlines(keepends=True)
    assert len(full_rows) == 12
    if interruption == 'partial-line':
        partial = gzip.compress(b''.join(full_rows[:6]) + full_rows[6][:15])
    else:
        compressor = zlib.compressobj(wbits=16+zlib.MAX_WBITS)
        partial = compressor.compress(b''.join(full_rows[:8]) + full_rows[8][:15])
        partial += compressor.flush(zlib.Z_SYNC_FLUSH)
        with pytest.raises(EOFError):
            gzip.decompress(partial)
    source, recovered = tmp_path / 'interrupted', tmp_path / 'recovered'
    make_interrupted(original_cell, source, partial)
    before = snapshot(source)
    finish(source, recovered, *JOB)
    assert snapshot(source) == before
    restored_rows = gzip.decompress((recovered / 'hands.jsonl.gz').read_bytes()).splitlines()
    assert list(map(json.loads, restored_rows)) == list(map(json.loads, full_rows))
    expected = json.loads((original_cell / 'results.json').read_text())
    result = json.loads((recovered / 'results.json').read_text())
    for key in ('deal_blocks', 'behavior', 'verification', 'candidate_sha256', 'field', 'hands'):
        assert result[key] == expected[key]
    assert result['recovery']['reused_card_blocks'] == 1
    assert result['recovery']['generated_card_blocks'] == 1
    assert result['complete'] and result['raw_hands_verified'] and result['terminal_payoffs_replayed']


@pytest.mark.parametrize('corruption', ('duplicate-seat', 'forged-zero-sum'))
def test_resume_rejects_invalid_complete_card_prefix(original_cell, tmp_path, corruption):
    rows = list(map(json.loads, gzip.decompress((original_cell / 'hands.jsonl.gz').read_bytes()).splitlines()[:6]))
    if corruption == 'duplicate-seat':
        rows[5] = rows[4]
    else:
        # Leave hero's result and conservation intact. Only trusted action replay reveals the change.
        rows[0]['utilities'][1] += 100
        rows[0]['utilities'][2] -= 100
    data = b''.join((json.dumps(row, separators=(',', ':'))+'\n').encode() for row in rows)
    source, recovered = tmp_path / 'bad-prefix', tmp_path / 'recovered'
    make_interrupted(original_cell, source, gzip.compress(data))
    before = snapshot(source)
    with pytest.raises(ValueError):
        finish(source, recovered, *JOB)
    assert not (recovered / 'results.json').exists()
    assert snapshot(source) == before
