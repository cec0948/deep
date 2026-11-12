"""Reuse archived frozen-replay evidence only for exactly restored learning code.

Lost historical model/replay bytes cannot be independently rechecked. This
explicit recovery path combines the unchanged, hash-checked archived reports
with newly executed three-seed Kuhn verification. It never claims the historical
six-player fitting experiment was rerun or restores nonexistent model weights.
"""
import gzip
import hashlib
import json
from pathlib import Path

from check_learning_gates import assess
from fullhouse_v3_common import ROOT, atomic_json, digest

ARCHIVE = ROOT/'reports/learning-validation/20261110T015233258042Z'
MANIFEST = ROOT/'reports/compact-v4/20261110T085438491435Z/source-manifest.json'


def verify_recovery(cfg):
    fresh = ROOT/cfg['recovery_validation']
    if fresh != ROOT/cfg['gate_run'] or cfg.get('verification_fixture'):
        raise ValueError('Invalid recovery validation destination')
    reference = json.loads((ROOT/'configs/fullhouse-compact-training.json').read_text())
    training_keys = ['revision','method','seeds','traversals','buffer_capacity','main_steps',
                     'batch_size','opening_steps','opening_max_steps','max_opening_tv',
                     'root_repetitions_first','root_repetitions','learning_rate','checkpoint_every']
    if any(cfg[k] != reference[k] for k in training_keys):
        raise ValueError('Archived evidence cannot authorize changed learning settings')
    source = ROOT/cfg['source_run']/'source'
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest['files'].items():
        path = source/name.removeprefix('source/') if name.startswith('source/') else ROOT/'scripts'/name.removeprefix('tools/')
        if digest(path) != expected:
            raise ValueError('Restored learning code differs: '+name)
    settings = json.loads((ARCHIVE/'config.json').read_text())
    fresh_settings = json.loads((ROOT/'configs/fullhouse-retraining-validation.json').read_text())
    expected_fresh = json.loads(json.dumps(settings))
    expected_fresh['kuhn'].update(iterations=240,milestones=[12,60,120,240])
    if fresh_settings != expected_fresh:
        raise ValueError('Recovery validation changed beyond the declared uniform iteration extension')
    prior = json.loads((ARCHIVE/'gates.json').read_text())
    if digest(ARCHIVE/'config.json') != prior['config_sha256']:
        raise ValueError('Archived validation config changed')
    if digest(ROOT/'scripts/check_learning_gates.py') != prior['verifier_sha256']:
        raise ValueError('Learning gate checker changed')
    frozen = {}
    for seed, _ in settings['frozen_pairs']:
        name = f'frozen-{seed}/results.json'
        data = gzip.decompress((ARCHIVE/(name+'.gz')).read_bytes())
        if hashlib.sha256(data).hexdigest() != prior['input_hashes'][name]:
            raise ValueError('Archived fitting report changed')
        report = json.loads(data)
        if report['config'] != settings:
            raise ValueError('Archived fitting config mismatch')
        for name, expected in report['source_hashes'].items():
            if digest(source/name) != expected:
                raise ValueError('Archived fitting source mismatch: '+name)
        frozen[seed] = report
    deployment = json.loads((ARCHIVE/'deployment-verification.json').read_text())
    for field, name in [('runtime_sha256','fullhouse_v4_runtime.py'),
                        ('compact_fitter_sha256','compact_regret.py'),
                        ('partition_sha256','regret_partition.py')]:
        if digest(ROOT/'scripts'/name) != deployment[field]:
            raise ValueError('Deployment code changed: '+name)
    if not deployment['passed']:
        raise ValueError('Archived deployment gate failed')
    small = {}
    for seed in settings['kuhn']['seeds']:
        folder = fresh/f'kuhn-compact-opening-expert-{seed}'
        report = json.loads((folder/'verified-results.json').read_text())
        verification = report['verification']
        if report['config'] != fresh_settings or not report['complete'] or not report['gate']['passed']:
            raise ValueError('Fresh small-game gate failed')
        if digest(folder/'latest.pt') != verification['checkpoint_sha256'] or digest(folder/'results.json') != verification['raw_report_sha256']:
            raise ValueError('Fresh small-game evidence changed')
        if digest(ROOT/'scripts/verify_kuhn_result.py') != verification['verifier_sha256']:
            raise ValueError('Fresh verifier changed')
        for name, expected in report['source_hashes'].items():
            if digest(source/name) != expected:
                raise ValueError('Fresh small-game source mismatch')
        small[('compact-opening-expert',seed)] = report
    result = assess(fresh_settings, frozen, small)
    if cfg['method'] not in result['eligible_methods']:
        raise ValueError('Unchanged numerical learning gates failed')
    provenance = dict(
        frozen_replay_basis='Archived verified reports; historical weights and buffers were lost and were not rechecked',
        small_game_basis='Fresh three-seed full learning runs with independently verified final checkpoints',
        source_basis='All 24 restored source/tool files match the last saved campaign manifest',
        archive=str(ARCHIVE.relative_to(ROOT)), archive_gate_sha256=digest(ARCHIVE/'gates.json'),
        source_manifest_sha256=digest(MANIFEST), recovery_verifier_sha256=digest(Path(__file__)),
        fresh_verifications={str(s):digest(fresh/f'kuhn-compact-opening-expert-{s}/verified-results.json') for s in settings['kuhn']['seeds']})
    provenance['validation_extension'] = 'All original Kuhn seeds rerun through 240 after one failed at 120; unchanged pass thresholds; both runs retained'
    result['recovery_provenance'] = provenance
    deployment['recovery_provenance'] = provenance
    atomic_json(fresh/'gates.json',result)
    atomic_json(fresh/'deployment-verification.json',deployment)
    return result
