"""Recovery snapshots fail closed on incomplete/corrupted stopped stages."""
import fcntl
import json
from pathlib import Path
import random
import sys
import tarfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fullhouse_v3_common import digest
from snapshot_learning_curve import snapshot, validate_checkpoint, validate_current_export, verify_archive


STAMP = '20261112T070000000000Z'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data)+'\n')


def checkpoint(cfg, manifest, seed, iteration=180):
    x = np.zeros((1, 51), np.float32); x[:, 46:51] = 1
    rng = np.random.default_rng(seed)
    return dict(seed=seed, iteration=iteration, config=cfg, source_manifest=manifest,
        net={'weight': torch.ones(3)},
        buffers=[dict(features=x.copy(), advantages=np.zeros((1, 5), np.float32),
            iterations=np.array([float(iteration)], np.float32), count=1, total_seen=1) for _ in range(6)],
        root_replay=dict(features=x.copy(), numerator=np.zeros((1, 5)),
            second_numerator=np.zeros((1, 5)), mass=np.ones(1), squared_mass=np.ones(1), count=np.ones(1, np.int64)),
        rng=dict(python=random.Random(seed).getstate(), numpy=np.random.RandomState(seed).get_state(),
            generator=rng.bit_generator.state, torch=torch.Generator().manual_seed(seed).get_state()),
        snapshots=[dict(iteration=n, weight=n, file=f'policy_{n:05d}.npz', sha256='a'*64)
                   for n in range(1, iteration + 1)])


def fixture(tmp_path):
    root = tmp_path
    study = root/'artifacts/continuation-study'/STAMP; campaign = study/'campaign'/STAMP
    campaign.mkdir(parents=True)
    script = root/'scripts/train_compact_selfplay.py'; script.parent.mkdir(); script.write_text('frozen-driver\n')
    original_registry = root/'artifacts/compact-v4/current.json'; write_json(original_registry, {'original': True})
    source = campaign/'source/frozen.py'; source.parent.mkdir(); source.write_text('frozen-source\n')
    document=root/'docs/retraining-learning-curve-20261112.md';document.parent.mkdir();document.write_text('preregistered plan\n')
    cfg = dict(seeds=[342, 343, 344], iterations=240, buffer_capacity=10, method='compact-opening-expert',
               gate_run='artifacts/retraining-validation/gate')
    manifest = dict(driver_sha256=digest(script), files={'source/frozen.py': digest(source)})
    write_json(campaign/'config.json', cfg); write_json(campaign/'source-manifest.json', manifest)
    for name in ('gates.json', 'deployment-verification.json'):
        write_json(root/cfg['gate_run']/name, {'passed': True})
    for name in ('config.json', 'method-audit.json', 'preparation.json', 'stage-ready.json', 'stage-180-verification.json'):
        write_json(study/name, {'complete': True})
    plan = dict(frozen_files={str(script): digest(script), str(original_registry): digest(original_registry)},
                protected_registry_files={str(original_registry): digest(original_registry)})
    write_json(study/'plan.json', plan)
    write_json(study/'status.json', {'status': 'running', 'supervisor_pid': 999,
        'plan_sha256': digest(study/'plan.json')})
    for seed in cfg['seeds']:
        folder = campaign/f'seed-{seed}'; folder.mkdir(); (folder/'worker.lock').touch()
        ck = checkpoint(cfg, manifest, seed)
        for item in ck['snapshots']:
            path = folder/item['file']; np.savez(path, weight=np.ones(2, np.float32)); item['sha256'] = digest(path)
        np.savez(folder/'current_00180.npz', weight=np.ones(2, np.float32),
            training_revision=np.array('r1-cpu-v4'), regret_fit_method=np.array('compact-opening-expert'),
            main_scales=np.ones(2, np.float32), opening_scale=np.array(1., np.float32))
        write_json(folder/'mixture_00180.json', dict(iteration=180, model_type='compact_snapshot_mixture', models=ck['snapshots']))
        torch.save(ck, folder/'latest.pt')
        write_json(folder/'status.json', dict(status='complete', iteration=180, through=180))
        (folder/'diagnostics.jsonl').write_text(''.join(json.dumps(dict(iteration=n, seed=seed))+'\n' for n in range(1, 181)))
    return root, study, campaign, cfg, manifest


def test_snapshot_restores_state_without_original_registry_or_live_process_claim(tmp_path):
    root, study, campaign, cfg, manifest = fixture(tmp_path)
    registry = root/'artifacts/compact-v4/current.json'; before = registry.read_bytes()
    result = snapshot(study, campaign, 180, root=root)
    assert result['iterations'] == {'342': 180, '343': 180, '344': 180}
    assert result['external_backup'] is False
    assert result['resumable_selfplay'] is True
    assert result['orchestration_resume_supported'] is False
    assert registry.read_bytes() == before
    assert json.loads((study/'stage-180-backup.json').read_text()) == result
    with tarfile.open(result['archive']) as tar:
        note = json.load(tar.extractfile('checkpoint-backup-manifest.json'))
        assert len(note['resume_worker_commands']) == 3
        assert all(command[-1] == '240' for command in note['resume_worker_commands'])
        assert 'artifacts/compact-v4/current.json' not in tar.getnames()
        restored = json.load(tar.extractfile(str(study.relative_to(root)/'status.json')))
        assert restored['status'] == 'interrupted' and restored['supervisor_pid'] is None
        assert restored['process_running'] is False
        old = json.load(tar.extractfile(str(study.relative_to(root)/'snapshot-provenance/status.json')))
        assert old['supervisor_pid'] == 999
    assert verify_archive(result['archive'], note) == result['verified_files']


@pytest.mark.parametrize('corruption', ['nan_replay', 'bad_rng', 'nonfinite_net', 'missing_history', 'illegal_regret'])
def test_checkpoint_payload_corruptions_are_rejected(corruption):
    cfg = dict(buffer_capacity=10); manifest = {'frozen': True}; ck = checkpoint(cfg, manifest, 342)
    if corruption == 'nan_replay': ck['buffers'][0]['advantages'][0, 0] = float('nan')
    if corruption == 'bad_rng': del ck['rng']['torch']
    if corruption == 'nonfinite_net': ck['net']['weight'][0] = float('inf')
    if corruption == 'missing_history': ck['snapshots'].pop()
    if corruption == 'illegal_regret':
        ck['buffers'][0]['features'][0, 46] = 0
        ck['buffers'][0]['advantages'][0, 0] = 1
    with pytest.raises(ValueError): validate_checkpoint(ck, cfg, manifest, 342, 180)


def test_live_worker_lock_prevents_snapshot(tmp_path):
    root, study, campaign, _, _ = fixture(tmp_path)
    with (campaign/'seed-343/worker.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): snapshot(study, campaign, 180, root=root)
    assert not list((root/'artifacts/exports').glob('*.tar.gz'))


def test_mixed_or_incomplete_stage_cannot_be_archived(tmp_path):
    root, study, campaign, _, _ = fixture(tmp_path)
    write_json(campaign/'seed-344/status.json', dict(status='running', iteration=174))
    with pytest.raises(ValueError, match='same stage'): snapshot(study, campaign, 180, root=root)
    assert not list((root/'artifacts/exports').glob('*.tar.gz'))


def test_archive_verifier_rejects_an_unlisted_member(tmp_path):
    path = tmp_path/'extra'; path.write_bytes(b'not in manifest')
    manifest = dict(files={})
    write_json(tmp_path/'checkpoint-backup-manifest.json', manifest)
    archive = tmp_path/'bad.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(path, arcname='extra')
        tar.add(tmp_path/'checkpoint-backup-manifest.json', arcname='checkpoint-backup-manifest.json')
    with pytest.raises(ValueError, match='Unexpected'): verify_archive(archive, manifest)


def test_export_allows_declared_unicode_metadata_and_rejects_numeric_nan(tmp_path):
    values = dict(weight=np.ones(2, np.float32), training_revision=np.array('r1-cpu-v4'),
        regret_fit_method=np.array('compact-opening-expert'), main_scales=np.ones(2), opening_scale=np.array(1.))
    path=tmp_path/'current.npz';np.savez(path, **values)
    validate_current_export(path, {'method': 'compact-opening-expert'})
    values['weight'][0] = np.nan;np.savez(path, **values)
    with pytest.raises(ValueError, match='tensor'): validate_current_export(path, {'method': 'compact-opening-expert'})


def test_actual_original120_export_metadata_when_source_available():
    root=Path(__file__).resolve().parents[1]
    parent=root/'artifacts/compact-v4/20261112T032318825281Z'
    if not (parent/'config.json').exists(): pytest.skip('Private source checkpoint is not checked into Git')
    cfg=json.loads((parent/'config.json').read_text())
    for seed in cfg['seeds']:
        validate_current_export(parent/f'seed-{seed}/current_00120.npz', cfg)
