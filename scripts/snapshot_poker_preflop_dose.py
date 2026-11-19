"""Export the fitted preflop policies and exercise each route from extracted source archives."""
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

from compact_lightweight_backup import safe_member
from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import StreetPolicy, write_policy
from run_poker_preflop_dose import LABELS, OUTPUT, verify_fit
from run_poker_preflop_pilot import SEEDS, SOURCE


MODE = 'preflop-fitting-dose-policy-bundle-v1'
ARCHIVE = ROOT / 'artifacts/exports/preflop-dose-20261119-480-policies.tar.gz'
FULL = 'f01399198fd4c3eb5fc58be569d726fcab5298995c0123dc531b108961af7f59'
SPECIALIST = '424b8e76a31f359f21de47ec02c58fd3a902ab2639f9f7e9a8785ff253d63f2e'
DEPENDENCIES = [
    dict(parent_checkpoint_sha256=FULL, scope='480-round postflop, replay, checkpoint and campaign source'),
    dict(parent_backup_sha256=SPECIALIST, scope='Confirmed 480-round preflop parent and zero-step control'),
]


def source_archives(index):
    verified = {}
    for sha in (FULL, SPECIALIST):
        records = [r for r in index['backups'] if r.get('sha256') == sha and not r.get('deleted_remote')]
        if len(records) != 1:
            raise ValueError('Required parent backup missing or ambiguous')
        record = records[0]
        path = Path(record['local_archive'])
        if (record.get('status') != 'complete' or record.get('restoration_verified') is not True
                or record.get('verified_remote_size') is not True or record.get('verified_owner_only') is not True
                or not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != sha):
            raise ValueError('Parent archive differs from its verified private index entry')
        verified[sha] = record
    return verified


def file_meta(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def study_files(fit_ready):
    seeds = {str(s) for s in SEEDS}
    if (fit_ready.get('fitting_complete') is not True or set(fit_ready['seed_evidence_sha256']) != seeds
            or set(fit_ready['policies']) != set(LABELS)
            or any(set(refs) != seeds for refs in fit_ready['policies'].values())):
        raise ValueError('Fitting must be complete for every arm and seed')
    files = {n: OUTPUT / n for n in ('fit-ready.json', 'design.json', 'config.json', 'evaluation-plan.json')}
    plan = read(files['evaluation-plan.json'])
    if plan['policies'] != fit_ready['policies'] or plan['frozen_files'].get(str(files['fit-ready.json'])) != digest(files['fit-ready.json']):
        raise ValueError('The evaluation and verified fitted profiles differ')
    verify_files(plan['frozen_files'])
    for name in ('fit_poker_preflop_dose.py', 'run_poker_preflop_dose.py', 'poker_preflop_specialist.py',
                 'snapshot_poker_preflop_online.py', 'train_poker_warm_preflop.py'):
        files[f'scripts/{name}'] = ROOT/'scripts'/name
    routes = {}
    for seed in sorted(seeds):
        directory = OUTPUT / f'seed-{seed}'
        evidence = directory / 'fit.json'
        verify_files({evidence: fit_ready['seed_evidence_sha256'][seed]})
        fit = read(evidence)
        policies = {k: fit_ready['policies'][k][seed] for k in LABELS}
        if (fit['policies'] != policies or fit['seed'] != int(seed) or fit['iteration'] != 480
                or fit['new_selfplay_iterations'] != 0 or fit['opening_unchanged'] is not True
                or fit['postflop_unchanged'] is not True or fit['scales_unchanged'] is not True):
            raise ValueError('Fitting evidence differs from verified study inputs')
        for key in ('source_plan', 'source_stage'):
            ref = fit[key]
            verify_files({ref['path']: ref['sha256']})
            name = f'provenance/{key}.json'
            source = Path(ref['path'])
            if files.setdefault(name, source) != source:
                raise ValueError('Seed fits used different source studies')
        files[f'seed-{seed}/fit.json'] = evidence
        verify_fit(int(seed), policies['steps0'], policies)
        member_routes = {}
        for label in LABELS:
            updates = int(label.removeprefix('steps'))
            record = fit['arms'][label]
            if (record['optimizer_steps'] != updates or record['training_draws'] != updates * 512
                    or (updates and record['prefix64_sha256'] != fit['arms']['steps64']['sample_order_sha256'])):
                raise ValueError('Optimizer steps or sample prefix differ from the fit plan')
            ref = policies[label]
            spec = read(ref['path'])
            preflop = f'seed-{seed}/specialist.npz' if not updates else f'seed-{seed}/{label}.npz'
            if updates:
                expected = directory / f'{label}.npz'
                if Path(spec['preflop']['path']) != expected or expected.is_symlink():
                    raise ValueError('Preflop weight escaped its fit folder')
                files[preflop] = expected
            files[f'seed-{seed}/{label}.json'] = Path(ref['path'])
            member_routes[label] = dict(
                preflop=dict(archive='dose' if updates else 'specialist480', member=preflop, sha256=spec['preflop']['sha256']),
                postflop=dict(archive='full480', member=f'seed-{seed}/current.npz', sha256=spec['postflop']['sha256']))
        routes[seed] = member_routes
    return files, routes


def restore_files(archive, destination, manifest_name, choose, limit, expected_manifest=None):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = {m.name for m in members}
        if (len(members) != len(names) or manifest_name not in names or sum(m.size for m in members) > limit
                or any(not m.isfile() or not safe_member(m.name) for m in members)):
            raise ValueError('Unsafe or incomplete archive')
        manifest = json.load(tar.extractfile(manifest_name))
        files = manifest['files']
        if names != set(files) | {manifest_name} or (expected_manifest is not None and manifest != expected_manifest):
            raise ValueError('Recovery manifest differs from source or file list')
        wanted = [m for m in members if m.name in files and choose(m.name)]
        if not wanted:
            raise ValueError('Recovery source supplied no requested members')
        tar.extractall(destination, members=wanted, filter='data')
        for m in wanted:
            if file_meta(destination / m.name) != files[m.name]:
                raise ValueError(f'Restored member differs from manifest: {m.name}')
    return len(wanted)


def verify_decisions(manifest, locations, fit_ready):
    comparisons = {}
    engine = Bridge()
    try:
        for seed, routes in manifest['policies'].items():
            comparisons[seed] = {}
            for label, route in routes.items():
                assets = {street: locations[ref['archive']] / ref['member'] for street, ref in route.items()}
                verify_files({p: route[street]['sha256'] for street, p in assets.items()})
                relocated = write_policy(locations['dose']/f'{seed}-{label}-restored.json', assets['preflop'], assets['postflop'])
                restored = StreetPolicy(relocated, locations['full480']/'source')
                original = StreetPolicy(fit_ready['policies'][label][seed]['path'], SOURCE)
                streets = dict(preflop=0, flop=0, turn=0, river=0)
                for deal in range(2037739900, 2037739904):
                    state = engine.call(dict(op='reset', seed=deal, button=0))
                    while not state['terminal']:
                        actor = state['seat_to_act']
                        key = semantic_seed(deal, actor, state)
                        a, b = original.decide(state, key), restored.decide(state, key)
                        if (a['distribution'], a['action']) != (b['distribution'], b['action']):
                            raise ValueError(f'Decision changed on restoration: {seed}/{label}')
                        streets[state['street']] += 1
                        move = dict(seat=actor, kind='check' if state['can_check'] else 'call')
                        state = engine.call(dict(op='step', action=move))
                if min(streets.values()) < 4:
                    raise ValueError('Insufficient restore coverage')
                comparisons[seed][label] = streets
    finally:
        engine.close()
    return comparisons


def snapshot():
    parents = source_archives(read(ROOT/'artifacts/offsite-backups.json'))
    fit_ready = read(OUTPUT/'fit-ready.json')
    files, routes = study_files(fit_ready)
    manifest = dict(kind='derived-policy-probe', checkpoint_mode=MODE, policies=routes,
                    run='20261119-480-preflop-dose', new_selfplay_iterations=0, resumable_selfplay=False,
                    archive_includes_final_evaluation_summary=False, restore_dependencies=DEPENDENCIES,
                    archive_labels=dict(full480=FULL, specialist480=SPECIALIST, dose='this archive'),
                    files={n: file_meta(p) for n, p in sorted(files.items())})
    with tempfile.TemporaryDirectory(dir=ARCHIVE.parent, prefix='.dose-restore-') as temp:
        root = Path(temp)
        if ARCHIVE.exists():
            archive = ARCHIVE
        else:
            archive = root / ARCHIVE.name
            with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
                data = json.dumps(manifest, indent=2, allow_nan=False).encode()
                item = tarfile.TarInfo('restore-manifest.json')
                item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
        locations = {key: root/key for key in manifest['archive_labels']}
        verified = restore_files(archive, locations['dose'], 'restore-manifest.json', lambda n: True, 100*1024**2, manifest)
        restore_files(parents[FULL]['local_archive'], locations['full480'], 'recovery-manifest.json',
                      lambda n: n.startswith('source/') or n.endswith('/current.npz'), 2*1024**3)
        restore_files(parents[SPECIALIST]['local_archive'], locations['specialist480'], 'restore-manifest.json',
                      lambda n: n.endswith('/specialist.npz'), 100*1024**2)
        decisions = verify_decisions(manifest, locations, fit_ready)
        if archive != ARCHIVE:
            os.link(archive, ARCHIVE)
    ready = dict(kind=manifest['kind'], checkpoint_mode=MODE, run=manifest['run'], archive=str(ARCHIVE),
                 **file_meta(ARCHIVE), new_selfplay_iterations=0, resumable_selfplay=False, fitting_complete=True,
                 archive_includes_final_evaluation_summary=False, fit_ready_sha256=digest(OUTPUT/'fit-ready.json'),
                 restore_dependencies=DEPENDENCIES, restoration_verified=True, external_backup=False,
                 restoration=dict(actual_extraction_verified=True, parent_extractions_verified=True,
                                  verified_payload_files=verified, decisions_by_seed_arm_street=decisions))
    atomic_json(OUTPUT/'backup-ready.json', ready)
    print(json.dumps({k: ready[k] for k in ('archive', 'bytes', 'sha256', 'restoration_verified')}))


if __name__ == '__main__':
    snapshot()
