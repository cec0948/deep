"""Archive both fitted street corrections and restore the selected playable route per seed."""
import argparse
from contextlib import closing
import gzip
import io
import json
import os
import subprocess
from pathlib import Path
import tarfile
import tempfile

from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import DEFAULT_BUDGET
from poker_preflop_specialist import file_ref
from poker_street_progress import StreetProgressPolicy
from publish_poker_street_progress import selected
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files, source_archives
import snapshot_poker_flop_repair as prior


NAME = 'street-progress-20261126'
FIT_COMMIT = '31cb9f80a1ba094f3e96b14168aacdd94cd74b61'
STREETS = ('flop', 'turn')
SEEDS = prior.SEEDS
REL = Path('preflop-followup/480-v1') / NAME
STUDY = ROOT / 'artifacts' / REL
ARCHIVE = ROOT / 'artifacts/exports' / (NAME + '-policies.tar.gz')
MODE = 'first-flop-and-turn-policy-bundle-v1'
TARGET = BackupTarget(STUDY, ARCHIVE, MODE)
V1 = '357c8aa6e30daa2f7c8f010e58b0b5bdf44b02f76d38d2125d2f1ff14220402b'
DEPENDENCIES = [*prior.DEPENDENCIES, dict(parent_backup_sha256=V1, scope='First-flop v1 parent and fallback')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dependencies(index):
    parents = source_archives(index)
    for sha in (V1, prior.ASSETS):
        rows = [r for r in index['backups'] if r.get('sha256') == sha and not r.get('deleted_remote')]
        require(len(rows) == 1, 'Missing or ambiguous recovery dependency')
        row = rows[0]
        require(row['verified_remote_size'] and row['verified_owner_only']
                and file_meta(Path(row['local_archive'])) == dict(bytes=row['bytes'], sha256=sha),
                'Recovery dependency does not match its local archive or verified remote receipt')
        parents[sha] = row
    return parents


def bundle(sources):
    sources.mkdir()
    design_file = STUDY / 'design.json'
    design = read(design_file)
    require(design['source_backup_sha256'] == V1, 'Wrong v1 predecessor')
    files = {'design.json': design_file}
    choices = {}
    prior_refs = {sid: rec['manifest_sha256'] for sid, rec in design['source_policies'].items()}
    for street in STREETS:
        directory = STUDY / street
        record, _ = selected(street)
        choices[street] = record['selected']
        frozen = read(directory / 'candidate.json')
        require(frozen['training_complete'] and frozen['fit_from_training_only'] and frozen['street'] == street
                and frozen['design_sha256'] == digest(design_file) and set(frozen['policies']) == set(prior_refs)
                and record['selected'] in ('parent', 'candidate'), 'Incomplete fit or selection')
        if street == 'turn':
            require(frozen['source_flop_choice_sha256'] == digest(STUDY / 'flop/selected.json'),
                    'Turn did not train on the selected flop route')
        for name in (*frozen['code_sha256'], 'snapshot_poker_flop_repair.py', 'run_poker.py',
                     'collect_poker_street_progress.py', 'score_poker_street_progress.py'):
            path = sources / name
            if not path.exists():
                path.write_bytes(subprocess.check_output(['git','show',f'{FIT_COMMIT}:scripts/{name}'], cwd=ROOT))
            files['scripts/' + name] = path
        verify_files({sources/name: sha for name, sha in frozen['code_sha256'].items()})
        files['scripts/snapshot_poker_street_progress.py'] = Path(__file__).resolve()
        for name in ('candidate.json', 'selected.json'):
            files[f'{street}/{name}'] = directory / name
        next_refs = {}
        for seed in SEEDS:
            sid = str(seed)
            folder = STUDY / 'models' / street / f'seed-{seed}'
            policy, critic = folder / 'policy.json', folder / 'critic.npz'
            fit = frozen['seeds'][sid]
            spec = read(policy)
            require(frozen['policies'][sid] == file_ref(policy) and fit['policy_sha256'] == digest(policy)
                    and fit['critic_sha256'] == digest(critic) and spec['seed'] == seed, 'Changed fitted policy')
            if street == 'flop':
                require(spec['base480']['sha256'] == design['source_policies'][sid]['base480_sha256']
                        and fit['warm_source_v1_weights_sha256'] == design['source_policies'][sid]['weights_sha256'],
                        'Flop did not descend from the registered fixed source')
            else:
                require(spec['flop_policy']['sha256'] == prior_refs[sid], 'Turn is linked to a different flop')
            require(prior.locate(spec['critic' if street == 'flop' else 'turn_critic'], policy) == critic,
                    'Candidate points outside the registered fitted critic')
            next_refs[sid] = digest(policy) if choices[street] == 'candidate' else prior_refs[sid]
            require(record['policies'][sid]['sha256'] == next_refs[sid], 'Selection points to a different policy')
            files[f'{street}/seed-{seed}/source-policy.json'] = policy
            files[f'{street}/seed-{seed}/critic.npz'] = critic
        prior_refs = next_refs
    play = read(STUDY/'play-ready.json')
    require(play['complete'] and play['selected_stage'] == record['selected_stage'], 'Final routed policy differs')
    verify_files({STUDY/'play'/f'seed-{sid}'/'policy.json': sha for sid, sha in prior_refs.items()})
    return files, choices


def restore(archive, v1, full480, specialist480, dst):
    verify_files({v1: V1})
    restore_files(archive, dst, 'restore-manifest.json', lambda name: True, 100*1024**2)
    with tarfile.open(archive, 'r:gz') as tar:
        manifest = json.load(tar.extractfile('restore-manifest.json'))
    require(manifest['checkpoint_mode'] == MODE and manifest['restore_dependencies'] == DEPENDENCIES,
            'Invalid street-progress restore manifest')
    prior.restore(v1, full480, specialist480, dst / 'flop-v1')
    choices = manifest['choices']
    require(set(choices) == set(STREETS) and all(v in ('candidate', 'parent') for v in choices.values()),
            'Unknown selected policy route')
    for seed in SEEDS:
        sub = f'seed-{seed}'
        parent = dst / 'flop-v1' / sub / 'restored-policy.json'
        parent_spec = read(parent)
        original_parent = read(parent.with_name('source-policy.json'))
        chosen = parent
        chosen_source_sha = digest(parent.with_name('source-policy.json'))
        for street in STREETS:
            folder = dst / street / sub
            original = folder / 'source-policy.json'
            spec = read(original)
            require(spec['seed'] == seed, 'Wrong restored seed')
            key = 'critic' if street == 'flop' else 'turn_critic'
            verify_files({folder/'critic.npz': spec[key]['sha256']})
            spec[key] = file_ref(folder/'critic.npz')
            if street == 'flop':
                for name in ('base480', 'feature_source'):
                    require(original_parent[name]['sha256'] == spec[name]['sha256'],
                            'Restored candidate has a different frozen parent or feature source')
                    spec[name] = parent_spec[name]
            else:
                require(spec['flop_policy']['sha256'] == chosen_source_sha, 'Turn linked to wrong restored flop')
                spec['flop_policy'] = file_ref(chosen)
            trained = folder/'restored-policy.json'
            atomic_json(trained, spec)
            if choices[street] == 'candidate':
                chosen = trained
                chosen_source_sha = digest(original)
            require(read(dst/street/'selected.json')['policies'][str(seed)]['sha256'] == chosen_source_sha,
                    'Restored route differs from the selection receipt')
        entry = dst / 'play' / sub / 'policy.json'
        entry.parent.mkdir(parents=True)
        entry.symlink_to(os.path.relpath(chosen, entry.parent))
    return manifest


def verify_decisions(restored):
    result = {}
    choices = read(STUDY/'turn/selected.json')['policies']
    for seed in SEEDS:
        sub, sid = f'seed-{seed}', str(seed)
        final = (StreetProgressPolicy(choices[sid]['path']), StreetProgressPolicy(restored/'play'/sub/'policy.json'))
        result[sid] = {}
        for street in STREETS:
            model = (StreetProgressPolicy(STUDY/'models'/street/sub/'policy.json'),
                     StreetProgressPolicy(restored/street/sub/'restored-policy.json'))
            mass = street + '_repair_mass'
            active = checked = 0
            with gzip.open(STUDY/street/'train'/sub/'positions.jsonl.gz', 'rt') as roots:
                for row in map(json.loads, roots):
                    view = row['root_observation']
                    key = semantic_seed(row['deal_seed'], row['hero'], view)
                    for live, recovered in (final, model):
                        first, second = live.decide(view, key), recovered.decide(view, key)
                        require(all(first.get(k) == second.get(k) for k in ('action','distribution',mass)),
                                'Restored probabilities or action differ from the original')
                    active += int(first.get(mass, 0.) > 0.)
                    checked += 1
                    if active >= 4 and checked >= 8:
                        break
            require(active > 0 and checked >= 8, 'Restore verification did not exercise the learned correction')
            result[sid][street] = dict(checked_source_positions=checked, active_candidate_corrections=active,
                                       selected_route_decisions_checked=checked)
        fallback_checks = dict(preflop=0, river=0)
        with closing(Bridge()) as engine:
            for deal in (103, 104):
                view = engine.call(dict(op='reset', seed=deal, button=0))
                while not view['terminal']:
                    seat = view['seat_to_act']
                    if view['street'] in fallback_checks:
                        key = semantic_seed(deal, seat, view)
                        live, recovered = [p.decide(view, key) for p in final]
                        require(all(live[k] == recovered[k] for k in ('action','distribution')),
                                'Selected fallback differs on a frozen street')
                        fallback_checks[view['street']] += 1
                    if view['street'] == 'river':
                        break
                    move = dict(seat=seat, kind='check' if view['can_check'] else 'call')
                    view = engine.call(dict(op='step', action=move))
        require(min(fallback_checks.values()) >= 2, 'Missing frozen-street restoration checks')
        result[sid]['frozen_streets'] = fallback_checks
    return result


def pack():
    parents = dependencies(read(INDEX))
    with tempfile.TemporaryDirectory(dir=ARCHIVE.parent, prefix='.street-progress-') as temp:
        work = Path(temp)
        files, choices = bundle(work/'fit-sources')
        manifest = dict(checkpoint_mode=MODE, kind='derived-policy-probe', parent_iteration=480,
                        resumable_selfplay=False, restore_dependencies=DEPENDENCIES, choices=choices,
                        fitting_code_commit=FIT_COMMIT, source_play_ready_sha256=digest(STUDY/'play-ready.json'),
                        files={name: file_meta(path) for name, path in sorted(files.items())})
        archive = ARCHIVE if ARCHIVE.exists() else work/ARCHIVE.name
        if not archive.exists():
            with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
                data = json.dumps(manifest, indent=2, allow_nan=False).encode()
                member = tarfile.TarInfo('restore-manifest.json')
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        dst = work/'restored'
        actual = restore(archive, *(Path(parents[h]['local_archive']) for h in (V1, FULL, SPECIALIST)), dst)
        require(actual == manifest, 'Archive differs from the current frozen fit or selection')
        checks = verify_decisions(dst)
        if archive != ARCHIVE:
            os.link(archive, ARCHIVE)
    receipts = {street: digest(STUDY/street/'selected.json') for street in STREETS}
    ready = dict(kind='derived-policy-probe', checkpoint_mode=MODE, run=NAME, parent_iteration=480,
                 parent_manifest_file_id=parents[FULL]['manifest_file_id'], archive=str(ARCHIVE), **file_meta(ARCHIVE),
                 restore_dependencies=DEPENDENCIES, fitting_complete=True, restoration_verified=True,
                 resumable_selfplay=False, new_selfplay_iterations=0, external_backup=False,
                 selection_receipts_sha256=receipts, archive_includes_stage_selections=True,
                 archive_includes_final_evaluation_summary=False, choices=choices, decision_restoration=checks)
    atomic_json(STUDY/'backup-ready.json', ready)
    return {k: ready[k] for k in ('bytes','sha256','choices','decision_restoration')}


def verified_ready():
    ready = read(STUDY/'backup-ready.json')
    require(ready['archive'] == str(ARCHIVE) and ready['checkpoint_mode'] == MODE and ready['fitting_complete']
            and ready['restoration_verified'] and ready['restore_dependencies'] == DEPENDENCIES
            and ready['external_backup'] is False and ready['kind'] == 'derived-policy-probe'
            and file_meta(ARCHIVE) == {k: ready[k] for k in ('bytes','sha256')}, 'Invalid backup-ready record')
    verify_files({STUDY/street/'selected.json': sha for street,sha in ready['selection_receipts_sha256'].items()})
    return ready


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('restore','pack','prepare','check','commit'))
    for name in ('archive','flop-v1','full480','specialist480','destination'):
        cli.add_argument('--' + name, type=Path)
    cli.add_argument('--sha256', help='Trusted local hash for the street-progress archive')
    cli.add_argument('--owner', default=os.environ.get('POKER_BACKUP_OWNER'))
    args = cli.parse_args()
    if args.action == 'restore':
        if not all((args.archive,args.flop_v1,args.full480,args.specialist480,args.destination)):
            cli.error('Supply all four archives and an empty destination')
        dst = args.destination.resolve()
        dst.mkdir(parents=True, exist_ok=True)
        require(not any(dst.iterdir()), 'Destination must be empty')
        if args.sha256:
            verify_files({args.archive: args.sha256})
        restore(args.archive, args.flop_v1, args.full480, args.specialist480, dst)
        result = dict(play=[str(dst/'play'/f'seed-{s}'/'policy.json') for s in SEEDS])
    elif args.action == 'pack':
        result = pack()
    elif args.action == 'prepare':
        size = sum(r['bytes'] + r.get('manifest_bytes', 0) for r in read(INDEX)['backups'] if not r.get('deleted_remote'))
        require(size + verified_ready()['bytes'] <= DEFAULT_BUDGET, 'Verified archive would exceed the 2 GiB budget')
        result = prepare(TARGET, verified_ready)
    elif args.owner:
        result = register(args.action, args.owner, TARGET, verified_ready, plan_retention)
        if args.action == 'commit':
            ready = verified_ready()
            result.update(choices=ready['choices'], decision_restoration=ready['decision_restoration'],
                          archive_includes_stage_selections=True)
            atomic_json(STUDY/'backup-verification.json', result)
            atomic_json(ROOT/'reports'/REL/'backup-verification.json', result)
    else:
        cli.error('Supply the approved Drive owner')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
