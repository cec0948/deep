"""Recover the shared river calibrator and its independently selected three playable routes."""
import argparse
import gzip
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import DEFAULT_BUDGET
from poker_preflop_specialist import file_ref
from poker_river_call_learning import FORMAT, RiverCallPolicy
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files
from snapshot_poker_street_progress import DEPENDENCIES, SEEDS, V1, dependencies, prior, require


NAME = 'river-call-learning-20261126'
REL = Path('preflop-followup/480-v1') / NAME
STUDY = ROOT/'artifacts'/REL
ARCHIVE = ROOT/'artifacts/exports'/(NAME+'-policy.tar.gz')
MODE = 'heads-up-river-call-calibration-bundle-v1'
TARGET = BackupTarget(STUDY, ARCHIVE, MODE)


def sources():
    design = read(STUDY/'design.json')
    fit = read(STUDY/'candidate.json')
    choice = read(STUDY/'selection.json')
    play = read(STUDY/'play-ready.json')
    require(choice['complete'] and choice['selected'] in ('candidate', 'parent')
            and play['complete'] and choice['design_sha256'] == digest(STUDY/'design.json')
            and play['selection_sha256'] == digest(STUDY/'selection.json')
            and fit['training_complete'] and fit['design_sha256'] == choice['design_sha256']
            and choice['candidate_sha256'] == digest(STUDY/'candidate.json')
            and (choice['selected'] == 'candidate') == (choice['paired_gain']['lower'] > 0.)
            and play['backend'] == choice['backend'] and play['policy_sha256'] == choice['policy_sha256'],
            'Missing or mismatched final policy selection')
    base = read(STUDY.parent/'street-progress-20261126/flop/selected.json')
    verify_files({STUDY.parent/'street-progress-20261126/flop/selected.json': design['flop_selection_sha256'],
                  ROOT/'scripts/poker_river_dominance.py': design['guard_sha256']})
    verify_files({ROOT/'scripts'/name: sha for name, sha in fit['code_sha256'].items()})
    verify_files({ROOT/'scripts/score_poker_river_call_learning.py': choice['scorer_sha256']})
    model = Path(fit['model']['path'])
    require(file_ref(model) == fit['model'] and read(model)['format'] == FORMAT
            and digest(model) == choice['model_sha256'] == fit['model_sha256'], 'Changed fitted model')
    files = {name:STUDY/name for name in ('design.json', 'candidate.json','selection.json','play-ready.json')}
    files['models/model.json'] = model
    for name in (*fit['code_sha256'], 'score_poker_river_call_learning.py', 'run_poker.py', 'poker_street_progress.py',
                 'serve_poker_flop_repair.py', 'snapshot_poker_river_calls.py'):
        files['scripts/'+name] = ROOT/'scripts'/name
    for seed in SEEDS:
        sid = str(seed)
        policy = Path(fit['policies'][sid]['path'])
        require(file_ref(policy) == fit['policies'][sid], 'Changed training policy')
        spec = read(policy)
        require(spec['format'] == FORMAT and spec['seed'] == seed and spec['model']['sha256'] == digest(model)
                and prior.locate(spec['model'], policy) == model
                and spec['baseline']['sha256'] == base['policies'][sid]['sha256'],
                'Different model or frozen parent')
        source = fit['policies'][sid] if choice['selected'] == 'candidate' else base['policies'][sid]
        require(choice['policies'][sid]['sha256'] == source['sha256']
                and play['policy_sha256'][sid] == source['sha256'], 'Different selected route')
        verify_files({source['path']: source['sha256']})
        files[f'models/seed-{seed}/source-policy.json'] = policy
    return files, choice


def restore(archive, v1, full480, specialist480, dst):
    verify_files({v1: V1})
    restore_files(archive, dst, 'restore-manifest.json', lambda name: True, 100*1024**2)
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile('restore-manifest.json'))
    require(manifest['checkpoint_mode'] == MODE and manifest['restore_dependencies'] == DEPENDENCIES
            and manifest['selected'] in ('candidate', 'parent'), 'Wrong recovery information')
    prior.restore(v1, full480, specialist480, dst/'flop-v1')
    model = dst/'models/model.json'
    for seed in SEEDS:
        folder = dst/'models'/f'seed-{seed}'
        source = folder/'source-policy.json'
        raw = read(source)
        parent = dst/'flop-v1'/f'seed-{seed}'/'restored-policy.json'
        verify_files({parent.with_name('source-policy.json'): raw['baseline']['sha256'], model: raw['model']['sha256']})
        require(raw['seed'] == seed and raw['format'] == FORMAT, 'Unexpected fitted seed manifest')
        fitted = folder/'restored-policy.json'
        atomic_json(fitted, dict(raw, baseline=file_ref(parent), model=file_ref(model)))
        candidate_selected = manifest['selected'] == 'candidate'
        chosen = fitted if candidate_selected else parent
        expected = digest(source) if candidate_selected else raw['baseline']['sha256']
        require(read(dst/'selection.json')['policy_sha256'][str(seed)] == expected,
                'Restored play route differs from frozen selection')
        entry = dst/'play'/f'seed-{seed}'/'policy.json'
        entry.parent.mkdir(parents=True)
        entry.symlink_to(os.path.relpath(chosen, entry.parent))
    return manifest


def check_decisions(dst):
    fit = read(STUDY/'candidate.json')
    selected = read(STUDY/'selection.json')
    results = {}
    for seed in SEEDS:
        sid, sub = str(seed), f'seed-{seed}'
        pairs = [(RiverCallPolicy(fit['policies'][sid]['path']), RiverCallPolicy(dst/'models'/sub/'restored-policy.json')),
                 (RiverCallPolicy(selected['policies'][sid]['path']), RiverCallPolicy(dst/'play'/sub/'policy.json'))]
        active = total = 0
        with gzip.open(STUDY/'train'/sub/'positions.jsonl.gz', 'rt') as roots:
            for row in map(json.loads, roots):
                view = row['root_observation']
                key = semantic_seed(row['deal_seed'], row['hero'], view)
                for number, (live, restored) in enumerate(pairs):
                    a, b = live.decide(view, key), restored.decide(view, key)
                    require(all(a.get(k) == b.get(k) for k in ('action','distribution','passive_probability_shift',
                                                                'call_over_fold_minimum_chips')),
                            'Recovered river probabilities or action differ')
                    if number == 0:
                        active += int(a.get('passive_probability_shift', 0.) > 0.)
                total += 1
                if total >= 12 and active >= 4:
                    break
        require(total >= 12 and active >= 1, 'No restored learned intervention exercised')
        results[sid] = dict(real_river_decisions=total, active_learned_updates=active,
                            selected_route_decisions=total)
    return results


def pack():
    parents = dependencies(read(INDEX))
    files, choice = sources()
    manifest = dict(checkpoint_mode=MODE, restore_dependencies=DEPENDENCIES, selected=choice['selected'],
                    selected_backend=choice['backend'], files={n:file_meta(p) for n,p in sorted(files.items())})
    with tempfile.TemporaryDirectory(dir=ARCHIVE.parent) as temp:
        stage = Path(temp)
        archive = ARCHIVE if ARCHIVE.exists() else stage/ARCHIVE.name
        if not archive.exists():
            with tarfile.open(archive, 'x:gz', compresslevel=9) as tar:
                for name, path in sorted(files.items()):
                    tar.add(path, arcname=name, recursive=False)
                data = json.dumps(manifest,allow_nan=False).encode()
                item = tarfile.TarInfo('restore-manifest.json')
                item.size = len(data)
                tar.addfile(item,io.BytesIO(data))
        dst = stage/'restored'
        actual = restore(archive, *(parents[sha]['local_archive'] for sha in (V1,FULL,SPECIALIST)), dst)
        require(actual == manifest, 'Archive differs from frozen source')
        checks = check_decisions(dst)
        if archive != ARCHIVE:
            os.link(archive, ARCHIVE)
    ready = dict(kind='derived-policy-probe', checkpoint_mode=MODE, run=NAME, parent_iteration=480,
                 parent_manifest_file_id=parents[FULL]['manifest_file_id'], archive=str(ARCHIVE), **file_meta(ARCHIVE),
                 restore_dependencies=DEPENDENCIES, fitting_complete=True, restoration_verified=True,
                 resumable_selfplay=False, new_selfplay_iterations=0, external_backup=False,
                 private_selection_sha256=digest(STUDY/'selection.json'), selected=choice['selected'],
                 selected_backend=choice['backend'], decision_restoration=checks,
                 archive_includes_final_evaluation_summary=True, archive_includes_raw_hands=False)
    atomic_json(STUDY/'backup-ready.json', ready)
    return {k:ready[k] for k in ('bytes','sha256','selected','selected_backend','decision_restoration')}


def verified_ready():
    row = read(STUDY/'backup-ready.json')
    require(row['checkpoint_mode'] == MODE and row['archive'] == str(ARCHIVE) and row['kind'] == 'derived-policy-probe'
            and row['fitting_complete'] and row['restoration_verified'] and row['restore_dependencies'] == DEPENDENCIES
            and row['private_selection_sha256'] == digest(STUDY/'selection.json')
            and file_meta(ARCHIVE) == {k:row[k] for k in ('bytes','sha256')}, 'Changed river backup')
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('restore','pack','prepare','check','commit'))
    for name in ('archive','flop-v1','full480','specialist480','destination'):
        cli.add_argument('--'+name, type=Path)
    cli.add_argument('--sha256')
    cli.add_argument('--owner', default=os.environ.get('POKER_BACKUP_OWNER'))
    args = cli.parse_args()
    if args.action == 'restore':
        if not all((args.archive,args.flop_v1,args.full480,args.specialist480,args.destination)):
            cli.error('Supply four archives and an empty destination')
        dst = args.destination.resolve()
        dst.mkdir(parents=True,exist_ok=True)
        require(not any(dst.iterdir()), 'Destination must be empty')
        if args.sha256:
            verify_files({args.archive:args.sha256})
        m = restore(args.archive,args.flop_v1,args.full480,args.specialist480,dst)
        result = dict(backend=m['selected_backend'], play=[str(dst/'play'/f'seed-{s}'/'policy.json') for s in SEEDS])
    elif args.action == 'pack':
        result = pack()
    elif args.action == 'prepare':
        active = sum(r['bytes']+r.get('manifest_bytes',0) for r in read(INDEX)['backups'] if not r.get('deleted_remote'))
        require(active+verified_ready()['bytes'] <= DEFAULT_BUDGET,'Verified backup exceeds the 2 GiB target')
        result = prepare(TARGET, verified_ready)
    elif args.owner:
        result = register(args.action,args.owner,TARGET,verified_ready,plan_retention)
        if args.action == 'commit':
            ready = verified_ready()
            # The shared registration helper reports a generic two-street probe.
            result.pop('restored_both_street_networks')
            result.update(restored_shared_river_model=True, archive_includes_final_evaluation_summary=True,
                          archive_includes_raw_hands=False, selected=ready['selected'],
                          selected_backend=ready['selected_backend'], decision_restoration=ready['decision_restoration'])
            atomic_json(STUDY/'backup-verification.json', result)
            atomic_json(ROOT/'reports'/REL/'backup-verification.json', result)
    else:
        cli.error('Supply the approved Drive owner')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
