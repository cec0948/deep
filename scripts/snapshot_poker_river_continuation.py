"""Archive the continued river policy and reattach it to its verified learned predecessor."""
import argparse
import gzip
import json
import os
from pathlib import Path
import tarfile

from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import DEFAULT_BUDGET
from poker_preflop_specialist import file_ref
from poker_river_call_diverse import DiverseRiverCallPolicy, FORMAT
from poker_river_call_learning import RiverCallPolicy
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files
import snapshot_poker_river_calls as predecessor
from snapshot_poker_scheduled_window import pack_and_restore
from snapshot_poker_street_progress import SEEDS, V1, dependencies, require


NAME = 'river-call-continuation-20261127'
REL = Path('preflop-followup/480-v1') / NAME
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
ARCHIVE = ROOT/'artifacts/exports'/(NAME+'-policy.tar.gz')
MODE = 'continued-heads-up-river-call-bundle-v1'
PARENT = 'dbf8ed7308e2ad7152daf3a8779babddaad2424d0c9f98b06ef2e5bf736efca3'
PARENTS = [*predecessor.DEPENDENCIES, dict(parent_backup_sha256=PARENT, scope='Selected learned river policy')]
TARGET = BackupTarget(STUDY, ARCHIVE, MODE)


def dependency_paths(index):
    parents = dependencies(index)
    matches = [r for r in index['backups'] if r.get('sha256') == PARENT and not r.get('deleted_remote')]
    require(len(matches) == 1, 'Expected one active predecessor')
    row = matches[0]
    require(row['status'] == 'complete' and row['restoration_verified'] and row['verified_owner_only']
            and row['verified_remote_size'] and row['selected'] == 'candidate'
            and file_meta(Path(row['local_archive'])) == {'bytes': row['bytes'], 'sha256': PARENT},
            'Predecessor unavailable or different from the verified river backup')
    parents[PARENT] = row
    return parents


def inputs():
    fit = read(STUDY/'candidate.json')
    result = read(STUDY/'selection.json')
    play = read(STUDY/'play-ready.json')
    base = read(predecessor.STUDY/'selection.json')
    chosen = result['selected'] == 'candidate'
    require(result['complete'] and result['selected'] in ('candidate','parent') and play['complete']
            and fit['training_complete'] and fit['design_sha256'] == result['design_sha256'] == digest(STUDY/'design.json')
            and result['candidate_sha256'] == digest(STUDY/'candidate.json')
            and play['selection_sha256'] == digest(STUDY/'selection.json')
            and chosen == (result['paired_gain']['lower'] > 0 and result['paired_gain']['observed_contribution_bb_per_100'] > 0)
            and result['backend'] == play['backend'] == ('river-call-continuation' if chosen else 'river-call-learning')
            and play['policy_sha256'] == result['policy_sha256'] and base['selected'] == 'candidate',
            'Selection, design, fit or served policy differs')
    files = {n:STUDY/n for n in ('candidate.json','selection.json','design.json','play-ready.json')}
    for name, sha in fit['code_sha256'].items():
        path = ROOT/'scripts'/name
        verify_files({path:sha})
        files['scripts/'+name] = path
    score = ROOT/'scripts'/'score_poker_river_continuation.py'
    verify_files({score:result['scorer_sha256']})
    for path in (score, Path(__file__).resolve(), ROOT/'scripts/run_poker.py'):
        files['scripts/'+path.name] = path
    model = Path(fit['model']['path'])
    require(file_ref(model) == fit['model'] and digest(model) == result['model_sha256'] == fit['model_sha256'],
            'Different shared fitted river parameters')
    files['models/model.json'] = model
    for seed in SEEDS:
        key = str(seed)
        candidate = Path(fit['policies'][key]['path'])
        policy = read(candidate)
        expected = fit['policies'][key] if chosen else base['policies'][key]
        require(file_ref(candidate) == fit['policies'][key] and policy['seed'] == seed and policy['format'] == FORMAT
                and policy['model']['sha256'] == digest(model) and policy['baseline']['sha256'] == base['policy_sha256'][key]
                and result['policy_sha256'][key] == expected['sha256']
                and file_ref(result['policies'][key]['path'])['sha256'] == expected['sha256'], 'Different fitted wrapper')
        files[f'models/seed-{seed}/source-policy.json'] = candidate
    return files, result


def reattach(dst, archive_paths):
    predecessor.restore(*archive_paths, dst/'parent')
    selected = read(dst/'selection.json')
    model = dst/'models/model.json'
    require(digest(model) == selected['model_sha256'], 'Wrong recovered weights')
    for seed in SEEDS:
        sub = f'seed-{seed}'
        folder = dst/'models'/sub
        source = folder/'source-policy.json'
        spec = read(source)
        base = dst/'parent/play'/sub/'policy.json'
        old_source = dst/'parent/models'/sub/'source-policy.json'
        require(spec['seed'] == seed and spec['format'] == FORMAT
                and spec['baseline']['sha256'] == digest(old_source) and spec['model']['sha256'] == digest(model),
                'Recovered ancestor or fitted seed manifest does not match')
        relocated = folder/'restored-policy.json'
        atomic_json(relocated, dict(spec, baseline=file_ref(base), model=file_ref(model)))
        candidate = selected['selected'] == 'candidate'
        require(selected['policy_sha256'][str(seed)] == digest(source if candidate else old_source),
                'Recovered selection differs from the frozen choice')
        alias = dst/'play'/sub/'policy.json'
        alias.parent.mkdir(parents=True)
        alias.symlink_to(os.path.relpath(relocated if candidate else base, alias.parent))


def restore(archive, archive_paths, dst):
    require(not dst.exists() or (dst.is_dir() and not any(dst.iterdir())), 'Expected an empty destination')
    verify_files(dict(zip(archive_paths, (PARENT,V1,FULL,SPECIALIST), strict=True)))
    restore_files(archive,dst,'recovery-manifest.json',lambda _: True,100*1024**2)
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile('recovery-manifest.json'))
    require(manifest['checkpoint_mode'] == MODE and manifest['restore_dependencies'] == PARENTS
            and manifest['selected_backend'] == read(dst/'selection.json')['backend'], 'Invalid continuation recovery')
    reattach(dst, archive_paths)
    return manifest


def check_decisions(dst):
    fit = read(STUDY/'candidate.json')
    selected = read(STUDY/'selection.json')
    factory = DiverseRiverCallPolicy if selected['selected'] == 'candidate' else RiverCallPolicy
    records = {}
    for seed in SEEDS:
        key, sub = str(seed), f'seed-{seed}'
        candidate = DiverseRiverCallPolicy(fit['policies'][key]['path'])
        restored_candidate = DiverseRiverCallPolicy(dst/'models'/sub/'restored-policy.json')
        actual = factory(selected['policies'][key]['path'])
        restored_play = factory(dst/'play'/sub/'policy.json')
        count = changed = 0
        for ledger in sorted((STUDY/'train').glob(f'*/{sub}/positions.jsonl.gz')):
            with gzip.open(ledger, 'rt') as rows:
                for row in map(json.loads, rows):
                    state = row['root_observation']
                    nonce = semantic_seed(row['deal_seed'], row['hero'], state)
                    candidate_distribution = None
                    for live, restored in ((candidate,restored_candidate),(actual,restored_play)):
                        a, b = live.decide(state,nonce), restored.decide(state,nonce)
                        require(a['action'] == b['action'] and a['distribution'] == b['distribution'],
                                'Restored distribution or seeded action mismatch')
                        if live is candidate:
                            candidate_distribution = a['distribution']
                    changed += int(candidate_distribution != candidate.parent.decide(state,nonce)['distribution'])
                    count += 1
                    if count >= 12 and changed >= 4:
                        break
            if count >= 12 and changed >= 4:
                break
        require(count >= 12 and changed > 0, 'Insufficient exercised river decisions')
        records[key] = dict(real_river_decisions=count, active_continued_updates=changed, selected_route_decisions=count)
    return records


def pack():
    parents = dependency_paths(read(INDEX))
    files, result = inputs()
    manifest = dict(checkpoint_mode=MODE, restore_dependencies=PARENTS, selected_backend=result['backend'],
                    files={k:file_meta(v) for k,v in sorted(files.items())})
    archive_paths = [Path(parents[sha]['local_archive']) for sha in (PARENT,V1,FULL,SPECIALIST)]

    def restored(dst):
        reattach(dst, archive_paths)
        return check_decisions(dst)

    _, checks = pack_and_restore(ARCHIVE,files,manifest,restored)
    ready = dict(kind='derived-policy-probe', checkpoint_mode=MODE, run=NAME, parent_iteration=480,
                 parent_manifest_file_id=parents[FULL]['manifest_file_id'], archive=str(ARCHIVE), **file_meta(ARCHIVE),
                 restore_dependencies=PARENTS, fitting_complete=True, restoration_verified=True,
                 resumable_selfplay=False, new_selfplay_iterations=0, external_backup=False,
                 private_selection_sha256=digest(STUDY/'selection.json'), selected=result['selected'],
                 selected_backend=result['backend'], decision_restoration=checks,
                 archive_includes_final_evaluation_summary=True, archive_includes_raw_hands=False)
    atomic_json(STUDY/'backup-ready.json',ready)
    return dict(bytes=ready['bytes'], sha256=ready['sha256'], decision_restoration=checks)


def ready_source():
    data = read(STUDY/'backup-ready.json')
    require(data['archive'] == str(ARCHIVE) and data['checkpoint_mode'] == MODE and data['kind'] == 'derived-policy-probe'
            and data['restoration_verified'] and data['fitting_complete'] and data['restore_dependencies'] == PARENTS
            and data['private_selection_sha256'] == digest(STUDY/'selection.json')
            and file_meta(ARCHIVE) == {k:data[k] for k in ('bytes','sha256')}, 'Frozen recovery changed')
    return data


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('pack','restore','prepare','check','commit'))
    for key in ('archive','river-parent','flop-v1','full480','specialist480','destination'):
        cli.add_argument('--'+key,type=Path)
    cli.add_argument('--sha256')
    cli.add_argument('--owner')
    args = cli.parse_args()
    if args.action == 'pack':
        result = pack()
    elif args.action == 'restore':
        deps = (args.river_parent,args.flop_v1,args.full480,args.specialist480)
        if not all((args.archive,args.destination,args.sha256,*deps)):
            cli.error('Supply the new archive, its local SHA-256, four dependency archives and an empty destination')
        verify_files({args.archive:args.sha256})
        result = restore(args.archive,deps,args.destination.resolve())['selected_backend']
    elif args.action == 'prepare':
        active = sum(row['bytes'] + row.get('manifest_bytes',0) for row in read(INDEX)['backups'] if not row.get('deleted_remote'))
        require(active + ready_source()['bytes'] <= DEFAULT_BUDGET, 'Upload exceeds target before replacement')
        result = prepare(TARGET,ready_source)
    else:
        if not args.owner:
            cli.error('Supply the approved Drive owner')
        result = register(args.action,args.owner,TARGET,ready_source,plan_retention)
        if args.action == 'commit':
            ready = ready_source()
            result.pop('restored_both_street_networks')
            result.update(restored_continued_river_policy=True, archive_includes_final_evaluation_summary=True,
                          archive_includes_raw_hands=False, selected=ready['selected'],
                          selected_backend=ready['selected_backend'], decision_restoration=ready['decision_restoration'])
            atomic_json(STUDY/'backup-verification.json',result)
            atomic_json(REPORT/'backup-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
