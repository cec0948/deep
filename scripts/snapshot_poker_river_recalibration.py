"""Package and verify the final recalibration against its archived predecessor."""
import argparse
from pathlib import Path
import gzip
import json
import os
import tarfile

from collect_poker_river_recalibration import PRIOR, REPORT, STUDY
from controlled_poker_refit import read, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import DEFAULT_BUDGET
from poker_preflop_specialist import file_ref
from poker_river_call_diverse import DiverseRiverCallPolicy
from poker_river_call_recalibrated import RecalibratedRiverCallPolicy
from register_poker_fixed_preflop_backup import BackupTarget, INDEX, plan_retention, prepare, register
from snapshot_poker_preflop_dose import FULL, SPECIALIST, file_meta, restore_files
from snapshot_poker_scheduled_window import pack_and_restore
from snapshot_poker_street_progress import SEEDS, V1, require
import snapshot_poker_river_continuation as parent


NAME = STUDY.name
ARCHIVE = ROOT/'artifacts/exports'/(NAME+'-policy.tar.gz')
MODE = 'cost-weighted-river-call-recalibration-v1'
PARENT_SHA = 'a789cabcc47d721a284abed739474cae7d2a9dd44dff00113807d870160501da'
PARENTS = [*parent.PARENTS,dict(parent_backup_sha256=PARENT_SHA,scope='Selected 11/27 river continuation')]
TARGET = BackupTarget(STUDY,ARCHIVE,MODE)
ORDER = (PARENT_SHA,parent.PARENT,V1,FULL,SPECIALIST)
LEARNER = 'river-call-recalibration'
INCUMBENT = 'river-call-continuation'


def reattach(dst, archives):
    verify_files(dict(zip(archives,ORDER,strict=True)))
    parent.restore(archives[0],archives[1:],dst/'parent')
    selected = read(dst/'selection.json')
    model = dst/'models/model.json'
    require(digest(model)==selected['model_sha256'],'Wrong restored shared calibration')
    for seed in SEEDS:
        sid = str(seed)
        folder = dst/'models'/f'seed-{sid}'
        source = folder/'source-policy.json'
        baseline = dst/'parent/play'/folder.name/'policy.json'
        baseline_source = dst/'parent/models'/folder.name/'source-policy.json'
        spec = read(source)
        require(spec['baseline']['sha256']==digest(baseline_source)
                and spec['model']['sha256']==digest(model),'Wrong restored dependencies')
        wrapper = folder/'restored-policy.json'
        atomic_json(wrapper,dict(spec,baseline=file_ref(baseline),model=file_ref(model)))
        candidate = selected['selected']=='candidate'
        require(selected['policy_sha256'][sid]==digest(source if candidate else baseline_source),
                'Wrong restored selection')
        play = dst/'play'/folder.name/'policy.json'
        play.parent.mkdir(parents=True)
        play.symlink_to(os.path.relpath(wrapper if candidate else baseline,play.parent))
    return selected['backend']


def check_decisions(dst, fit, selected):
    checks = {}
    selected_type = RecalibratedRiverCallPolicy if selected['selected']=='candidate' else DiverseRiverCallPolicy
    for seed in SEEDS:
        sid, sub = str(seed), f'seed-{seed}'
        models = (
            (RecalibratedRiverCallPolicy(fit['policies'][sid]['path']),
             RecalibratedRiverCallPolicy(dst/'models'/sub/'restored-policy.json')),
            (selected_type(selected['policies'][sid]['path']),selected_type(dst/'play'/sub/'policy.json')))
        count = 0
        raw = next((STUDY/'train').glob(f'*/{sub}/positions.jsonl.gz'))
        with gzip.open(raw,'rt') as stream:
            for row in map(json.loads,stream):
                for original, recovered in models:
                    a,b = [policy.decide(row['root_observation'],row['deal_seed']) for policy in (original,recovered)]
                    require((a['action'],a['distribution'])==(b['action'],b['distribution']),
                            'Recovered policy differs on a real river')
                count += 1
                if count==16:
                    break
        require(count==16,'Too few restored decisions')
        checks[sid] = count
    return checks


def pack():
    idx = read(INDEX)
    previous = parent.dependency_paths(idx)
    matches = [r for r in idx['backups'] if r.get('sha256')==PARENT_SHA and not r.get('deleted_remote')]
    require(len(matches)==1 and all(matches[0][key] for key in
            ('restoration_verified','verified_remote_size','verified_owner_only')),
            'Previous remote checkpoint cannot be verified')
    previous[PARENT_SHA] = matches[0]
    archives = [Path(previous[h]['local_archive']) for h in ORDER]
    verify_files(dict(zip(archives,ORDER,strict=True)))
    fit, result, play = [read(STUDY/name) for name in ('candidate.json','selection.json','play-ready.json')]
    accepted = result['paired_gain']['lower']>0 and result['paired_gain']['observed_contribution_bb_per_100']>0
    require(fit['complete'] and result['complete'] and play['complete']
            and result['candidate_sha256']==digest(STUDY/'candidate.json')
            and play['selection_sha256']==digest(STUDY/'selection.json')
            and fit['model_sha256']==result['model_sha256']
            and result['selected']==('candidate' if accepted else 'parent')
            and result['backend']==play['backend']==(LEARNER if accepted else INCUMBENT),
            'Fit, score and playable policy disagree')
    files = {name:STUDY/name for name in ('design.json','candidate.json','selection.json','play-ready.json')}
    files['models/model.json'] = Path(fit['model']['path'])
    verify_files({files['models/model.json']:fit['model_sha256']})
    for sid, ref in fit['policies'].items():
        path = Path(ref['path'])
        verify_files({path:fit['policy_sha256'][sid]})
        files[f'models/seed-{sid}/source-policy.json'] = path
    source = {**fit['sources'],'score_poker_river_recalibration.py':result['scorer_sha256']}
    verify_files({ROOT/'scripts'/name:h for name,h in source.items()})
    for name in (*source,Path(__file__).name,'run_poker.py','collect_poker_river_recalibration.py'):
        files['scripts/'+name] = ROOT/'scripts'/name
    manifest = dict(checkpoint_mode=MODE,restore_dependencies=PARENTS,selected_backend=result['backend'],
                    files={name:file_meta(path) for name,path in sorted(files.items())})

    def restore_and_check(dst):
        reattach(dst,archives)
        return check_decisions(dst,fit,result)

    _, checks = pack_and_restore(ARCHIVE,files,manifest,restore_and_check)
    ready = dict(kind='derived-policy-probe',checkpoint_mode=MODE,run=NAME,parent_iteration=480,
                 parent_manifest_file_id=previous[FULL]['manifest_file_id'],archive=str(ARCHIVE),**file_meta(ARCHIVE),
                 restore_dependencies=PARENTS,fitting_complete=True,restoration_verified=True,
                 resumable_selfplay=False,new_selfplay_iterations=0,external_backup=False,
                 selected=result['selected'],selected_backend=result['backend'],decision_restoration=checks,
                 archive_includes_final_evaluation_summary=True,archive_includes_raw_hands=False)
    atomic_json(STUDY/'backup-ready.json',ready)
    return {key:ready[key] for key in ('bytes','sha256','decision_restoration')}


def ready_source():
    data = read(STUDY/'backup-ready.json')
    require(data['archive']==str(ARCHIVE) and data['checkpoint_mode']==MODE and data['restoration_verified']
            and data['restore_dependencies']==PARENTS and data['fitting_complete']
            and file_meta(ARCHIVE)=={k:data[k] for k in ('bytes','sha256')},'Recovery file changed')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('pack','prepare','check','commit','restore'))
    parser.add_argument('--owner')
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--dependencies',type=Path,nargs=5)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.action=='pack':
        result=pack()
    elif args.action=='restore':
        if not all((args.archive,args.dependencies,args.destination,args.sha256)):
            parser.error('Supply archive, SHA-256, dependency archives in manifest order, and empty destination')
        verify_files({args.archive:args.sha256})
        require(not args.destination.exists() or args.destination.is_dir() and not any(args.destination.iterdir()),
                'Use an empty restore destination')
        restore_files(args.archive,args.destination,'recovery-manifest.json',lambda _: True,100*1024**2)
        with tarfile.open(args.archive) as archive:
            manifest=json.load(archive.extractfile('recovery-manifest.json'))
        require(manifest['checkpoint_mode']==MODE and manifest['restore_dependencies']==PARENTS,
                'Restore archive differs from checkpoint format')
        result=dict(backend=reattach(args.destination.resolve(),args.dependencies))
    elif args.action=='prepare':
        active=sum(r['bytes']+r.get('manifest_bytes',0) for r in read(INDEX)['backups'] if not r.get('deleted_remote'))
        require(active+ready_source()['bytes']<=DEFAULT_BUDGET,'Upload would exceed target size')
        result=prepare(TARGET,ready_source)
    else:
        if not args.owner:
            parser.error('Supply the approved Drive owner')
        result=register(args.action,args.owner,TARGET,ready_source,plan_retention)
        if args.action=='commit':
            ready=ready_source()
            result.pop('restored_both_street_networks',None)
            result.update(archive_includes_final_evaluation_summary=True,selected=ready['selected'],
                          selected_backend=ready['selected_backend'],decision_restoration=ready['decision_restoration'])
            for folder in (STUDY,REPORT):
                atomic_json(folder/'backup-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
