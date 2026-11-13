"""Private retention plan with independent legacy and rich-family round counts.

Never uploads or deletes. A rich round12 is not a legacy round372. Completed
rich runs retain their latest stage and one newer recovery, while the original
360 recovery remains protected as a parent reference. Unknown records remain
opaque and partial records stop cleanup planning.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re

from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_poker_backup_retention import (COMPLETE, INCOMPLETE, KNOWN_KINDS,
    DEFAULT_BUDGET, plan_retention as legacy_plan, validate_record)
from snapshot_rich_selfplay import FAMILY, FORMAT, SEEDS


def _rich_record(record,root):
    if (record.get('kind')!='training-checkpoint' or record.get('training_family')!=FAMILY
            or record.get('checkpoint_mode')!=FORMAT or record.get('parent_iteration')!=360
            or record.get('resumable_selfplay') is not True
            or record.get('resumable_worker_family')!=FAMILY
            or record.get('exact_legacy_resume') is not False
            or record.get('warm_start_state_verified') is not True
            or record.get('historical_mixture_available') is not False
            or record.get('restoration_verified') is not True):
        raise ValueError('Incomplete rich-family recovery capability verification')
    iterations=record.get('family_iterations',{})
    if set(iterations)!=set(map(str,SEEDS)) or any(type(n) is not int or not 0<n<=96 for n in iterations.values()):
        raise ValueError('Rich family iterations must cover all three seeds separately')
    if 'iterations' in record:
        raise ValueError('Rich family must not masquerade as legacy iteration counts')
    if type(record.get('training_stage_completed')) is not bool:
        raise ValueError('Rich recovery must explicitly identify completed-stage status')
    if record['training_stage_completed'] and (len(set(iterations.values()))!=1 or next(iter(iterations.values())) not in (12,48,96)):
        raise ValueError('Rich completed stage must be12,48or96 for all three seeds')
    run=record.get('run','')
    if not re.fullmatch(r'\d{8}T\d{6,12}Z',run):raise ValueError('Invalid rich run identifier')
    path=Path(record.get('local_archive','')).resolve()
    if not re.fullmatch(r'rich-selfplay-'+re.escape(run)+r'-stage-\d{5}-\d{8}T\d{6,12}Z\.tar\.gz',path.name):
        raise ValueError('Rich archive filename/run mismatch')
    # Reuse all remote size/permission, split order/size and path validation,
    # without applying the legacy training-family filename/round assumptions.
    shadow=copy.deepcopy(record);shadow['kind']='evaluation-assets'
    return validate_record(shadow,root)


def _remote_ids(record):
    return {v for v in [record.get('file_id'),record.get('manifest_file_id')]
            +[p.get('file_id') for p in record.get('parts',[]) if isinstance(p,dict)] if v}


def _dependencies(value,in_dependencies=False):
    ids=set();shas=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('parent_manifest_file_id','parent_file_id') or (in_dependencies and key=='file_id'):
                if not isinstance(item,str) or not item:raise ValueError('Malformed restore dependency')
                ids.add(item)
            elif key in ('parent_backup_sha256','parent_checkpoint_sha256'):
                if not isinstance(item,str) or not re.fullmatch('[0-9a-f]{64}',item):raise ValueError('Malformed parent checkpoint SHA-256')
                shas.add(item)
            else:
                a,b=_dependencies(item,in_dependencies or key in ('restore_dependencies','backup_dependencies'))
                ids|=a;shas|=b
    elif isinstance(value,list):
        for item in value:
            a,b=_dependencies(item,in_dependencies);ids|=a;shas|=b
    return ids,shas


def _local_proof(number,record):
    p=Path(record['local_archive'])
    if not p.is_file() or p.stat().st_size!=record['bytes'] or digest(p)!=record['sha256']:
        raise ValueError('Retained latest rich checkpoint local size/SHA-256 mismatch')
    return dict(backup_index=number,archive=str(p),bytes=record['bytes'],sha256=record['sha256'])


def plan_retention(index,root=ROOT,budget_bytes=DEFAULT_BUDGET):
    root=Path(root).resolve();records=index.get('backups')
    if not isinstance(records,list):raise ValueError('Private backup index required')
    rich={};active={};owners={};shaowners={}
    shadow=copy.deepcopy(index)
    for number,record in enumerate(records):
        if not isinstance(record,dict) or type(record.get('deleted_remote',False)) is not bool:
            raise ValueError('Malformed private backup record')
        if record.get('deleted_remote'):continue
        active[number]=record
        if record.get('kind') in KNOWN_KINDS and record.get('status') in INCOMPLETE:
            raise ValueError('Partial/incomplete backup blocks cleanup planning')
        for identifier in _remote_ids(record):
            if identifier in owners:raise ValueError('Remote ID shared by multiple active backup records')
            owners[identifier]=number
        if record.get('sha256'):shaowners.setdefault(record['sha256'],set()).add(number)
        is_rich=record.get('checkpoint_mode')==FORMAT or record.get('training_family')==FAMILY
        if is_rich:
            if record.get('status') not in COMPLETE:raise ValueError('Unknown rich status blocks cleanup')
            rich[number]=_rich_record(record,root)
            # The old planner must still see ID ownership so an older retained
            # probe can explicitly depend on a rich manifest. Hide only the new
            # family's schema; wrapper dependency closure below uses real rows.
            shadow['backups'][number]=dict(kind='rich-family-shadow',status='verified',
                file_id=record.get('file_id'),manifest_file_id=record.get('manifest_file_id'),
                parts=[{'file_id':p.get('file_id')} for p in record.get('parts',[])])
    legacy=legacy_plan(shadow,root,budget_bytes)
    items={p['backup_index']:p for p in legacy['retained']+legacy['candidates']}
    keep={p['backup_index']:list(p['reasons']) for p in legacy['retained'] if p['backup_index'] not in rich}
    # The user's latest storage policy protects the original full120 even if
    # newer probe records no longer happen to reference its restore manifest.
    full120=[n for n,r in active.items() if r.get('kind')=='training-checkpoint'
        and r.get('checkpoint_mode') is None and r.get('status') in COMPLETE
        and set(r.get('iterations',{}).values())=={120}]
    if not full120:raise ValueError('The required original full120 recovery must remain available')
    first120=min(full120,key=lambda n:datetime.datetime.fromisoformat(records[n]['recorded_at']))
    keep.setdefault(first120,[]).append('required_original_full120_recovery')
    for number,(files,stamp,path) in rich.items():
        record=records[number]
        items[number]=dict(backup_index=number,kind='training-checkpoint',run=record['run'],
            training_family=FAMILY,family_iterations=record['family_iterations'],parent_iteration=360,
            sha256=record['sha256'],local_archive=str(path),remote_files=files,
            remote_bytes=sum(f['expected_bytes'] for f in files),checkpoint_mode=FORMAT,
            exact_legacy_resume=False,resumable_selfplay=True,resumable_worker_family=FAMILY,
            warm_start_state_verified=True,historical_mixture_available=False)
    local=[]
    runs={records[n]['run'] for n in rich}
    for run in runs:
        members=[n for n in rich if records[n]['run']==run]
        completed=[n for n in members if records[n]['training_stage_completed']]
        if completed:
            latest=max(completed,key=lambda n:(min(records[n]['family_iterations'].values()),rich[n][1]))
            keep.setdefault(latest,[]).append('latest_completed_stage_of_rich_run')
            local.append(_local_proof(latest,records[latest]))
            newer=[n for n in members if max(records[n]['family_iterations'].values())>min(records[latest]['family_iterations'].values())]
        else:
            # Before the first declared stage, retain one advanced recovery.
            newer=members
        if newer:
            latest=max(newer,key=lambda n:(min(records[n]['family_iterations'].values()),
                max(records[n]['family_iterations'].values()),rich[n][1]))
            keep.setdefault(latest,[]).append('latest_poststage_rich_recovery')
            local.append(_local_proof(latest,records[latest]))
    if rich:
        parents=[n for n,r in active.items() if r.get('kind')=='training-checkpoint'
            and r.get('checkpoint_mode')=='compact-bounded-replay-recovery-v1'
            and set(r.get('iterations',{}).values())=={360}]
        if not parents:raise ValueError('Original compact360 parent recovery must remain available')
        for n in parents:keep.setdefault(n,[]).append('original360_parent_of_rich_family')
    pending=list(keep)
    while pending:
        number=pending.pop();ids,shas=_dependencies(records[number])
        dependencies=set()
        for identifier in ids:
            if identifier not in owners:raise ValueError('Retained backup references missing/deleted restore dependency')
            dependencies.add(owners[identifier])
        for sha in shas:
            if sha not in shaowners:raise ValueError('Retained backup references missing/deleted parent SHA-256')
            dependencies|=shaowners[sha]
        for parent in dependencies:
            if parent not in keep:
                keep[parent]=['required_restore_dependency'];pending.append(parent)
    retained=[];candidates=[]
    for number in sorted(active):
        item=items[number]
        item['reasons']=keep.get(number,['superseded_verified_backup_without_retained_dependents'])
        (retained if number in keep else candidates).append(item)
    known=sum(p['remote_bytes'] or 0 for p in retained)
    total=None if any(p['remote_bytes'] is None for p in retained) else known
    return dict(format='private-rich-poker-retention-plan-v1',dry_run=True,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        remote_operations_performed=False,
        index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        retained=retained,candidates=candidates,
        ignored_deleted_records=[n for n,r in enumerate(records) if r.get('deleted_remote')],
        candidate_remote_files=sum(len(p['remote_files']) for p in candidates),
        expected_reclaimed_bytes=sum(p['remote_bytes'] for p in candidates),
        planned_retained_bytes=total,known_retained_bytes=known,
        within_budget=total<=budget_bytes if total is not None else None,
        policy=dict(budget_bytes=budget_bytes,preserve_legacy_policy=True,
            preserve_required_full120=True,preserve_original360_parent=True,latest_completed_stage_per_rich_run=1,
            latest_poststage_recovery_per_rich_run=1,rich_rounds_are_not_legacy_rounds=True),
        retained_latest_rich_local_verification=local,
        retained_latest_legacy_local_verification=legacy['retained_latest_local_verification'],
        execution_requirements=legacy['execution_requirements'],
        budget_note=('Unknown preserved records prevent an exact storage total' if total is None else
            'Essential retained dependencies exceed budget; do not delete additional files' if total>budget_bytes else
            'Retained indexed files fit the budget'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=ROOT/'artifacts/offsite-backups.json')
    parser.add_argument('--output',type=Path);parser.add_argument('--budget-bytes',type=int,default=DEFAULT_BUDGET)
    args=parser.parse_args();result=plan_retention(json.loads(args.index.read_text()),budget_bytes=args.budget_bytes)
    if args.output:
        path=args.output.resolve()
        if not path.is_relative_to(ROOT/'artifacts') or path==args.index.resolve() or path.suffix!='.json':
            raise ValueError('Plan output must be a distinct private JSON file in artifacts')
        atomic_json(path,result)
        print(json.dumps({k:result[k] for k in ['candidate_remote_files','planned_retained_bytes','within_budget']}))
    else:print(json.dumps(result,indent=2))
