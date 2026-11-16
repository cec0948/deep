"""Extend the existing safe retention planner for self-contained refit bundles.

Never uploads/deletes. Preserves the same full60/full120,360,rich96,research and
dependency rules; the latest TWO derived records are counted across families.
"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_rich_backup_retention import plan_retention as rich_plan, _dependencies, _remote_ids
from plan_poker_backup_retention import validate_record, COMPLETE, DEFAULT_BUDGET
from export_controlled_poker_probe import MODE


def plan_retention(index,root=ROOT,budget_bytes=DEFAULT_BUDGET):
    root=Path(root).resolve();shadow=copy.deepcopy(index);special={}
    for n,r in enumerate(index['backups']):
        if r.get('deleted_remote') or r.get('checkpoint_mode')!=MODE:
            continue
        if (r.get('kind')!='derived-policy-probe' or r.get('status') not in COMPLETE
                or r.get('resumable_selfplay') is not False
                or r.get('new_selfplay_iterations')!=0 or r.get('restoration_verified') is not True
                or len(r.get('restore_dependencies',[]))!=2):
            raise ValueError('Incomplete controlled refit backup blocks cleanup')
        as_asset=copy.deepcopy(r);as_asset['kind']='evaluation-assets'
        special[n]=validate_record(as_asset,root)
        shadow['backups'][n]=as_asset
    base=rich_plan(shadow,root,budget_bytes)
    items={r['backup_index']:r for r in base['retained']+base['candidates']}
    active={n:r for n,r in enumerate(index['backups']) if not r.get('deleted_remote')}
    keep={n:list(item['reasons']) for n,item in items.items() if item in base['retained']}
    # Derived records are ranked together, not two old probes PLUS every new family.
    probes=sorted([n for n,r in active.items() if r.get('kind')=='derived-policy-probe'
                   and r.get('status') in COMPLETE],
                  key=lambda n:datetime.datetime.fromisoformat(active[n]['recorded_at']),reverse=True)
    for n in probes:
        old=keep.get(n,[])
        remaining=[reason for reason in old if reason not in ['one_of_latest_two_derived_probes','evaluation_assets']]
        if n in probes[:2]:remaining.append('one_of_latest_two_derived_probes_across_families')
        if remaining:keep[n]=remaining
        else:keep.pop(n,None)
    owners={fid:n for n,r in active.items() for fid in _remote_ids(r)}
    shaowners={r['sha256']:n for n,r in active.items() if r.get('sha256')}
    pending=list(keep)
    while pending:
        n=pending.pop();ids,shas=_dependencies(active[n]);dependencies=[]
        for key,table in [(fid,owners) for fid in ids]+[(sha,shaowners) for sha in shas]:
            if key not in table:raise ValueError('A retained probe restore dependency is missing')
            dependencies.append(table[key])
        for parent in dependencies:
            if parent not in keep:
                keep[parent]=['required_restore_dependency'];pending.append(parent)
    retained=[];candidates=[]
    for n,item in sorted(items.items()):
        item['kind']=active[n]['kind']
        if n in special:item['checkpoint_mode']=MODE
        item['reasons']=keep.get(n,['superseded_verified_backup_without_retained_dependents'])
        (retained if n in keep else candidates).append(item)
    for n in probes[:2]:
        if n in special:
            r=active[n];p=Path(r['local_archive'])
            if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
                raise ValueError('Latest controlled probe local size/hash mismatch')
    known=sum(x['remote_bytes'] or 0 for x in retained)
    total=None if any(x['remote_bytes'] is None for x in retained) else known
    base.update(format='private-controlled-poker-retention-plan-v1',retained=retained,candidates=candidates,
        index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        planned_retained_bytes=total,known_retained_bytes=known,
        within_budget=total<=budget_bytes if total is not None else None,
        candidate_remote_files=sum(len(x['remote_files']) for x in candidates),
        expected_reclaimed_bytes=sum(x['remote_bytes'] for x in candidates))
    base['budget_note']=('Unknown preserved records prevent an exact total' if total is None else
        'Retained indexed files fit the budget' if total<=budget_bytes else
        'Essential retained dependencies exceed budget; do not delete additional files')
    return base


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'artifacts'):
        raise ValueError('Retention plans are private artifacts only')
    result=plan_retention(json.loads((ROOT/'artifacts/offsite-backups.json').read_text()));atomic_json(a.output,result)
    print(json.dumps({k:result[k] for k in ['candidate_remote_files','planned_retained_bytes','within_budget']}))
