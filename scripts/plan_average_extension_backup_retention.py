"""Private plan for replacing a verified420 recovery with its self-contained480 successor.

No remote deletion is performed here. All older mandatory families, unknown
records and any retained restore dependencies remain protected.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_controlled_backup_retention import plan_retention as prior_plan
from plan_poker_backup_retention import COMPLETE, DEFAULT_BUDGET, validate_record
from plan_rich_backup_retention import _dependencies, _remote_ids
from train_poker_average_continuation import CHECKPOINT_FORMAT as MODE


ONLINE_MODE='preflop-online-comparison-recovery-v1'
WARM_MODE='preflop-warm-start-recovery-v1'


def stage(record):
    values=record.get('iterations',{})
    if (set(values)!={'342','343','344'} or len(set(values.values()))!=1
            or any(type(n) is not int or n not in (420,480) for n in values.values())):
        raise ValueError('Unknown window-average stage blocks cleanup')
    n=values['342']
    if (record.get('kind')!='training-checkpoint' or record.get('status') not in COMPLETE
            or record.get('training_family')!='legacy-window-average'
            or record.get('averaging_window')!=[361,n]
            or record.get('restoration_verified') is not True
            or record.get('warm_start_state_verified') is not True
            or record.get('resumable_selfplay') is not False
            or record.get('exact_legacy_resume') is not False
            or record.get('average_policy_available') is not True
            or record.get('historical_mixture_available') is not False):
        raise ValueError('Incomplete window-average recovery blocks cleanup')
    return n


def replacement_candidates(active):
    """Only an explicitly replaced predecessor, after complete verification."""
    windows={n:r for n,r in active.items() if r.get('checkpoint_mode')==MODE}
    stages={n:stage(r) for n,r in windows.items()}
    candidates=set()
    for n,r in windows.items():
        if (stages[n]!=480 or r.get('verified_remote_size') is not True
                or r.get('verified_owner_only') is not True
                or r.get('parent_comparison_policies_in_archive') is not True):
            continue
        for old,previous in windows.items():
            if stages[old]==420 and r.get('replaces_archive_sha256')==previous.get('sha256'):
                candidates.add(old)
    # Conservatively keep a predecessor if ANY other active record references
    # it, including an otherwise superseded record. Never break dependency chains.
    for old in list(candidates):
        for n,r in active.items():
            if n==old:continue
            ids,shas=_dependencies(r)
            if windows[old].get('sha256') in shas or ids&_remote_ids(windows[old]):
                candidates.remove(old);break
    return candidates


def plan_retention(index,root=ROOT,budget_bytes=DEFAULT_BUDGET):
    root=Path(root);shadow=copy.deepcopy(index)
    active={n:r for n,r in enumerate(index['backups']) if not r.get('deleted_remote')}
    windows={n:r for n,r in active.items() if r.get('checkpoint_mode')==MODE}
    online={n:r for n,r in active.items() if r.get('checkpoint_mode') in (ONLINE_MODE,WARM_MODE)}
    for r in online.values():
        warm=r['checkpoint_mode']==WARM_MODE
        expected_arms=['warm492'] if warm else ['control492','specialist492']
        if (r.get('kind')!='training-checkpoint' or r.get('status') not in COMPLETE
                or r.get('iterations')!={'342':492,'343':492,'344':492}
                or r.get('arms')!=expected_arms
                or r.get('restoration_verified') is not True
                or r.get('training_stage_completed') is not True):
            raise ValueError('Incomplete 492-round preflop recovery blocks cleanup')
        if warm:
            deps=r.get('restore_dependencies',[])
            parent_ids,shas=_dependencies({'restore_dependencies':deps})
            training_parents={rec['sha256'] for rec in windows.values() if stage(rec)==480}
            specialist_parents={rec['sha256'] for rec in active.values()
                                if rec.get('checkpoint_mode')=='preflop-specialist-policy-bundle-v1'
                                and rec.get('kind')=='derived-policy-probe' and rec.get('status') in COMPLETE}
            if (parent_ids or len(shas)!=2
                    or not any(dep.get('parent_checkpoint_sha256') in training_parents for dep in deps)
                    or not any(dep.get('parent_backup_sha256') in specialist_parents for dep in deps)
                    or r.get('warm_start_state_verified') is not True
                    or r.get('verification',{}).get('both_street_weights_verified') is not True):
                raise ValueError('Unverified warm-start weights or missing source recoveries')
    for n,r in windows.items():
        stage(r)
    for n,r in {**windows, **online}.items():
        item=copy.deepcopy(r);item['kind']='evaluation-assets'
        validate_record(item,root);shadow['backups'][n]=item
    base=prior_plan(shadow,root,budget_bytes)
    removed=replacement_candidates(active);retained=[];candidates=[]
    for item in base['retained']+base['candidates']:
        n=item['backup_index']
        if n in windows or n in online:
            r=active[n];path=Path(r['local_archive'])
            if not path.is_file() or path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:
                raise ValueError('Retained recovery local size/SHA mismatch')
            item['kind']='training-checkpoint';item['checkpoint_mode']=r['checkpoint_mode']
            if n in online:
                item['reasons']=['retained_warm_preflop_recovery' if r['checkpoint_mode']==WARM_MODE
                                 else 'retained_fixed_492_comparison_recovery']
            else:
                item['reasons']=['explicitly_replaced420_with_verified_self_contained480'] if n in removed else ['retained_window_recovery_or_dependency']
            (candidates if n in removed else retained).append(item)
        elif item in base['retained']:retained.append(item)
        else:candidates.append(item)
    total=None if any(r['remote_bytes'] is None for r in retained) else sum(r['remote_bytes'] for r in retained)
    base.update(format='private-window-average-extension-retention-plan-v1',retained=retained,candidates=candidates,
        index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        planned_retained_bytes=total,known_retained_bytes=sum(r['remote_bytes'] or 0 for r in retained),
        within_budget=total<=budget_bytes if total is not None else None,
        candidate_remote_files=sum(len(r['remote_files']) for r in candidates),
        expected_reclaimed_bytes=sum(r['remote_bytes'] for r in candidates))
    return base


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'artifacts'):raise ValueError('Private artifacts path required')
    result=plan_retention(json.loads((ROOT/'artifacts/offsite-backups.json').read_text()))
    atomic_json(args.output,result)
    print(json.dumps({k:result[k] for k in ('within_budget','planned_retained_bytes','candidate_remote_files')}))
