"""Retain one latest window-average recovery plus all existing mandatory assets."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path

from fullhouse_v3_common import ROOT,atomic_json,digest
from plan_controlled_backup_retention import plan_retention as prior_plan
from plan_poker_backup_retention import validate_record,COMPLETE,DEFAULT_BUDGET
from snapshot_poker_average_baseline import MODE


def plan_retention(index,root=ROOT,budget_bytes=DEFAULT_BUDGET):
    root=Path(root);shadow=copy.deepcopy(index);special={}
    for n,r in enumerate(index['backups']):
        if r.get('deleted_remote') or r.get('checkpoint_mode')!=MODE:continue
        if (r.get('kind')!='training-checkpoint' or r.get('status') not in COMPLETE
                or r.get('resumable_selfplay') is not False or r.get('restoration_verified') is not True
                or r.get('warm_start_state_verified') is not True or r.get('averaging_window')!=[361,420]
                or r.get('average_policy_available') is not True
                or r.get('historical_mixture_available') is not False
                or r.get('iterations')!={str(s):420 for s in (342,343,344)}):
            raise ValueError('Incomplete average recovery blocks cleanup')
        item=copy.deepcopy(r);item['kind']='evaluation-assets'
        validate_record(item,root);shadow['backups'][n]=item;special[n]=r
    base=prior_plan(shadow,root,budget_bytes)
    # The current study has one final stage, so duplicate runs are not silently
    # discarded. Keep the newest verified copy per run, and protect dependencies.
    latest={}
    for n,r in special.items():
        stamp=datetime.datetime.fromisoformat(r['recorded_at']);run=r['run']
        if run not in latest or stamp>latest[run][0]:latest[run]=(stamp,n)
    items=base['retained']+base['candidates'];retained=[];candidates=[]
    for item in items:
        n=item['backup_index']
        if n in special:
            item['kind']='training-checkpoint';item['checkpoint_mode']=MODE
            if n==latest[special[n]['run']][1]:
                r=special[n];path=Path(r['local_archive'])
                if not path.is_file() or path.stat().st_size!=r['bytes'] or digest(path)!=r['sha256']:
                    raise ValueError('Latest average recovery local size/hash mismatch')
                item['reasons']=['latest_completed_window_average_recovery'];retained.append(item)
            else:
                # Replacing duplicate exact-stage archives requires explicit
                # dependency closure; conservatively preserve them by default.
                item['reasons']=['same_stage_duplicate_preserved'];retained.append(item)
        elif item in base['retained']:retained.append(item)
        else:candidates.append(item)
    total=None if any(r['remote_bytes'] is None for r in retained) else sum(r['remote_bytes'] for r in retained)
    base.update(format='private-window-average-retention-plan-v1',retained=retained,candidates=candidates,
        index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        planned_retained_bytes=total,known_retained_bytes=sum(r['remote_bytes'] or 0 for r in retained),
        within_budget=total<=budget_bytes if total is not None else None,
        candidate_remote_files=sum(len(r['remote_files']) for r in candidates),
        expected_reclaimed_bytes=sum(r['remote_bytes'] for r in candidates))
    return base


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'artifacts'):raise ValueError('Retention plans must stay private')
    result=plan_retention(json.loads((ROOT/'artifacts/offsite-backups.json').read_text()))
    atomic_json(a.output,result);print(json.dumps({k:result[k] for k in ('planned_retained_bytes','within_budget','candidate_remote_files')}))
