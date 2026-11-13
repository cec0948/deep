"""Publish observed bounded-study progress; never train, back up or promote."""
import argparse
import datetime
import fcntl
import math
from pathlib import Path
import time

from bounded_poker_study_support import check_frozen,read,validate_config
from evaluate_compact_learning_curve import adoption_conditions
from fullhouse_v3_common import ROOT,atomic_json,digest
from publish_continuation_progress import external_backup,process_matches,verified_decisions


def collect(root=ROOT):
    root=Path(root);now=datetime.datetime.now(datetime.timezone.utc)
    registry=read(root/'artifacts/bounded-continuation/current.json');study=Path(registry['folder']).resolve()
    if study.parent!=root/'artifacts/bounded-continuation':raise ValueError('Unexpected study registry')
    cfg=read(study/'config.json');validate_config(cfg);plan=read(study/'plan.json');state=read(study/'status.json')
    if (plan['config']!=cfg or state['plan_sha256']!=digest(study/'plan.json')
            or plan.get('no_model_promotion') is not True or plan.get('no_automatic_retry') is not True
            or plan.get('planned_evaluations')!=36 or plan.get('planned_hands')!=4320000
            or not plan.get('frozen_files')):raise ValueError('Plan mismatch')
    check_frozen(plan['frozen_files']);campaign=study/'campaign'
    live=process_matches(registry['pid'],registry['process_start'],root/'scripts/run_bounded_poker_study.py','--output',study)
    if registry['pid']!=state['supervisor_pid'] or registry['process_start']!=state['supervisor_start']:
        raise ValueError('Supervisor registry/status mismatch')
    saved=state['status'];observed='interrupted_or_stale' if saved=='running' and not live else saved
    if state['completed_stages'] not in ([],[300],[300,360]):raise ValueError('Invalid completed stages')
    progress={}
    for seed in cfg['seeds']:
        folder=campaign/f'seed-{seed}';pointer=read(folder/'current.json');iteration=pointer['iteration']
        if type(iteration) is not int or not 240<=iteration<=360:raise ValueError('Invalid committed iteration')
        worker=state.get('running',{}).get(f'train/{seed}',{})
        alive=process_matches(worker.get('pid'),worker.get('process_start'),root/'scripts/train_compact_bounded.py','--run-dir',campaign)
        row=read(folder/'status.json')
        latest=max(iteration,state.get('latest_completed_rounds',{}).get(str(seed),iteration))
        if latest>360:raise ValueError('Unplanned training extension')
        progress[str(seed)]=dict(checkpoint_iteration=iteration,latest_completed_round=latest,
            additional_saved_selfplay_iterations=iteration-240,worker_alive=alive,recorded_state=row['status'])
    e=read(study/'evaluation-status.json') if (study/'evaluation-status.json').exists() else {}
    if e:
        ep=read(study/'evaluation-plan.json')
        if (ep['study_plan_sha256']!=digest(study/'plan.json') or ep['config']!=cfg
                or e['evaluation_plan_sha256']!=digest(study/'evaluation-plan.json') or e['planned_evaluations']!=36
                or e['completed_evaluations']!=len(e['finished']) or len(set(e['finished']))!=len(e['finished'])
                or e['verified_hands']!=e['completed_evaluations']*120000
                or set(e['verified_result_files'])!=set(e['finished'])
                or not set(e['finished']).issubset({j['id'] for j in ep['jobs']})
                or len(ep['jobs'])!=36):raise ValueError('Evaluation progress provenance mismatch')
        for name,p in e['verified_result_files'].items():
            for file,key in [('results.json','results_sha256'),('hands.jsonl.gz','hands_sha256')]:
                if digest(study/'evaluation'/name/file)!=p[key]:raise ValueError('Verified result bytes changed')
            if digest(study/'evaluation'/(name+'.provenance.json'))!=p['provenance_sha256']:raise ValueError('Cell provenance changed')
        evaluator_alive=process_matches(e['evaluator_pid'],e['evaluator_start'],root/'scripts/evaluate_bounded_poker_study.py','--study',study)
    else:evaluator_alive=False
    stages={};ready=read(study/'stage-ready.json')
    for stage in state['completed_stages']:
        p=study/f'stage-{stage}-verification.json';proof=read(p)
        if not proof.get('passed') or ready['checkpoints'][str(stage)]['verification_sha256']!=digest(p):
            raise ValueError('Stage verification is missing')
        b=read(study/f'stage-{stage}-backup.json') if (study/f'stage-{stage}-backup.json').exists() else None
        if b and (b['checkpoint_mode']!='compact-bounded-replay-recovery-v1' or digest(b['archive'])!=b['sha256']
                  or Path(b['archive']).stat().st_size!=b['bytes']):raise ValueError('Local compact archive changed')
        stages[str(stage)]=dict(checkpoint_verified=True,evaluation_complete=stage in e.get('completed_checkpoint_evaluations',[]),
                               local_recovery_verified=b is not None,local_recovery_bytes=b['bytes'] if b else None)
    decisions={};complete=False
    summary_path=study/'evaluation-summary.json'
    if not summary_path.exists() and e.get('completed_checkpoint_evaluations'):
        summary_path=study/f"stage-{max(e['completed_checkpoint_evaluations'])}-evaluation.json"
    if summary_path.exists():
        summary=read(summary_path)
        decisions=verified_decisions(summary,cfg,summary['through_iteration'])
        if summary_path.name=='evaluation-summary.json' and (study/'summary.json').exists():
            final=read(study/'summary.json')
            complete=bool(final.get('complete') and final['plan_sha256']==digest(study/'plan.json')
                and final['evaluation_summary_sha256']==digest(study/'evaluation-summary.json')
                and final['selfplay_iteration']==360 and final['additional_selfplay_iterations']==120
                and final.get('no_model_promotion') and final.get('no_automatic_retry')
                and e.get('status')=='complete' and e.get('completed_evaluations')==36
                and summary.get('raw_hands_verified') and summary.get('source_provenance_verified')
                and summary.get('total_hands')==4320000 and summary.get('independent_card_blocks')==20000
                and state['completed_stages']==[300,360] and all(v['local_recovery_verified'] for v in stages.values()))
    if saved=='complete' and not complete:observed='unverified_complete'
    index=read(root/'artifacts/offsite-backups.json');maximum=max(x['checkpoint_iteration'] for x in progress.values())
    backup=external_backup(index,study.name,maximum)
    if backup:backup['scope']='This bounded study'
    else:
        previous=[]
        for row in index['backups']:
            if row.get('kind')=='training-checkpoint' and not row.get('deleted_remote') and set(row.get('iterations',{}).values())=={240}:
                item=external_backup(index,row['run'],240)
                if item:previous.append(item)
        if previous:
            backup=max(previous,key=lambda x:x['recorded_at']);backup['scope']='Previously verified parent240 recovery; no new study stage claimed'
    stamp=datetime.datetime.fromisoformat(state['updated_utc'])
    if stamp.tzinfo is None or (stamp-now).total_seconds()>60:raise ValueError('Invalid or future observation timestamp')
    return dict(observed_utc=now.isoformat(),run=study.name,status=observed,phase=state['phase'],
        status_updated_utc=state['updated_utc'],status_age_seconds=max(0,(now-stamp).total_seconds()),supervisor_alive=live,
        seed_progress=progress,parent_selfplay_iteration=240,planned_selfplay_iteration=360,
        completed_training_stages=state['completed_stages'],training_stages=stages,
        training_complete=state['completed_stages']==[300,360],study_complete=complete and saved=='complete',
        evaluation=dict(recorded_status=e.get('status','pending'),evaluator_alive=evaluator_alive,
            completed_evaluations=e.get('completed_evaluations',0),planned_evaluations=36,verified_hands=e.get('verified_hands',0),
            status_updated_utc=e.get('updated_utc')),
        completed_evaluations=e.get('completed_evaluations',0),planned_evaluations=36,planned_hands=4320000,
        independent_card_blocks=20000,adoption_decisions=decisions,external_checkpoint=backup,
        source_provenance_verified=True,connection_audit_passed=True,original_registries_unchanged=True,
        historical_mixture_created=False,replay_capacity_rows_per_seat=100000,model_replaced=False,
        error_present=bool(state.get('error')),failure_stage=state.get('failure_stage'),
        next_action='Analyze completed fixed-budget study; no automatic promotion or extension' if complete else
                    'Observe existing supervised study; no duplicate launch or threshold change' if observed=='running' else
                    'Inspect failure or stale execution; no automatic retry',
        limitations=['Liveness is observed only at observed_utc; recorded timestamps do not prove continued execution',
            'Training stage completion is separate from the full4320000-hand evaluation',
            'Shared cards across seats and fixed training seeds are20000 statistical blocks, not4320000 independent observations',
            'Offsite backup facts are prior private-index verifications; this publisher does not request Drive checksums'])


def main(once=False,interval=30):
    lock=(ROOT/'artifacts/bounded-continuation/publisher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:row=collect()
        except (OSError,ValueError,KeyError,TypeError) as error:
            row=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='verification_failed',
                study_complete=False,error_category=type(error).__name__,next_action='Inspect source evidence; do not restart automatically')
        atomic_json(ROOT/'reports/bounded-continuation-status.json',row)
        print({k:row.get(k) for k in ['observed_utc','status','phase','completed_evaluations']},flush=True)
        if once or row['status']!='running':return
        time.sleep(interval)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=30)
    a=p.parse_args();main(a.once,a.interval)
