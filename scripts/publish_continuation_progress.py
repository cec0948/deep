"""Publish factual local study progress without Git, Drive or private references.

Study completion requires completed training stages AND all enlarged evaluation
results. A verified local stage snapshot is not an off-site backup. Liveness is
observed from the actual PID, start token and command, never from an old timestamp.
"""
import argparse
import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import time

from fullhouse_v3_common import ROOT, atomic_json, digest, process_start


STAMP = re.compile(r'\d{8}T\d{6,12}Z')
PHASES = {'prepared', 'training', 'verifying_stage', 'snapshotting', 'evaluating',
          'verifying', 'complete', 'failed', 'interrupted'}
CONDITIONS = ('public_lower_above_zero', 'at_least_two_public_seeds_improve',
              'mixed_lower_above_minus_25', 'noregrets_lower_above_minus_50')
HASH_CACHE = {}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.datetime.fromisoformat(value)
        return result.astimezone(datetime.timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def checked_hash(path, expected):
    """Hash once per unchanged inode/size/mtime/ctime; first observation is full."""
    path = Path(path).resolve()
    before = path.stat()
    def fingerprint(row):
        return (str(path), row.st_dev, row.st_ino, row.st_size,
                row.st_mtime_ns, row.st_ctime_ns)
    key = fingerprint(before)
    actual = HASH_CACHE.get(key)
    if actual is None:
        actual = digest(path)
        if fingerprint(path.stat()) != key:
            raise ValueError('Evidence changed while hashing')
        HASH_CACHE[key] = actual
    if not isinstance(expected, str) or actual != expected:
        raise ValueError('Frozen evidence hash mismatch')


def process_matches(pid, start, script, flag, folder):
    if (type(pid) is not int or pid <= 0 or not isinstance(start, str)
            or not start.isdigit() or process_start(pid) != start):
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        if not any(Path(arg).resolve() == Path(script).resolve() for arg in args if arg):
            return False
        return (Path(args[args.index(flag)+1]).resolve() == Path(folder).resolve()
                and process_start(pid) == start)
    except (OSError, UnicodeError, ValueError, IndexError):
        return False


def runner_alive(root, registry, status, folder):
    pid, start = registry.get('pid'), registry.get('process_start')
    return (status.get('supervisor_pid') == pid and status.get('supervisor_start') == start
            and process_matches(pid, start, root/'scripts/run_compact_learning_curve.py', '--output', folder))


def verify_sources(root, folder, cfg, plan, status):
    if (plan.get('config') != cfg or plan.get('parent_iteration') != 120
            or status.get('plan_sha256') != digest(folder/'plan.json')
            or plan.get('planned_evaluations') != 54 or plan.get('planned_hands') != 6480000
            or plan.get('no_model_promotion') is not True
            or plan.get('no_automatic_retry') is not True):
        raise ValueError('Study plan/status mismatch')
    frozen = plan.get('frozen_files')
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError('Missing frozen study inputs')
    for name, sha in frozen.items():
        path = Path(name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Frozen path is outside the source environment')
        checked_hash(path, sha)
    protected = plan.get('protected_registry_files', {})
    if set(protected) != {str(root/'artifacts/compact-v4/current.json'),
                          str(root/'artifacts/retraining-loop.json')}:
        raise ValueError('Original registry/controller protection is missing')
    for name, sha in protected.items():
        if frozen.get(name) != sha:
            raise ValueError('Protected registry does not match frozen plan')
    audit = read(folder/'method-audit.json', {})
    checked_hash(folder/'method-audit.json', plan.get('method_audit_sha256'))
    checks = audit.get('checks', {})
    if (audit.get('complete') is not True or audit.get('no_blocking_implementation_faults') is not True
            or audit.get('parent') != plan.get('parent') or not checks
            or any(value is not True for value in checks.values())
            or not audit.get('source_hashes')):
        raise ValueError('Completed passing method audit is missing')
    for name, sha in audit['source_hashes'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Audit source escapes repository')
        checked_hash(path, sha)
    checked_hash(folder/'preparation.json', plan.get('preparation_sha256'))
    preparation = read(folder/'preparation.json', {})
    if (preparation.get('complete') is not True or preparation.get('parent') != plan.get('parent')
            or preparation.get('campaign') != plan.get('campaign')
            or preparation.get('original_registries_unchanged') is not True
            or preparation.get('method_audit_sha256') != plan.get('method_audit_sha256')):
        raise ValueError('Study preparation provenance mismatch')


def external_backup(index, campaign_name, maximum_iteration):
    """Expose only verified index facts; no remote verification happens here."""
    eligible = []
    for row in index.get('backups', []):
        if (not isinstance(row, dict) or row.get('kind') != 'training-checkpoint'
                or row.get('run') != campaign_name or row.get('deleted_remote', False) is not False
                or row.get('status') not in (None, 'complete', 'verified', 'uploaded')
                or row.get('verified_remote_size') is not True or row.get('verified_owner_only') is not True
                or type(row.get('bytes')) is not int or row['bytes'] <= 0
                or type(row.get('remote_checksum_available')) is not bool
                or timestamp(row.get('recorded_at')) is None):
            continue
        iterations = row.get('iterations', {})
        if (set(iterations) != {'342', '343', '344'}
                or any(type(value) is not int or not 120 <= value <= maximum_iteration for value in iterations.values())):
            continue
        parts = row.get('parts')
        if parts is None:
            if not row.get('file_id'):
                continue
        elif (not isinstance(parts, list) or not parts or not row.get('manifest_file_id')
              or type(row.get('manifest_bytes')) is not int or row['manifest_bytes'] <= 0
              or sum(p.get('bytes', 0) for p in parts) != row['bytes']
              or any(not p.get('file_id') or p.get('verified_remote_size') is not True
                     or p.get('verified_owner_only') is not True for p in parts)):
            continue
        eligible.append(row)
    if not eligible:
        return None
    row = max(eligible, key=lambda item:(min(item['iterations'].values()), timestamp(item['recorded_at'])))
    result = dict(kind='training-checkpoint', run=row['run'], iterations=row['iterations'],
        bytes=row['bytes'], recorded_at=row['recorded_at'], verified_remote_size=True,
        verified_owner_only=True, remote_checksum_available=row['remote_checksum_available'],
        resumable_selfplay=row.get('resumable_selfplay', True),
        verification_scope='Previously recorded private-index verification; this publisher makes no Drive request')
    if row.get('checkpoint_mode') == 'compact-bounded-replay-recovery-v1':
        if (row.get('resumable_selfplay') is not False or row.get('exact_legacy_resume') is not False
                or row.get('historical_mixture_available') is not False
                or row.get('warm_start_state_verified') is not True):
            raise ValueError('Lightweight recovery capabilities are unverified')
        result.update(checkpoint_mode=row['checkpoint_mode'], exact_legacy_resume=False,
            warm_start_state_verified=True, historical_mixture_available=False)
    return result


def stage_evidence(folder, campaign, stage, ready, seeds):
    entry = ready.get('checkpoints', {}).get(str(stage), {})
    path = folder/f'stage-{stage}-verification.json'
    if entry.get('iteration') != stage:
        raise ValueError('Stage-ready iteration mismatch')
    checked_hash(path, entry.get('verification_sha256'))
    report = read(path, {})
    rows = report.get('seeds', [])
    if (report.get('complete') is not True or report.get('passed') is not True
            or report.get('stage') != stage or len(rows) != len(seeds)
            or {row.get('seed') for row in rows} != set(seeds)):
        raise ValueError('Completed training-stage verification is missing')
    for row in rows:
        if (row.get('iteration') != stage or row.get('snapshots') != stage
                or any(row.get(key) is not True for key in
                       ('finite_state', 'root_weighting_verified', 'pre_update_history_verified'))):
            raise ValueError('Training-stage verification disagrees')
        # latest.pt can advance in the next stage; stage policies remain immutable.
        seed_folder = campaign/f"seed-{row['seed']}"
        checked_hash(seed_folder/f'current_{stage:05d}.npz', row.get('current_sha256'))
        checked_hash(seed_folder/f'mixture_{stage:05d}.json', row.get('mixture_sha256'))
    return dict(iteration=stage, checkpoint_verified=True, evaluation_complete=False,
                local_snapshot_verified=False, external_backup=None)


def verified_decisions(summary, cfg, through):
    expected = {f'{kind}-{iteration}' for iteration in cfg['stages'] if iteration <= through
                for kind in cfg['variants']}
    supplied = summary.get('decisions', {})
    if set(supplied) != expected:
        raise ValueError('Candidate decision coverage mismatch')
    result = {}
    for candidate in sorted(expected):
        changes = {field:summary['paired_changes'][f'{field}/{candidate}-minus-current-120']
                   for field in cfg['evaluation_fields']}
        for row in changes.values():
            if (any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                    for key in ('mean', 'lower', 'upper', 'ci95_half_width'))
                    or {r.get('seed') for r in row.get('per_seed', [])} != set(cfg['seeds'])
                    or len(row['per_seed']) != len(cfg['seeds'])
                    or any(type(r.get('mean')) not in (int, float) or not math.isfinite(r['mean'])
                           for r in row['per_seed'])):
                raise ValueError('Invalid candidate comparison')
        conditions = dict(public_lower_above_zero=changes['public']['lower'] > 0,
            at_least_two_public_seeds_improve=sum(row['mean'] > 0 for row in changes['public']['per_seed']) >= 2,
            mixed_lower_above_minus_25=changes['mixed']['lower'] >= -25,
            noregrets_lower_above_minus_50=changes['noregrets']['lower'] >= -50)
        decision = supplied[candidate]
        if (decision.get('conditions') != conditions
                or any(type(value) is not bool for value in decision['conditions'].values())
                or decision.get('original_four_conditions_pass') is not all(conditions.values())
                or decision.get('deployment_authorized') is not False):
            raise ValueError('Reported adoption gates disagree with paired evidence')
        result[candidate] = dict(conditions=conditions,
            original_four_conditions_pass=all(conditions.values()), deployment_authorized=False)
    return result


def evaluation_evidence(folder, campaign, cfg, plan_sha):
    state = read(folder/'evaluation-status.json')
    if state is None:
        return dict(recorded_status='pending', completed_evaluations=0,
                    planned_evaluations=54, verified_hands=0, completed_checkpoint_evaluations=[]), {}
    plan = read(folder/'evaluation-plan.json', {})
    evaluation_sha = digest(folder/'evaluation-plan.json')
    if (plan.get('config') != cfg or plan.get('study_plan_sha256') != plan_sha
            or state.get('evaluation_plan_sha256') != evaluation_sha
            or state.get('status') not in {'running', 'complete', 'failed', 'interrupted'}
            or state.get('planned_evaluations') != 54):
        raise ValueError('Evaluation plan/status provenance mismatch')
    for name, sha in plan.get('frozen_files', {}).items():
        checked_hash(name, sha)
    jobs = {row['id']:row for row in plan.get('jobs', [])}
    expected_ids = {f'{field}-seed-{seed}-{kind}-{iteration}' for iteration in cfg['evaluation_iterations']
                    for kind in cfg['variants'] for seed in cfg['seeds'] for field in cfg['evaluation_fields']}
    finished = state.get('finished', [])
    if (set(jobs) != expected_ids or len(plan['jobs']) != 54
            or not isinstance(finished, list) or len(finished) != len(set(finished))
            or not set(finished).issubset(expected_ids)
            or state.get('completed_evaluations') != len(finished)
            or state.get('verified_hands') != len(finished)*cfg['evaluation_deals']*6):
        raise ValueError('Evaluation coverage mismatch')
    verified_files = state.get('verified_result_files', {})
    if set(verified_files) != set(finished):
        raise ValueError('Completed result byte provenance is missing')
    for name in finished:
        job = jobs[name]; cell = folder/'evaluation'/name
        proof_file = folder/'evaluation'/(name+'.provenance.json')
        files = verified_files[name]
        checked_hash(cell/'results.json', files.get('results_sha256'))
        checked_hash(cell/'hands.jsonl.gz', files.get('hands_sha256'))
        checked_hash(proof_file, files.get('provenance_sha256'))
        proof = read(proof_file, {}); row = read(cell/'results.json', {})
        model = campaign/f"seed-{job['seed']}"/f"{job['variant']}_{job['iteration']:05d}.{'json' if job['variant']=='mixture' else 'npz'}"
        if (proof.get('job') != job or Path(job['model']).resolve() != model
                or row.get('complete') is not True or row.get('field') != job['field']
                or row.get('hands') != cfg['evaluation_deals']*6 or row.get('deals') != cfg['evaluation_deals']
                or row.get('deal_seed') != cfg['evaluation_deal_seed']
                or Path(row.get('candidate', '')).resolve() != model):
            raise ValueError('Completed result configuration mismatch')
        checked_hash(model, proof.get('expected', {}).get('candidate'))
        for path, sha in proof.get('policy_files', {}).items():
            checked_hash(path, sha)
        if row.get('candidate_sha256') != proof['expected']['candidate']:
            raise ValueError('Evaluated policy provenance mismatch')
        for key in ('evaluator_sha256', 'runtime_sha256', 'engine_sha256', 'source_features_sha256', 'opponents'):
            if row.get(key) != proof['expected'].get(key):
                raise ValueError('Completed evaluation source provenance mismatch')
    stages = state.get('completed_checkpoint_evaluations', [])
    if stages not in ([], [180], [180, 240]):
        raise ValueError('Invalid completed evaluation stages')
    summaries = {}
    for stage in stages:
        count = 36 if stage == 180 else 54
        required = {name for name, job in jobs.items() if job['iteration'] <= stage}
        summary = read(folder/f'stage-{stage}-evaluation.json', {})
        if (not required.issubset(finished) or summary.get('complete') is not True
                or summary.get('config') != cfg or summary.get('through_iteration') != stage
                or summary.get('evaluation_plan_sha256') != evaluation_sha
                or summary.get('raw_hands_verified') is not True
                or summary.get('source_provenance_verified') is not True
                or summary.get('completed_evaluations') != count
                or summary.get('total_hands') != count*cfg['evaluation_deals']*6
                or summary.get('independent_card_blocks') != cfg['evaluation_deals']):
            raise ValueError('Completed stage evaluation is unverified')
        summaries[stage] = verified_decisions(summary, cfg, stage)
    return dict(recorded_status=state['status'], completed_evaluations=len(finished),
        planned_evaluations=54, verified_hands=state['verified_hands'],
        completed_checkpoint_evaluations=stages, evaluation_plan_sha256=evaluation_sha,
        status_updated_utc=state.get('updated_utc'), evaluator_pid=state.get('evaluator_pid'),
        evaluator_start=state.get('evaluator_start')), summaries


def collect(root=ROOT):
    root = Path(root).resolve(); now = datetime.datetime.now(datetime.timezone.utc)
    registry = read(root/'artifacts/continuation-study/current.json', {})
    folder = Path(registry.get('folder', '')).resolve()
    if folder.parent != root/'artifacts/continuation-study' or not STAMP.fullmatch(folder.name):
        raise ValueError('Invalid isolated study registry')
    cfg, plan, status = (read(folder/name, {}) for name in ('config.json', 'plan.json', 'status.json'))
    if (cfg.get('seeds') != [342, 343, 344] or cfg.get('stages') != [180, 240]
            or cfg.get('evaluation_iterations') != [120, 180, 240]
            or cfg.get('evaluation_fields') != ['public', 'mixed', 'noregrets']
            or cfg.get('variants') != ['current', 'mixture'] or cfg.get('evaluation_deals') != 20000
            or cfg.get('promotion') != 'none'):
        raise ValueError('Unrecognized study configuration')
    campaign = folder/'campaign'/folder.name
    if plan.get('campaign') != str(campaign) or status.get('campaign') != str(campaign):
        raise ValueError('Campaign path mismatch')
    verify_sources(root, folder, cfg, plan, status)
    plan_sha = digest(folder/'plan.json')
    saved = status.get('status'); phase = status.get('phase')
    if saved not in {'running', 'complete', 'failed', 'interrupted'} or phase not in PHASES:
        raise ValueError('Invalid recorded study state')
    recorded_time = timestamp(status.get('updated_utc'))
    if recorded_time is None or (recorded_time-now).total_seconds() > 60:
        raise ValueError('Missing or future study timestamp')
    live = runner_alive(root, registry, status, folder)
    observed = 'interrupted_or_stale' if saved == 'running' and not live else saved
    ready = read(folder/'stage-ready.json', {})
    stages = status.get('completed_stages', [])
    # Ready is atomically written just before status at a stage boundary. A newer
    # ready record may legitimately lead the status snapshot by one write.
    if (stages not in ([], [180], [180, 240])
            or ready.get('completed_stages') not in ([], [180], [180, 240])
            or not set(stages).issubset(ready['completed_stages'])):
        raise ValueError('Completed training stages disagree')
    training_stages = {str(stage):stage_evidence(folder, campaign, stage, ready, cfg['seeds']) for stage in stages}
    progress = {}
    for seed in cfg['seeds']:
        row = read(campaign/f'seed-{seed}/status.json', {})
        iteration = row.get('iteration')
        if type(iteration) is not int or not 120 <= iteration <= 240:
            raise ValueError('Invalid saved checkpoint iteration')
        worker = status.get('running', {}).get(f'train/{seed}', {})
        running = process_matches(worker.get('pid'), worker.get('process_start'),
            root/'scripts/train_compact_selfplay.py', '--run-dir', campaign)
        progress[str(seed)] = dict(checkpoint_iteration=iteration, additional_selfplay_iterations=iteration-120,
            worker_alive=running, checkpoint_state=row.get('status') if row.get('status') in
            {'running', 'complete', 'failed', 'interrupted'} else 'recorded')
    evaluation, decisions = evaluation_evidence(folder, campaign, cfg, plan_sha)
    evaluator_alive = process_matches(evaluation.pop('evaluator_pid', None), evaluation.pop('evaluator_start', None),
        root/'scripts/evaluate_compact_learning_curve.py', '--study', folder)
    evaluation['evaluator_alive'] = evaluator_alive
    if evaluation['recorded_status'] == 'running' and not evaluator_alive:
        evaluation['observed_status'] = 'interrupted_or_stale'
    else:
        evaluation['observed_status'] = evaluation['recorded_status']
    for stage in stages:
        training_stages[str(stage)]['evaluation_complete'] = stage in decisions
        backup = read(folder/f'stage-{stage}-backup.json')
        if backup is not None:
            archive = Path(backup.get('archive', '')).resolve()
            if (not archive.is_relative_to(root/'artifacts/exports')
                    or backup.get('iterations') != {str(seed):stage for seed in cfg['seeds']}
                    or backup.get('resumable_selfplay') is not True
                    or type(backup.get('verified_files')) is not int or backup['verified_files'] <= 0
                    or archive.stat().st_size != backup.get('bytes')):
                raise ValueError('Local stage snapshot is unverified')
            checked_hash(archive, backup.get('sha256'))
            training_stages[str(stage)].update(local_snapshot_verified=True, local_snapshot_bytes=backup['bytes'])
    summary = read(folder/'summary.json', {})
    complete = bool(summary.get('complete') is True and summary.get('plan_sha256') == plan_sha
        and summary.get('config') == cfg and summary.get('parent') == plan.get('parent')
        and summary.get('campaign') == str(campaign) and summary.get('completed_stages') == [180, 240]
        and stages == [180, 240] and all(row['checkpoint_iteration'] == 240 for row in progress.values())
        and all(row['local_snapshot_verified'] for row in training_stages.values())
        and summary.get('selfplay_iteration') == 240 and summary.get('additional_selfplay_iterations') == 120
        and summary.get('raw_hands_verified') is True and summary.get('source_provenance_verified') is True
        and summary.get('method_audit_passed') is True and summary.get('original_registries_unchanged') is True
        and summary.get('no_model_promotion') is True and summary.get('no_automatic_retry') is True
        and summary.get('completed_evaluations') == 54 and summary.get('total_hands') == 6480000
        and evaluation['recorded_status'] == 'complete' and evaluation['completed_evaluations'] == 54
        and set(decisions) == {180, 240})
    if complete:
        checked_hash(folder/'evaluation-summary.json', summary.get('evaluation_summary_sha256'))
        final = read(folder/'evaluation-summary.json', {})
        complete = bool(final.get('complete') is True and final.get('config') == cfg
            and final.get('through_iteration') == 240 and final.get('completed_evaluations') == 54
            and final.get('total_hands') == 6480000 and final.get('independent_card_blocks') == 20000
            and final.get('evaluation_plan_sha256') == evaluation.get('evaluation_plan_sha256')
            and final.get('raw_hands_verified') is True and final.get('source_provenance_verified') is True
            and verified_decisions(final, cfg, 240) == decisions[240])
    if saved == 'complete' and not complete:
        observed = 'unverified_complete'; phase = 'verification_required'
    if observed == 'interrupted_or_stale':
        phase = 'not_running'
    external = external_backup(read(root/'artifacts/offsite-backups.json', {}), campaign.name,
                               max(row['checkpoint_iteration'] for row in progress.values()))
    if external and len(set(external['iterations'].values())) == 1:
        stage = str(next(iter(external['iterations'].values())))
        if stage in training_stages:
            training_stages[stage]['external_backup'] = external
    result = dict(observed_utc=now.isoformat(), run=folder.name, status=observed, phase=phase,
        recorded_status=saved, recorded_phase=status['phase'], status_updated_utc=recorded_time.isoformat(),
        status_age_seconds=max(0., (now-recorded_time).total_seconds()), supervisor_alive=live,
        liveness_evidence='Matching registry/status PID, non-zombie /proc start token, runner command and output argument',
        plan_sha256=plan_sha, source_provenance_verified=True, method_audit_passed=True,
        original_selfplay_iteration=120, planned_selfplay_iteration=240,
        original_model_replaced=False, original_registries_unchanged=True,
        seed_progress=progress, completed_training_stages=stages, training_stages=training_stages,
        training_complete=stages == [180, 240], study_complete=complete and saved == 'complete',
        evaluation=evaluation, completed_evaluations=evaluation['completed_evaluations'],
        planned_evaluations=54, planned_hands=6480000, independent_card_blocks=20000,
        adoption_decisions=decisions[max(decisions)] if decisions else {}, external_checkpoint=external,
        limitations=['A completed training checkpoint does not mean all large evaluations or the research study are complete',
            'Observed process liveness applies only at observed_utc; timestamps never prove continuing execution',
            'This study collects new self-play through240 and does not reuse the failed offline-refit correction',
            'Original adoption gates are unchanged; interim gate results do not alter the authorized fixed study budget',
            'Common cards across stages are paired research comparisons, not independent final holdouts',
            'Fixed opponents and three seeds do not establish professional strength or general robustness',
            'Off-site backup facts come from the private index; this publisher makes no remote metadata or checksum request'])
    if observed == 'complete':
        result['total_hands'] = 6480000
        result['next_action'] = 'Analyze completed fixed-budget study; no automatic promotion, retry or extension'
    elif observed != 'running':
        result['next_action'] = 'Inspect private evidence; do not automatically restart'
    else:
        result['next_action'] = 'Observe the existing fixed-budget study; do not start a duplicate'
    if status.get('error'):
        result['recorded_error_present'] = True
        if status.get('failure_stage') in PHASES:
            result['failure_stage'] = status['failure_stage']
    return result


def main(once=False, interval=30):
    if type(interval) is not int or interval < 1:
        raise ValueError('Polling interval must be positive')
    (ROOT/'reports').mkdir(exist_ok=True)
    lock = (ROOT/'artifacts/continuation-study-publisher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            result = collect()
        except (OSError, ValueError, KeyError, TypeError) as error:
            result = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status='verification_failed', phase='verification_required', supervisor_alive=False,
                study_complete=False, error_category=type(error).__name__,
                next_action='Inspect local source evidence; no execution or backup claim is verified')
        atomic_json(ROOT/'reports/continuation-study-status.json', result)
        print(json.dumps({key:result.get(key) for key in ('observed_utc', 'run', 'status', 'phase',
            'supervisor_alive', 'completed_training_stages', 'completed_evaluations', 'planned_evaluations')}), flush=True)
        if once or result['status'] != 'running':
            return
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args(); main(args.once, args.interval)
