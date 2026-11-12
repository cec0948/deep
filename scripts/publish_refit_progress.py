"""Write local, privacy-safe refit progress; never commit, push or upload.

Live status requires the recorded PID and /proc start token to agree with the
current registry and the actual runner command. Timestamps are only timestamps.
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
PHASES = {'fitting', 'evaluating', 'verifying', 'complete', 'failed', 'interrupted'}
CONDITIONS = ('public_lower_above_zero', 'at_least_two_public_seeds_improve',
              'mixed_lower_above_minus_25', 'noregrets_lower_above_minus_50')
METRICS = ('normalized_training_mse', 'normalized_teacher_score_mse',
           'normalized_anchored_objective', 'unraised_preflop_conditional_target_tv',
           'positive_regret_absent_fraction')
DRIFT = ('teacher_policy_tv', 'teacher_policy_tv_p95',
         'teacher_policy_tv_over_half_fraction', 'newly_absent_fraction',
         'resolved_absent_fraction')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def utc_stamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.datetime.fromisoformat(value)
        return result.astimezone(datetime.timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def finite_metrics(value, keys):
    return {k: value[k] for k in keys if isinstance(value.get(k), (int, float))
            and not isinstance(value[k], bool) and math.isfinite(value[k])}


def runner_alive(registry, status, folder):
    pid, start = registry.get('pid'), registry.get('process_start')
    if (type(pid) is not int or pid <= 0 or not isinstance(start, str)
            or not start.isdigit() or status.get('supervisor_pid') != pid
            or status.get('supervisor_start') != start):
        return False
    try:
        if process_start(pid) != start:
            return False
        args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        runners = {'run_compact_refit_probe.py', 'run_compact_anchored_probe.py'}
        if not any(Path(arg).name in runners for arg in args):
            return False
        position = args.index('--output')
        return Path(args[position + 1]).resolve() == folder and process_start(pid) == start
    except (OSError, ValueError, IndexError, UnicodeError):
        return False


def external_backup(index, run):
    """Only whitelist public metadata from a matching, complete private record."""
    eligible = []
    for item in index.get('backups', []):
        if (not isinstance(item, dict) or item.get('kind') != 'derived-policy-probe'
                or item.get('run') != run or item.get('deleted_remote', False) is not False
                or item.get('status') not in (None, 'complete', 'verified', 'uploaded')
                or item.get('verified_remote_size') is not True
                or item.get('verified_owner_only') is not True
                or item.get('resumable_selfplay') is not False
                or type(item.get('remote_checksum_available')) is not bool
                or type(item.get('bytes')) is not int or item['bytes'] <= 0
                or utc_stamp(item.get('recorded_at')) is None):
            continue
        parts = item.get('parts')
        if parts is not None and (not isinstance(parts, list) or not parts
                or not item.get('manifest_file_id') or not item.get('manifest_bytes')
                or any(p.get('verified_remote_size') is not True
                       or p.get('verified_owner_only') is not True
                       or not p.get('file_id') for p in parts)):
            continue
        if parts is None and not item.get('file_id'):
            continue
        eligible.append(item)
    if not eligible:
        return None
    item = max(eligible, key=lambda b: utc_stamp(b['recorded_at']))
    # Checksum availability is not proof that a remote checksum was checked.
    return dict(kind='derived-policy-probe', run=run, bytes=item['bytes'],
                recorded_at=utc_stamp(item['recorded_at']).isoformat(),
                verified_remote_size=True, verified_owner_only=True,
                remote_checksum_available=item['remote_checksum_available'],
                resumable_selfplay=False,
                verification_scope='Previously recorded verification; no fresh Drive request by this publisher')


def collect(root=ROOT):
    root = Path(root).resolve()
    now = datetime.datetime.now(datetime.timezone.utc)
    registry = read(root/'artifacts/refit-probe/current.json', {})
    folder = Path(registry.get('folder', '')).resolve()
    if folder.parent != root/'artifacts/refit-probe' or not STAMP.fullmatch(folder.name):
        raise ValueError('Invalid local probe registry')
    cfg, plan, status = (read(folder/name, {}) for name in ('config.json', 'plan.json', 'status.json'))
    plan_sha = digest(folder/'plan.json')
    if (plan.get('config') != cfg or plan.get('parent_iteration') != 120
            or status.get('plan_sha256') != plan_sha):
        raise ValueError('Probe plan/status provenance mismatch')
    seeds, variants, fields = cfg['seeds'], cfg['variants'], cfg['evaluation_fields']
    if (any(type(seed) is not int for seed in seeds)
            or any(v not in {'current', 'mixture', 'refit', 'anchored'} for v in variants)
            or set(fields) != {'public', 'mixed', 'noregrets'}):
        raise ValueError('Unrecognized probe configuration')
    planned = len(seeds) * len(variants) * len(fields)
    completed = status.get('completed_evaluations')
    if (status.get('planned_evaluations') != planned or type(completed) is not int
            or not 0 <= completed <= planned):
        raise ValueError('Invalid evaluation progress')
    live = runner_alive(registry, status, folder)
    saved_state, saved_phase = status.get('status'), status.get('phase')
    if saved_phase not in PHASES or saved_state not in {'running', 'complete', 'failed', 'interrupted'}:
        raise ValueError('Unrecognized recorded run status')
    observed_state, phase = saved_state, saved_phase
    if saved_state == 'running' and not live:
        observed_state, phase = 'interrupted_or_stale', 'not_running'
    observed_time = utc_stamp(status.get('updated_utc'))
    summary = read(folder/'summary.json')
    summary_verified = bool(isinstance(summary, dict) and summary.get('complete') is True
        and summary.get('raw_hands_verified') is True and summary.get('source_provenance_verified') is True
        and summary.get('plan_sha256') == plan_sha and summary.get('config') == cfg
        and summary.get('parent') == plan.get('parent')
        and summary.get('completed_evaluations') == planned
        and type(summary.get('total_hands')) is int
        and summary['total_hands'] == planned * cfg['evaluation_deals'] * 6)
    if saved_state == 'complete' and not summary_verified:
        observed_state, phase = 'unverified_complete', 'verification_required'
    index = read(root/'artifacts/offsite-backups.json', {})
    result = dict(observed_utc=now.isoformat(), run=folder.name, status=observed_state,
                  phase=phase, recorded_status=saved_state, recorded_phase=saved_phase,
                  status_updated_utc=observed_time.isoformat() if observed_time else None,
                  status_age_seconds=(now-observed_time).total_seconds() if observed_time else None,
                  supervisor_alive=live,
                  liveness_evidence='Matching registry/status PID, /proc start token and runner output argument',
                  completed_evaluations=completed, planned_evaluations=planned,
                  parent_iteration=120, resumable_selfplay=False,
                  artifact_kind='derived-offline-anchored-refit' if 'anchored' in variants else 'derived-offline-refit',
                  plan_sha256=plan_sha, candidate_fits={}, decisions=None,
                  summary_verified=summary_verified,
                  external_backup=external_backup(index, folder.name),
                  limitations=['Process liveness was checked only at observed_utc; old timestamps never prove current execution',
                               'Candidate fitting is offline reuse of the original120 replay, not additional self-play rounds',
                               'Fit metrics and Torch policy diagnostics are not held-out poker strength or deployment behavior',
                               'Backup verification is read from the private index; no fresh remote metadata or checksum check is performed'])
    for seed in seeds:
        fit = read(folder/f'seed-{seed}/results.json', {})
        fit_complete = fit.get('complete') is True
        if fit_complete:
            expected = fit.get('files', {}).get('candidate.npz', {})
            candidate = folder/f'seed-{seed}/candidate.npz'
            if (fit.get('seed') != seed or fit.get('parent_iteration') != 120
                    or fit.get('parent') != plan.get('parent') or fit.get('resumable_selfplay') is not False
                    or fit.get('extra_steps') != cfg['extra_steps']
                    or not candidate.is_file() or candidate.stat().st_size != expected.get('bytes')
                    or digest(candidate) != expected.get('sha256')):
                raise ValueError('Completed candidate provenance mismatch')
        fit_state = ('complete' if fit_complete else 'failed' if fit.get('error') else
                     'reported_running' if live and f'fit/{seed}' in status.get('running', {}) else
                     'pending' if live else 'not_running')
        item = dict(status=fit_state, complete=fit_complete,
                    planned_steps=cfg['extra_steps'],
                    completed_steps=cfg['extra_steps'] if fit_complete else None)
        for moment in ('before', 'after'):
            metrics = fit.get(moment, {})
            if metrics:
                item[moment] = finite_metrics(metrics, METRICS)
                item[moment]['policy_drift'] = finite_metrics(metrics.get('policy_drift', {}), DRIFT)
        result['candidate_fits'][str(seed)] = item
    result['completed_candidate_fits'] = sum(row['complete'] for row in result['candidate_fits'].values())
    result['planned_candidate_fits'] = len(seeds)
    if 'anchored' in variants:
        result['limitations'].append('Score anchoring does not guarantee a bound on policy drift or poker performance')
    if summary_verified:
        decisions = {}
        for variant in variants:
            if variant == 'current':
                continue
            decision = summary.get('decisions', {}).get(variant, {})
            conditions = decision.get('conditions', {})
            if (set(conditions) != set(CONDITIONS) or any(type(conditions[k]) is not bool for k in CONDITIONS)
                    or type(decision.get('original_four_conditions_pass')) is not bool
                    or decision.get('original_four_conditions_pass') != all(conditions.values())
                    or decision.get('deployment_authorized') is not False):
                raise ValueError('Invalid summary decision')
            decisions[variant] = dict(conditions={k: conditions[k] for k in CONDITIONS},
                original_four_conditions_pass=decision['original_four_conditions_pass'], deployment_authorized=False)
        result.update(decisions=decisions, total_hands=summary['total_hands'])
        result['limitations'] += [
            'Nominal confidence intervals are not adjusted for repeated experimentation',
            'Shared cards are one block across fixed seeds, not independent seed-by-deal samples',
            'The fixed opponent pool does not establish general or professional poker strength']
    if observed_state == 'complete':
        result['next_action'] = ('independent_validation_required; no automatic promotion'
            if all(x['original_four_conditions_pass'] for x in result['decisions'].values()) else
            'paused_for_analysis; failed original conditions, no automatic retrial or promotion')
    elif observed_state in {'failed', 'interrupted', 'interrupted_or_stale', 'unverified_complete'}:
        result['next_action'] = 'inspect private run evidence; do not automatically restart'
    else:
        result['next_action'] = 'observe the existing bounded experiment; do not start a duplicate'
    if status.get('error'):
        result['recorded_error_present'] = True
        result['failure_detail'] = 'Runner recorded an error; inspect its private status.json for details'
    return result


def main(once=False, interval=30):
    if interval < 1:
        raise ValueError('Polling interval must be positive')
    lock = (ROOT/'artifacts/refit-progress-publisher.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            result = collect()
        except (OSError, ValueError, KeyError, TypeError) as error:
            # Fail closed without publishing private exception text or stale liveness.
            result = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          status='verification_failed', phase='verification_required',
                          supervisor_alive=False, error_category=type(error).__name__,
                          next_action='Inspect local source evidence; no execution or backup claim is verified')
        atomic_json(ROOT/'reports/refit-probe-status.json', result)
        print(json.dumps({k: result.get(k) for k in ('observed_utc', 'run', 'status', 'phase',
            'supervisor_alive', 'completed_candidate_fits', 'completed_evaluations', 'planned_evaluations')}), flush=True)
        if once or result['status'] != 'running':
            return
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    main(args.once, args.interval)
