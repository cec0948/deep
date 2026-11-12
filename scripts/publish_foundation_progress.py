"""Write a privacy-safe local status snapshot; do not commit or upload."""
import argparse
import datetime
import json
from pathlib import Path
import re
import time

from fullhouse_v3_common import ROOT, atomic_json, digest, process_start


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def collect():
    registry = read(ROOT/'artifacts/foundation-study/current.json', {})
    folder = Path(registry.get('folder', '')).resolve()
    if folder.parent != ROOT/'artifacts/foundation-study' or not re.fullmatch(r'\d{8}T\d{6,12}Z', folder.name):
        raise ValueError('Invalid study registry')
    status = read(folder/'status.json', {})
    plan = read(folder/'plan.json', {})
    plan_sha = digest(folder/'plan.json')
    if status.get('plan_sha256') != plan_sha:
        raise ValueError('Study plan/status hash mismatch')
    pid, start = registry.get('pid'), registry.get('process_start')
    live = False
    if (type(pid) is int and pid > 0 and start is not None
            and status.get('supervisor_pid') == pid and status.get('supervisor_start') == start
            and process_start(pid) == start):
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
            live = ('scripts/run_poker_foundation_study.py' in args
                    and Path(args[args.index('--output')+1]).resolve() == folder
                    and process_start(pid) == start)
        except (OSError, ValueError, IndexError):
            pass
    saved = status['status']
    observed = 'interrupted_or_stale' if saved == 'running' and not live else saved
    cfg = plan['config']
    result = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        run=folder.name, status=observed, recorded_status=saved, phase=status['phase'],
        status_updated_utc=status.get('updated_utc'), supervisor_alive=live,
        liveness_evidence='Recorded PID, /proc start token, runner command and output argument',
        plan_sha256=plan_sha, study_kind='research-validation', selfplay_iteration=120,
        new_six_player_selfplay_samples=0, six_player_model_replaced=False,
        completed_benchmarks=status['completed_benchmarks'], planned_benchmarks=status['planned_benchmarks'],
        completed_evaluations=status['completed_evaluations'], planned_evaluations=status['planned_evaluations'],
        benchmark_progress={}, external_backup=None,
        limitations=['Leduc is a two-player reference benchmark, not validation of the six-player architecture',
            'Reference opponents are transparent untrained heuristics, not proven strong poker agents',
            'Process liveness is observed only at the recorded observation time'])
    for job in plan['benchmark_jobs']:
        state = read(folder/'benchmark'/job['name']/'status.json', {})
        report = read(folder/'benchmark'/job['name']/'results.json', {})
        curve = report.get('curve', [])
        result['benchmark_progress'][job['name']] = dict(status=state.get('status', 'pending'),
            iteration=state.get('iteration', 0), planned_iterations=cfg['leduc'][job['method']+'_iterations'],
            last_evaluated_iteration=curve[-1]['iteration'] if curve else None,
            last_metrics=curve[-1]['metrics'] if curve else None)
    summary = read(folder/'summary.json')
    if saved == 'complete':
        if not (summary and summary.get('complete') is True
                and summary.get('plan_sha256') == plan_sha and summary.get('config') == cfg
                and summary.get('source_provenance_verified') is True
                and summary.get('no_model_promotion') is True
                and summary['benchmarks'].get('saved_policy_metrics_recomputed') is True
                and summary['reference_evaluation'].get('raw_hands_verified') is True
                and summary['reference_evaluation']['completed_evaluations'] == status['planned_evaluations']):
            raise ValueError('Completion evidence is inconsistent')
        result.update(reference_checks=summary['benchmarks']['checks'],
            total_reference_hands=summary['reference_evaluation']['total_hands'],
            source_provenance_verified=True,
            next_action='Analyze completed evidence; no automatic retrial, self-play continuation or promotion')
    elif observed != 'running':
        result['next_action'] = 'Inspect private study logs; no automatic restart'
    else:
        result['next_action'] = 'Observe the existing bounded study; do not start a duplicate'
    index = read(ROOT/'artifacts/offsite-backups.json', {})
    for item in reversed(index.get('backups', [])):
        if (item.get('kind') == 'research-validation' and item.get('run') == folder.name
                and not item.get('deleted_remote') and item.get('verified_remote_size') is True
                and item.get('verified_owner_only') is True and item.get('resumable_selfplay') is False
                and item.get('status') == 'complete'):
            result['external_backup'] = {key:item.get(key) for key in
                ('kind', 'run', 'bytes', 'recorded_at', 'verified_remote_size', 'verified_owner_only',
                 'remote_checksum_available', 'resumable_selfplay')}
            result['external_backup']['verification_scope'] = 'Recorded private-index verification; this publisher performs no Drive request'
            break
    if status.get('error'):
        result['recorded_error_present'] = True
    return result


def main(once):
    while True:
        try:
            result = collect()
        except (OSError, KeyError, ValueError, TypeError):
            result = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                status='verification_failed', supervisor_alive=False,
                next_action='Inspect local study evidence; no execution or backup claim is verified')
        atomic_json(ROOT/'reports/foundation-study-status.json', result)
        print(json.dumps({k:result.get(k) for k in ('run','status','phase','supervisor_alive',
            'completed_benchmarks','completed_evaluations')}), flush=True)
        if once or result['status'] != 'running':
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--once', action='store_true')
    main(parser.parse_args().once)
