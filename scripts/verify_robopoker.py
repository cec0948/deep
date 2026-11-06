#!/usr/bin/env python3
"""Run bounded, existing upstream checks and retain logs and exit statuses.

These checks establish engine/research capabilities, not poker playing strength.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache' / 'robopoker'
COMMIT = 'b6bf9e2788d43a129c0938f24b278ac2f910a5ed'
CHECKS = {
    'six-player': ['-p', 'kicker', '--features', 'pokerkit/sixmax', '--lib', 'six_player'],
    'multiway': ['-p', 'kicker', '--features', 'pokerkit/sixmax', '--lib', 'multiway'],
    'default': ['-p', 'kicker', '-p', 'mccfr', '--lib'],
    'kuhn': ['-p', 'kuhn', '--lib', 'cfr::solver::tests::nash_equilibrium',
             '--', '--exact', '--nocapture'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checks', nargs='*', metavar='CHECK',
                        help='One or more of: ' + ', '.join(CHECKS))
    parser.add_argument('--timeout', type=int, default=300,
                        help='Maximum seconds per build + check; default 300')
    args = parser.parse_args()
    if any(name not in CHECKS for name in args.checks):
        parser.error('Unknown check; choose from: ' + ', '.join(CHECKS))
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    if commit != COMMIT:
        parser.error('Upstream is not at the reviewed commit; run fetch_robopoker.sh')
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'],
                                    text=True).strip()
    if dirty:
        parser.error('Upstream has local changes; review them before validation')
    env = os.environ.copy()
    env.setdefault('CARGO_HOME', '/workspace/.tools/cargo')
    env.setdefault('RUSTUP_HOME', '/workspace/.tools/rustup')
    env.setdefault('CARGO_BUILD_JOBS', '4')
    env.setdefault('RAYON_NUM_THREADS', '4')
    cargo = shutil.which('cargo') or str(Path(env['CARGO_HOME']) / 'bin' / 'cargo')
    if not Path(cargo).is_file():
        parser.error('Rust 1.90.0 is required; see docs/robopoker-assessment.md')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts' / 'verification' / stamp
    out.mkdir(parents=True)
    results = []
    for name in args.checks or CHECKS:
        command = [cargo, '+1.90.0', 'test', '--locked', *CHECKS[name]]
        log = out / (name + '.log')
        print(f'Running {name}; log: {log}', flush=True)
        started = time.monotonic()
        timed_out = False
        with log.open('w') as stream:
            # New process group lets a deadline stop both Cargo and its test children.
            proc = subprocess.Popen(command, cwd=UPSTREAM, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                import signal
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        content = log.read_text(errors='replace')
        summaries = re.findall(r'^test result: .+$', content, flags=re.M)
        passed = sum(int(x) for x in re.findall(r'test result: ok\. (\d+) passed;', content))
        result = dict(name=name, command=command, returncode=code,
                      timed_out=timed_out, elapsed_seconds=round(time.monotonic() - started, 3),
                      summaries=summaries, passed=passed,
                      verified=code == 0 and passed > 0,
                      log=log.name, sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        results.append(result)
        (out / 'results.json').write_text(json.dumps(
            dict(upstream_commit=commit, created_at_utc=stamp, checks=results), indent=2) + '\n')
        print(json.dumps(result), flush=True)
    print(f'Results: {out / "results.json"}')
    return 0 if all(x['verified'] for x in results) else 1


if __name__ == '__main__':
    sys.exit(main())
