"""Bounded, isolated frozen-replay refit and fresh-card evaluation experiment.

Never changes the self-play registry, original checkpoints, or deployment.
All comparisons use common card blocks; seeds are averaged within each block.
"""
import argparse
from bisect import bisect_right
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import time

from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def interval(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('At least two finite deal blocks required')
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** .5
    return dict(mean=mean, ci95_half_width=half, lower=mean-half, upper=mean+half)


def previous_ranges(output):
    """Find saved evaluation ranges, not merely different starting seeds."""
    records = {}
    for base in (ROOT/'artifacts', ROOT/'reports'):
        for path in base.rglob('config.json'):
            if path.is_relative_to(output):
                continue
            cfg = json.loads(path.read_text())
            start, count = cfg.get('evaluation_deal_seed'), cfg.get('evaluation_deals')
            if type(start) is int and type(count) is int and count > 0:
                records[str(path.relative_to(ROOT))] = [start, start+count]
    return records


def check_card_range(start, count, previous):
    if type(start) is not int or type(count) is not int or count < 2:
        raise ValueError('Invalid card range')
    if start < 0 or start+count > 2**32:
        raise ValueError('Card range is outside the engine seed range')
    overlaps = [name for name, (lo, hi) in previous.items()
                if max(start, lo) < min(start+count, hi)]
    if overlaps:
        raise ValueError('Evaluation cards overlap prior saved runs: '+', '.join(overlaps))


def verify_result(path, job, cfg, expected):
    """Recompute results from every raw hand, with exact seat/deal coverage."""
    row = json.loads((path/'results.json').read_text())
    deals, card_seed = cfg['evaluation_deals'], cfg['evaluation_deal_seed']
    if not row.get('complete') or row['field'] != job['field']:
        raise ValueError('Incomplete or wrong-field result')
    if (row['deals'], row['hands'], row['deal_seed']) != (deals, deals*6, card_seed):
        raise ValueError('Wrong evaluation budget or card seed')
    if Path(row['candidate']).resolve() != Path(job['model']).resolve():
        raise ValueError('Wrong candidate path')
    if row['candidate_sha256'] != expected['candidate'] or digest(job['model']) != expected['candidate']:
        raise ValueError('Candidate changed')
    for name in ('evaluator_sha256', 'runtime_sha256', 'engine_sha256', 'source_features_sha256'):
        if row.get(name) != expected[name]:
            raise ValueError('Evaluation provenance mismatch: '+name)
    if row['opponents'] != expected['opponents']:
        raise ValueError('Opponent models changed')
    mixture = job['variant'] == 'mixture'
    if row['candidate_type'] != ('snapshot_mixture' if mixture else 'compact_advantage'):
        raise ValueError('Unexpected policy type')
    cumulative = []
    if mixture:
        spec = json.loads(Path(job['model']).read_text())
        total = 0
        for model in spec['models']:
            total += model['weight']; cumulative.append(total)
    blocks = [[None]*6 for _ in range(deals)]
    count = 0
    with gzip.open(path/'hands.jsonl.gz', 'rt') as stream:
        for line in stream:
            hand = json.loads(line); deal, hero = hand['deal'], hand['hero']
            if type(deal) is not int or type(hero) is not int or not 0 <= deal < deals or not 0 <= hero < 6:
                raise ValueError('Out-of-range raw hand')
            if blocks[deal][hero] is not None:
                raise ValueError('Duplicate raw hand')
            utilities = hand['utilities']
            if len(utilities) != 6 or not all(math.isfinite(v) for v in utilities):
                raise ValueError('Invalid utilities')
            if hand['seed'] != card_seed+deal or sum(utilities) != 0 or hand['chips'] != utilities[hero]:
                raise ValueError('Invalid hand seed or payoff')
            hand_seed = int.from_bytes(hashlib.sha256(f'compact-mixture:{card_seed}:{deal}:{hero}'.encode()).digest()[:8], 'big')
            if hand['hand_seed'] != hand_seed:
                raise ValueError('Wrong per-hand random seed')
            if mixture:
                selected = bisect_right(cumulative, random.Random(hand_seed).randrange(cumulative[-1]))
                if hand['policy_iteration'] != spec['models'][selected]['iteration']:
                    raise ValueError('Mixture selection mismatch')
            blocks[deal][hero] = hand['chips']; count += 1
    if count != deals*6 or any(None in block for block in blocks):
        raise ValueError('Incomplete raw hand coverage')
    means = [statistics.mean(block) for block in blocks]
    if len(row['deal_blocks']) != deals or any(abs(a-b) > 1e-9 for a, b in zip(means, row['deal_blocks'])):
        raise ValueError('Raw deal-block mismatch')
    metrics = interval(means)
    if abs(metrics['mean']-row['bb_per_100']) > 1e-9 or abs(metrics['ci95_half_width']-row['ci95_half_width']) > 1e-9:
        raise ValueError('Reported score or interval mismatch')
    return row


def summarize(rows, cfg):
    absolute, comparisons = {}, {}
    seeds = cfg['seeds']
    for variant in cfg['variants']:
        for field in cfg['evaluation_fields']:
            per = [rows[f'{field}-seed-{s}-{variant}']['deal_blocks'] for s in seeds]
            pooled = [statistics.mean(values) for values in zip(*per)]
            absolute[f'{field}/{variant}'] = dict(**interval(pooled),
                per_seed=[dict(seed=s, **interval(values)) for s, values in zip(seeds, per)])
            if variant != 'current':
                changes = [[b-a for a, b in zip(rows[f'{field}-seed-{s}-current']['deal_blocks'], values)]
                           for s, values in zip(seeds, per)]
                comparisons[f'{field}/{variant}-minus-current'] = dict(
                    **interval([statistics.mean(values) for values in zip(*changes)]),
                    per_seed=[dict(seed=s, **interval(values)) for s, values in zip(seeds, changes)])
    decisions = {}
    for variant in cfg['variants']:
        if variant == 'current':
            continue
        public, mixed, weak = [comparisons[f'{f}/{variant}-minus-current'] for f in ('public', 'mixed', 'noregrets')]
        checks = dict(public_lower_above_zero=public['lower'] > 0,
                      at_least_two_public_seeds_improve=sum(s['mean'] > 0 for s in public['per_seed']) >= 2,
                      mixed_lower_above_minus_25=mixed['lower'] >= -25,
                      noregrets_lower_above_minus_50=weak['lower'] >= -50)
        decisions[variant] = dict(conditions=checks, original_four_conditions_pass=all(checks.values()),
                                 deployment_authorized=False)
    return dict(complete=True, completed_evaluations=len(rows),
                total_hands=sum(r['hands'] for r in rows.values()), absolute_scores=absolute,
                paired_changes=comparisons, decisions=decisions,
                limitations=['Exploratory two-candidate comparison; nominal intervals are not multiplicity-adjusted',
                             'Shared cards are one block across seeds, not independent seed-by-deal samples',
                             'Fixed trained seeds and a fixed two-model opponent pool',
                             'No automatic model promotion, self-play continuation, or professional-strength claim'])


def run(parent, output, config):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    cfg = json.loads(Path(config).read_text()); parent_cfg = json.loads((parent/'config.json').read_text())
    if cfg['seeds'] != parent_cfg['seeds'] or len(cfg['seeds']) < 2:
        raise ValueError('Evaluate every declared parent training seed in its original order')
    if set(cfg['variants']) != {'current', 'mixture', 'refit'} or len(cfg['variants']) != 3:
        raise ValueError('Exactly current, mixture and refit are required')
    if set(cfg['evaluation_fields']) != {'public', 'mixed', 'noregrets'} or len(cfg['evaluation_fields']) != 3:
        raise ValueError('All three fixed opponent fields are required')
    if not 1 <= cfg['parallelism'] <= 3 or not 1 <= cfg['extra_steps'] <= 8192:
        raise ValueError('Invalid bounded compute budget')
    if cfg['batch_size'] != parent_cfg['batch_size'] or cfg['learning_rate'] != parent_cfg['learning_rate']:
        raise ValueError('Only the extra main-fit budget may change')
    if json.loads((parent/'campaign-status.json').read_text())['status'] != 'complete':
        raise ValueError('Parent campaign must be complete')
    if not json.loads((parent/'verification.json').read_text())['passed']:
        raise ValueError('Parent checkpoint verification must pass')
    prior = previous_ranges(output)
    check_card_range(cfg['evaluation_deal_seed'], cfg['evaluation_deals'], prior)
    parent_manifest = json.loads((parent/'source-manifest.json').read_text())
    frozen = {str(parent/name): sha for name, sha in parent_manifest['files'].items()}
    for name in ('config.json', 'source-manifest.json', 'verification.json'):
        frozen[str(parent/name)] = digest(parent/name)
    iteration = parent_cfg['iterations']
    for seed in cfg['seeds']:
        folder = parent/f'seed-{seed}'
        for name in ('latest.pt', f'current_{iteration:05d}.npz', f'mixture_{iteration:05d}.json'):
            frozen[str(folder/name)] = digest(folder/name)
        spec = json.loads((folder/f'mixture_{iteration:05d}.json').read_text())
        if [m['iteration'] for m in spec['models']] != list(range(1, iteration+1)) or any(m['weight'] != m['iteration'] for m in spec['models']):
            raise ValueError('Invalid parent policy mixture')
        for row in spec['models']:
            path = (folder/row['file']).resolve()
            if path.parent != folder:
                raise ValueError('Snapshot escaped parent seed folder')
            frozen[str(path)] = row['sha256']
    for name in ('probe_compact_refit.py', 'evaluate_compact.py', 'fullhouse_v4_runtime.py',
                 'fullhouse_runtime.py', 'fullhouse_v3_runtime.py', 'regret_partition.py'):
        frozen[str(ROOT/'scripts'/name)] = digest(ROOT/'scripts'/name)
    frozen[str(Path(__file__).resolve())] = digest(Path(__file__))
    registry_path = ROOT/'artifacts/selfplay/current.json'
    registry = json.loads(registry_path.read_text()); opponent = ROOT/registry['model']
    frozen[str(registry_path)] = digest(registry_path); frozen[str(opponent)] = registry['sha256']
    engine = ROOT/'.cache/noregrets/target/release/poker-bridge'
    frozen[str(engine)] = digest(engine)
    for name, sha in frozen.items():
        if digest(name) != sha:
            raise ValueError('Frozen artifact changed: '+name)
    from fullhouse_runtime import Fullhouse
    public = Fullhouse()
    frozen[str(public.model)] = public.digest
    output.mkdir(parents=True, exist_ok=False)
    lock = (output/'probe.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output/'logs').mkdir(); (output/'evaluation').mkdir()
    plan = dict(created_utc=utc(), parent=str(parent), parent_iteration=iteration, config=cfg,
                config_source_sha256=digest(config), frozen_files=frozen, prior_card_ranges=prior,
                card_range=[cfg['evaluation_deal_seed'], cfg['evaluation_deal_seed']+cfg['evaluation_deals']],
                note='No original run, self-play registry, controller, or deployment modification')
    atomic_json(output/'plan.json', plan); plan_sha = digest(output/'plan.json')
    atomic_json(output/'config.json', cfg)
    began = time.monotonic(); active = {}; finished = []; phase = 'fitting'
    event_log = (output/'probe.log').open('a', buffering=1)
    def status(state, pending=(), **extra):
        value = dict(status=state, phase=phase, updated_utc=utc(), seconds=time.monotonic()-began,
                     supervisor_pid=os.getpid(), supervisor_start=process_start(os.getpid()),
                     plan_sha256=plan_sha, finished=finished[:], pending=list(pending),
                     running={name:p['process'].pid for name,p in active.items()},
                     completed_evaluations=sum(name.startswith('eval/') for name in finished),
                     planned_evaluations=len(cfg['seeds'])*3*3, **extra)
        atomic_json(output/'status.json', value)
    def terminate(*unused):
        raise InterruptedError('Probe interrupted')
    previous_signals = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    def execute(jobs):
        pending = list(jobs)
        while pending or active:
            while pending and len(active) < cfg['parallelism']:
                job = pending.pop(0); name = job['id']
                log = (output/'logs'/(name.replace('/', '-')+'.log')).open('w')
                child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[name] = dict(process=child, log=log, started=time.monotonic(), timeout=job['timeout'])
                event_log.write(json.dumps(dict(event='started', job=name, pid=child.pid, utc=utc()))+'\n')
            for name, row in list(active.items()):
                result = row['process'].poll()
                if result is None:
                    if time.monotonic()-row['started'] > row['timeout']:
                        raise TimeoutError('Probe job exceeded budget: '+name)
                    continue
                row['log'].close(); del active[name]
                if result:
                    raise RuntimeError(f'Probe job failed: {name}, exit {result}')
                finished.append(name); event_log.write(json.dumps(dict(event='finished', job=name, utc=utc()))+'\n')
            status('running', [j['id'] for j in pending])
            if pending or active:
                time.sleep(2)
    try:
        fit_jobs = [dict(id=f'fit/{seed}', timeout=1200,
                        command=[str(PYTHON), str(ROOT/'scripts/probe_compact_refit.py'), '--parent', str(parent),
                                 '--output', str(output/f'seed-{seed}'), '--seed', str(seed),
                                 '--extra-steps', str(cfg['extra_steps'])]) for seed in cfg['seeds']]
        execute(fit_jobs)
        for seed in cfg['seeds']:
            result = json.loads((output/f'seed-{seed}/results.json').read_text())
            if not result.get('complete') or not result.get('opening_unchanged') or not result.get('scales_unchanged'):
                raise ValueError('Refit provenance/preservation checks failed')
        phase = 'evaluating'; jobs = []; expected = {}
        common = dict(evaluator_sha256=frozen[str(ROOT/'scripts/evaluate_compact.py')],
                      runtime_sha256=frozen[str(ROOT/'scripts/fullhouse_v4_runtime.py')],
                      engine_sha256=frozen[str(engine)], source_features_sha256=digest(parent/'source/bot/features.py'))
        for variant in cfg['variants']:
            for seed in cfg['seeds']:
                model = output/f'seed-{seed}/candidate.npz' if variant == 'refit' else parent/f'seed-{seed}'/(
                    f'current_{iteration:05d}.npz' if variant == 'current' else f'mixture_{iteration:05d}.json')
                for field in cfg['evaluation_fields']:
                    name = f'{field}-seed-{seed}-{variant}'
                    job = dict(id='eval/'+name, name=name, field=field, variant=variant, seed=seed, model=str(model), timeout=1900,
                               command=[str(PYTHON), str(ROOT/'scripts/evaluate_compact.py'), '--model', str(model),
                                        '--source', str(parent/'source'), '--field', field,
                                        '--deals', str(cfg['evaluation_deals']), '--deal-seed', str(cfg['evaluation_deal_seed']),
                                        '--output', str(output/'evaluation'/name)])
                    jobs.append(job)
                    expected[name] = dict(common, candidate=digest(model),
                        opponents=dict(noregrets=registry['sha256'], public=public.digest if field in ('public','mixed') else None))
        atomic_json(output/'evaluation-plan.json', dict(jobs=jobs, expected=expected))
        evaluation_plan_sha = digest(output/'evaluation-plan.json')
        execute(jobs)
        phase = 'verifying'; status('running')
        rows = {job['name']: verify_result(output/'evaluation'/job['name'], job, cfg, expected[job['name']]) for job in jobs}
        for name, sha in frozen.items():
            if digest(name) != sha:
                raise ValueError('Frozen input/code changed during probe: '+name)
        if digest(output/'plan.json') != plan_sha or digest(output/'evaluation-plan.json') != evaluation_plan_sha:
            raise ValueError('Predeclared experiment plan changed')
        summary = summarize(rows, cfg)
        summary.update(plan_sha256=plan_sha, evaluation_plan_sha256=evaluation_plan_sha, config=cfg,
                       parent=str(parent), completed_utc=utc(), seconds=time.monotonic()-began,
                       raw_hands_verified=True, source_provenance_verified=True)
        atomic_json(output/'summary.json', summary)
        phase = 'complete'; status('complete')
        print(json.dumps(dict(output=str(output), complete=True, evaluations=len(rows), decisions=summary['decisions'])), flush=True)
        return summary
    except BaseException as error:
        phase = 'interrupted' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'failed'
        status(phase, error=str(error)); raise
    finally:
        for row in active.values():
            if row['process'].poll() is None:
                try: os.killpg(row['process'].pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for row in active.values():
            try: row['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(row['process'].pid, signal.SIGKILL)
                except ProcessLookupError: pass
                row['process'].wait()
            row['log'].close()
        for sig, handler in previous_signals.items(): signal.signal(sig, handler)
        event_log.close(); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(args.parent, args.output, args.config)
