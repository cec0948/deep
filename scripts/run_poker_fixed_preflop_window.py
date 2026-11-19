"""Run a fixed 24-round online postflop learning window from the confirmed 480 policy."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
from statistics import NormalDist

import run_poker_preflop_confirmation as confirmation
from controlled_poker_refit import read, utc, verify_files
from evaluate_compact_learning_curve import adoption_conditions
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from run_poker_preflop_pilot import CAMPAIGN, FIELDS, SEEDS, paired
from train_poker_fixed_preflop_window import ARM, FINAL, INITIAL, OUTPUT


WORKER = Path(__file__).with_name('train_poker_fixed_preflop_window.py').resolve()
PARENT = 'specialist480'
START, DEALS = 2037800000, 20000
POSTFLOP_STEPS, BATCH = 768, 512
CONFIRM_SUMMARY = '52d1a10ff493680189afd12fdef4408b4bd8dc980ce56786d7ad7dcba4dfdff1'
CONFIRM_AUDIT = 'b847f113a1965302619517d2395706f0d2070915c5447de417b453b0ef720930'
DOSE_SUMMARY = '6298d5e5a89f3212bed1f706376e41b298ec06d60f70e0f32d474358946e9610'
DOSE_AUDIT = '5f502fe9459f7f985166ba5b138ce35c9460161042e49c2847d4004170aa50c3'
PARENT_BUNDLE = '424b8e76a31f359f21de47ec02c58fd3a902ab2639f9f7e9a8785ff253d63f2e'


def frozen_inputs():
    models, opponents, frozen = confirmation.frozen_pilot()
    parents = models['specialist']
    dose = confirmation.FOLLOWUP/'preflop-dose'
    confirm = confirmation.OUTPUT
    reference = confirm/'backup-reference.json'
    frozen.update({str(dose/'summary.json'): DOSE_SUMMARY, str(dose/'final-audit.json'): DOSE_AUDIT,
                   str(confirm/'summary.json'): CONFIRM_SUMMARY, str(confirm/'final-audit.json'): CONFIRM_AUDIT,
                   str(reference): '48b1d97868a096e0cdde59b52f8c4ec7a3754de14e3487070aa0513c1c10b069'})
    verify_files(frozen)
    conf_summary, dose_summary = read(confirm/'summary.json'), read(dose/'summary.json')
    for directory, summary in ((confirm, conf_summary), (dose, dose_summary)):
        frozen[str(directory/'evaluation-plan.json')] = summary['evaluation_plan_sha256']
    verify_files(frozen)
    conf_plan = read(confirm/'evaluation-plan.json')
    dose_plan = read(dose/'evaluation-plan.json')
    parent_hashes = {r['sha256'] for r in read(reference)['existing_backups']}
    if (not conf_summary['independent_confirmation_pass']
            or read(confirm/'final-audit.json')['summary_sha256'] != CONFIRM_SUMMARY
            or read(dose/'final-audit.json')['summary_sha256'] != DOSE_SUMMARY
            or conf_plan['policies']['specialist'] != parents or dose_plan['policies']['steps0'] != parents
            or conf_plan['opponents'] != opponents or dose_plan['opponents'] != opponents
            or not {PARENT_BUNDLE, confirmation.pilot.CFG['restore_dependency_480_sha256']} <= parent_hashes):
        raise ValueError('Confirmed parent or paired diagnosis does not match the selected 480 specialist')
    cfg = read(CAMPAIGN/'config.json')
    frozen.update({p: h for p, h in cfg['frozen_files'].items()
                   if Path(p).is_relative_to(CAMPAIGN/'source') or Path(p).is_relative_to(CAMPAIGN/'tools')})
    stage = read(CAMPAIGN.parent/'stage-ready.json')
    for seed, ref in parents.items():
        folder = CAMPAIGN/f'seed-{seed}'
        parent = stage['seeds'][seed]
        frozen[str(folder/'current.json')] = digest(folder/'current.json')
        frozen.update({str(Path(parent['checkpoint'])/name): sha for name, sha in parent['files'].items()})
        for asset in (ref, *[read(ref['path'])[street] for street in ('preflop', 'postflop')]):
            frozen[asset['path']] = asset['sha256']
    scripts = ('run_poker_preflop_pilot','run_poker_preflop_online','train_poker_warm_preflop',
               'train_poker_average_continuation','poker_strategy_average','compact_lightweight_backup',
               'snapshot_poker_preflop_online','controlled_poker_refit','poker_preflop_specialist',
               'evaluate_compact_learning_curve','fullhouse_v3_common','fullhouse_runtime')
    for path in [Path(__file__).resolve(), WORKER, CAMPAIGN/'config.json',
                 *(ROOT/'scripts'/f'{name}.py' for name in scripts)]:
        frozen[str(path)] = digest(path)
    verify_files(frozen)
    return parents, opponents, frozen


def check_workers():
    confirmation.check_no_jobs()
    names = {WORKER.name, Path(__file__).name, 'train_poker_warm_preflop.py',
             'run_poker_warm_preflop.py', 'run_poker_preflop_dose.py'}
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid() or not process_start(int(proc.name)):
            continue
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'status' in argv:
            continue
        if len(argv) > 1 and os.fsdecode(argv[1]).split('/')[-1] in names:
            raise RuntimeError(f'Existing poker worker or supervisor {proc.name}')


def verify_training(plan):
    assets, summaries = {}, {}
    stage = read(CAMPAIGN.parent/'stage-ready.json')
    cfg = read(CAMPAIGN/'config.json')
    for seed in SEEDS:
        key = str(seed)
        folder = OUTPUT/ARM/f'seed-{seed}'
        ready, saved = read(folder/'ready.json'), read(folder/'current.json')
        start = read(folder/'start-verification.json')
        checkpoint = folder/'checkpoints'/f'iter-{FINAL:05d}'
        verify_files({folder/'diagnostics.jsonl': ready['diagnostics_sha256'],
                      folder/'start-verification.json': ready['start_sha256'],
                      ready['policy']['path']: ready['policy']['sha256'],
                      ready['preflop_export']['path']: ready['preflop_export']['sha256'],
                      **{checkpoint/name: sha for name, sha in saved['files'].items()}})
        ref = plan['parent_policies'][key]
        parent = read(ref['path'])
        final = read(ready['policy']['path'])
        if (saved['iteration'] != FINAL or Path(ready['checkpoint']) != checkpoint
                or saved['directory'] != checkpoint.name or ready['files'] != saved['files']
                or start['deployed_policy_sha256'] != ref['sha256']
                or start['checkpoint_files'] != stage['seeds'][key]['files']
                or not start['replay_root_reference_and_rng_verified']
                or start['seed'] != seed or start['iteration'] != INITIAL
                or start['max_probability_error'] > 1e-6 or min(start['action_probability_probes'].values()) < 1
                or final['format'] != parent['format'] or final['preflop'] != ready['preflop_export']
                or final['preflop']['sha256'] != parent['preflop']['sha256']
                or final['postflop'] != dict(path=str(checkpoint/'current.npz'), sha256=saved['files']['current.npz'])
                or final['postflop']['sha256'] == parent['postflop']['sha256']):
            raise ValueError('Final checkpoint, street exports or restored parent evidence differs')
        rows = [json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
        if [row['iteration'] for row in rows] != list(range(INITIAL+1, FINAL+1)):
            raise ValueError('Missing or duplicate online iteration')
        previous = start['weights']
        for row in rows:
            current = row['output_weights']
            sampled, selected = row['fit_draws_by_street'], row['fit_rows_by_street']
            if (row['seed'] != seed or row['input_weights'] != previous
                    or current['preflop'] != start['weights']['preflop'] or current['general'] == previous['general']
                    or row['frozen_preflop_max_probability_error'] != 0 or row['frozen_preflop_probes'] <= 0
                    or row['general_steps'] != POSTFLOP_STEPS or row['preflop_steps'] != 0 or row['opening_steps'] != 0
                    or not math.isfinite(row['main_loss']) or len(sampled) != 3 or len(selected) != 3
                    or min(sampled) <= 0 or min(selected) <= 0 or sum(selected) > 600000
                    or sum(sampled) != row['fit_draws'] or row['fit_draws'] != POSTFLOP_STEPS*BATCH
                    or row['traverser'] != (row['iteration']-1)%6 or row['traversals'] != cfg['traversals']
                    or row['new_regret_rows'] <= 0 or row['root_rows'] != 169*cfg['root_repetitions']
                    or row['buffer_rows'] != [100000]*6 or row['reference_collection']['rows'] != 100000
                    or row['reference_collection']['hands'] != cfg['reference_hands']
                    or (row['iteration'] % 6 == 0 and row.get('checkpoint_iteration') != row['iteration'])):
                raise ValueError(f'Seed {seed}, iteration {row["iteration"]}: invalid training or fitting evidence')
            previous = current
        summaries[key] = dict(start=start, iteration=FINAL, updates=len(rows), final_weights=previous,
            general_steps=len(rows)*POSTFLOP_STEPS, preflop_steps=0, opening_steps=0,
            traversals=sum(r['traversals'] for r in rows), new_regret_rows=sum(r['new_regret_rows'] for r in rows),
            opening_samples=sum(r['root_rows'] for r in rows), reference_hands=sum(r['reference_collection']['hands'] for r in rows),
            fit_draws_by_street=[sum(r['fit_draws_by_street'][i] for r in rows) for i in range(3)],
            checkpoints=[r['checkpoint_iteration'] for r in rows if r['iteration']%6 == 0],
            final_average=rows[-1]['average_fit'], all_preflop_unchanged=True)
        assets[key] = ready
    return dict(training_complete=True, iteration=FINAL, arms={ARM: assets}, completed_utc=utc(),
                warm_start_weight_chain_verified=True, seed_verification=summaries)


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT/'summary.json').is_file():
            return read(OUTPUT/'summary.json')
        if OUTPUT.exists():
            raise RuntimeError(f'Study already exists; inspect its status and original process in {OUTPUT}')
        check_workers()
        previous = confirmation.check_fresh_cards(OUTPUT, START, DEALS)
        parents, opponents, frozen = frozen_inputs()
        if shutil.disk_usage(OUTPUT.parent).free < 1024**3:
            raise RuntimeError('Insufficient local space for verified raw records and recovery')
        OUTPUT.mkdir(exist_ok=False)
        try:
            plan = dict(created_utc=utc(), initial_iteration=INITIAL, final_iteration=FINAL, arms=[ARM],
                seeds=SEEDS, fields=FIELDS, campaign=str(CAMPAIGN), parent_policies=parents,
                parent_archive_sha256=confirmation.pilot.CFG['restore_dependency_480_sha256'],
                parent_policy_bundle_sha256=PARENT_BUNDLE, frozen_files=frozen,
                source_dose_summary_sha256=DOSE_SUMMARY, source_dose_audit_sha256=DOSE_AUDIT,
                source_confirmation_summary_sha256=CONFIRM_SUMMARY, source_confirmation_audit_sha256=CONFIRM_AUDIT,
                general_steps=POSTFLOP_STEPS, preflop_steps=0, opening_steps=0, main_batch=BATCH,
                checkpoint_every=6, training='Fixed full preflop specialist, online traversal, postflop-only replay fitting',
                statistical_unit='Original card block, average over its six seats and three fixed seeds',
                evaluation_deal_seed=START, evaluation_deals=DEALS, planned_evaluation_cells=18, planned_hands=2160000,
                recommendation_criteria='504 minus specialist480: nominal public lower > 0, at least two positive public seeds, '
                                        'nominal mixed lower >= -25, nominal NoRegrets lower >= -50 bb/100',
                interval='Nominal 95% intervals by field and Bonferroni simultaneous 95% across three fields',
                decision='Passing qualifies for an independent evaluation. No automatic promotion or extension.',
                automatic_retry=False, automatic_extension=False, model_promotion=False)
            atomic_json(OUTPUT/'plan.json', plan)
            atomic_json(OUTPUT/'config.json', dict(evaluation_deal_seed=START, evaluation_deals=DEALS,
                prior_card_ranges=previous, initial_iteration=INITIAL, final_iteration=FINAL))
            plan_sha = digest(OUTPUT/'plan.json')
            commands = {f'{ARM}-seed-{s}': [str(PYTHON), str(WORKER), 'worker', str(s)] for s in SEEDS}
            confirmation.run_jobs(OUTPUT, commands, len(SEEDS), 'training')
            verify_files({**frozen, str(OUTPUT/'plan.json'): plan_sha})
            stage = verify_training(plan)
            atomic_json(OUTPUT/'stage-ready.json', stage)
            policies = {PARENT: parents, ARM: {s: v['policy'] for s, v in stage['arms'][ARM].items()}}
            evidence = {**frozen, str(OUTPUT/'stage-ready.json'): digest(OUTPUT/'stage-ready.json'), str(OUTPUT/'plan.json'): plan_sha}
            for ready in stage['arms'][ARM].values():
                policy = ready['policy']
                final = read(policy['path'])
                for ref in (policy, final['preflop'], final['postflop']):
                    evidence[ref['path']] = ref['sha256']
            verify_files(evidence)
            rows, outcome = confirmation.evaluate(OUTPUT, policies, opponents, evidence, START, DEALS)
            change = paired(rows, PARENT, ARM, DEALS)
            z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
            simultaneous = {}
            for field, estimate in change.items():
                half = estimate['ci95_half_width']*z/1.96
                simultaneous[field] = dict(lower=estimate['mean']-half, upper=estimate['mean']+half)
            conditions = adoption_conditions(change)['conditions']
            outcome.update(completed=True, completed_utc=utc(), new_selfplay_iterations_per_seed=FINAL-INITIAL,
                new_cells=18, reused_cells=0, new_hands=2160000, trained_minus_frozen_parent=change,
                simultaneous_95_bonferroni=simultaneous, selection_conditions=conditions,
                candidate_recommended=all(conditions.values()), recommendation_scope='independent confirmation vs parent480',
                warm_start_weight_chain_verified=True, entire_preflop_frozen=True, model_promoted=False, automatic_extension=False)
            atomic_json(OUTPUT/'summary.json', outcome)
            return outcome
        except BaseException as exc:
            atomic_json(OUTPUT/'failure.json', dict(observed_utc=utc(), error=f'{type(exc).__name__}: {exc}'))
            raise


def observed_process(pid, token, expected_argv=None):
    proc = Path('/proc')/str(pid)
    try:
        argv = [os.fsdecode(arg) for arg in (proc/'cmdline').read_bytes().split(b'\0') if arg]
        if expected_argv is None:
            matches = any((proc/'cwd'/arg).resolve() == Path(__file__).resolve()
                          for arg in argv[1:] if arg.endswith('.py')) and 'status' not in argv
        else:
            matches = argv == expected_argv
        return dict(pid=pid, start_token=token, live=bool(token and process_start(pid) == token and matches),
                    argv=argv if matches else None)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return dict(pid=pid, start_token=token, live=False, argv=None)


def status():
    report = dict(checked_utc=utc(), study=OUTPUT.name)
    if (OUTPUT/'failure.json').is_file():
        return dict(report, state='failed', failure=read(OUTPUT/'failure.json'))
    if (OUTPUT/'summary.json').is_file():
        summary = read(OUTPUT/'summary.json')
        return dict(report, state='complete', completed_utc=summary['completed_utc'],
                    verified_hands=summary['verified_hands'], completed_cells=summary['completed_cells'],
                    selection_conditions=summary['selection_conditions'])
    for phase in ('evaluation', 'training'):
        filename = OUTPUT/f'{phase}-status.json'
        if not filename.is_file():
            continue
        record = read(filename)
        workers = {}
        for name, child in record['active'].items():
            if phase == 'training':
                folder = OUTPUT/ARM/('seed-'+name.rsplit('-',1)[1])
            else:
                folder = OUTPUT/'evaluation'/name
            progress = read(folder/'status.json') if (folder/'status.json').is_file() else {}
            worker = observed_process(child['pid'], child['process_start'], child['argv'])
            worker['progress'] = {k: progress[k] for k in ('status','phase','iteration',
                                  'last_committed_iteration','completed_card_blocks','verified_hands') if k in progress}
            workers[name] = worker
        supervisor = observed_process(record['pid'], record['process_start'])
        return dict(report, state=record['state'] if supervisor['live'] else 'source_process_not_confirmed',
                    phase=phase, recorded_utc=record['observed_utc'], supervisor=supervisor, workers=workers,
                    completed=record['completed'], planned=record['planned'])
    return dict(report, state='preparing' if OUTPUT.exists() else 'not_started')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('run', 'status'), default='run')
    action = parser.parse_args().action
    result = status() if action == 'status' else run()['trained_minus_frozen_parent']
    print(json.dumps(result, indent=2, allow_nan=False))
