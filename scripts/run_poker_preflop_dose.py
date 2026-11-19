"""Compare three additional nonopening preflop fitting budgets on fixed 480-round replay."""
import fcntl
import os
from pathlib import Path
from statistics import NormalDist

import run_poker_preflop_confirmation as confirmation
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest, process_start
from run_poker_preflop_pilot import FIELDS, SEEDS, paired


OUTPUT = confirmation.FOLLOWUP / 'preflop-dose'
START, DEALS = 2037700000, 20000
LABELS = ('steps0', 'steps64', 'steps256')
CONTRASTS = {'64_minus_0': ('steps0', 'steps64'), '256_minus_0': ('steps0', 'steps256'),
             '256_minus_64': ('steps64', 'steps256')}
SUMMARY_SHA = '263c20db53b9ef78a0ba1ac6081f58c323b8c3a2d3500a218ca8ab7bfd874dcf'
AUDIT_SHA = '7c476c13e5b91f377a896716762a03269ec2045eb083b44efd642fe12edb1a0a'


def verify_fit(seed, parent, policies):
    import numpy as np
    if set(policies) != set(LABELS):
        raise ValueError('Incomplete fitting budgets')
    original = read(parent['path'])
    for label, ref in policies.items():
        verify_files({ref['path']: ref['sha256']})
        spec = read(ref['path'])
        if (spec['format'] != original['format'] or spec['postflop'] != original['postflop']
                or (label == 'steps0' and spec != original)):
            raise ValueError(f'{seed}/{label} changed the frozen parent or postflop route')
        weight = spec['preflop']
        verify_files({weight['path']: weight['sha256']})
        with np.load(original['preflop']['path'], allow_pickle=False) as base, np.load(weight['path'], allow_pickle=False) as candidate:
            opening = [k for k in base.files if k.startswith('opening_')]
            if (not opening or set(base.files) != set(candidate.files)
                    or any(not np.array_equal(base[k], candidate[k]) for k in opening)
                    or any(not np.isfinite(candidate[k]).all() for k in candidate.files if candidate[k].dtype.kind in "fc")):
                raise ValueError('Opening weights, architecture or finite-value check failed')
    if len({read(ref['path'])['preflop']['sha256'] for ref in policies.values()}) != len(LABELS):
        raise ValueError('Expected three distinct preflop exports')


def run():
    from fit_poker_preflop_dose import fit
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT/'summary.json').is_file():
            return read(OUTPUT/'summary.json')
        confirmation.check_no_jobs()
        ranges = confirmation.check_fresh_cards(OUTPUT, START, DEALS)
        models, opponents, frozen = confirmation.frozen_pilot()
        swap = confirmation.FOLLOWUP/'preflop-swap'
        frozen.update({str(swap/'summary.json'): SUMMARY_SHA, str(swap/'final-audit.json'): AUDIT_SHA})
        for path in (Path(__file__).resolve(), Path(__file__).with_name('fit_poker_preflop_dose.py').resolve()):
            frozen[str(path)] = digest(path)
        verify_files(frozen)
        prior_audit = read(swap/'final-audit.json')
        parents = models['specialist']
        if prior_audit['summary_sha256'] != SUMMARY_SHA or any(
                read(ref['path'])['preflop']['sha256'] != prior_audit['per_seed_street_network_sha256'][s]['specialist480']['preflop']
                for s, ref in parents.items()):
            raise ValueError('Selected 480 preflop candidate differs from the audited comparison')
        OUTPUT.mkdir(exist_ok=False)
        try:
            design = dict(created_utc=utc(), evaluation_deal_seed=START, evaluation_deals=DEALS,
                          prior_card_ranges=ranges, steps=[0,64,256], batch=512, seeds=SEEDS, fields=FIELDS,
                          frozen_parent='480-round confirmed preflop specialist', source_summary_sha256=SUMMARY_SHA,
                          source_audit_sha256=AUDIT_SHA, fixed_replay_iteration=480, new_selfplay_iterations=0,
                          sample_rng="MT19937, seed 20261119 + training seed", optimizer="continuous Adam at 0.001",
                          shared_prefix=True, freeze_opening_and_postflop=True, model_promotion=False,
                          contrasts=list(CONTRASTS), planned_evaluation_cells=27, planned_hands=3240000,
                          intervals='Paired by original card block after averaging 6 seats and 3 fixed seeds; Bonferroni across all 9 estimates')
            atomic_json(OUTPUT/'design.json', design)
            atomic_json(OUTPUT/'config.json',dict(evaluation_deal_seed=START,evaluation_deals=DEALS))
            frozen[str(OUTPUT/'design.json')] = digest(OUTPUT/'design.json')
            policies = {label: {} for label in LABELS}
            proofs = {}
            for seed in SEEDS:
                atomic_json(OUTPUT/'fit-status.json', dict(state='fitting',seed=seed,observed_utc=utc(),
                           pid=os.getpid(),process_start=process_start(os.getpid()),completed_seeds=len(proofs)))
                folder = OUTPUT/f'seed-{seed}'
                refs = fit(seed, folder, parents[str(seed)])
                verify_fit(seed, parents[str(seed)], refs)
                for label, ref in refs.items():
                    policies[label][str(seed)] = ref
                    frozen[ref['path']] = ref['sha256']
                    for street in ('preflop','postflop'):
                        asset = read(ref['path'])[street]
                        frozen[asset['path']] = asset['sha256']
                proofs[str(seed)] = digest(folder/'fit.json')
                frozen[str(folder/'fit.json')] = proofs[str(seed)]
            verify_files(frozen)
            ready = dict(completed_utc=utc(),fitting_complete=True,policies=policies,seed_evidence_sha256=proofs)
            atomic_json(OUTPUT/'fit-ready.json', ready)
            frozen[str(OUTPUT/'fit-ready.json')] = digest(OUTPUT/'fit-ready.json')
            atomic_json(OUTPUT/'fit-status.json', dict(state='complete', completed_seeds=len(proofs), observed_utc=utc()))
            rows, result = confirmation.evaluate(OUTPUT, policies, opponents, frozen, START, DEALS)
            changes = {key: paired(rows, before, after, DEALS) for key, (before, after) in CONTRASTS.items()}
            z = NormalDist().inv_cdf(1-.05/(2*len(FIELDS)*len(changes)))
            simultaneous = {}
            for key, values in changes.items():
                simultaneous[key] = {}
                for field, estimate in values.items():
                    half = z * estimate['ci95_half_width']/1.96
                    simultaneous[key][field] = dict(lower=estimate['mean']-half, upper=estimate['mean']+half)
            verify_files(frozen)
            result.update(completed=True, completed_utc=utc(), new_cells=27, reused_cells=0, new_hands=3240000,
                          source_summary_sha256=SUMMARY_SHA, source_audit_sha256=AUDIT_SHA,
                          fit_ready_sha256=frozen[str(OUTPUT/'fit-ready.json')], comparisons=changes,
                          simultaneous_95=simultaneous, new_selfplay_iterations=0, model_promoted=False)
            atomic_json(OUTPUT/'summary.json', result)
            return result
        except BaseException as exc:
            atomic_json(OUTPUT/'failure.json', dict(observed_utc=utc(), error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__ == '__main__':
    run()
