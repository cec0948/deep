"""Split the registered one-decision contrast by fold/check/call/raise using the same terminal returns."""
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from score_poker_flop_diagnostic import LINEUPS, action_values, json_lines, probabilities

GROUPS = ('fold', 'check', 'call', 'raise')
REL = Path('preflop-followup/480-v1/flop-action-diagnostic-20261123')
DESIGN = ROOT/'reports'/REL/'action-attribution-registration.json'
DESIGN_SHA = 'aefcdb0121496aa2287fbe6fb7e395f340cdb4d1a4c82e49566f74d8b0dad3f7'
SEEDS = (342,343,344)
REPEATS = 8
PER_SEED = 1024


def totals_by_kind(weights, kinds):
    return np.bincount(kinds, weights=weights, minlength=len(GROUPS))


def decompose(study):
    verify_files({DESIGN:DESIGN_SHA, study/'registration.json':read(DESIGN)['parent_registration_sha256']})
    official = read(study/'summary.json')
    mass = np.zeros((len(SEEDS),2,len(GROUPS)))
    contribution = np.zeros((len(SEEDS),len(LINEUPS),len(GROUPS)))
    seen = set()
    input_hashes = {}
    for s, seed in enumerate(SEEDS):
        folder = study/f'seed-{seed}'
        sample, returns = folder/'states.jsonl', folder/'values.jsonl.gz'
        for path in (sample,returns):
            name = str(path.relative_to(study))
            verify_files({path:official['input_sha256'][name]})
            input_hashes[name] = official['input_sha256'][name]
        outcomes = json_lines([returns])
        n = 0
        for state in json_lines([sample]):
            if state['seed'] != seed or state['deal_seed'] in seen:
                raise ValueError('Unexpected seed or duplicated original deal')
            seen.add(state['deal_seed'])
            p = probabilities(state)
            parent, mixture = p[0], p[1:].mean(axis=0)
            difference = mixture-parent
            kinds = [GROUPS.index(action['kind']) for action in state['legal_actions']]
            mass[s,0] += totals_by_kind(parent,kinds)
            mass[s,1] += totals_by_kind(mixture,kinds)
            rows = list(islice(outcomes,len(LINEUPS)))
            if Counter(r['lineup'] for r in rows) != Counter(LINEUPS):
                raise ValueError('Missing or duplicated matched opponent field')
            for row in rows:
                if any(row[k] != state[k] for k in ('seed','deal_seed','hero','state_id','legal_actions')):
                    raise ValueError('Root identity or action order differs from the sample')
                q = action_values(row,np.any(p>0,axis=0),REPEATS,100)
                baseline = np.dot(parent,q)
                terms = difference*(q-baseline)
                if not np.isclose(terms.sum(),difference@q,rtol=0,atol=1e-10):
                    raise ValueError('Centered contributions failed to reconstruct one root effect')
                contribution[s,LINEUPS.index(row['lineup'])] += totals_by_kind(terms,kinds)
            n += 1
        if n != PER_SEED or next(outcomes,None) is not None:
            raise ValueError('Attribution requires exactly the registered roots and lineups')
        mass[s] /= n
        contribution[s] /= n
    pct = 100*mass.mean(axis=0)
    by_lineup = contribution.mean(axis=0)
    if not np.allclose(pct.sum(axis=1),100,rtol=0,atol=1e-9):
        raise ValueError('Grouped root action frequencies do not reconcile')
    comparisons = official['primary_mixture_minus_480']['contrasts']
    if not all(np.isclose(by_lineup[f].sum(),comparisons[name]['mean_bb'],rtol=0,atol=1e-9)
               for f,name in enumerate(LINEUPS)):
        raise ValueError('Attribution differs from the official primary contrast')
    return dict(verified_utc=utc(), registration_sha256=DESIGN_SHA, script_sha256=digest(__file__),
                source_summary_sha256=digest(study/'summary.json'), input_sha256=input_hashes,
                original_eligible_deals=len(seen), fixed_training_seeds=list(SEEDS), repeats_per_action=REPEATS,
                primary_mixture_action_mass={kind:dict(baseline_pct=pct[0,i],mixture_pct=pct[1,i],
                    difference_percentage_points=pct[1,i]-pct[0,i]) for i,kind in enumerate(GROUPS)},
                contribution_bb_per_first_flop_decision={lineup:{
                    kind:float(by_lineup[f,i]) for i,kind in enumerate(GROUPS)} for f,lineup in enumerate(LINEUPS)},
                reconciled_primary_mean_bb={lineup:comparisons[lineup]['mean_bb'] for lineup in LINEUPS},
                interpretation='Signed accounting centered on the same-state 480 continuation value; '
                               'components depend on that reference and are not independent causal effects.')


if __name__ == '__main__':
    result = decompose(ROOT/'artifacts'/REL)
    atomic_json(ROOT/'reports'/REL/'action-attribution.json',result)
