from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_postflop_averaging import SLOTS, VerifiedMixture, first_difference, paired_stats, private_slot


class SelectionProbe:
    def __init__(self, iteration):
        self.iteration = iteration

    def decide(self, state, seed):
        return self.iteration


def test_private_selection_lasts_one_hand_and_keeps_fixed_preflop():
    policy = VerifiedMixture.__new__(VerifiedMixture)
    policy.preflop = SelectionProbe(480)
    policy.snapshots = {t:SelectionProbe(t) for t in SLOTS}
    values = set()
    for hand_seed in range(100):
        policy.begin_hand(hand_seed)
        assert policy.decide({'street':'preflop'}, 1) == 480
        for seed, street in enumerate(('flop', 'turn', 'river')):
            assert policy.decide({'street':street}, seed) == private_slot(hand_seed)
        values.add(private_slot(hand_seed))
    assert values == set(SLOTS)


def test_paired_stats_clusters_repeated_seeds_by_original_card():
    baseline = np.array([[12,-1,4],[17,-4,3],[41,20,-18]], dtype=float)
    change = np.array([[1,5,2],[10,12,15],[-5,1,-11]], dtype=float)
    stats = paired_stats(baseline, baseline+change, 2)
    clusters = [2,6,2]
    mean = np.mean(clusters)
    radius = 2*np.std(clusters,ddof=1)/np.sqrt(3)
    assert stats['mean'] == pytest.approx(mean)
    assert stats['interval'] == pytest.approx([mean-radius,mean+radius])
    assert stats['fixed_seed_changes'] == pytest.approx(change.mean(axis=1))
    with pytest.raises(ValueError):
        paired_stats(baseline.ravel(),baseline.ravel(), 2)


def test_first_divergence_requires_same_state_and_hero_action():
    head = [('preflop',2,'call',None)]
    left = head+[('turn',4,'call',None)]
    right = head+[('turn',4,'raise',400)]
    assert first_difference(left,right,4) == 'turn'
    assert first_difference(left,left,4) == 'same_trace'
    with pytest.raises(ValueError):
        first_difference(left,right,2)
    with pytest.raises(ValueError):
        first_difference(left,head,4)


def test_streaming_ledger_audit_rejects_balanced_but_incorrect_payoffs(tmp_path):
    import gzip
    import json
    from contextlib import closing
    from audit_postflop_averaging import Bridge, verify_terminals
    from fullhouse_runtime import BRIDGE

    if not BRIDGE.exists():
        pytest.skip('Requires the pinned rules checkout and rule engine')
    row = dict(seed=918,hero=0,actions=[])
    with closing(Bridge()) as engine:
        state = engine.call(dict(op='reset',seed=row['seed'],button=0))
        while not state['terminal']:
            act = dict(kind='check' if state['can_check'] else 'fold', seat=state['seat_to_act'])
            row['actions'].append(act)
            state = engine.call(dict(op='step',action=act))
    row.update(utilities=state['utilities'], chips=state['utilities'][0], steps=len(row['actions']))
    file = tmp_path/'trace.jsonl.gz'

    def write_row():
        with gzip.open(file, 'wt') as stream:
            stream.write(json.dumps(row)+'\n')

    write_row()
    assert verify_terminals(file) == dict(hands=1,decisions=row['steps'])
    row['utilities'][1] += 1
    row['utilities'][2] -= 1
    write_row()
    with pytest.raises(ValueError,match='terminal payout mismatch'):
        verify_terminals(file)


def test_audit_accepts_capped_tradeoff_and_rejects_wrong_uncertainty():
    from audit_postflop_averaging import END, PARENT, SEEDS, audit_pair
    from score_poker_population import score_panel

    groups = {'public':dict(role='primary',family='public',deal_seed=10,deals=4),
              'strong-win':dict(role='primary',family='other',deal_seed=10,deals=4)}
    offsets = np.arange(12).reshape(3,4).astype(float)
    data = {'public':{PARENT:offsets-80,END:offsets-60},
            'strong-win':{PARENT:offsets+400,END:offsets+300}}
    panel = {f:{label:dict(zip(SEEDS,matrix)) for label,matrix in models.items()} for f,models in data.items()}
    summary = score_panel(panel,groups,list(SEEDS),baseline=PARENT,candidate=END)
    audit_pair(summary,groups,data,PARENT,END)
    changed = summary['roles']['primary']['family_balanced_capped_bb100']['100']['paired_change']
    assert changed['mean'] == pytest.approx(10)
    changed['lower'] -= 10
    with pytest.raises(ValueError,match='capped confidence interval'):
        audit_pair(summary,groups,data,PARENT,END)
