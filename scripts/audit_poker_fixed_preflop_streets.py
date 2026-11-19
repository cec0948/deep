"""Attribute paired endpoint payoffs to the street of the first differing hero decision."""
from contextlib import closing

import numpy as np

from audit_poker_fixed_preflop_window import (
    ARM, DEALS, FIELDS, PARENT, SEATS, SEEDS, START, STUDY, FirstActionDifferences,
    Bridge, atomic_json, check, digest, match, read, utc, verified_cell,
)


class StreetDifferences(FirstActionDifferences):
    def __init__(self, engine):
        super().__init__(engine)
        for street in ('flop', 'turn', 'river'):
            self.chips[street] = np.zeros(DEALS, dtype=np.int64)

    def hero_group(self, prefix, hero):
        # With a fixed button and starting stacks, an identical public betting
        # history reaches the same street and acting seat for every deck.
        key = tuple((a['seat'], a['kind'], a.get('to')) for a in prefix)
        if key not in self.nodes:
            state = self.bridge.call(dict(op='reset', seed=START, button=0))
            for i, action in enumerate(prefix, 1):
                check(not state['terminal'] and state['seat_to_act'] == action['seat'], 'Invalid common public history')
                state = self.bridge.call(dict(op='step', action=action))
                check(not state['terminal'], 'Common history ended before divergence')
                self.nodes[key[:i]] = state['street'], state['seat_to_act']
        street, actor = self.nodes[key]
        check(actor == hero and street in ('flop', 'turn', 'river'), 'Unexpected street or actor at first hero difference')
        return street


def audit_streets():
    plan = read(STUDY/'evaluation-plan.json')
    outcome = read(STUDY/'summary.json')
    fields = {}
    postflop = ('flop', 'turn', 'river')
    with closing(Bridge()) as engine:
        for field in FIELDS:
            traces = StreetDifferences(engine)
            deltas = []
            for seed in SEEDS:
                cells = [verified_cell(plan, outcome['result_sha256'], label, seed, field) for label in (PARENT, ARM)]
                before, after = match(cells, traces, engine)
                deltas.append(after-before)
            paired = np.mean(deltas, axis=0)
            groups, _ = traces.results(paired)
            fields[field] = {street: groups[street] for street in postflop}
            other_groups = ('same_trace', 'opponent_first_after_hero_fold', 'opponent_first_other')
            for group in other_groups:
                fields[field][group] = groups[group]
    return fields


def write_result(fields):
    audit = read(STUDY/'final-audit.json')
    check(audit['summary_sha256'] == digest(STUDY/'summary.json'), 'Main audit is stale')
    streets = ('flop', 'turn', 'river')
    for field, groups in fields.items():
        main = audit['first_action_divergence'][field]
        for key in ('paired_hands', 'nonzero_hero_payoff_changes', 'contribution_bb100'):
            np.testing.assert_allclose(sum(groups[s][key] for s in streets), main['postflop'][key], rtol=0, atol=1e-9)
        check(all(groups[k] == main[k] for k in groups if k not in streets), 'Opponent-first or common-trace group mismatch')
    result = dict(checked_utc=utc(), summary_sha256=audit['summary_sha256'], final_audit_sha256=digest(STUDY/'final-audit.json'),
                  independent_card_blocks=DEALS, raw_hand_records=2*DEALS*SEATS*len(SEEDS)*len(FIELDS),
                  paired_hands_per_field=DEALS*SEATS*len(SEEDS), endpoint_minus_parent_by_first_divergence=fields,
                  scope='Exploratory changes in terminal payoff per 100 of all hands; street is the first differing '
                        'hero action in a common public betting history. Opponent-first pairs are included separately.')
    atomic_json(STUDY/'postflop-street-audit.json', result)


if __name__ == '__main__':
    write_result(audit_streets())
