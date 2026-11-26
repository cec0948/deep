"""Preserve the selected policy except for provably dominated heads-up river folds."""
from itertools import combinations

import eval7

from poker_street_progress import StreetProgressPolicy

DECK = tuple(eval7.Deck().cards)
STARTING_STACK = 10_000  # Fixed stack used by this six-seat rules engine.


def closing_call_floor(state):
    """Guaranteed chips won by calling instead of folding, or zero if unproved."""
    if state['terminal'] or state['street'] != 'river' or state['can_check']:
        return 0
    players = state['players']
    active = [p for p in players if not p['is_folded']]
    if len(active) != 2 or len(players) != 6:
        return 0
    hero = next(p for p in active if p['seat'] == state['seat_to_act'])
    other = next(p for p in active if p is not hero)
    owed = state['amount_owed']
    if not 0 < owed <= hero['stack'] or other['bet_this_street'] - hero['bet_this_street'] != owed:
        return 0

    # In heads-up play a full call of the outstanding river bet closes action.
    # Both live seats must have matched total contributions after calling and
    # be eligible for every chip, including dead bets from folded players.
    final_commit = STARTING_STACK - other['stack']
    if (hero['stack'] - owed != other['stack']
            or any(STARTING_STACK - p['stack'] > final_commit for p in players)
            or sum(STARTING_STACK - p['stack'] for p in players) != state['pot']):
        return 0

    board = [eval7.Card(c) for c in state['community_cards']]
    hole = [eval7.Card(c) for c in state['your_cards']]
    known = set(board + hole)
    if len(board) != 5 or len(hole) != 2 or len(known) != 7:
        raise ValueError('River decision needs seven distinct visible cards')
    hero_rank = eval7.evaluate(board + hole)
    unseen = [card for card in DECK if card not in known]
    tie_possible = False
    for first, second in combinations(unseen, 2):
        rank = eval7.evaluate(board + [first, second])
        if rank > hero_rank:
            return 0
        if rank == hero_rank:
            tie_possible = True
    final_pot = state['pot'] + owed
    payout = final_pot
    if tie_possible:
        # The rules distribute odd chips separately in each contribution tier.
        commitments = [STARTING_STACK - p['stack'] + (owed if p is hero else 0) for p in players]
        previous = 0
        payout = 0
        for level in sorted(set(commitments) - {0}):
            layer = (level - previous) * sum(amount >= level for amount in commitments)
            payout += layer // 2
            previous = level
    return max(0, payout - owed)


def guard_fold(state, reply):
    fold = next((item['probability'] for item in reply['distribution']
                 if item['action']['kind'] == 'fold'), 0.)
    if fold <= 0:
        return reply
    floor = closing_call_floor(state)
    if floor <= 0:
        return reply
    distribution = [dict(row) for row in reply['distribution'] if row['action']['kind'] != 'fold']
    for row in distribution:
        if row['action']['kind'] == 'call':
            row['probability'] += fold
            break
    else:
        distribution.append(dict(action=dict(kind='call'), probability=fold))
    action = {'kind': 'call'} if reply['action']['kind'] == 'fold' else reply['action']
    return dict(reply, action=action, distribution=distribution, policy_source='river_dominated_fold',
                redirected_fold_probability=fold, call_over_fold_minimum_chips=floor)


class RiverDominancePolicy(StreetProgressPolicy):
    def decide(self, state, seed):
        return guard_fold(state, super().decide(state, seed))


if __name__ == '__main__':
    from serve_poker_flop_repair import main
    main(RiverDominancePolicy, 'street-progress-river')
