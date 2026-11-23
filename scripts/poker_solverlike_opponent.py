"""Adapter for the fixed public SolverLike behavioral benchmark."""
import importlib.util
from pathlib import Path
import subprocess

from fullhouse_runtime import SOURCE
from fullhouse_v3_common import digest
from poker_reference_opponents import checked_action


UPSTREAM_COMMIT = 'e504793d480b1b975f25258d25939b45c6dbd5a4'
MODEL = SOURCE / 'bots/adversarial/solver_like.py'
MODEL_SHA256 = '76ef638a9a7e9762316ba34d54eca435bd37086afd9bb30433423346282a5240'
# Only fields read by the original bot are supplied. The trusted bridge already
# uses its action_log vocabulary: blind/deal entries, checks, calls and raises.
PUBLIC_FIELDS = ('your_cards', 'community_cards', 'street', 'pot', 'your_stack',
                 'your_bet_this_street', 'amount_owed', 'current_bet', 'action_log', 'can_check')


class SolverLikeOpponent:
    def __init__(self):
        actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != UPSTREAM_COMMIT or digest(MODEL) != MODEL_SHA256:
            raise ValueError('Unexpected upstream SolverLike policy')
        spec = importlib.util.spec_from_file_location('upstream_solver_like', MODEL)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        self.upstream = upstream
        self.model = MODEL
        self.digest = MODEL_SHA256
        self.dependency_hashes = {str(MODEL): MODEL_SHA256, str(Path(__file__).resolve()): digest(__file__)}

    def decide(self, state, seed):
        """Act using the source bot's deterministic hash of the visible cards."""
        if state['terminal']:
            raise ValueError('No decision in a terminal hand')
        view = {key: state[key] for key in PUBLIC_FIELDS}
        bounds = state['raise_bounds']
        # The original protocol always supplies a numeric minimum. When a raise
        # is unavailable its size is immaterial because it becomes a check/call.
        view['min_raise_to'] = bounds[0] if bounds else view['your_stack'] + view['your_bet_this_street']
        # The source entry point silently folds on every exception. Call its
        # same decision functions directly so a broken protocol fails the cell.
        decide = self.upstream._decide_preflop if view['street'] == 'preflop' else self.upstream._decide_postflop
        raw = decide(view)
        kind = raw['action']
        passive = {'kind': 'check' if view['can_check'] else 'call'}
        if kind in ('raise', 'all_in') and bounds:
            target = bounds[1] if kind == 'all_in' else raw['amount']
            move = {'kind': 'raise', 'to': target}
        elif kind == 'fold' and not view['can_check']:
            move = {'kind': 'fold'}
        elif kind in ('fold', 'check', 'call', 'all_in', 'raise'):
            move = passive
        else:
            raise ValueError(f'Unexpected SolverLike action: {kind}')
        action = checked_action(state, move)
        return dict(ok=True, action=action, distribution=[dict(action=action, probability=1.0)])
