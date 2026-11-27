"""Play the continued river-call model on top of the previously selected policy."""
from pathlib import Path

import numpy as np
from scipy.special import expit

from controlled_poker_refit import read
from fullhouse_v3_common import digest
from poker_flop_repair import _ref
from poker_river_call_learning import (
    FEATURES, FORMAT as PARENT_FORMAT, MASS, RiverCallPolicy, eligible, features, passive_update,
)
from poker_river_dominance import closing_call_floor


FORMAT = 'selfplay-river-call-diverse-v2'
SOURCE_MODEL = '64b79018289644c66c2f80e9162d1fa852b0b9c90920b3f87518c9fb232ee26a'
EDGE = .05


class DiverseRiverCallPolicy:
    def __init__(self, manifest):
        path = Path(manifest).resolve()
        self.digest = digest(path)
        spec = read(path)
        if spec['format'] != FORMAT:
            raise ValueError('Expected a continued river-call policy')
        baseline = _ref(spec['baseline'], path.parent)
        parent_spec = read(baseline)
        if (parent_spec['format'] != PARENT_FORMAT or parent_spec['seed'] != spec['seed']
                or parent_spec['model']['sha256'] != SOURCE_MODEL):
            raise ValueError('Continued river model requires its selected learned parent')
        self.parent = RiverCallPolicy(baseline)
        self.model = read(_ref(spec['model'], path.parent))
        m = self.model
        if (m['format'] != FORMAT or tuple(m['features']) != FEATURES
                or m['source_model_sha256'] != SOURCE_MODEL or m['passive_shift'] != MASS
                or m['minimum_equity_margin'] != EDGE
                or m['center'] != self.parent.model['center'] or m['scale'] != self.parent.model['scale']):
            raise ValueError('Invalid continued river model parameters or parent normalization')
        self.center, self.scale = self.parent.center, self.parent.scale
        self.coef = np.asarray(m['coefficients'], dtype=np.float64)
        self.intercept = float(m['intercept'])
        if self.coef.shape != (len(FEATURES),) or not np.isfinite(self.coef).all() or not np.isfinite(self.intercept):
            raise ValueError('Invalid continued river weights')

    def decide(self, state, seed):
        before = self.parent.decide(state, seed)
        if not eligible(state) or closing_call_floor(state) > 0:
            return before
        x = features(state)
        q = float(expit(x[0] + self.intercept + self.coef @ ((x-self.center)/self.scale)))
        odds = state['amount_owed'] / (state['pot'] + state['amount_owed'])
        if q >= odds + EDGE:
            prefer = 'call'
        elif q <= odds - EDGE:
            prefer = 'fold'
        else:
            return before
        # Preserve the parent's realized raise as well as its distribution. The
        # second coupled fold/call draw has a different namespace from the first.
        after = passive_update(before, prefer, f'continuation:{seed}')
        if after is before:
            return before
        return dict(after, policy_source='continued_river_call', predicted_showdown_share=q)


if __name__ == '__main__':
    from serve_poker_flop_repair import main
    main(DiverseRiverCallPolicy, 'river-call-continuation')
