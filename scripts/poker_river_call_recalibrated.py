"""Replace the final river logit while preserving the deployed action budget."""
from pathlib import Path

import numpy as np
from scipy.special import expit

from controlled_poker_refit import read
from fullhouse_v3_common import digest
from poker_flop_repair import _ref
from poker_river_call_diverse import DiverseRiverCallPolicy, EDGE
from poker_river_call_learning import FEATURES, MASS, eligible, features, passive_update
from poker_river_dominance import closing_call_floor


FORMAT = 'selfplay-river-call-recalibrated-v1'
INTERACTIONS = ((0, 1), (0, 2), (0, 5), (1, 9))
FEATURE_NAMES = (*FEATURES, *(f'{FEATURES[a]} * {FEATURES[b]}' for a, b in INTERACTIONS))


def basis(x, center, scale):
    standardized = (np.asarray(x) - center) / scale
    clipped = np.clip(standardized, -3., 3.)
    return np.concatenate((standardized, np.stack([clipped[..., a] * clipped[..., b]
                                                   for a, b in INTERACTIONS], axis=-1)), axis=-1)


class RecalibratedRiverCallPolicy:
    def __init__(self, manifest):
        path = Path(manifest).resolve()
        spec = read(path)
        self.digest = digest(path)
        baseline = _ref(spec['baseline'], path.parent)
        parent_spec = read(baseline)
        self.parent = DiverseRiverCallPolicy(baseline)
        self.model = read(_ref(spec['model'], path.parent))
        self.coef = np.asarray(self.model['coefficients'])
        self.intercept = float(self.model['intercept'])
        if (spec['format'] != FORMAT or self.model['format'] != FORMAT
                or spec['seed'] != parent_spec['seed']
                or self.model['source_model_sha256'] != parent_spec['model']['sha256']
                or self.model['passive_shift'] != MASS or self.model['minimum_equity_margin'] != EDGE
                or tuple(self.model['features']) != FEATURE_NAMES
                or self.coef.shape != (len(FEATURE_NAMES),) or not np.isfinite(self.coef).all()
                or not np.isfinite(self.intercept)):
            raise ValueError('Recalibrated river model differs from its validated parent or fit')

    def decide(self, state, seed):
        base = self.parent.parent.decide(state, seed)
        if not eligible(state) or closing_call_floor(state) > 0:
            return base
        x = features(state)
        share = float(expit(x[0] + self.intercept + self.coef @ basis(x, self.parent.center, self.parent.scale)))
        odds = state['amount_owed'] / (state['pot'] + state['amount_owed'])
        if share >= odds + EDGE:
            prefer = 'call'
        elif share <= odds - EDGE:
            prefer = 'fold'
        else:
            return base
        # Match the incumbent's draw, including identical sampled raises.
        return passive_update(base, prefer, f'continuation:{seed}')


if __name__ == '__main__':
    from serve_poker_flop_repair import main
    main(RecalibratedRiverCallPolicy, 'river-call-recalibration')
