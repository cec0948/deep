"""Play the selected first-flop strategy and an optional first-turn correction."""
from pathlib import Path
import random

import numpy as np

from controlled_poker_refit import read
from fullhouse_v3_common import digest
from poker_flop_repair import (FORMAT as FLOP_FORMAT, HIDDEN, INPUT_DIM, MAX_UPDATE, MODELS,
                               FlopRepairPolicy, _ref, critic_input, improved_distribution)
from poker_river_runtime import action_key, legal_menu


FORMAT = 'selfplay-street-progress-v1'


def first_turn_decision(state):
    if state['street'] != 'turn' or state.get('terminal'):
        return False
    for event in reversed(state['action_log']):
        if event['action'] == 'deal':
            return True
        if event.get('seat') == state['seat_to_act']:
            return False
    raise ValueError('Missing deal marker for the turn')


class StreetProgressPolicy:
    """A flop-only or composed flop+turn policy, fixed for the entire hand."""

    def __init__(self, model):
        self.model = Path(model).resolve()
        self.digest = digest(self.model)
        spec = read(self.model)
        self.model_type = spec['format']
        self.turn_weights = None
        if self.model_type == FLOP_FORMAT:
            self.flop = FlopRepairPolicy(self.model)
        elif self.model_type == FORMAT:
            seed = spec['seed']
            if seed not in (342, 343, 344) or spec['max_update_mass'] != MAX_UPDATE:
                raise ValueError('Invalid turn model seed or probability cap')
            parent = _ref(spec['flop_policy'], self.model.parent)
            if read(parent)['seed'] != seed:
                raise ValueError('Selected flop and turn seeds differ')
            self.flop = FlopRepairPolicy(parent)
            critic = _ref(spec['turn_critic'], self.model.parent)
            shapes = {'center':(INPUT_DIM,), 'scale':(INPUT_DIM,),
                      'w1':(MODELS,HIDDEN,INPUT_DIM), 'b1':(MODELS,HIDDEN),
                      'w2':(MODELS,5,HIDDEN), 'b2':(MODELS,5)}
            with np.load(critic, allow_pickle=False) as archive:
                if set(archive.files) != set(shapes):
                    raise ValueError('Invalid turn network arrays')
                self.turn_weights = {name: archive[name] for name in shapes}
            if (any(array.shape != shapes[name] or not np.isfinite(array).all()
                    for name, array in self.turn_weights.items()) or np.any(self.turn_weights['scale'] <= 0)):
                raise ValueError('Invalid turn critic weights or normalization')
        else:
            raise ValueError(f'Unsupported policy format {self.model_type!r}')
        self.dependency_hashes = dict(self.flop.dependency_hashes)
        self.dependency_hashes[str(self.model)] = self.digest
        self.dependency_hashes[str(Path(__file__).resolve())] = digest(__file__)
        if self.turn_weights is not None:
            self.dependency_hashes[str(critic)] = spec['turn_critic']['sha256']

    def decide(self, state, seed):
        reply = self.flop.decide(state, seed)
        if self.turn_weights is None or not first_turn_decision(state):
            return reply
        menu = list(legal_menu(state).values())
        by_action = {action_key(row['action']): row['probability'] for row in reply['distribution']}
        baseline = np.array([by_action.get(action_key(action), 0.) for action in menu])
        inputs, heads = critic_input(self.flop.base.policies['postflop'], state, menu, baseline)
        probs, shift = improved_distribution(inputs, heads, baseline, self.turn_weights)
        if shift == 0:
            return dict(reply, turn_repair_mass=0.)
        dist = [dict(action=action, probability=float(prob))
                for action, prob in zip(menu, probs, strict=True) if prob > 0.]
        chosen = random.Random(seed).choices([row['action'] for row in dist],
                                             weights=[row['probability'] for row in dist], k=1)[0]
        return dict(reply, action=chosen, distribution=dist, policy_source='first_turn_repair',
                    turn_repair_mass=shift, model_type=FORMAT, model_sha256=self.digest)


if __name__ == '__main__':
    from serve_poker_flop_repair import main
    main(StreetProgressPolicy, 'street-progress')
