"""A bounded, one-decision flop improvement using a supervised self-play action critic."""
from pathlib import Path
import random

import numpy as np

from controlled_poker_refit import read
from fullhouse_v3_common import digest
from poker_preflop_specialist import StreetPolicy
from poker_river_runtime import action_key, legal_menu
from sample_poker_flop_diagnostic import abstract_indices


FORMAT = 'selfplay-flop-action-repair-v1'
MAX_UPDATE = 0.25
INPUT_DIM = 56
HIDDEN = 32
MODELS = 2


def first_flop_decision(state):
    if state['street'] != 'flop' or state.get('terminal'):
        return False
    for event in reversed(state['action_log']):
        if event['action'] == 'deal':
            return True
        if event.get('seat') == state['seat_to_act']:
            return False
    raise ValueError('Missing deal marker for the flop')


def critic_input(postflop, state, menu, concrete_probs):
    """Use only incumbent observation features and the incumbent's actual action distribution."""
    x, _ = postflop.features.encode_state_dict(state,postflop.lookup._equity_tables)
    mapped = abstract_indices(state,menu)
    # A clamped raise can occupy two or three nominal size heads: train and play
    # the smallest index representing this concrete action, once per position.
    heads = np.asarray([mapped.index(i) for i in range(len(menu))],dtype=int)
    base = np.zeros(5)
    base[heads] = concrete_probs
    features = np.concatenate([x,base])
    if features.shape!=(INPUT_DIM,) or not np.isfinite(features).all() or not np.isclose(base.sum(),1):
        raise ValueError('Unexpected critic input')
    return features, heads


def improved_distribution(features, heads, baseline, weights):
    x = np.clip((features-weights['center'])/weights['scale'],-6.,6.)
    hidden = np.maximum(np.einsum('mhi,i->mh',weights['w1'],x)+weights['b1'],0.)
    predictions = np.einsum('moh,mh->mo',weights['w2'],hidden)+weights['b2']
    action_q = predictions[:,heads]
    best = int(np.argmax(action_q.mean(axis=0)))
    gain = float(np.min(action_q[:,best] - action_q @ baseline))
    if not np.isfinite(gain):
        raise ValueError('Nonfinite flop action estimates')
    # Conservative linear ramp up to the registered cap. The units are the
    # larger of the current pot and five big blinds, fixed at this decision.
    alpha = MAX_UPDATE * np.clip((gain-.05)/.25,0.,1.)
    probs = (1.-alpha)*baseline
    probs[best] += alpha
    return probs, float(alpha)


def _ref(ref, folder, *, directory=False):
    path = Path(ref['path'])
    path = (folder/path).resolve() if not path.is_absolute() else path
    if digest(path/'bot/features.py' if directory else path)!=ref['sha256']:
        raise ValueError('Repair policy dependency hash mismatch')
    return path


class FlopRepairPolicy:
    model_type = FORMAT

    def __init__(self, model, source=None):
        self.model = Path(model).resolve()
        self.digest = digest(self.model)
        spec = read(self.model)
        if spec.get('format')!=FORMAT or spec['max_update_mass']!=MAX_UPDATE or spec['seed'] not in (342,343,344):
            raise ValueError('Unsupported flop repair policy')
        folder = self.model.parent
        if source is None:
            source = _ref(spec['feature_source'],folder,directory=True)
        elif digest(Path(source)/'bot/features.py') != spec['feature_source']['sha256']:
            raise ValueError('Repair feature encoder changed')
        base = _ref(spec['base480'],folder)
        critic = _ref(spec['critic'],folder)
        self.base = StreetPolicy(base,source)
        self.dependency_hashes = dict(self.base.dependency_hashes)
        for path in (self.model,base,critic,Path(source)/'bot/features.py',Path(__file__).resolve()):
            self.dependency_hashes[str(path)] = digest(path)
        shapes={'center':(INPUT_DIM,), 'scale':(INPUT_DIM,), 'w1':(MODELS,HIDDEN,INPUT_DIM),
                'b1':(MODELS,HIDDEN,), 'w2':(MODELS,5,HIDDEN),'b2':(MODELS,5)}
        with np.load(critic,allow_pickle=False) as archive:
            if set(archive.files)!=set(shapes):
                raise ValueError('Unexpected critic arrays')
            self.weights = {k:archive[k] for k in shapes}
        if (any(self.weights[k].shape != shape or not np.isfinite(self.weights[k]).all()
                for k,shape in shapes.items()) or np.any(self.weights['scale']<=0)):
            raise ValueError('Invalid critic weights')

    def decide(self, state, seed):
        reply = self.base.decide(state,seed)
        if not first_flop_decision(state):
            return reply
        menu = sorted(legal_menu(state).values(),key=lambda a:(('fold','check','call','raise').index(a['kind']),
                                                               a.get('to',0)))
        by_action = {action_key(row['action']):row['probability'] for row in reply['distribution']}
        baseline = np.array([by_action.get(action_key(a),0.) for a in menu])
        x,heads = critic_input(self.base.policies['postflop'],state,menu,baseline)
        probs,alpha = improved_distribution(x,heads,baseline,self.weights)
        if alpha == 0.:
            return dict(reply, flop_repair_mass=0.)
        dist = [dict(action=a,probability=float(p)) for a,p in zip(menu,probs,strict=True) if p>0]
        chosen = random.Random(seed).choices([r['action'] for r in dist],weights=[r['probability'] for r in dist])[0]
        return dict(reply,action=chosen,distribution=dist,model_type=FORMAT,model_sha256=self.digest,
                    policy_source='first_flop_repair',flop_repair_mass=alpha)
