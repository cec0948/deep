"""Play the frozen 480 specialist through the flop and one diverse snapshot thereafter."""
from pathlib import Path

from fullhouse_v3_common import atomic_json, digest
from poker_capture_postflop_snapshots import ScheduledPostflopMixture, read
from poker_preflop_specialist import evaluator_policy


FORMAT = 'specialist480-flop-diverse504-turn-river-v1'
ITERATIONS = (486, 492, 498, 504)


def _checked_ref(ref):
    path = Path(ref['path']).resolve()
    if digest(path) != ref['sha256']:
        raise ValueError(f'Policy reference changed: {path}')
    return dict(path=str(path), sha256=ref['sha256'])


def _spec(training_seed, parent_ref, mixture_ref):
    parent = _checked_ref(parent_ref)
    mixture = _checked_ref(mixture_ref)
    capture_path = Path(mixture['path']).parent/'plan.json'
    capture = read(capture_path)
    slots = read(mixture['path'])
    if (slots['plan_sha256'] != digest(capture_path)
            or capture['parent_policy_sha256'] != parent['sha256']
            or slots['preflop']['sha256'] != read(parent['path'])['preflop']['sha256']
            or type(training_seed) is not int
            or (slots['seed'], slots['arm']) != (training_seed, 'pool504')
            or tuple(slot['iteration'] for slot in slots['postflop']) != ITERATIONS):
        raise ValueError('Rollback sources differ in parent, seed, arm or snapshot schedule')
    return dict(format=FORMAT, training_seed=training_seed, specialist480=parent, diverse504=mixture)


def write_policy(path, training_seed, parent_ref, mixture_ref):
    path = Path(path)
    spec = _spec(training_seed, parent_ref, mixture_ref)
    if path.exists():
        if read(path) != spec:
            raise ValueError('Existing rollback policy has different sources')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, spec)
    return path


class FlopRollbackPolicy:
    model_type = FORMAT

    def __init__(self, model, source):
        self.model = Path(model).resolve()
        spec = read(self.model)
        if spec != _spec(spec['training_seed'], spec['specialist480'], spec['diverse504']):
            raise ValueError('Invalid rollback policy manifest')
        parent = spec['specialist480']
        mixture = spec['diverse504']
        self.parent = evaluator_policy(parent['path'], source)
        self.mixture = ScheduledPostflopMixture(mixture['path'], source)
        self.postflop = self.mixture.postflop
        self.dependency_hashes = dict(self.parent.dependency_hashes)
        self.dependency_hashes.update(self.mixture.dependency_hashes)
        self.dependency_hashes.update({ref['path']:ref['sha256'] for ref in (parent, mixture)})
        folder = Path(mixture['path']).parent
        receipts = [folder/f'iter-{t:05d}'/'receipt.json' for t in ITERATIONS]
        for path in [self.model, Path(__file__).resolve(), folder/'plan.json', *receipts]:
            self.dependency_hashes[str(path)] = digest(path)

    @property
    def active(self):
        return self.mixture.active

    @property
    def snapshot_iteration(self):
        return None if self.active is None else ITERATIONS[self.postflop.index(self.active)]

    def begin_hand(self, hand_seed):
        self.mixture.begin_hand(hand_seed)

    def decide(self, state, seed):
        if self.active is None:
            raise ValueError('Call begin_hand with a private hand seed before deciding')
        if state['street'] in ('preflop', 'flop'):
            return self.parent.decide(state, seed)
        return self.mixture.decide(state, seed)
