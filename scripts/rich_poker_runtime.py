"""Explicit, NumPy-only runtime for a rich residual research policy.

The descriptor identifies immutable residual weights and the frozen teacher.
This entry point never changes a registry or promotes a policy for deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from fullhouse_runtime import Bridge, validate_request
from fullhouse_v4_runtime import FullhouseV4, policy_for as legacy_policy_for
from regret_partition import base_strata, is_opening
from rich_poker_observation import RICH_DIM, SCHEMA, encode_bridge

DESCRIPTOR_FORMAT = 'rich-policy-export-v1'
WEIGHT_FORMAT = 'rich-regret-residual-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_descriptor(path):
    path = Path(path).resolve()
    row = json.loads(path.read_text())
    allowed = {'format','model','model_sha256','teacher','teacher_sha256','source',
               'iteration','seed','feature_schema','parent_iteration','average_policy_available'}
    if (not isinstance(row, dict) or set(row)-allowed
            or row.get('format') != DESCRIPTOR_FORMAT or row.get('feature_schema') != SCHEMA
            or type(row.get('iteration')) is not int or row['iteration'] < 0
            or type(row.get('seed')) is not int or not 0 <= row['seed'] < 2**32):
        raise ValueError('Invalid rich policy descriptor')
    if (('parent_iteration' in row and (type(row['parent_iteration']) is not int
                                      or row['parent_iteration'] != 360))
            or ('average_policy_available' in row and row['average_policy_available'] is not False)):
        raise ValueError('Invalid rich parent/average-policy capability metadata')
    for key in ('model','teacher'):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError('Missing descriptor path: '+key)
        resolved = (path.parent/row[key]).resolve()
        expected = row.get(key+'_sha256')
        if (not isinstance(expected, str) or len(expected) != 64
                or digest(resolved) != expected):
            raise ValueError('Rich policy dependency changed: '+key)
        row[key] = str(resolved)
    if 'source' in row:
        row['source'] = str((path.parent/row['source']).resolve())
    return row


def load_residual(path, teacher_sha256):
    """Validate the complete small numeric export; never deserialize pickle."""
    with np.load(path, allow_pickle=False) as saved:
        fields = {'format','teacher_sha256','feature_dim','width','opening_frozen',
                  'average_policy_available'}
        fields |= {f'residual_{kind}{index}' for kind in ('w','b') for index in range(3)}
        if set(saved.files) != fields:
            raise ValueError('Residual export has missing or unexpected arrays')
        if (str(saved['format']) != WEIGHT_FORMAT
                or str(saved['teacher_sha256']) != teacher_sha256
                or int(saved['feature_dim']) != RICH_DIM
                or bool(saved['opening_frozen']) is not True
                or bool(saved['average_policy_available']) is not False):
            raise ValueError('Residual export family/teacher/feature mismatch')
        width = int(saved['width'])
        if not 1 <= width <= 2048:
            raise ValueError('Invalid residual width')
        shapes = ((width, RICH_DIM), (width, width), (5, width))
        layers = []
        for index, shape in enumerate(shapes):
            weight = saved[f'residual_w{index}'].copy()
            bias = saved[f'residual_b{index}'].copy()
            if (weight.shape != shape or bias.shape != (shape[0],)
                    or weight.dtype != np.float32 or bias.dtype != np.float32
                    or not np.isfinite(weight).all() or not np.isfinite(bias).all()):
                raise ValueError('Invalid finite float32 residual layer')
            layers.append((weight, bias))
    return tuple(layers)


class RichPolicy(FullhouseV4):
    def __init__(self, descriptor, source=None):
        descriptor = Path(descriptor).resolve()
        self.spec = load_descriptor(descriptor)
        selected_source = source or self.spec.get('source')
        if selected_source is None:
            raise ValueError('A frozen feature source is required')
        if ('source' in self.spec and source is not None
                and Path(source).resolve() != Path(self.spec['source']).resolve()):
            raise ValueError('Explicit feature source differs from descriptor')
        self.layers = load_residual(self.spec['model'], self.spec['teacher_sha256'])
        self.zero_residual = not any(np.any(value) for value in self.layers[-1])
        super().__init__(self.spec['teacher'], selected_source)
        self.teacher_sha256 = self.digest
        self.model = descriptor
        self.digest = digest(descriptor)
        self.model_type = 'rich_regret_residual'
        self.source = Path(selected_source).resolve()
        self.dependency_hashes = {
            str(descriptor): self.digest,
            self.spec['model']: self.spec['model_sha256'],
            self.spec['teacher']: self.spec['teacher_sha256'],
            str(self.source/'bot/features.py'): digest(self.source/'bot/features.py'),
            str(Path(__file__).resolve()): digest(__file__),
            str(Path(__file__).with_name('rich_poker_observation.py').resolve()):
                digest(Path(__file__).with_name('rich_poker_observation.py')),
            str(Path(__file__).with_name('regret_partition.py').resolve()):
                digest(Path(__file__).with_name('regret_partition.py')),
        }
        if self.teacher_sha256 != self.spec['teacher_sha256']:
            raise ValueError('Loaded teacher changed')

    def _strategy(self, state):
        old, mask = self.features.encode_state_dict(state, self.lookup._equity_tables)
        old = np.asarray(old, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        rich = encode_bridge(state, old)
        selected = self.opening_model if bool(is_opening(old)) else self.lookup._models[0]
        if self.zero_residual or bool(is_opening(old)):
            return self.lookup._model_strategy(old, mask, selected)
        raw = self.lookup._forward_single(old, mask, selected).astype(np.float64)
        scales = self.lookup._models[0]['main_scales']
        scale = float(scales[int(base_strata(old[None])[0])])
        if not np.isfinite(scale) or scale <= 0:
            raise FloatingPointError('Invalid teacher BB scale')
        hidden = rich
        for index, (weight, bias) in enumerate(self.layers):
            hidden = np.dot(hidden, weight.T)+bias
            if index < 2:
                hidden = np.where(hidden > 0., hidden, .01*hidden)
        scores = raw*scale+hidden.astype(np.float64)*100.
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite rich residual regrets')
        positive = np.maximum(scores, 0.)*mask
        probabilities = positive/positive.sum() if positive.sum() > 0 else mask/mask.sum()
        return probabilities.astype(np.float32)

    def decide(self, state, seed):
        reply = super().decide(state, seed)
        reply.update(policy_source='rich_residual_policy', model_family=WEIGHT_FORMAT,
                     new_family_iteration=self.spec['iteration'], seed=self.spec['seed'],
                     teacher_sha256=self.teacher_sha256, feature_schema=SCHEMA,
                     feature_dimension=RICH_DIM, average_policy_available=False,
                     strength_verified=False)
        if 'parent_iteration' in self.spec:
            reply['parent_iteration'] = self.spec['parent_iteration']
        return reply


def policy_for(model, source=None):
    path = Path(model).resolve()
    if path.suffix == '.json':
        spec = json.loads(path.read_text())
        if spec.get('format') == DESCRIPTOR_FORMAT:
            return RichPolicy(path, source)
    policy = legacy_policy_for(path, source)
    policy.dependency_hashes = {str(path): digest(path)}
    if source is not None:
        policy.dependency_hashes[str(Path(source).resolve()/'bot/features.py')] = digest(Path(source)/'bot/features.py')
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    policy = RichPolicy(args.model, args.source)
    engine = Bridge()
    try:
        for line in sys.stdin:
            try:
                if len(line) > 1048576:
                    raise ValueError('Request exceeds1MiB')
                request = json.loads(line)
                validate_request(request)
                reply = policy.decide(engine.call(dict(request, op='replay')), request.get('seed',0))
            except (ValueError, KeyError, TypeError, FloatingPointError) as error:
                reply = dict(ok=False, error=str(error))
            print(json.dumps(reply, allow_nan=False), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
