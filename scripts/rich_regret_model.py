"""A frozen compact360 regret prior plus a lossless-input neural residual.

This is a new approximation family, not exact continuation of the legacy CFR
worker. The residual starts at zero; untrained and first-opening decisions use
the exact scalar teacher policy path. No historical policy mixture is claimed.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from regret_partition import base_strata, is_opening

FORMAT = 'rich-regret-residual-v1'
BB_SCALE = 100.


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ResidualNet(nn.Module):
    def __init__(self, feature_dim, width=128):
        super().__init__()
        if type(feature_dim) is not int or feature_dim < 51 or feature_dim > 10000:
            raise ValueError('Invalid lossless feature dimension')
        self.layers = nn.Sequential(nn.Linear(feature_dim, width), nn.LeakyReLU(.01),
                                    nn.Linear(width, width), nn.LeakyReLU(.01),
                                    nn.Linear(width, 5))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return self.layers(features) * BB_SCALE


class RichRegretModel:
    def __init__(self, teacher_npz, feature_dim, source=None, width=128):
        if source is not None and str(Path(source).resolve()) not in sys.path:
            sys.path.insert(0, str(Path(source).resolve()))
        from bot.deep_cfr_lookup import DeepCFRLookup
        self.teacher_path = Path(teacher_npz).resolve()
        self.teacher_sha256 = sha256(self.teacher_path)
        with np.load(self.teacher_path, allow_pickle=False) as data:
            self.teacher = {key: data[key].copy() for key in data.files}
        if str(self.teacher.get('training_revision')) != 'r1-cpu-v4':
            raise ValueError('An existing compact360 teacher is required')
        self.teacher['_model_type'] = 'advantage'
        self.opening = {key[len('opening_'):]: value for key, value in self.teacher.items()
                        if key.startswith('opening_')}
        if 'trunk_w0' not in self.opening or 'main_scales' not in self.teacher:
            raise ValueError('Teacher experts and BB scales are required')
        self.opening['_model_type'] = 'advantage'
        # No public state, equity table or hidden cards are needed for a
        # pre-encoded scalar forward pass.
        self.lookup = DeepCFRLookup.__new__(DeepCFRLookup)
        self.feature_dim, self.width = feature_dim, width
        self.residual = ResidualNet(feature_dim, width)
        self._base_cache = OrderedDict()
        self._sync_numpy()

    def _sync_numpy(self):
        self.numpy_layers = tuple((layer.weight.detach().cpu().numpy().copy(),
                                   layer.bias.detach().cpu().numpy().copy())
                                  for layer in self.residual.layers if isinstance(layer, nn.Linear))
        self.zero_residual = not any(np.any(a) for a in self.numpy_layers[-1])

    def reset_residual(self):
        self.residual = ResidualNet(self.feature_dim, self.width)
        self._sync_numpy()

    @staticmethod
    def _validate(features, legal, dimension):
        x = np.asarray(features, dtype=np.float32)
        mask = np.asarray(legal, dtype=np.float32)
        if (x.shape != (dimension,) or mask.shape != (5,) or not np.isfinite(x).all()
                or not np.isin(mask, [0., 1.]).all() or not mask.any()
                or not np.array_equal(x[46:51], mask)):
            raise ValueError('Finite rich vector with matching legal mask required')
        return x, mask

    def _expert(self, old):
        return self.opening if bool(is_opening(old)) else self.teacher

    def teacher_strategy(self, old, legal):
        return self.lookup._model_strategy(np.asarray(old, np.float32),
                                           np.asarray(legal, np.float32), self._expert(old))

    def base_regrets(self, old, legal):
        old = np.asarray(old, dtype=np.float32)
        mask = np.asarray(legal, dtype=np.float32)
        key = old.tobytes() + mask.tobytes()
        if key in self._base_cache:
            self._base_cache.move_to_end(key)
            return self._base_cache[key].copy()
        raw = self.lookup._forward_single(old, mask, self._expert(old))
        if bool(is_opening(old)):
            scale = float(self.teacher['opening_scale'])
        else:
            scale = float(self.teacher['main_scales'][int(base_strata(old[None])[0])])
        if not np.isfinite(scale) or scale <= 0:
            raise FloatingPointError('Teacher scale must be positive and finite')
        value = raw.astype(np.float64) * scale
        if not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite teacher BB regrets')
        self._base_cache[key] = value.copy()
        if len(self._base_cache) > 60000:
            self._base_cache.popitem(last=False)
        return value

    def residual_numpy(self, features):
        h = np.asarray(features, dtype=np.float32)
        for index, (weight, bias) in enumerate(self.numpy_layers):
            h = np.dot(h, weight.T) + bias
            if index < len(self.numpy_layers)-1:
                h = np.where(h > 0., h, .01*h)
        return h.astype(np.float64) * BB_SCALE

    def strategy(self, features, legal):
        x, mask = self._validate(features, legal, self.feature_dim)
        old = x[:51]
        if self.zero_residual or bool(is_opening(old)):
            return self.teacher_strategy(old, mask)
        raw = self.base_regrets(old, mask) + self.residual_numpy(x)
        if not np.isfinite(raw).all():
            raise FloatingPointError('Nonfinite residual regrets')
        positive = np.maximum(raw, 0.) * mask
        if positive.sum() > 0:
            result = positive / positive.sum()
        else:
            result = mask / mask.sum()
        return result.astype(np.float32)

    def fit(self, buffers, *, steps=256, batch_size=128, learning_rate=.001):
        if not 1 <= steps <= 4096 or not 1 <= batch_size <= 1024 or not 0 < learning_rate <= .01:
            raise ValueError('Invalid bounded residual fitting budget')
        pieces = [(b.features[:len(b)], b.advantages[:len(b)], b.iterations[:len(b)])
                  for b in buffers if len(b)]
        if not pieces:
            raise ValueError('Fresh rich replay must be collected before fitting')
        x = np.concatenate([v[0] for v in pieces])
        targets = np.concatenate([v[1] for v in pieces])
        weights = np.concatenate([v[2] for v in pieces]).astype(np.float64)
        if (x.shape[1] != self.feature_dim or targets.shape != (len(x), 5)
                or not np.isfinite(x).all() or not np.isfinite(targets).all()
                or not np.isfinite(weights).all() or np.any(weights <= 0)):
            raise ValueError('Malformed finite positive-weight rich replay')
        eligible = ~is_opening(x[:, :51])
        if not eligible.any():
            self.reset_residual()
            return dict(loss=0., rows=len(x), eligible_rows=0, steps=0,
                        opening_frozen=True, objective='iteration_weighted_raw_BB_MSE')
        x, targets, weights = x[eligible], targets[eligible], weights[eligible]
        teacher = np.stack([self.base_regrets(row[:51], row[46:51]) for row in x]).astype(np.float32)
        weights = (weights/weights.mean()).astype(np.float32)
        tx = torch.from_numpy(x); ty = torch.from_numpy(targets)
        tb = torch.from_numpy(teacher); tw = torch.from_numpy(weights)
        legal_masks = tx[:,46:51]
        monitor = np.linspace(0, len(x)-1, min(512, len(x)), dtype=int)
        with torch.no_grad():
            before = tb[monitor]+self.residual(tx[monitor])
            before_mse = float((tw[monitor]*((before-ty[monitor])*legal_masks[monitor]).square().mean(1)).mean())
            prior_mse = float((tw[monitor]*((tb[monitor]-ty[monitor])*legal_masks[monitor]).square().mean(1)).mean())
        self.reset_residual()
        optimizer = torch.optim.Adam(self.residual.parameters(), lr=learning_rate)
        self.residual.train()
        for _ in range(steps):
            ids = np.random.randint(0, len(x), size=batch_size)
            values = tb[ids] + self.residual(tx[ids])
            loss = (tw[ids] * (((values-ty[ids])*legal_masks[ids])/BB_SCALE).square().mean(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite rich fitting loss')
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.residual.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite residual gradient')
            optimizer.step()
        self.residual.eval()
        if any(not torch.isfinite(value).all() for value in self.residual.state_dict().values()):
            raise FloatingPointError('Nonfinite residual parameters')
        self._sync_numpy()
        with torch.no_grad():
            after = tb[monitor]+self.residual(tx[monitor])
            after_mse = float((tw[monitor]*((after-ty[monitor])*legal_masks[monitor]).square().mean(1)).mean())
        def probabilities(raw):
            legal = x[monitor,46:51]
            positive = np.maximum(raw.numpy(), 0.)*legal
            total = positive.sum(1, keepdims=True)
            return np.divide(positive, total, out=legal/legal.sum(1, keepdims=True), where=total>0)
        tv = float(np.abs(probabilities(before)-probabilities(after)).sum(1).mean()/2)
        return dict(loss=float(loss.detach()), rows=int(sum(len(b) for b in buffers)),
                    eligible_rows=len(x), steps=steps, opening_frozen=True,
                    objective='uniform_replay_sampling_iteration_weighted_raw_BB_MSE',
                    fixed_BB_normalizer=BB_SCALE,
                    fixed_monitor_rows=len(monitor), replay_mse_before_bb2=before_mse,
                    replay_mse_after_bb2=after_mse, teacher_prior_mse_bb2=prior_mse,
                    relative_policy_tv=tv,
                    monitor_scope='Deterministic retained-replay subset, not independent strength or heldout performance')

    def state_dict(self):
        return dict(format=FORMAT, teacher_sha256=self.teacher_sha256,
                    feature_dim=self.feature_dim, width=self.width,
                    residual=self.residual.state_dict())

    def load_state_dict(self, state):
        if (state.get('format') != FORMAT or state.get('teacher_sha256') != self.teacher_sha256
                or state.get('feature_dim') != self.feature_dim or state.get('width') != self.width):
            raise ValueError('Rich model teacher/architecture mismatch')
        self.residual.load_state_dict(state['residual'], strict=True)
        if any(not torch.isfinite(value).all() for value in self.residual.state_dict().values()):
            raise ValueError('Nonfinite restored residual parameters')
        self._sync_numpy()

    def export(self, path):
        arrays = dict(format=np.array(FORMAT), teacher_sha256=np.array(self.teacher_sha256),
                      feature_dim=np.array(self.feature_dim), width=np.array(self.width),
                      opening_frozen=np.array(True), average_policy_available=np.array(False))
        for index, (weight, bias) in enumerate(self.numpy_layers):
            arrays[f'residual_w{index}'] = weight
            arrays[f'residual_b{index}'] = bias
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, **arrays)
