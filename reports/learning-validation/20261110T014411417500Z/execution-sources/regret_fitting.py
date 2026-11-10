"""Experimental frozen-replay fitter; no automatic change to the V3 trainer.

Uniform positive-mass strata, time-proportional sampling within each stratum.
For unopened preflop states, exactly identical inputs are replaced by their
weighted conditional means. This reduces label variance without changing the
within-stratum squared-loss gradient in expectation. Equal stratum mass DOES
change the global finite-network projection objective. No poker-hand rules.
"""
import numpy as np
import torch


class ConditionalRegretSampler:
    def __init__(self, buffers, strata_fn):
        features = np.concatenate([b.features[:len(b)] for b in buffers])
        targets = np.concatenate([b.advantages[:len(b)] for b in buffers])
        weights = np.concatenate([b.iterations[:len(b)] for b in buffers]).astype(float)
        if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Positive finite replay iteration weights required')
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError('Nonfinite replay')
        groups = strata_fn(features)
        xs, ys, ws, gs = [], [], [], []
        for group in np.unique(groups):
            mask = groups == group
            x, y, w = features[mask], targets[mask], weights[mask]
            if group < 6:
                keys, inv = np.unique(x, axis=0, return_inverse=True)
                mass = np.bincount(inv, weights=w, minlength=len(keys))
                y = np.stack([np.bincount(inv, weights=w*y[:, a], minlength=len(keys))
                              for a in range(y.shape[1])], axis=1) / mass[:, None]
                x, w = keys, mass
            xs.append(x)
            ys.append(y)
            ws.append(w/w.sum()/len(np.unique(groups)))
            gs.append(np.full(len(x), group))
        self.features = np.concatenate(xs).astype(np.float32)
        self.targets = np.concatenate(ys).astype(np.float32)
        self.probabilities = np.concatenate(ws)
        self.groups = np.concatenate(gs)
        self.cdf = np.cumsum(self.probabilities)
        self.cdf[-1] = 1.
        self.draws = np.zeros(len(self.features), dtype=np.int64)

    def sample(self, size):
        ids = np.searchsorted(self.cdf, np.random.random(size), side='right')
        np.add.at(self.draws, ids, 1)
        return self.features[ids], self.targets[ids]

    def diagnostics(self):
        aa = ((self.features[:, 17] == 1) & (self.features[:, 24] == 1)
              & np.isclose(self.features[:, 27], .015)
              & (self.features[:, 10] == 1) & (self.features[:, 7] > .99))
        return dict(draws=int(self.draws.sum()), entries=int(len(self.features)),
                    distinct_entries=int((self.draws > 0).sum()),
                    aa_root_draws=int(self.draws[aa].sum()))


def train_conditional(net, sampler, optimizer, steps, batch_size):
    net.train()
    loss_sum = 0.
    for _ in range(steps):
        x, y = sampler.sample(batch_size)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        legal = x[:, 46:51]
        loss = ((net(x, legal) - y*legal).square()).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite regret fitting loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        loss_sum += float(loss.detach())
    net.eval()
    return loss_sum / steps
