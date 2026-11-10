"""Positive, input-defined regret scaling and conditional replay fitting.

The scale for a feature stratum is shared by ALL its replay iterations. Thus
E_t[r/s(I)] = E_t[r]/s(I), and regret matching is unchanged. This is not
per-row normalization. Equal feature mass changes the shared-network projection
objective, so its poker strength requires a separate experiment.
"""
import numpy as np
import torch

from regret_fitting import ConditionalRegretSampler

METHODS = ('global-scaled-conditional', 'stratum-scaled-feature-conditional')


class ScaledRegretSampler(ConditionalRegretSampler):
    def __init__(self, buffers, strata_fn, method):
        if method not in METHODS:
            raise ValueError('Unsupported scaled regret method')
        super().__init__(buffers, strata_fn)
        self.strata_fn = strata_fn
        self.method = method
        self.raw_targets = self.targets.copy()
        present = np.unique(self.groups)
        if method == METHODS[1]:
            # Give every distinct unraised-preflop input equal training mass.
            # Its time-weighted conditional target was computed BEFORE this.
            for group in present[present < 6]:
                ids = self.groups == group
                self.probabilities[ids] = 1./len(present)/int(ids.sum())
        self.cdf = np.cumsum(self.probabilities); self.cdf[-1] = 1.
        self.scales = np.ones(22, np.float32)
        if method == METHODS[0]:
            self.scales[:] = 100.  # One fixed 100BB stack, not a sample stddev.
        else:
            for group in present:
                ids = self.groups == group
                second = np.square(self.raw_targets[ids].astype(float)).mean(1)
                self.scales[group] = max(1., float(np.sqrt(np.average(second, weights=self.probabilities[ids]))))
        self.targets = (self.raw_targets/self.scales[self.groups, None]).astype(np.float32)
        if not np.isfinite(self.targets).all() or not np.all(self.scales > 0):
            raise ValueError('Invalid normalized regret targets')

    def diagnostics(self):
        return dict(super().diagnostics(), method=self.method, scales=self.scales.tolist(),
                    target_rms=float(np.sqrt(np.mean(np.square(self.targets)))))


class BBPredictions(torch.nn.Module):
    """Diagnostic view of normalized NN scores in the replay's original BB units."""
    def __init__(self, net, sampler):
        super().__init__()
        self.net = net
        self.sampler = sampler

    def forward(self, features, legal):
        groups = self.sampler.strata_fn(features.detach().cpu().numpy())
        scale = torch.as_tensor(self.sampler.scales[groups], device=features.device)
        return self.net(features, legal)*scale[:, None]
