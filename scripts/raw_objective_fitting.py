"""Preserve a retained replay's iteration-weighted raw-unit squared loss.

For conditional record i with original mass p_i and fixed positive scale s_i,
define q*_i = p_i*s_i**2/Z. Keep the original balanced sampling q and apply
the fixed per-example loss weight q*_i/q_i. The expected gradient is the
original raw-unit gradient divided by Z. Exact
conditional means add only a parameter-independent variance constant. This is
an empirical replay objective, not a six-player equilibrium guarantee.
"""
import numpy as np


def conditional_raw_objective(features, targets, weights, groups, scales, legal,
                              collapse_groups=()):
    """Generic finite-population construction; all mathematical arrays float64.

    Features may have any width and targets any action count. Groups appear in
    sorted order. Collapsed groups use lexicographically sorted exact feature
    vectors; other groups retain their original row order.
    """
    x, y = np.asarray(features), np.asarray(targets, dtype=np.float64)
    w, g = np.asarray(weights, dtype=np.float64), np.asarray(groups)
    s, mask = np.asarray(scales, dtype=np.float64), np.asarray(legal, dtype=np.float64)
    n = x.shape[0] if x.ndim == 2 else 0
    if (x.ndim != 2 or not n or y.ndim != 2 or y.shape[0] != n
            or mask.shape != y.shape or w.shape != (n,) or g.shape != (n,)
            or not np.issubdtype(g.dtype, np.integer) or np.any(g < 0)
            or s.ndim != 1 or len(s) <= int(g.max())):
        raise ValueError('Invalid objective array shapes or group indices')
    if (not all(np.isfinite(a).all() for a in (x,y,w,s,mask))
            or np.any(w <= 0) or np.any(s <= 0)
            or not np.isin(mask,[0.,1.]).all() or np.any(mask.sum(1)==0)):
        raise ValueError('Positive finite weights/scales and legal masks required')
    collapse = set(collapse_groups)
    xs, ys, ws, gs, masks = [], [], [], [], []
    row_to_record = np.empty(n, dtype=np.int64)
    variance_sum = 0.; offset = 0
    for group in np.unique(g):
        original = np.flatnonzero(g == group)
        gx, gy, gw, gm = x[original], y[original], w[original], mask[original]
        if int(group) in collapse:
            keys, inverse = np.unique(gx, axis=0, return_inverse=True)
            mass = np.bincount(inverse, weights=gw, minlength=len(keys))
            means = np.stack([np.bincount(inverse, weights=gw*gy[:,a], minlength=len(keys))
                              for a in range(y.shape[1])], axis=1) / mass[:,None]
            _, first = np.unique(inverse, return_index=True)
            compressed_mask = gm[first]
            if not np.array_equal(gm, compressed_mask[inverse]):
                raise ValueError('Identical features have different legal masks')
            residual = (gy-means[inverse])*gm
            variance_sum += float(np.dot(gw, np.square(residual).mean(1)))
            gx, gy, gw, gm = keys, means, mass, compressed_mask
            row_to_record[original] = offset + inverse
        else:
            row_to_record[original] = offset + np.arange(len(original))
        xs.append(gx);ys.append(gy);ws.append(gw);masks.append(gm)
        gs.append(np.full(len(gx),group,dtype=np.int64));offset+=len(gx)
    mass=np.concatenate(ws); groups_out=np.concatenate(gs)
    total=float(mass.sum()); p=mass/total; record_scales=s[groups_out]
    tilted=p*np.square(record_scales); normalizer=float(tilted.sum()); q=tilted/normalizer
    if (not np.isfinite(total) or not np.isfinite(normalizer) or normalizer <= 0
            or not np.isfinite(q).all() or np.any(q <= 0)
            or abs(float(q.sum())-1.) > 1e-12):
        raise ValueError('Objective probabilities do not have finite full support')
    raw=np.concatenate(ys)
    return dict(features=np.concatenate(xs),raw_targets=raw,legal=np.concatenate(masks),
        groups=groups_out,mass=mass,original_probabilities=p,target_probabilities=q,
        record_scales=record_scales,normalized_targets=raw/record_scales[:,None],
        normalizer_z=normalizer,conditional_variance_constant=variance_sum/total,
        row_to_record=row_to_record,original_rows=n)


class RawObjectiveRegretSampler:
    """Align the exact objective with an immutable ScaledRegretSampler layout.

    Production targets deliberately retain the frozen sampler's float32
    rounding. The conservation identity is exact in real arithmetic and is
    verified numerically in float64; float32 target-rounding errors are reported.
    """
    def __init__(self,buffers,strata_fn,base_sampler,expected_scales):
        x=np.concatenate([b.features[:len(b)] for b in buffers])
        y=np.concatenate([b.advantages[:len(b)] for b in buffers])
        w=np.concatenate([b.iterations[:len(b)] for b in buffers])
        groups=strata_fn(x)
        scales=np.asarray(expected_scales)
        if not np.array_equal(base_sampler.scales,scales):
            raise ValueError('Sampler scales differ from immutable parent scales')
        proof=conditional_raw_objective(x,y,w,groups,scales,x[:,46:51],
                                       collapse_groups=set(int(g) for g in np.unique(groups) if g<6))
        if (not np.array_equal(base_sampler.features,proof['features'].astype(np.float32))
                or not np.array_equal(base_sampler.groups,proof['groups'])
                or not np.array_equal(base_sampler.raw_targets,proof['raw_targets'].astype(np.float32))):
            raise ValueError('Frozen conditional record/target alignment mismatch')
        expected=(proof['raw_targets'].astype(np.float32)/scales[proof['groups'],None]).astype(np.float32)
        if not np.array_equal(expected,base_sampler.targets):
            raise ValueError('Frozen normalized targets differ')
        self.features=base_sampler.features
        self.targets=base_sampler.targets
        self.raw_targets=proof['raw_targets']
        self.groups=base_sampler.groups
        self.scales=scales.copy()
        self.original_probabilities=proof['original_probabilities']
        self.target_probabilities=proof['target_probabilities']
        self.probabilities=base_sampler.probabilities.copy()
        self.original_sampler_probabilities=base_sampler.probabilities.copy()
        self.original_sampler_cdf=base_sampler.cdf.copy()
        self.cdf=base_sampler.cdf.copy()
        self.loss_weights=self.target_probabilities/self.probabilities
        if not np.isfinite(self.loss_weights).all() or np.any(self.loss_weights<=0):
            raise ValueError('Loss weights must have finite full support')
        self.normalizer_z=proof['normalizer_z']
        self.conditional_variance_constant=proof['conditional_variance_constant']
        self.original_rows=proof['original_rows']
        self.raw_mean_rounding_max_abs=float(np.max(np.abs(self.raw_targets-base_sampler.raw_targets)))
        self.normalized_target_rounding_max_abs=float(np.max(np.abs(proof['normalized_targets']-self.targets)))
        self.draws=np.zeros(len(self.features),dtype=np.int64)

    def sample(self,size):
        if type(size) is not int or size<=0:
            raise ValueError('Positive integer batch size required')
        ids=np.searchsorted(self.cdf,np.random.random(size),side='right')
        self.last_sample_ids=ids.copy()
        np.add.at(self.draws,ids,1)
        return self.features[ids],self.targets[ids],self.loss_weights[ids].astype(np.float32)

    def diagnostics(self):
        ratio=self.loss_weights
        return dict(method='importance-corrected-balanced',original_rows=self.original_rows,
            conditional_records=len(self.features),draws=int(self.draws.sum()),
            distinct_entries=int((self.draws>0).sum()),normalizer_z=self.normalizer_z,
            conditional_variance_constant=self.conditional_variance_constant,
            target_probability_min=float(self.target_probabilities.min()),target_probability_max=float(self.target_probabilities.max()),
            old_to_corrected_importance_min=float(ratio.min()),old_to_corrected_importance_max=float(ratio.max()),
            importance_second_moment=float(np.dot(self.probabilities,np.square(self.loss_weights))),
            effective_batch_fraction=1./float(np.dot(self.probabilities,np.square(self.loss_weights))),
            raw_mean_rounding_max_abs=self.raw_mean_rounding_max_abs,
            normalized_target_rounding_max_abs=self.normalized_target_rounding_max_abs,
            importance_weight_rounding_max_abs=float(np.max(np.abs(self.loss_weights-self.loss_weights.astype(np.float32)))),
            by_stratum={str(int(g)):dict(records=int((self.groups==g).sum()),
                retained_iteration_mass=float(self.original_probabilities[self.groups==g].sum()),
                previous_sampler_mass=float(self.original_sampler_probabilities[self.groups==g].sum()),
                actual_sampler_mass=float(self.probabilities[self.groups==g].sum()),
                objective_normalized_score_mass=float(self.target_probabilities[self.groups==g].sum()),scale=float(self.scales[g]))
                for g in np.unique(self.groups)},
            identity='Expected q-sampled (q*/q)-weighted normalized-MSE gradient = retained-row raw-unit iteration-weighted-MSE gradient / Z, up to float32 rounding',
            limitations=['Empirical retained replay objective; not a full-game or six-player convergence guarantee',
                'Correcting objective mass may reduce deliberate coverage of rare contexts',
                'Conditional variance and lost state information cannot be recovered by extra fitting'])
