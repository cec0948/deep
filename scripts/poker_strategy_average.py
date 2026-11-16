"""Bounded neural approximation of a reach-weighted continuation-window strategy.

Reference trajectories use uniform legal actions for EVERY player. Thus their
history probability q(h) is fixed across learning iterations. Adding sigma_i^t
with weight t*pi_i^t(h) yields, at a perfect-recall information set I,
q(I)*sum_t t*pi_i^t(I)*sigma_i^t(I). The common q(I) cancels on normalization.
There is no inverse opponent reach, clipping, policy-weight averaging, or claim
that a lossy51-feature abstraction is a perfect-recall information state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FORMAT = 'compact-window-average-network-v1'


class StrategyReplay:
    def __init__(self, capacity=100000, dimension=51, seed=0):
        if not 1 <= capacity <= 100000 or dimension != 51:
            raise ValueError('Fixed strategy replay capacity/dimension required')
        self.capacity, self.dimension = capacity, dimension
        self.x = np.zeros((capacity, dimension), np.float32)
        self.y = np.zeros((capacity, 5), np.float32)
        self.mass = np.zeros(capacity, np.float64)
        self.size = self.total_seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, features, probabilities, mass):
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(probabilities, dtype=np.float32)
        legal = x[46:51] if x.shape == (self.dimension,) else np.zeros(5)
        if (x.shape != (self.dimension,) or y.shape != (5,) or not np.isfinite(x).all()
                or not np.isfinite(y).all() or not np.isfinite(mass) or mass < 0
                or not np.isin(legal, [0, 1]).all() or not legal.any()
                or np.any(y < 0) or np.any(y[legal == 0] != 0)
                or not np.isclose(y.sum(), 1., atol=1e-6)):
            raise ValueError('Invalid strategy sample')
        if mass == 0:
            return
        self.total_seen += 1
        if self.size < self.capacity:
            index = self.size; self.size += 1
        else:
            index = int(self.rng.integers(self.total_seen))
            if index >= self.capacity:
                return
        self.x[index], self.y[index], self.mass[index] = x, y, mass

    def state_dict(self):
        return dict(capacity=self.capacity, dimension=self.dimension, size=self.size,
            total_seen=self.total_seen, x=self.x[:self.size].copy(), y=self.y[:self.size].copy(),
            mass=self.mass[:self.size].copy(), rng=self.rng.bit_generator.state)

    @classmethod
    def from_state(cls, state):
        result = cls(state['capacity'], state['dimension'])
        n = state['size']
        if (type(n) is not int or not 0 <= n <= result.capacity or state['total_seen'] < n
                or state['x'].shape != (n,51) or state['y'].shape != (n,5)
                or state['mass'].shape != (n,) or np.any(state['mass'] <= 0)
                or any(not np.isfinite(state[k]).all() for k in ('x','y','mass'))):
            raise ValueError('Invalid strategy replay state')
        legal = state['x'][:,46:51]
        if (not np.isin(legal,[0,1]).all() or np.any(legal.sum(1)<1)
                or np.any(state['y']<0) or np.any(state['y'][legal==0]!=0)
                or (n and not np.allclose(state['y'].sum(1),1.,atol=1e-6))):
            raise ValueError('Invalid restored strategy probabilities')
        result.x[:n],result.y[:n],result.mass[:n] = state['x'],state['y'],state['mass']
        result.size,result.total_seen = n,state['total_seen']
        result.rng.bit_generator.state = state['rng']
        return result


def collect_reference(initial_state, players, replay, iteration_weight, rng,
                      *, terminal, chance, outcomes, actor, observe, strategy, child):
    """One fixed-reference trajectory; chance law must not depend on the policy.

    Game adapters supply immutable child states; private simulation cards must
    never be included in observe(). Own reach is the evaluated policy's reach,
    while the actual sampled actions are uniform. This distinction is essential
    in multiplayer games.
    """
    if players < 2 or not np.isfinite(iteration_weight) or iteration_weight <= 0:
        raise ValueError('Invalid reference collection budget')
    reach = np.ones(players, dtype=np.float64)
    state = initial_state; count = 0
    while not terminal(state):
        count += 1
        if count > 1024:
            raise ValueError('Reference trajectory did not terminate')
        if chance(state):
            values, probabilities = zip(*outcomes(state))
            state = child(state, values[int(rng.choice(len(values),p=probabilities))])
            continue
        seat = actor(state); x, legal = observe(state)
        sigma = np.asarray(strategy(state,x,legal),dtype=np.float64)
        if (sigma.shape != (5,) or not np.isfinite(sigma).all() or np.any(sigma<0)
                or np.any(sigma[legal==0]!=0) or not np.isclose(sigma.sum(),1.,atol=1e-6)):
            raise ValueError('Invalid reference policy probabilities')
        sigma /= sigma.sum()
        replay.add(x,sigma,iteration_weight*reach[seat])
        action = int(rng.choice(np.flatnonzero(legal)))
        reach[seat] *= sigma[action]
        state = child(state,action)
    return count


class AverageNet(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(51,width),torch.nn.ReLU(),
            torch.nn.Linear(width,width),torch.nn.ReLU(),torch.nn.Linear(width,5))

    def forward(self, x):
        logits = self.body(x)
        return torch.log_softmax(logits.masked_fill(x[:,46:51]==0,-1e9),dim=1)


def fit_average(replay, *, seed, steps=2048, batch_size=512, width=128):
    """Weighted cross-entropy with soft strategy labels; isolate all training RNG.

    Identical features are combined by their exact weighted conditional means.
    This preserves the entire retained-replay cross-entropy objective/gradient;
    it is variance reduction, not a change of stratum or information-set mass.
    """
    if not replay.size or not 1 <= steps <= 8192 or not 1 <= batch_size <= 1024:
        raise ValueError('Invalid fixed average fitting budget')
    x,inv = np.unique(replay.x[:replay.size],axis=0,return_inverse=True)
    mass = np.bincount(inv,weights=replay.mass[:replay.size],minlength=len(x))
    y = np.stack([np.bincount(inv,weights=replay.mass[:replay.size]*replay.y[:replay.size,a],
                             minlength=len(x)) for a in range(5)],1)/mass[:,None]
    if not np.isfinite(y).all() or not np.allclose(y.sum(1),1.,atol=1e-6):
        raise ValueError('Nonfinite conditional strategy targets')
    cdf = np.cumsum(mass/mass.sum()); cdf[-1] = 1.
    fit_rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = AverageNet(width); optimizer = torch.optim.Adam(net.parameters(),lr=.001)
        tx,ty = torch.from_numpy(x),torch.from_numpy(y.astype(np.float32))
        for _ in range(steps):
            ids = np.searchsorted(cdf,fit_rng.random(batch_size),side='right')
            logp = net(tx[ids]); loss = -(ty[ids]*logp).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite average-policy fitting loss')
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite average-policy gradient')
            optimizer.step()
        net.eval()
        errors=[]; divergences=[]
        with torch.no_grad():
            for start in range(0,len(x),4096):
                p=net(tx[start:start+4096]).exp().numpy().astype(np.float64)
                target=y[start:start+4096]
                errors.extend(np.abs(p-target).sum(1)/2)
                divergences.extend(np.sum(np.where(target>0,target*np.log(np.maximum(target,1e-30)/np.maximum(p,1e-30)),0.),axis=1))
    report=dict(rows=replay.size,total_seen=replay.total_seen,distinct_inputs=len(x),steps=steps,
        batch_size=batch_size,weighted_replay_tv=float(np.average(errors,weights=mass)),
        weighted_replay_kl=float(np.average(divergences,weights=mass)),
        objective='Retained replay mass times soft-label cross-entropy; no stratum rebalance',
        scope='Replay approximation consistency; not independent poker strength',
        parameters=sum(p.numel() for p in net.parameters()))
    if not all(np.isfinite(report[k]) for k in ('weighted_replay_tv','weighted_replay_kl')):
        raise FloatingPointError('Nonfinite average diagnostics')
    return net,report


def export_average(net,path,**metadata):
    arrays={k:v.detach().cpu().numpy() for k,v in net.state_dict().items()}
    arrays['format']=np.array(FORMAT)
    import json
    arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))
    np.savez_compressed(path,**arrays)



from poker_average_inference import load_average, average_strategy
