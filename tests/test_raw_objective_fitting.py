"""Finite-population conservation and matched-control draw regression tests."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from raw_objective_fitting import conditional_raw_objective, RawObjectiveRegretSampler


def example():
    # Duplicated observations carry unequal, deliberately conflicting labels.
    x=np.array([[.2,.4],[.2,.4],[.8,-.3],[.1,.7],[.1,.7],[.9,.6]],dtype=np.float64)
    y=np.array([[3.,-2.,9.],[7.,2.,-8.],[-4.,6.,1.],[1.,-3.,2.],
                [-2.,8.,6.],[9.,-5.,3.]],dtype=np.float64)
    w=np.array([1.,3.,2.,7.,1.,5.]);groups=np.array([0,0,0,1,1,1])
    legal=np.array([[1,1,0],[1,1,0],[1,0,1],[0,1,1],[0,1,1],[1,1,1]],dtype=float)
    scales=np.array([2.,17.])
    return x,y,w,groups,scales,legal


def test_weighted_conditional_objective_and_full_network_gradient_match():
    x,y,w,g,s,legal=example()
    proof=conditional_raw_objective(x,y,w,g,s,legal,collapse_groups=(0,1))
    torch.manual_seed(9)
    network=torch.nn.Sequential(torch.nn.Linear(2,5),torch.nn.Tanh(),torch.nn.Linear(5,3)).double()
    raw_scores=network(torch.from_numpy(x))*torch.from_numpy(legal)
    raw_loss=(torch.from_numpy(w/w.sum())*(raw_scores*torch.from_numpy(s[g,None])-torch.from_numpy(y*legal)).square().mean(1)).sum()
    gradients=torch.autograd.grad(raw_loss,tuple(network.parameters()))
    compressed=network(torch.from_numpy(proof['features']))*torch.from_numpy(proof['legal'])
    error=(compressed-torch.from_numpy(proof['normalized_targets']*proof['legal'])).square().mean(1)
    # An intentionally skewed proposal; fixed weights must undo it exactly.
    q=np.arange(1,len(error)+1,dtype=float);q/=q.sum()
    importance=proof['target_probabilities']/q
    proposal_loss=(torch.from_numpy(q)*torch.from_numpy(importance)*error).sum()
    corrected_gradients=torch.autograd.grad(proposal_loss,tuple(network.parameters()),retain_graph=True)
    assert float(raw_loss.detach()) == pytest.approx(float(proposal_loss.detach())*proof['normalizer_z']+proof['conditional_variance_constant'],abs=1e-12)
    for before,after in zip(gradients,corrected_gradients):
        torch.testing.assert_close(before/proof['normalizer_z'],after,rtol=1e-12,atol=1e-12)
    # Regression trap: preserving p alone omits the s² correction.
    wrong=(torch.from_numpy(proof['original_probabilities'])*error).sum()
    wrong_gradients=torch.autograd.grad(wrong,tuple(network.parameters()))
    assert max(float((a-b).abs().max()) for a,b in zip(wrong_gradients,corrected_gradients)) > .01


def test_collapse_preserves_record_mass_order_and_variance_constant():
    x,y,w,g,s,legal=example()
    r=conditional_raw_objective(x,y,w,g,s,legal,collapse_groups=(0,))
    assert r['features'].shape==(5,2)
    np.testing.assert_array_equal(r['row_to_record'],[0,0,1,2,3,4])
    np.testing.assert_allclose(r['mass'],[4,2,7,1,5])
    np.testing.assert_allclose(r['raw_targets'][0],(y[0]+3*y[1])/4)
    expected=np.sum(w[:2,None]*np.square((y[:2]-r['raw_targets'][0])*legal[:2]))/3/w.sum()
    assert r['conditional_variance_constant']==pytest.approx(expected)
    assert r['target_probabilities'].sum()==pytest.approx(1.)


@pytest.mark.parametrize('key,value',[
    ('weights',np.array([1.,3.,0.,7.,1.,5.])),
    ('weights',np.array([1.,3.,np.nan,7.,1.,5.])),
    ('scales',np.array([2.,0.])),
    ('scales',np.array([2.,np.inf])),
])
def test_invalid_objective_mass_rejected(key,value):
    x,y,w,g,s,legal=example()
    args=dict(features=x,targets=y,weights=w,groups=g,scales=s,legal=legal,collapse_groups=(0,1))
    args[key]=value
    with pytest.raises(ValueError):conditional_raw_objective(**args)


def test_identical_features_cannot_collapse_conflicting_legal_masks():
    x,y,w,g,s,legal=example();legal[1]=[1,0,1]
    with pytest.raises(ValueError,match='legal masks'):
        conditional_raw_objective(x,y,w,g,s,legal,collapse_groups=(0,1))


class Buffer:
    def __init__(self,x,y,w):self.features=x;self.advantages=y;self.iterations=w
    def __len__(self):return len(self.features)


def test_production_sampler_matches_old_draws_and_uses_fixed_weights():
    from scaled_regret_fitting import ScaledRegretSampler
    x=np.zeros((7,51),dtype=np.float32);x[:,0]=[1,1,2,3,4,5,6]
    x[:,1]=[0,0,0,6,6,6,6];x[:,46:51]=[1,1,0,1,0]
    y=np.array([[1,3,0,-2,0],[5,-1,0,2,0],[3,2,0,8,0],
                [50,1,0,-10,0],[2,-60,0,8,0],[9,20,0,5,0],[-4,3,0,2,0]],dtype=np.float32)
    w=np.array([1,3,5,2,9,6,8],dtype=np.float32)
    buffers=[Buffer(x,y,w)];groups=lambda a:a[:,1].astype(int)
    old=ScaledRegretSampler(buffers,groups,'stratum-scaled-feature-conditional')
    corrected=RawObjectiveRegretSampler(buffers,groups,old,old.scales.copy())
    np.random.seed(23);old_x,old_y=old.sample(500)
    np.random.seed(23);new_x,new_y,weights=corrected.sample(500)
    np.testing.assert_array_equal(old_x,new_x);np.testing.assert_array_equal(old_y,new_y)
    assert np.isfinite(weights).all() and (weights>0).all()
    assert np.dot(corrected.probabilities,corrected.loss_weights)==pytest.approx(1.)
    np.testing.assert_allclose(corrected.probabilities*corrected.loss_weights,corrected.target_probabilities)
    assert corrected.diagnostics()['method']=='importance-corrected-balanced'
    damaged=SimpleNamespace(**vars(old));damaged.targets=old.targets.copy();damaged.targets[0,0]+=1
    with pytest.raises(ValueError,match='targets'):
        RawObjectiveRegretSampler(buffers,groups,damaged,old.scales.copy())
