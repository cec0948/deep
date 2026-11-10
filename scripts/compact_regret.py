"""Two small regret experts; exactly one is evaluated for a deployment decision."""
from pathlib import Path

import numpy as np
import torch

from training.network import DuelingAdvantageNet
from training.traverse_fast import NumpyPolicy as BaseNumpyPolicy
from regret_partition import is_opening,base_strata
from regret_fitting import train_conditional
from scaled_regret_fitting import ScaledRegretSampler

METHOD='compact-opening-expert'


class ReplayView:
    def __init__(self,x,y,w):
        self.features=x;self.advantages=y;self.iterations=w
    def __len__(self):return len(self.features)


class CompactRegretNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main=DuelingAdvantageNet();self.opening=DuelingAdvantageNet()
        self.register_buffer('main_scales',torch.ones(22))
        self.register_buffer('opening_scale',torch.ones(()))

    def forward(self,x,legal):
        selected=torch.as_tensor(is_opening(x.detach().cpu().numpy()),device=x.device)
        output=torch.empty((len(x),5),dtype=x.dtype,device=x.device)
        if selected.any():output[selected]=self.opening(x[selected],legal[selected])
        if (~selected).any():output[~selected]=self.main(x[~selected],legal[~selected])
        return output

    @torch.no_grad()
    def get_strategy(self,x,legal):
        # Regret matching is discontinuous when scores cross zero. FP32 batch
        # GEMM and one-observation GEMV can land on different sides of zero.
        # Strategy collection, traversal and deployment use one NumPy policy;
        # differentiable supervised fitting still uses forward() above.
        policy=CompactNumpyPolicy(self)
        result=np.stack([policy.strategy(a,b) for a,b in zip(x.detach().cpu().numpy(),legal.detach().cpu().numpy())])
        return torch.from_numpy(result).to(x.device)


class CompactNumpyPolicy:
    def __init__(self,net):
        if isinstance(net,CompactRegretNet):
            self.main=BaseNumpyPolicy(net.main);self.opening=BaseNumpyPolicy(net.opening)
        else:self.main=BaseNumpyPolicy(net);self.opening=None
    def strategy(self,features,legal):
        selected=self.opening if self.opening is not None and bool(is_opening(features)) else self.main
        return selected.strategy(features,legal)


def install_policy():
    import training.traverse_fast as traverse
    traverse.NumpyPolicy=CompactNumpyPolicy


class BBCompactPredictions(torch.nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x,legal):
        raw=x.detach().cpu().numpy();selected=is_opening(raw)
        scale=self.net.main_scales[torch.as_tensor(base_strata(raw),device=x.device)]
        scale=torch.where(torch.as_tensor(selected,device=x.device),self.net.opening_scale,scale)
        return self.net(x,legal)*scale[:,None]


def split_replay(buffers,root_replay=None):
    other=[];xs=[];ys=[];ws=[]
    for b in buffers:
        x=b.features[:len(b)];mask=is_opening(x)
        other.append(ReplayView(x[~mask],b.advantages[:len(b)][~mask],b.iterations[:len(b)][~mask]))
        if root_replay is None:
            xs.append(x[mask]);ys.append(b.advantages[:len(b)][mask]);ws.append(b.iterations[:len(b)][mask])
    if root_replay is not None:return other,root_replay.means()
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws).astype(float)
    if not len(x):return other,(np.empty((0,51),np.float32),np.empty((0,5)),np.empty(0))
    keys,inv=np.unique(x,axis=0,return_inverse=True)
    mass=np.bincount(inv,weights=w,minlength=len(keys))
    targets=np.stack([np.bincount(inv,weights=w*y[:,a],minlength=len(keys)) for a in range(5)],1)/mass[:,None]
    return other,(keys,targets,mass)


def fit_opening(net,root,steps,lr=.001,optimizer=None):
    x,y,w=root
    if not len(x):return dict(steps=0,classes=0)
    scale=max(1.,float(np.sqrt(np.average(np.square(y).mean(1),weights=w))))
    x=torch.from_numpy(x);legal=x[:,46:51]
    target=torch.from_numpy((y/scale).astype(np.float32));weights=torch.from_numpy((w/w.mean()).astype(np.float32))
    net.opening_scale.fill_(scale);opt=optimizer or torch.optim.Adam(net.opening.parameters(),lr=lr)
    net.opening.train()
    for _ in range(steps):
        loss=(weights*(net.opening(x,legal)-target).square().mean(1)).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.opening.parameters(),1.);opt.step()
    net.opening.eval()
    return dict(steps=steps,classes=len(x),scale=scale,loss=float(loss.detach()))


def fit_compact(net,buffers,steps,batch_size,root_replay=None,opening_steps=None,lr=.001):
    other,root=split_replay(buffers,root_replay)
    if sum(len(b) for b in other):
        sampler=ScaledRegretSampler(other,base_strata,'stratum-scaled-feature-conditional')
        loss=train_conditional(net.main,sampler,torch.optim.Adam(net.main.parameters(),lr=lr),steps,batch_size)
        net.main_scales.copy_(torch.from_numpy(sampler.scales))
    else:loss=0.
    opening=fit_opening(net,root,opening_steps or steps,lr)
    net.eval();return dict(main_loss=loss,opening=opening)


ARRAY_MAP={'trunk.0.weight':'trunk_w0','trunk.0.bias':'trunk_b0','trunk.1.weight':'trunk_ln0_g','trunk.1.bias':'trunk_ln0_b',
           'trunk.3.weight':'trunk_w1','trunk.3.bias':'trunk_b1','trunk.4.weight':'trunk_ln1_g','trunk.4.bias':'trunk_ln1_b',
           'value_head.0.weight':'val_w0','value_head.0.bias':'val_b0','value_head.2.weight':'val_w1','value_head.2.bias':'val_b1',
           'advantage_head.0.weight':'adv_w0','advantage_head.0.bias':'adv_b0','advantage_head.2.weight':'adv_w1','advantage_head.2.bias':'adv_b1'}


def load_main(net,path):
    with np.load(path) as data:
        net.main.load_state_dict({key:torch.from_numpy(data[name].copy()) for key,name in ARRAY_MAP.items()})
        net.main_scales.copy_(torch.from_numpy(data['regret_scale_by_stratum'].copy()))


def export_compact(net,path):
    arrays={name:net.main.state_dict()[key].detach().numpy() for key,name in ARRAY_MAP.items()}
    arrays.update({'opening_'+name:net.opening.state_dict()[key].detach().numpy() for key,name in ARRAY_MAP.items()})
    np.savez(path,**arrays,training_revision=np.asarray('r1-cpu-v4'),regret_fit_method=np.asarray(METHOD),
             main_scales=net.main_scales.numpy(),opening_scale=net.opening_scale.numpy())
