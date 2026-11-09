"""Frozen-buffer diagnostics and controlled refits, never a training resume."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/fullhouse-medium/20261106T023201457778Z'
sys.path.insert(0, str(RUN / 'source'))
from training.network import AverageStrategyNet, DuelingAdvantageNet
from training.train import _export_strategy_npz, _export_advantage_npz
from training.r1_support import weight_diagnostics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(state, horizon, strategy=True):
    x = torch.from_numpy(state['features'])
    y = torch.from_numpy(state['advantages'])
    w = state['iterations'].astype(float) / horizon
    if strategy:
        w *= state['sample_weights']
        y = y * x[:, 46:51]
        y = y / y.sum(1, keepdim=True).clamp(min=1e-12)
    return x, y, w


def score(net, x, y, w, ids, strategy=True):
    sums = np.zeros(4)
    mass = w[ids].sum()
    with torch.no_grad():
        for start in range(0, len(ids), 2048):
            ix = ids[start:start + 2048]
            a, b = x[ix], y[ix]
            pred = net(a, a[:, 46:51])
            if strategy:
                lp = torch.log_softmax(pred, 1)
                ce = -(b * lp).sum(1)
                kl = ce + (b * b.clamp(min=1e-12).log()).sum(1)
                tv = (lp.exp() - b).abs().sum(1) / 2
                vals = torch.stack([ce, kl, tv, -(lp.exp() * lp).sum(1)], 1).numpy()
            else:
                vals = torch.stack([(pred - b).square().mean(1), b.square().mean(1),
                                    pred.square().mean(1), (pred - b).abs().mean(1)], 1).numpy()
            sums += (vals * w[ix, None]).sum(0)
    names = ['ce', 'kl_to_stored_target', 'tv_to_stored_target', 'prediction_entropy'] if strategy else ['mse', 'zero_mse', 'prediction_square', 'mae']
    return dict(zip(names, (sums / mass).tolist()))


def fit(mode, seed, x, y, w, ids, valid, output, export=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = AverageStrategyNet()
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    cdf = np.cumsum(w[ids] / w[ids].sum())
    cdf[-1] = 1.
    grad_norms, masses, rows = [], [], []
    start = time.monotonic()
    touched = np.zeros(len(x), dtype=bool)
    for step in range(1, 2049):
        if mode == 'pps':
            idx = ids[np.searchsorted(cdf, rng.random(128), side='right')]
            weights = torch.ones(128)
        else:
            idx = ids[rng.integers(0, len(ids), 128)]
            scale = w[ids].mean() if mode == 'uniform-normalized' else 1.
            weights = torch.tensor(w[idx] / scale, dtype=torch.float32)
        touched[idx] = True
        a, b = x[idx], y[idx]
        ce = -(b * torch.log_softmax(net(a, a[:, 46:51]), 1)).sum(1)
        loss = (weights * ce).mean()
        opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        grad_norms.append(float(norm))
        masses.append(float(w[idx].sum()))
        if step in (128, 512, 2048):
            row = dict(mode=mode, seed=seed, steps=step, seconds=time.monotonic()-start,
                       train=score(net,x,y,w,ids), validation=score(net,x,y,w,valid),
                       clip_fraction=float(np.mean(np.asarray(grad_norms)>1)),
                       gradient_norm_p50=float(np.median(grad_norms)),
                       gradient_norm_p99=float(np.quantile(grad_norms,.99)),
                       distinct_rows=int(touched.sum()), seen_weight_fraction=float(w[touched].sum()/w[ids].sum()))
            rows.append(row)
            if export and step in (128, 2048):
                path = output / f'{mode}-seed{seed}-{step}.npz'
                _export_strategy_npz(net,path)
                row['export_path'] = str(path)
                row['export_sha256'] = digest(path)
            print(json.dumps(row), flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint=RUN/'training/latest.pt'
    ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    x,y,w=arrays(ck['strategy_buffer'],600)
    ids=np.arange(len(x))
    report=dict(checkpoint_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),
                objective='sum_i (iteration_i/600)*importance_i*cross_entropy_i / sum_i weights; PPS samples in proportion to those exact weights, no clipping of sample weights.',
                limitations=['Frozen final replay, not new self play.', 'Row holdout has correlated histories and extreme weight imbalance; not poker generalization.', 'PPS retains the same empirical objective, but Adam and gradient clipping make optimizer paths different.'],
                buffer=weight_diagnostics(w),groups={},models={},refits=[])
    def save():
        (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    street=x[:,17:21].argmax(1).numpy()
    for name,mask in [(f'street-{s}',street==s) for s in range(4)]+[
            ('pot-under-10bb',x[:,27].numpy()<.1),
            ('pot-10-to-50bb',(x[:,27].numpy()>=.1)&(x[:,27].numpy()<.5)),
            ('pot-50-to-200bb',(x[:,27].numpy()>=.5)&(x[:,27].numpy()<2)),
            ('pot-over-200bb',x[:,27].numpy()>=2)]:
        report['groups'][name]=dict(weight_diagnostics(w[mask]),mass_fraction=float(w[mask].sum()/w.sum()),
                                    mean_target=np.average(y[mask].numpy(),axis=0,weights=w[mask]).tolist())
    for it in (120,300,600):
        model=torch.load(RUN/f'training/iter_{it:05d}.pt',weights_only=False,map_location='cpu')
        net=AverageStrategyNet();net.load_state_dict(model['strategy_net']);net.eval()
        row={'on_final_replay':score(net,x,y,w,ids),'by_street':{str(s):score(net,x,y,w,ids[street==s]) for s in range(4)}}
        # Numpy deployment export versus Torch inference, same input and mask.
        from bot.deep_cfr_lookup import DeepCFRLookup
        lookup=DeepCFRLookup(RUN/f'training/iter_{it:05d}.npz')
        sample=ids[np.random.default_rng(771).choice(len(x),512,replace=False)]
        a=net.get_strategy(x[sample],x[sample,46:51]).numpy()
        b=np.asarray([lookup._model_strategy(x[i].numpy(),x[i,46:51].numpy(),lookup._models[0]) for i in sample])
        row['numpy_torch_max_abs_error']=float(np.max(np.abs(a-b)))
        report['models'][str(it)]=row
        adv=DuelingAdvantageNet();adv.load_state_dict(model['net']);adv.eval()
        path=args.output/f'advantage-{it}.npz';_export_advantage_npz(adv,path)
        row['advantage_export']=str(path);row['advantage_sha256']=digest(path)
    adv=DuelingAdvantageNet();adv.load_state_dict(ck['net']);adv.eval()
    report['final_advantage_replay']={}
    for seat,state in enumerate(ck['buffers']):
        a,b,v=arrays(state,600,False)
        report['final_advantage_replay'][str(seat)]=score(adv,a,b,v,np.arange(len(a)),False)
    save()
    split=np.random.default_rng(701).permutation(ids)
    train,valid=split[:160000],split[160000:]
    report['holdout_weight_fraction']=float(w[valid].sum()/w.sum())
    for seed in (7,8,9):
        for mode in ('uniform-raw','uniform-normalized','pps'):
            report['refits']+=fit(mode,seed,x,y,w,train,valid,args.output)
            save()
    # Fixed before inspecting match results: all-buffer seed-7 controls for game evaluation.
    report['all_buffer_refits']=[]
    for mode in ('uniform-raw','pps'):
        report['all_buffer_refits']+=fit(mode,7,x,y,w,ids,ids,args.output,True)
        save()
    report['complete']=True;save()


if __name__=='__main__':
    main()
