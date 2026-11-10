"""Check actual NumPy deployment against frozen targets and Torch score graphs."""
import argparse
import json
from pathlib import Path
import sys

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    cfg=json.loads((a.folder/'config.json').read_text());source=ROOT/cfg['source_run']/'source';sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from compact_regret import CompactRegretNet,ARRAY_MAP
    from regret_partition import is_opening
    from fullhouse_v4_runtime import FullhouseV4
    torch.set_num_threads(1);rows=[]
    for seed,_ in cfg['frozen_pairs']:
        report=json.loads((a.folder/f'frozen-{seed}/results.json').read_text())
        row=next(r for r in report['refits'] if r['steps']==cfg['fixed_buffer_gates']['eligible_steps'])
        path=ROOT/row['model'];runtime=FullhouseV4(path,source)
        net=CompactRegretNet()
        with np.load(path) as data:
            net.main.load_state_dict({key:torch.from_numpy(data[name].copy()) for key,name in ARRAY_MAP.items()})
            net.opening.load_state_dict({key:torch.from_numpy(data['opening_'+name].copy()) for key,name in ARRAY_MAP.items()})
            net.main_scales.copy_(torch.from_numpy(data['main_scales'].copy()));net.opening_scale.copy_(torch.from_numpy(data['opening_scale'].copy()))
        net.eval();table=row['metrics']['root_table'];x=np.array([r['features'] for r in table],np.float32)
        target=np.array([r['target'] for r in table]);mass=np.array([r['mass'] for r in table])
        expected=np.maximum(target,0)*x[:,46:51];total=expected.sum(1,keepdims=True);legal=x[:,46:51].astype(float)
        expected=np.divide(expected,total,out=legal/legal.sum(1,keepdims=True),where=total>0)
        pred=[];raw_error=0.
        for sample in x:
            model=runtime.opening_model if bool(is_opening(sample)) else runtime.lookup._models[0]
            pred.append(runtime.lookup._model_strategy(sample,sample[46:51],model))
            with torch.no_grad():raw=net(torch.from_numpy(sample[None]),torch.from_numpy(sample[None,46:51])).numpy()[0]
            raw_error=max(raw_error,float(np.max(np.abs(raw-runtime.lookup._forward_single(sample,sample[46:51],model)))))
        pred=np.asarray(pred);canonical=net.get_strategy(torch.from_numpy(x),torch.from_numpy(x[:,46:51])).numpy()
        tv=float(np.average(np.abs(pred-expected).sum(1)/2,weights=mass));aa=np.flatnonzero((x[:,10]==1)&(x[:,7]>.99))[0]
        error=float(np.max(np.abs(pred-canonical)))
        passed=tv<=cfg['fixed_buffer_gates']['max_root_weighted_tv'] and pred[aa,0]<=cfg['fixed_buffer_gates']['max_aa_fold'] and error<1e-5 and raw_error<1e-5
        rows.append(dict(seed=seed,model_sha256=digest(path),model_bytes=path.stat().st_size,root_weighted_tv=tv,aa_fold=float(pred[aa,0]),
                         max_deployment_policy_error=error,max_torch_numpy_score_error=raw_error,passed=bool(passed)))
    result=dict(passed=all(r['passed'] for r in rows),rows=rows,runtime_sha256=digest(ROOT/'scripts/fullhouse_v4_runtime.py'),
                compact_fitter_sha256=digest(ROOT/'scripts/compact_regret.py'),partition_sha256=digest(ROOT/'scripts/regret_partition.py'),
                verifier_sha256=digest(Path(__file__)))
    atomic_json(a.folder/'deployment-verification.json',result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
