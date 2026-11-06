"""Reproduce review evidence against the pinned trainer; do not modify models."""
import json
import statistics
import sys
from pathlib import Path
from unittest.mock import patch
from fullhouse_runtime import ROOT, SOURCE, validate_source

def main():
    validate_source();sys.path.insert(0,str(SOURCE))
    import numpy as np
    import torch
    from training.train import train_network
    from training.network import DuelingAdvantageNet
    from training.buffer import ReservoirBuffer
    from training.traverse_fast import _mix_with_uniform
    torch.manual_seed(7);np.random.seed(7)
    x=np.zeros(51,dtype=np.float32);x[46:49]=1
    raw=np.array([2.,1.,-.375,0.,0.],dtype=np.float32)
    assert np.isclose(np.dot(raw,[.1,.1,.8,0,0]),0)
    b=ReservoirBuffer(2,51,5);b.add(x,raw,1)
    captured=[]
    class Capture(torch.nn.MSELoss):
        def forward(self,pred,target):
            captured.append(target.detach().cpu().numpy().copy())
            return super().forward(pred,target)
    net=DuelingAdvantageNet(51,16,8)
    with patch('training.train.nn.MSELoss',Capture):
        train_network(net,[b],torch.optim.Adam(net.parameters()),1,1,torch.device('cpu'))
    def regret_matching(a):
        v=np.maximum(a,0);return (v/v.sum()).tolist()
    before=regret_matching(raw);after=regret_matching(captured[0][0])
    assert not np.allclose(before,after)
    b.add(x,raw,4)
    with patch('numpy.random.randint',return_value=np.array([0])):old=b.sample(1)[2].tolist()
    with patch('numpy.random.randint',return_value=np.array([1])):new=b.sample(1)[2].tolist()
    with patch('numpy.random.randint',return_value=np.array([0,1])):together=b.sample(2)[2].tolist()
    sigma=np.array([.9,.1,0,0,0],dtype=np.float32);mask=np.array([1,1,0,0,0],dtype=np.float32)
    q=_mix_with_uniform(sigma,mask,.2)
    values=np.array([0,10,0,0,0])
    result={'target_normalization':{'raw_regrets':raw.tolist(),'actual_training_targets':captured[0][0].tolist(),'raw_policy':before,'normalized_policy':after},
            'batch_weighting':{'old_alone':old,'new_alone':new,'same_batch':together},
            'opponent_exploration':{'sigma':sigma.tolist(),'q':q.tolist(),'expected_under_sigma':float(sigma@values),'expected_under_q':float(q@values)}}
    folder=ROOT/'artifacts/fullhouse-evaluation/20261106T013048502614Z'
    rows=[json.loads(x) for x in (folder/'hands.jsonl').read_text().splitlines()]
    reported=json.loads((folder/'results.json').read_text())
    assert len(rows)==1800 and all(sum(x['utilities'])==0 for x in rows)
    result['recomputed_evaluation']={}
    for arm in ('fullhouse','noregrets','same-noregrets'):
        blocks={}
        for r in rows:
            if r['arm']==arm:blocks.setdefault(r['deal'],[]).append(r)
        assert len(blocks)==100 and all(sorted(x['hero'] for x in group)==list(range(6)) for group in blocks.values())
        means=[statistics.mean(x['chips'] for x in group) for group in blocks.values()]
        mean=statistics.mean(means);half=1.96*statistics.stdev(means)/len(means)**.5
        assert abs(mean-reported['summary'][arm]['bb_per_100'])<1e-8
        assert abs(half-reported['summary'][arm]['deal_ci95_half_width'])<1e-8
        result['recomputed_evaluation'][arm]={'mean':mean,'ci95':[mean-half,mean+half]}
    out=ROOT/'artifacts/research-review';out.mkdir(exist_ok=True)
    (out/'r1-r2-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
