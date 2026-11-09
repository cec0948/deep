"""Check fitting budget on the frozen final regret replay, with row holdout."""
import argparse
import json
import time
from pathlib import Path

from diagnose_fullhouse_replay import RUN, torch, np, arrays, score, digest
from training.network import DuelingAdvantageNet
from training.buffer import ReservoirBuffer
from training.train import train_network, _export_advantage_npz
from training import r1_support


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r1_support.WEIGHT_HORIZON=600
    ck=torch.load(RUN/'training/latest.pt',weights_only=False,map_location='cpu')
    rng=np.random.default_rng(701)
    buffers=[];validation=[];training=[]
    for state in ck['buffers']:
        order=rng.permutation(state['count']);cut=int(.8*len(order));a,b=order[:cut],order[cut:]
        selected={key:state[key][a] for key in ('features','advantages','iterations')}
        selected.update(count=len(a),total_seen=len(a))
        buf=ReservoirBuffer(len(a),51);buf.load_state_dict(selected);buffers.append(buf)
        x,y,w=arrays(state,600,False)
        training.append((x,y,w,a));validation.append((x,y,w,b))
    def measure(net,parts):
        rows=[score(net,x,y,w,ids,False) for x,y,w,ids in parts]
        masses=np.array([w[ids].sum() for x,y,w,ids in parts])
        return {key:float(np.average([r[key] for r in rows],weights=masses)) for key in rows[0]}
    existing=DuelingAdvantageNet();existing.load_state_dict(ck['net'])
    result=dict(script_sha256=digest(Path(__file__)),checkpoint_sha256=digest(RUN/'training/latest.pt'),
                existing_final_model=measure(existing,validation),refits=[],
                limitation='Seeded row holdout, histories correlated. Additional optimizer steps use frozen labels; fitting quality is not poker strength or exploitability.')
    for seed in (7,8,9):
        np.random.seed(seed);torch.manual_seed(seed)
        net=DuelingAdvantageNet();opt=torch.optim.Adam(net.parameters(),lr=.001)
        previous=0;start=time.monotonic()
        for steps in (512,2048,8192):
            train_network(net,buffers,opt,steps-previous,128,torch.device('cpu'))
            row=dict(seed=seed,steps=steps,seconds=time.monotonic()-start,
                     train=measure(net,training),validation=measure(net,validation))
            if seed==7:
                path=args.output/f'advantage-refit-{steps}.npz';_export_advantage_npz(net,path)
                row.update(export_path=str(path),export_sha256=digest(path))
            result['refits'].append(row)
            (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            previous=steps
    result['complete']=True
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
