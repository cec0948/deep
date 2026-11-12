"""Verify a retained-replay loss identity on official two-player Leduc samples.

This gate measures an empirical objective and its gradients, not poker strength.
The original balanced draw distribution is kept. Fixed importance factors
restore iteration-weighted raw-unit MSE, including the square of target scales.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import random
import time

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_config(cfg):
    if cfg['game']!='leduc_poker' or cfg['method']!='importance-corrected-balanced':
        raise ValueError('Expected the predeclared Leduc importance-correction gate')
    limits={'iterations':32,'traversals_per_player':128,'advantage_steps':512,
            'batch_size':512,'memory_capacity':100000,'max_seconds':120,
            'sampling_draws':1000000}
    for key,maximum in limits.items():
        if type(cfg[key]) is not int or not 1<=cfg[key]<=maximum:
            raise ValueError('Invalid bounded setting: '+key)
    if cfg['sampling_draws']<10000 or cfg['batch_size']>cfg['memory_capacity']:
        raise ValueError('Insufficient sampling check or invalid batch capacity')
    if (not cfg['network_layers'] or len(cfg['network_layers'])>3
            or any(type(n) is not int or not 1<=n<=128 for n in cfg['network_layers'])
            or not 0<cfg['learning_rate']<=.01):
        raise ValueError('Invalid reference network settings')
    for key in ('seed','sampling_seed'):
        if type(cfg[key]) is not int or not 0<=cfg[key]<2**32:
            raise ValueError('Invalid deterministic seed')
    if len(cfg['probe_seeds'])!=3 or any(type(s) is not int for s in cfg['probe_seeds']):
        raise ValueError('Three fixed probe networks required')
    for key in ('gradient_relative_tolerance','loss_relative_tolerance'):
        if cfg[key]!=1e-9:
            raise ValueError('Exact identity tolerance is fixed')
    if cfg['monte_carlo_sigma']!=8 or cfg['negative_control_min_relative_error']!=1e-4:
        raise ValueError('Fixed Monte Carlo and negative-control criteria required')


def information_metadata(game):
    """Map exact full information states to legal masks and betting rounds."""
    from open_spiel.python import policy
    table=policy.TabularPolicy(game)
    result={}
    patterns=sorted(set(tuple(int(v) for v in row) for row in table.legal_actions_mask))
    if len(patterns)>6:
        raise ValueError('Unexpected number of Leduc legal-action patterns')
    for state in table.states:
        x=np.asarray(state.information_state_tensor(),np.float32)
        legal=np.zeros(game.num_distinct_actions(),np.float64);legal[state.legal_actions()]=1
        round_number=state.round() if callable(state.round) else state.round
        if round_number not in (1,2):
            raise ValueError('Unexpected Leduc round')
        group=6*(round_number-1)+patterns.index(tuple(int(v) for v in legal))
        entry=(legal,group,state.current_player())
        if x.tobytes() in result:
            raise ValueError('Full information-state tensor collision')
        result[x.tobytes()]=entry
    return result


def collect_replay(cfg,progress=None,deadline=None):
    """Keep the official traversal, regret matcher, reservoir and learner."""
    import torch
    import pyspiel
    from open_spiel.python.pytorch import deep_cfr
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    game=pyspiel.load_game(cfg['game'],{'players':2});metadata=information_metadata(game)
    solver=deep_cfr.DeepCFRSolver(game,policy_network_layers=tuple(cfg['network_layers']),
        advantage_network_layers=tuple(cfg['network_layers']),num_iterations=cfg['iterations'],
        num_traversals=cfg['traversals_per_player'],learning_rate=cfg['learning_rate'],
        batch_size_advantage=cfg['batch_size'],batch_size_strategy=cfg['batch_size'],
        memory_capacity=cfg['memory_capacity'],advantage_network_train_steps=cfg['advantage_steps'],
        policy_network_train_steps=1,reinitialize_advantage_networks=True,seed=cfg['seed'],device='cpu')
    for iteration in range(1,cfg['iterations']+1):
        if deadline is not None and time.monotonic()>deadline:
            raise TimeoutError('Predeclared sampling gate time budget exhausted')
        for player in range(2):
            for _ in range(cfg['traversals_per_player']):
                solver._traverse_game_tree(solver._root_node,player)
            solver._reinitialize_advantage_network(player)
            solver._learn_advantage_network(player)
        solver._iteration+=1
        if progress:
            progress(iteration)
    xs=[];ys=[];ws=[];ls=[];gs=[];counts=[]
    for player,buffer in enumerate(solver.advantage_buffers):
        n=len(buffer);counts.append(n)
        x=buffer.experience.info_state[:n].copy()
        y=buffer.experience.advantage[:n].astype(np.float64)
        w=buffer.experience.iteration[:n].reshape(-1).astype(np.float64)
        legal=[];groups=[]
        for row in x:
            mask,group,observing_player=metadata[row.tobytes()]
            if observing_player!=player:
                raise ValueError('Replay sample has the wrong player information state')
            legal.append(mask);groups.append(group)
        legal=np.asarray(legal)
        if (not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0)
                or np.any(y[legal==0]!=0)):
            raise ValueError('Malformed official retained regret sample')
        xs.append(x);ys.append(y);ws.append(w);ls.append(legal);gs.append(groups)
    data=dict(features=np.concatenate(xs),targets=np.concatenate(ys),weights=np.concatenate(ws),
              legal=np.concatenate(ls),groups=np.concatenate(gs).astype(np.int64))
    scales=np.ones(int(data['groups'].max())+1,np.float64)
    for g in np.unique(data['groups']):
        # Input-defined, fixed unequal scales exercise the required s^2 factor.
        scales[g]=1.+2.*(int(g)//6)+.5*data['legal'][data['groups']==g][0].sum()
    data['scales']=scales
    return data,dict(retained_rows_per_player=counts,full_information_states=len(metadata),
                     retained_rows=len(data['weights']),input_dimensions=data['features'].shape[1],
                     game=str(game),illegal_nonzero_regret_targets=0,
                     group_definition='Betting round and exact legal-action pattern; groups below6 are round1')


def balanced_probabilities(proof):
    """Same balancing pattern as the production conditional sampler."""
    groups=proof['groups'];mass=proof['mass'];present=np.unique(groups)
    q=np.zeros(len(mass),np.float64)
    for group in present:
        ids=groups==group
        q[ids]=(1./ids.sum() if group<6 else mass[ids]/mass[ids].sum())/len(present)
    if not np.isfinite(q).all() or np.any(q<=0) or abs(q.sum()-1)>1e-12:
        raise ValueError('Balanced distribution lacks finite full support')
    return q


def verify_objective(data,cfg):
    """Compare exact gradients against the uncompressed retained population."""
    import torch
    from raw_objective_fitting import conditional_raw_objective
    x,y,w,legal,g,scales=[data[k] for k in ('features','targets','weights','legal','groups','scales')]
    proof=conditional_raw_objective(x,y,w,g,scales,legal,
        collapse_groups=[int(group) for group in np.unique(g) if group<6])
    q=balanced_probabilities(proof);importance=proof['target_probabilities']/q
    if not np.isfinite(importance).all() or np.any(importance<=0):
        raise ValueError('Invalid fixed importance weights')
    cdf=np.cumsum(q);cdf[-1]=1.
    rng=np.random.default_rng(cfg['sampling_seed'])
    sampled=np.searchsorted(cdf,rng.random(cfg['sampling_draws']),side='right')
    empirical=np.bincount(sampled,minlength=len(q))/cfg['sampling_draws']
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float64)
    raw_x,raw_y,raw_w,raw_legal=tensor(x),tensor(y),tensor(w/w.sum()),tensor(legal)
    raw_scales=tensor(scales[g])
    cx,cy,cl=tensor(proof['features']),tensor(proof['normalized_targets']),tensor(proof['legal'])
    Z=proof['normalizer_z'];constant=proof['conditional_variance_constant']
    rows=[]
    for seed in cfg['probe_seeds']:
        torch.manual_seed(seed)
        net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],13),torch.nn.Tanh(),
                                torch.nn.Linear(13,y.shape[1])).double()
        def gradient(loss):
            return torch.cat([v.flatten() for v in torch.autograd.grad(loss,tuple(net.parameters()),retain_graph=True)])
        raw_scores=net(raw_x)*raw_legal*raw_scales[:,None]
        raw_loss=(raw_w*(raw_scores-raw_y*raw_legal).square().mean(1)).sum()
        normalized=(net(cx)*cl-cy*cl).square().mean(1)
        weighted=(tensor(q)*tensor(importance)*normalized).sum()
        directed=(tensor(proof['target_probabilities'])*normalized).sum()
        wrong=(tensor(proof['original_probabilities'])*normalized).sum()
        raw_gradient=gradient(raw_loss)/Z
        corrected_gradient=gradient(weighted)
        direct_gradient=gradient(directed)
        wrong_gradient=gradient(wrong)
        denom=max(float(torch.linalg.vector_norm(raw_gradient)),1e-15)
        gradient_error=float(torch.linalg.vector_norm(raw_gradient-corrected_gradient))/denom
        direct_error=float(torch.linalg.vector_norm(raw_gradient-direct_gradient))/denom
        wrong_error=float(torch.linalg.vector_norm(raw_gradient-wrong_gradient))/denom
        loss_error=abs(float(raw_loss.detach())/Z-float(weighted.detach())-constant/Z)/max(abs(float(raw_loss.detach())/Z),1e-15)
        observations=importance*normalized.detach().numpy()
        expected=float(np.dot(q,observations));estimate=float(np.dot(empirical,observations))
        variance=float(np.dot(q,np.square(observations-expected)))
        standard_error=(variance/cfg['sampling_draws'])**.5
        monte_carlo_ok=abs(estimate-expected)<=cfg['monte_carlo_sigma']*standard_error+1e-12
        rows.append(dict(probe_seed=seed,raw_loss=float(raw_loss.detach()),
            normalized_corrected_loss=float(weighted.detach()),
            conditional_variance_constant=constant,normalizer_z=Z,
            gradient_relative_error=gradient_error,direct_sampling_gradient_relative_error=direct_error,
            missing_scale_squared_relative_gradient_error=wrong_error,loss_relative_error=loss_error,
            monte_carlo=dict(draws=cfg['sampling_draws'],expected=expected,estimate=estimate,
                exact_standard_error=standard_error,within_predeclared_sigma=monte_carlo_ok)))
    checks=dict(all_exact_gradient_identities=all(r['gradient_relative_error']<=cfg['gradient_relative_tolerance'] for r in rows),
        direct_sampling_equivalence=all(r['direct_sampling_gradient_relative_error']<=cfg['gradient_relative_tolerance'] for r in rows),
        conditional_loss_identity=all(r['loss_relative_error']<=cfg['loss_relative_tolerance'] for r in rows),
        missing_scale_squared_negative_control_detected=all(r['missing_scale_squared_relative_gradient_error']>=cfg['negative_control_min_relative_error'] for r in rows),
        importance_sampling_monte_carlo=all(r['monte_carlo']['within_predeclared_sigma'] for r in rows),
        fixed_weights_have_full_support=bool(np.all(importance>0) and np.isfinite(importance).all()),
        compression_actually_exercised=len(proof['mass'])<len(w))
    summary=dict(original_rows=len(w),conditional_records=len(proof['mass']),
        normalizer_z=Z,conditional_variance_constant=constant,
        balanced_q_min=float(q.min()),balanced_q_max=float(q.max()),
        importance_min=float(importance.min()),importance_max=float(importance.max()),
        by_group={str(int(group)):dict(raw_rows=int((g==group).sum()),
            conditional_records=int((proof['groups']==group).sum()),
            retained_iteration_mass=float(proof['original_probabilities'][proof['groups']==group].sum()),
            balanced_draw_mass=float(q[proof['groups']==group].sum()),
            corrected_loss_mass=float(proof['target_probabilities'][proof['groups']==group].sum()),
            scale=float(scales[group])) for group in np.unique(g)})
    return dict(passed=all(checks.values()),gate=dict(passed=all(checks.values()),checks=checks),
                gradient_checks=rows,sampling_check=summary),proof


def run(config,output):
    import torch
    import pyspiel
    from open_spiel.python import policy
    from open_spiel.python.pytorch import deep_cfr
    import raw_objective_fitting
    cfg=json.loads(Path(config).read_text());validate_config(cfg)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Existing sampling gate is immutable')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    sources={str(Path(__file__).resolve()):digest(__file__),str(Path(config).resolve()):digest(config)}
    for module in (raw_objective_fitting,policy,deep_cfr,pyspiel):
        path=Path(inspect.getfile(module)).resolve();sources[str(path)]=digest(path)
    began=time.monotonic()
    report=dict(format='leduc-sampling-objective-gate-v1',complete=False,passed=False,
        method=cfg['method'],config=cfg,source_hashes=sources,started_utc=utc(),
        versions={k:importlib.metadata.version(k) for k in ('open_spiel','torch','numpy','dm-tree')},
        objective='Original retained-row iteration-weighted raw-unit legal-action mean squared error, divided by one fixed positive Z',
        limitations=['Exact objective identity on retained empirical replay, not an equilibrium or poker-strength guarantee',
            'This fixed Leduc data collection is a mathematical validation fixture; no candidate selection or strength gate',
            'Full Leduc information states do not validate the compressed51feature six-player input',
            'Float64 identities are separate from production float32 rounding and stochastic optimization effects',
            'No importance clipping, minibatch self-normalization, or new six-player self-play'])
    def status(state,iteration=0,**extra):
        write_json(output/'status.json',dict(status=state,iteration=iteration,planned_iterations=cfg['iterations'],
            updated_utc=utc(),seconds=time.monotonic()-began,**extra))
    write_json(output/'config.json',cfg);write_json(output/'results.json',report)
    try:
        status('collecting')
        data,collection=collect_replay(cfg,lambda i:status('collecting',i),began+cfg['max_seconds'])
        status('verifying',cfg['iterations'])
        result,proof=verify_objective(data,cfg)
        archive=output/'retained-replay.npz';np.savez_compressed(archive,**data)
        for name,expected in sources.items():
            if digest(name)!=expected:
                raise ValueError('Gate source/config changed during execution: '+name)
        if time.monotonic()-began>cfg['max_seconds']:
            raise TimeoutError('Predeclared sampling gate time budget exhausted')
        report.update(result,complete=True,completed_utc=utc(),seconds=time.monotonic()-began,
            collection=collection,replay_file=dict(name=archive.name,bytes=archive.stat().st_size,sha256=digest(archive)))
        write_json(output/'results.json',report);status('complete',cfg['iterations'],passed=report['passed'])
        return report
    except BaseException as error:
        report.update(error=str(error),stopped_utc=utc(),seconds=time.monotonic()-began)
        write_json(output/'results.json',report);status('failed',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.config,args.output)
    print(json.dumps({k:result[k] for k in ('complete','passed','seconds','gate')}),flush=True)
