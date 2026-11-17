import sys,json,os,subprocess,time
from pathlib import Path
sys.path.insert(0,'scripts')
import numpy as np
from train_poker_average_baseline import load_checkpoint
from poker_strategy_average import StrategyReplay
from poker_average_diagnostics import conditional_targets
from fullhouse_v3_common import ROOT,PYTHON,atomic_json,digest,process_start
from controlled_poker_refit import utc
out=ROOT/'artifacts/average-repair-diagnostic/20261117T031500Z'
out.mkdir(parents=True,exist_ok=False);(out/'inputs').mkdir();(out/'logs').mkdir()
old=ROOT/'artifacts/average-baseline/20261116T084130Z/campaign'
inputs={};frozen={};jobs=[]
for seed in [342,343,344]:
 state,path=load_checkpoint(old/f'seed-{seed}')
 replay=StrategyReplay.from_state(state['strategy_replay']);x,y,mass=conditional_targets(replay)
 order=np.random.default_rng(seed+1711000).permutation(len(x));train=np.ones(len(x),bool);train[order[:len(x)//5]]=False
 p=out/'inputs'/f'seed-{seed}.npz';np.savez_compressed(p,x=x,y=y,mass=mass,train=train)
 inputs[str(seed)]=dict(iteration=state['iteration'],rows=replay.size,distinct_inputs=len(x),heldout_groups=int((~train).sum()),input_sha256=digest(p),checkpoint=str(path))
 frozen.update({str(p):digest(p) for p in path.iterdir() if p.is_file()})
 for width,norm in [(128,False),(128,True),(256,False),(256,True)]:
  jobs.append(dict(seed=seed,fit_seed=seed*100000+state['iteration'],width=width,normalize=norm,id=f'{seed}-w{width}-norm{int(norm)}',input=str(p)))
for n in ['poker_average_diagnostics.py','poker_strategy_average.py','poker_average_inference.py']:
 p=ROOT/'scripts'/n;frozen[str(p)]=digest(p)
plan=dict(created_utc=utc(),authorization='User2026-11-17:check carefully and proceed',purpose='Fixed-data approximation diagnosis before authorized continuation',inputs=inputs,jobs=jobs,frozen_files=frozen,steps=16384,observations=[8192,16384],batch_size=512,heldout='20percent unique input groups; no identical feature group in both partitions; normalization fit on train only',selection='At16384steps choose minimum mean heldout TV across3datasets; prefer smaller width then no normalization on exact tie. Report training TV<=0.15 and heldout regressions separately; no claim of poker improvement.',automatic_retry=False,new_selfplay_iterations=0,max_parallel=3,max_seconds_per_job=900,max_total_seconds=3600)
atomic_json(out/'plan.json',plan)
atomic_json(out.parent/'current.json',dict(folder=str(out),pid=os.getpid(),process_start=process_start(os.getpid()),started_utc=utc()))
queue=jobs.copy();active={};done={};began=time.monotonic()
try:
 while queue or active:
  if time.monotonic()-began>3600:raise TimeoutError('Diagnostic total budget exceeded')
  while queue and len(active)<3:
   j=queue.pop(0);cmd=[str(PYTHON),str(ROOT/'scripts/poker_average_diagnostics.py'),'--input',j['input'],'--output',str(out/j['id']),'--seed',str(j['fit_seed']),'--width',str(j['width'])]
   if j['normalize']:cmd.append('--normalize')
   log=(out/'logs'/f"{j['id']}.log").open('w');p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   active[j['id']]=dict(p=p,log=log,j=j,start=time.monotonic(),command=cmd)
  for key,v in list(active.items()):
   if time.monotonic()-v['start']>900:raise TimeoutError('Diagnostic job budget exceeded')
   code=v['p'].poll()
   if code is None:continue
   v['log'].close();del active[key]
   if code:raise RuntimeError(key+' failed; no automatic retry')
   done[key]=json.loads((out/key/'result.json').read_text());print(key,done[key]['checkpoints'],flush=True)
  atomic_json(out/'status.json',dict(observed_utc=utc(),status='running',completed=len(done),planned=12,active={k:dict(pid=v['p'].pid,process_start=process_start(v['p'].pid),command=v['command']) for k,v in active.items()}))
  time.sleep(1)
 for p,sha in frozen.items():
  if digest(p)!=sha:raise ValueError('Diagnostic frozen input changed: '+p)
 arms=[]
 for width,norm in [(128,False),(128,True),(256,False),(256,True)]:
  results=[done[f'{s}-w{width}-norm{int(norm)}']['checkpoints'][-1] for s in [342,343,344]]
  arms.append(dict(width=width,normalize=norm,steps=16384,mean_heldout_tv=float(np.mean([r['heldout']['weighted_tv'] for r in results])),per_seed=results,all_train_tv_within_original_reference=all(r['training']['weighted_tv']<=.15 for r in results)))
 selected=min(arms,key=lambda r:(r['mean_heldout_tv'],r['width'],r['normalize']))
 summary=dict(complete=True,completed_utc=utc(),arms=arms,selected=selected,new_selfplay_iterations=0,poker_strength_evaluated=False,source_inputs_unchanged=True)
 atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',dict(status='complete',observed_utc=utc(),completed=12,planned=12,active={}))
 print(json.dumps(summary),flush=True)
except BaseException as e:
 for v in active.values():
  v['p'].terminate()
 for v in active.values():
  try:v['p'].wait(timeout=10)
  except subprocess.TimeoutExpired:v['p'].kill();v['p'].wait()
 atomic_json(out/'failure.json',dict(observed_utc=utc(),error=str(e),automatic_retry=False));raise
