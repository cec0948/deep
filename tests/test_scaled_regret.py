"""Numerical invariants for normalization across replay iterations."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from scaled_regret_fitting import ScaledRegretSampler, METHODS


def test_scaling_preserves_conditional_regret_matching():
    class Buffer:
        features = np.zeros((4,51), np.float32)
        features[:,46:49] = 1
        features[:2,17] = 1;features[:2,24] = 1;features[:2,27] = .015;features[:2,45] = 1
        features[2,0] = 1; features[3,0] = 2
        advantages = np.array([[2,1,-.375,0,0],[6,-3,1.625,0,0],[-2,2,0,0,0],[1,3,-1,0,0]],np.float32)
        iterations = np.array([1,3,100,400],np.float32)
        def __len__(self): return 4
    def strata(x): return (x[:,0] == 2).astype(int)*6
    def rm(x):
        p = np.maximum(x,0); return p/p.sum(1,keepdims=True)
    for method in METHODS:
        s = ScaledRegretSampler([Buffer()], strata, method)
        np.testing.assert_allclose(s.raw_targets[0], [5,-2,1.125,0,0])
        np.testing.assert_allclose(s.targets*s.scales[s.groups,None],s.raw_targets,rtol=1e-6)
        np.testing.assert_allclose(rm(s.targets),rm(s.raw_targets),rtol=1e-6)
        assert np.isclose(s.probabilities.sum(),1)
    s = ScaledRegretSampler([Buffer()],strata,METHODS[1])
    # Two distinct preflop inputs get equal mass regardless of replay frequency.
    np.testing.assert_allclose(s.probabilities,[.25,.25,.5])
    focused = ScaledRegretSampler([Buffer()],strata,METHODS[2])
    np.testing.assert_allclose(focused.probabilities,[1/3,1/3,1/3])


def test_root_private_card_classes_partition_all_physical_hands():
    from root_regret_replay import hand_classes,card_pairs,conditional_deck
    classes = hand_classes()
    assert len(classes) == 169
    pairs = [tuple(sorted(pair)) for hand in classes for pair in card_pairs(hand)]
    assert len(pairs) == len(set(pairs)) == 1326
    rng = np.random.default_rng(41)
    for hero in range(6):
        for cards in ((0,13),(12,25),(0,51)):
            deck = conditional_deck(cards,hero,rng)
            assert list(deck[2*hero:2*hero+2]) == list(cards)
            np.testing.assert_array_equal(np.sort(deck),np.arange(52))


def test_root_sufficient_statistics_preserve_time_mean_and_resume():
    from root_regret_replay import RootReplay
    r = RootReplay();x = np.zeros(51,np.float32)
    r.add(x,np.array([2,1,-.375,0,0]),1)
    saved = r.state_dict();restored = RootReplay();restored.load_state_dict(saved)
    for replay in (r,restored): replay.add(x,np.array([6,-3,1.625,0,0]),3)
    a,b,w = r.means();np.testing.assert_allclose(b,[[5,-2,1.125,0,0]]);np.testing.assert_allclose(w,[4])
    for key,value in r.state_dict().items(): np.testing.assert_array_equal(value,restored.state_dict()[key])
    assert r.diagnostics()['minimum_observations'] == 2
    # Changing repetitions must change estimator variance, not the intended
    # iteration weights of the cumulative regret target.
    different=RootReplay()
    for _ in range(16):different.add(x,np.array([2,1,-.375,0,0]),1,sample_weight=1/16)
    for _ in range(4):different.add(x,np.array([6,-3,1.625,0,0]),3,sample_weight=1/4)
    np.testing.assert_allclose(different.means()[1],b)


def test_root_rollout_matches_nonuniform_exact_tree():
    import subprocess,os
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys
sys.path[:0]=['scripts','artifacts/fullhouse-v3/20261109T034714104421Z/source']
import numpy as np
from kuhn_validation import game_adapter
from root_regret_replay import rollout_action_values
from training import game as g
class NeedDraw(Exception):
 def __init__(self,p):self.p=p
class Rng:
 def __init__(self,path):self.path=path;self.i=0
 def choice(self,n,p):
  if self.i==len(self.path):raise NeedDraw(p)
  a=self.path[self.i];self.i+=1;return a
class Policy:
 def strategy(self,x,legal):
  i=int(np.argmax(x[:12]));p={2:[.4,.6],3:[.3,.7],11:[.2,.8],6:[.9,.1]}[i]
  return np.array(p+[0,0,0],float)
with game_adapter():
 from training.traverse_fast import _encode_fast as encode
 state=np.zeros(g.STATE_SIZE,np.int32);state[g.S_NPLAYERS]=2
 todo=[([],1.)];mean=np.zeros(5);mass=0.
 while todo:
  path,weight=todo.pop()
  try:values=rollout_action_values(state,None,np.array([2,0]),0,Policy(),Rng(path),g,encode)
  except NeedDraw as e:
   todo.extend((path+[a],weight*p) for a,p in enumerate(e.p) if p>0)
  else:mean+=weight*values;mass+=weight
 assert abs(mass-1)<1e-12
 np.testing.assert_allclose(mean[:2],[1.28,1.1],atol=1e-12)
 sigma=np.array([.4,.6,0,0,0]);regret=(mean-sigma@mean)*np.array([1,1,0,0,0])
 np.testing.assert_allclose(regret[:2],[.108,-.072],atol=1e-12)
 assert abs(sigma@regret)<1e-12
'''
    result = subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,
                            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),timeout=60)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


def test_compact_expert_torch_numpy_and_export_match():
    import subprocess,os
    root=Path(__file__).resolve().parents[1]
    code='''
import sys,tempfile
from pathlib import Path
sys.path[:0]=['scripts','artifacts/fullhouse-v3/20261109T034714104421Z/source']
import numpy as np,torch
from compact_regret import CompactRegretNet,CompactNumpyPolicy,export_compact
from fullhouse_v4_runtime import FullhouseV4
from fullhouse_runtime import Bridge
from regret_partition import is_opening
torch.set_num_threads(1);torch.manual_seed(71)
net=CompactRegretNet();net.eval();rng=np.random.default_rng(71)
x=rng.random((80,51),dtype=np.float32);x[:,46:51]=[1,1,1,1,0]
x[:40,17]=1;x[:40,24]=1;x[:40,27]=.015;x[:40,45]=1
expected=net.get_strategy(torch.from_numpy(x),torch.from_numpy(x[:,46:51])).numpy()
policy=CompactNumpyPolicy(net)
observed=np.array([policy.strategy(a,a[46:51]) for a in x])
np.testing.assert_allclose(observed,expected,rtol=1e-5,atol=1e-5)
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'compact.npz';export_compact(net,path);runtime=FullhouseV4(path)
 deployed=[]
 for a in x:
  m=runtime.opening_model if bool(is_opening(a)) else runtime.lookup._models[0]
  deployed.append(runtime.lookup._model_strategy(a,a[46:51],m))
  with torch.no_grad():raw=net(torch.from_numpy(a[None]),torch.from_numpy(a[None,46:51])).numpy()[0]
  np.testing.assert_allclose(runtime.lookup._forward_single(a,a[46:51],m),raw,atol=3e-6,rtol=3e-5)
 np.testing.assert_allclose(deployed,expected,rtol=1e-5,atol=1e-5)
 bridge=Bridge()
 try:
  state=bridge.call(dict(op='replay',hero=3,button=0,hole=['Ac','Ad'],board=[],actions=[]))
  answer=runtime.decide(state,81)
  assert answer['ok'] and np.isclose(sum(p['probability'] for p in answer['distribution']),1)
 finally:bridge.close()
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,
                          env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),timeout=60)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
