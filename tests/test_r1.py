"""Regression and exact finite-tree oracle for the corrected CPU reference."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/fullhouse-r1'))
from training import r1_support as r1
from training.buffer import ReservoirBuffer
from training.train import train_network
from training.network import DuelingAdvantageNet
from training.game import S_TERMINAL,S_TOACT

class R1Tests(unittest.TestCase):
    def test_weight_diagnostics(self):
        a=r1.weight_diagnostics([1,1,1,1]);self.assertEqual(a['ess'],4)
        b=r1.weight_diagnostics([0,0,0,100]);self.assertEqual(b['ess'],1)
        self.assertEqual(b['max_mass_fraction'],1)
        self.assertEqual(r1.weight_diagnostics([0,0])['ess'],0)
        with self.assertRaises(FloatingPointError):r1.weight_diagnostics([float('inf')])
    def test_fixed_opponent_average_oracle(self):
        # Opponent sigma changes, but the sampling opponent remains uniform.
        # Player 0 acts twice; each node has its own fixed visitation factor.
        mass={1:0.,2:0.};num={1:np.zeros(5),2:np.zeros(5)}
        expected_mass={1:0.,2:0.};expected_num={1:np.zeros(5),2:np.zeros(5)}
        for t,p,own,deep in [(1,.9,.2,.4),(2,.1,.8,.7),(3,.4,0.,.9)]:
            sigma={0:np.array([p,1-p,0,0,0]),1:np.array([own,1-own,0,0,0]),2:np.array([deep,1-deep,0,0,0])}
            for node,reach in [(1,1.),(2,own)]:
                expected_mass[node]+=t*reach;expected_num[node]+=t*reach*sigma[node]
            def state():
                s=np.zeros(59,np.int32);s[S_TOACT]=1;return s
            def step(s,d,h,a):
                if a==1 or s[58]==2:s[S_TERMINAL]=1
                else:s[58]+=1;s[S_TOACT]=0
            class Policy:
                def __init__(self,net):pass
                def strategy(self,x,legal):
                    assert int(x[0])!=0,'Opponent policy must not be queried'
                    return sigma[int(x[0])]
            for path in [(1,),(0,1),(0,0,0),(0,0,1)]:
                prob=.5
                for node,action in enumerate(path[1:],1):prob*=sigma[node][action]
                if prob==0:continue
                class Rng:
                    def __init__(self):self.path=iter(path)
                    def integers(self,*a):return 0
                    def choice(self,*a,**k):return next(self.path)
                class Buffer:
                    def __init__(self):self.rows=[]
                    def add(self,f,s,t,sample_weight):
                        self.rows.append((int(f[0]),s.copy(),sample_weight))
                b=Buffer()
                with patch('training.traverse_fast.NumpyPolicy',Policy),patch('training.traverse_fast._encode_fast',lambda s,*a:np.array([s[58]],np.float32)),patch('training.game.new_hand',return_value=(state(),None,None)),patch('training.game.legal_actions',return_value=np.array([1,1,0,0,0])),patch('training.game.step',step):
                    r1.collect_fixed_opponent(None,Rng(),1,None,b,t)
                for node,s,w in b.rows:
                    self.assertEqual(w,1.)
                    mass[node]+=t*prob*w;num[node]+=t*prob*w*s
        for node in (1,2):
            self.assertAlmostEqual(mass[node],.5*expected_mass[node])
            np.testing.assert_allclose(num[node],.5*expected_num[node],rtol=1e-6)
            np.testing.assert_allclose(num[node]/mass[node],expected_num[node]/expected_mass[node],rtol=1e-6)

    def test_regret_target_counterexample(self):
        x=np.zeros(51,np.float32);x[46:49]=1
        raw=np.array([2,1,-.375,0,0],np.float32)
        buf=ReservoirBuffer(1,51);buf.add(x,raw,1)
        captured=[]
        class Capture(torch.nn.MSELoss):
            def forward(self,pred,target):
                captured.append(target.detach().numpy().copy());return super().forward(pred,target)
        net=DuelingAdvantageNet(51,16,8)
        with patch('training.train.nn.MSELoss',Capture):
            train_network(net,[buf],torch.optim.Adam(net.parameters()),1,1,torch.device('cpu'))
        np.testing.assert_array_equal(captured[0][0],raw)
    def test_weight_batch_independence(self):
        r1.WEIGHT_HORIZON=4
        b=ReservoirBuffer(2,51);b.add(np.zeros(51),np.zeros(5),1);b.add(np.zeros(51),np.zeros(5),4)
        with patch('numpy.random.randint',return_value=np.array([0])):a=b.sample(1)[2]
        with patch('numpy.random.randint',return_value=np.array([0,1])):both=b.sample(2)[2]
        self.assertEqual(a[0],.25);np.testing.assert_equal(both,[.25,1])
    def test_failed_save_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'latest.pt';r1.atomic_save({'value':42},p);before=p.read_bytes()
            with patch('training.r1_support.torch.save',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):r1.atomic_save({'value':99},p)
            self.assertEqual(before,p.read_bytes());self.assertEqual(len(list(Path(d).iterdir())),1)
    def test_all_rng_roundtrip(self):
        import random
        rng=np.random.default_rng(4);state=r1.pack_rng(rng)
        def draw():return random.random(),np.random.rand(),rng.random(),float(torch.rand(()))
        a=draw();r1.restore_rng(state,rng);self.assertEqual(a,draw())
    def test_exact_average_tree_oracle(self):
        # Opponent reach differs by iteration; the same player acts twice later.
        # Enumerate every sampling path, weight its actual collector output by q.
        expected_num={1:np.zeros(5),2:np.zeros(5)};expected_mass={1:0.,2:0.}
        observed_num={1:np.zeros(5),2:np.zeros(5)};observed_mass={1:0.,2:0.}
        for iteration,p,own,deep in [(1,.9,.2,.4),(2,.1,.8,.7)]:
            sigma={0:np.array([p,1-p,0,0,0]),1:np.array([own,1-own,0,0,0]),2:np.array([deep,1-deep,0,0,0])}
            q={k:.5*s[:2]+.25 for k,s in sigma.items()}
            for node,mass in [(1,1.),(2,own)]:
                expected_mass[node]+=iteration*mass;expected_num[node]+=iteration*mass*sigma[node]
            def state(node):
                s=np.zeros(59,dtype=np.int32);s[58]=node;s[S_TOACT]=1 if node==0 else 0;return s
            def step(s,d,h,a):
                node=s[58]
                if a==1 or node==2:s[S_TERMINAL]=1
                else:s[58]+=1;s[S_TOACT]=0
            def encode(s,*args):return np.array([s[58]],dtype=np.float32)
            class Policy:
                def __init__(self,net):pass
                def strategy(self,x,legal):return sigma[int(x[0])]
            for path in [(1,),(0,1),(0,0,0),(0,0,1)]:
                class Rng:
                    def __init__(self):self.path=iter(path)
                    def integers(self,*a):return 0
                    def choice(self,*a,**k):return next(self.path)
                class Buffer:
                    def __init__(self):self.rows=[]
                    def add(self,f,s,t,sample_weight):self.rows.append((int(f[0]),s.copy(),sample_weight))
                b=Buffer();prob=1.
                for node,action in enumerate(path):prob*=q[node][action]
                with patch('training.traverse_fast.NumpyPolicy',Policy),patch('training.traverse_fast._encode_fast',encode),patch('training.game.new_hand',return_value=(state(0),None,None)),patch('training.game.legal_actions',return_value=np.array([1,1,0,0,0])),patch('training.game.step',step):
                    r1.collect_average(None,Rng(),1,None,b,iteration)
                for node,s,w in b.rows:
                    if node in observed_mass:
                        observed_mass[node]+=iteration*prob*w
                        observed_num[node]+=iteration*prob*w*s
        for node in (1,2):
            np.testing.assert_allclose(observed_num[node],expected_num[node],rtol=1e-6)
            self.assertAlmostEqual(observed_mass[node],expected_mass[node])
        # Without opponent-visit correction the first state's average would be .3091, not .6.
        self.assertAlmostEqual(observed_num[1][0]/observed_mass[1],.6,places=6)

if __name__=='__main__':unittest.main()
