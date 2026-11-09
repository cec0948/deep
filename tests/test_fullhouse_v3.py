"""V3 runs in subprocesses so legacy and revised bot modules cannot collide."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class FullhouseV3Tests(unittest.TestCase):
    def run_probe(self,code):
        result=subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'scripts');"+code],cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),capture_output=True,text=True,timeout=180)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        return result.stdout

    def test_rules_counterexamples_and_terminal_payoffs(self):
        self.run_probe('from verify_fullhouse_v3 import rules; print(rules(100))')

    def test_conditional_average_and_sampling_rng(self):
        self.run_probe('from verify_fullhouse_v3 import sampling; print(sampling())')

    def test_layered_pots_and_single_live_actor_runout(self):
        # An isolated settlement fixture, not support for variable-stack training.
        self.run_probe('''
from verify_fullhouse_v3 import g,np
s=np.zeros(g.STATE_SIZE,np.int32);s[g.S_NPLAYERS]=6
initial=np.array([1000,5000,10000,10000,10000,10000],np.int32)
invested=np.array([1000,5000,5000,0,0,0],np.int32)
s[g.S_INITIAL_STACKS:g.S_INITIAL_STACKS+6]=initial
s[g.S_INVESTED:g.S_INVESTED+6]=invested
s[g.S_STACKS:g.S_STACKS+6]=initial-invested
s[g.S_FOLDED:g.S_FOLDED+6]=[0,0,0,1,1,1]
s[g.S_ALLIN:g.S_ALLIN+6]=[1,1,0,0,0,0]
s[g.S_POT]=11000
prefix=[12,25,11,24,44,13,0,2,3,4,5,6,1,15,32,46,8]
deck=np.array(prefix+[c for c in range(52) if c not in prefix],np.int8)
hole=deck[:12].reshape(6,2).copy()
g._advance_street(s,deck,hole)
assert s[g.S_TERMINAL] and s[g.S_STREET]==3
np.testing.assert_array_equal(s[g.S_PAYOFFS:g.S_PAYOFFS+6],[2000,3000,-5000,0,0,0])
''')

    def test_exact_resume_and_legacy_rejection(self):
        self.run_probe('from verify_fullhouse_v3 import resume; print(resume())')

    def test_runtime_encoder_and_export_do_not_change_legacy_policy(self):
        self.run_probe('''
from pathlib import Path
import numpy as np,torch
from fullhouse_runtime import Fullhouse,Bridge
from fullhouse_v3_runtime import FullhouseV3
torch.set_num_threads(1)
old=Fullhouse();engine=Bridge()
try:
 state=engine.call(dict(op='replay',hero=3,button=0,hole=['Ac','Ad'],board=[],actions=[]))
 before=old.decide(state,81)
 new=FullhouseV3('artifacts/fullhouse-v3-verification/smoke-iter12.npz')
 assert old.decide(state,81)==before
 import bot.features
 assert '/fullhouse-bot/' in str(Path(bot.features.__file__).resolve())
 sys.path.insert(0,'.cache/fullhouse-v3')
 from training.network import AverageStrategyNet
 ck=torch.load('artifacts/fullhouse-v3-verification/smoke-iter12.pt',weights_only=False,map_location='cpu')
 net=AverageStrategyNet();net.load_state_dict(ck['strategy_net']);net.eval()
 x,legal=new.features.encode_state_dict(state,new.lookup._equity_tables)
 expected=net.get_strategy(torch.from_numpy(x[None]),torch.from_numpy(legal.astype(np.float32)[None])).numpy()[0]
 np.testing.assert_allclose(new.lookup.get_strategy(state),expected,rtol=1e-5,atol=1e-6)
 assert np.isclose(sum(v['probability'] for v in new.decide(state,81)['distribution']),1.)
finally:engine.close()
''')


if __name__=='__main__':unittest.main()
