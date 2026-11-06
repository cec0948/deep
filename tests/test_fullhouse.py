"""Cross the real Rust replay / native Fullhouse feature / learned policy boundary."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fullhouse_runtime import Bridge, Fullhouse, validate_request

class FullhouseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bridge=Bridge();cls.policy=Fullhouse()
    @classmethod
    def tearDownClass(cls):cls.bridge.close()
    def request(self,**kw):
        r=dict(hero=3,button=0,hole=['As','Kd'],board=[],actions=[],seed=42);r.update(kw);validate_request(r);return r,self.bridge.call(r)
    def test_all_streets_and_native_parity(self):
        import numpy as np
        from bot.features import encode_state_dict
        actions=[dict(seat=s,kind='call') for s in (3,4,5,0,1)]+[dict(seat=2,kind='check')]
        for n in (3,4,5):
            r,state=self.request(hero=1,board=['2c','7d','Jh','9c','Qs'][:n],actions=actions)
            a=self.policy.decide(state,42);self.assertEqual(a,self.policy.decide(state,42))
            self.assertAlmostEqual(sum(x['probability'] for x in a['distribution']),1.)
            # Equivalent native state without street markers must recover identical features.
            native=copy.deepcopy(state);native['action_log']=[x for x in native['action_log'] if x['action']!='deal']
            np.testing.assert_allclose(encode_state_dict(state)[0],encode_state_dict(native)[0])
            np.testing.assert_allclose(self.policy.lookup.get_strategy(state),self.policy.lookup.get_strategy(native))
            actions += [dict(seat=s,kind='check') for s in (1,2,3,4,5,0)]
    def test_placeholder_independence_and_rotation(self):
        for button in range(6):
            r,s=self.request(button=button,hero=(button+3)%6)
            other=self.bridge.call(dict(r,placeholder_seed=99))
            self.assertEqual(s,other)
            self.assertEqual(s['dealer'],button)
            self.assertNotIn('hole',s['players'][0])
    def test_call_log_is_incremental(self):
        _,s=self.request(hero=2,actions=[dict(seat=i,kind='call') for i in (3,4,5,0,1)])
        self.assertEqual(s['action_log'][-1]['amount'],50)
    def test_short_raise_does_not_reopen(self):
        # Raise to 9900, caller all-in to 10000 is an incomplete raise.
        a=[dict(seat=3,kind='raise',to=9900),dict(seat=4,kind='raise',to=10000),dict(seat=5,kind='fold'),dict(seat=0,kind='fold'),dict(seat=1,kind='fold'),dict(seat=2,kind='fold')]
        _,s=self.request(actions=a)
        self.assertIsNone(s['raise_bounds'])
        reply=self.policy.decide(s,1)
        self.assertTrue(all(x['action']['kind'] in ('call','fold') for x in reply['distribution']))
    def test_bad_input_and_recovery(self):
        for change in ({'hole':['As','As']},{'actions':[dict(seat=3,kind='raise',to=1)]},{'board':['2c','3d','4h']}):
            with self.assertRaises(ValueError):self.request(**change)
        with self.assertRaises(ValueError):validate_request(dict(hero=3,button=0,hole=['As','Kd'],opponent_holes=[]))
        self.request()
    def test_jsonl_recovery(self):
        r=dict(hero=3,button=0,hole=['As','Kd'],board=[],actions=[])
        p=subprocess.run([sys.executable,str(ROOT/'scripts/fullhouse_runtime.py')],input='bad\n'+json.dumps(r)+'\n',capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr);rows=[json.loads(s) for s in p.stdout.splitlines()]
        self.assertFalse(rows[0]['ok']);self.assertTrue(rows[1]['ok'])
    def test_model_missing_is_fatal(self):
        with self.assertRaises(FileNotFoundError):Fullhouse(ROOT/'artifacts/not-a-model.npz')

if __name__=='__main__':unittest.main()
