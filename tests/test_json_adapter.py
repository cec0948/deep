"""Integration checks against the real pinned engine and saved smoke blueprint."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / '.cache/noregrets/target/release/poker-json'
MODEL = Path(os.environ.get('POKER_TEST_MODEL', str(ROOT / 'artifacts/noregrets-smoke/run-NK70RShx/resumed.bin')))


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BIN.is_file() or not MODEL.is_file():
            raise RuntimeError('Build the adapter and restore the smoke model first; see docs/json-api.md')
        cls.proc = subprocess.Popen([str(BIN),str(MODEL)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,text=True,bufsize=1)

    @classmethod
    def tearDownClass(cls):
        cls.proc.stdin.close()
        cls.proc.wait(timeout=10)
        cls.proc.stdout.close()
        error = cls.proc.stderr.read()
        cls.proc.stderr.close()
        if cls.proc.returncode:
            raise AssertionError(error)

    def request(self, **updates):
        r = dict(hero=3,button=0,hole=['As','Kd'],board=[],actions=[],seed=42)
        r.update(updates)
        self.proc.stdin.write(json.dumps(r)+'\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        self.assertTrue(line, self.proc.poll())
        return json.loads(line)

    def valid(self, result):
        self.assertTrue(result['ok'], result)
        self.assertFalse(result['strength_verified'])
        self.assertEqual(result['players'],6)
        self.assertAlmostEqual(sum(x['probability'] for x in result['distribution']),1.,places=5)
        for x in result['distribution']:
            self.assertGreaterEqual(x['probability'],0.)
            a = x['action']
            if a['kind']=='raise':
                lo,hi=result['raise_bounds']
                self.assertLessEqual(lo,a['to'])
                self.assertLessEqual(a['to'],hi)
            if a['kind']=='check': self.assertEqual(result['to_call'],0)
            if a['kind'] in ('call','fold'): self.assertGreater(result['to_call'],0)

    def test_preflop_and_repeatability(self):
        a=self.request()
        self.valid(a)
        self.assertEqual(a,self.request())
        self.assertEqual(a['pot'],150)
        self.assertEqual(a['to_call'],100)

    def test_all_postflop_streets(self):
        actions=[dict(seat=s,kind='call') for s in (3,4,5,0,1)]+[dict(seat=2,kind='check')]
        for n,street in ((3,'Flop'),(4,'Turn'),(5,'River')):
            with self.subTest(street=street):
                a=self.request(hero=1,board=['2c','7d','Jh','9c','Qs'][:n],actions=actions)
                self.valid(a)
                self.assertEqual(a['street'],street)
            actions=actions+[dict(seat=s,kind='check') for s in (1,2,3,4,5,0)]

    def test_off_tree_raise(self):
        self.valid(self.request(hero=4,actions=[dict(seat=3,kind='raise',to=357)]))

    def test_model_hits_and_fallback_are_distinguished(self):
        sources=set()
        for pair in itertools.combinations(['2c','3d','4h','5s','6c','7d','8h','9s','Tc','Jd','Qh','Ks','Ac'],2):
            a=self.request(hole=list(pair))
            self.valid(a)
            sources.add(a['policy_source'])
        self.assertIn('blueprint',sources)
        # The small smoke fixture has gaps here; a larger trained model can
        # legitimately cover every preflop observation in this panel.
        self.assertTrue(sources <= {'blueprint','unseen_or_zero_mass_fallback','menu_mismatch_fallback'})
        if MODEL.resolve() == (ROOT / 'artifacts/noregrets-smoke/run-NK70RShx/resumed.bin').resolve():
            self.assertIn('unseen_or_zero_mass_fallback',sources)

    def test_invalid_observations(self):
        cases=[dict(hole=['As','As']),dict(hero=6),dict(board=['2c']),
               dict(board=['2c','7d','Jh']),dict(actions=[dict(seat=0,kind='call')]),
               dict(actions=[dict(seat=3,kind='check')]),
               dict(actions=[dict(seat=3,kind='raise',to=1)]),
               dict(opponent_holes=[['Ac','Ad']]),dict(players=2)]
        for case in cases:
            with self.subTest(case=case):self.assertFalse(self.request(**case)['ok'])
        self.valid(self.request())

    def test_corrupt_json_does_not_break_process(self):
        self.proc.stdin.write('not-json\n')
        self.proc.stdin.flush()
        self.assertFalse(json.loads(self.proc.stdout.readline())['ok'])
        self.valid(self.request())

    def test_missing_model_fails_startup(self):
        p=subprocess.run([str(BIN),str(ROOT/'artifacts/no-such-model.bin')],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(p.stdout,'')


if __name__=='__main__':unittest.main()
