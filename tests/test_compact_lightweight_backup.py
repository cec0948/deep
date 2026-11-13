import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compact_lightweight_backup import (FORMAT, SEEDS, equal_state, load_state,
    save_state, sha256, validate_state, verify_archive)


def fixture(seed):
    features = np.zeros((2, 51), np.float32); features[:, 46:48] = 1
    return dict(seed=seed, iteration=240, config={'buffer_capacity': 100000},
        net={'weight':torch.tensor([[1., -2.]], dtype=torch.float32)},
        buffers=[dict(features=features.copy(), advantages=np.zeros((2, 5), np.float32),
                      iterations=np.array([120., 240.], np.float32), count=2, total_seen=999999) for _ in range(6)],
        root_replay={'features':features[:1].copy(),'numerator':np.zeros((1,5),np.float64),
            'second_numerator':np.ones((1,5),np.float64),'mass':np.array([7.]),
            'squared_mass':np.array([49.]),'count':np.array([7],np.int64)},
        rng={'python':random.Random(123).getstate(),
             'numpy':np.random.RandomState(123).get_state(),
             'generator':np.random.default_rng(123).bit_generator.state,
             'torch':torch.Generator().manual_seed(123).get_state()})


class LightweightBackupTests(unittest.TestCase):
    def test_lossless_dtype_rng_tensor_and_replay_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'seed'; original=fixture(342); save_state(original,d)
            restored=load_state(d)
            self.assertTrue(equal_state(original,restored))
            self.assertEqual(validate_state(restored,342,240),[2]*6)
            self.assertFalse((d/'latest.pt').exists())

    def test_nonfinite_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=fixture(342);s['buffers'][0]['features'][0,0]=np.nan
            with self.assertRaises(ValueError):save_state(s,Path(tmp)/'seed')

    def test_illegal_replay_and_excess_capacity_are_rejected(self):
        s=fixture(342);s['buffers'][0]['advantages'][0,4]=1
        with self.assertRaises(ValueError):validate_state(s,342,240)
        s=fixture(342);s['config']['buffer_capacity']=100001
        with self.assertRaises(ValueError):validate_state(s,342,240)

    def test_missing_npz_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'seed';save_state(fixture(342),d);np.savez_compressed(d/'arrays.npz')
            with self.assertRaises(ValueError):load_state(d)

    def test_later_iteration_does_not_require_history_or_larger_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            sizes=[]
            for iteration in [240,480]:
                state=fixture(342);state['iteration']=iteration;d=Path(tmp)/str(iteration)
                save_state(state,d);restored=load_state(d)
                self.assertEqual(validate_state(restored,342,iteration),[2]*6)
                self.assertTrue(equal_state(state,restored))
                sizes.append((d/'arrays.npz').stat().st_size)
            self.assertEqual(sizes[0],sizes[1])

    def make_archive(self,root,extra=None,tamper=False):
        stage=root/'stage';stage.mkdir()
        for seed in SEEDS:save_state(fixture(seed),stage/f'seed-{seed}')
        files={str(p.relative_to(stage)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in stage.rglob('*') if p.is_file()}
        manifest=dict(format=FORMAT,exact_legacy_resume=False,historical_mixture_available=False,
                      iterations={str(s):240 for s in SEEDS},files=files)
        (stage/'recovery-manifest.json').write_text(json.dumps(manifest))
        if tamper:(stage/'seed-342/state.json').write_text('{}')
        archive=root/'test.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for p in stage.rglob('*'):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(stage)))
            if extra:
                entry=tarfile.TarInfo(extra);entry.size=1;tar.addfile(entry,io.BytesIO(b'x'))
        return archive

    def test_archive_restore_checks_all_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=self.make_archive(root);out=root/'restore'
            self.assertTrue(verify_archive(archive,out)['passed'])
            self.assertTrue(equal_state(fixture(344),load_state(out/'seed-344')))
            with self.assertRaises(FileExistsError):verify_archive(archive,out)

    def test_tampered_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive=self.make_archive(Path(tmp),tamper=True)
            with self.assertRaises(ValueError):verify_archive(archive)

    def test_extra_and_traversal_members_are_rejected(self):
        for extra in ['extra', '../escape']:
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as tmp:
                archive=self.make_archive(Path(tmp),extra=extra)
                with self.assertRaises(ValueError):verify_archive(archive)


if __name__=='__main__':unittest.main()
