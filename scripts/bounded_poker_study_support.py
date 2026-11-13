"""Verification and fixed-size snapshots for the authorized240-to360 study."""
import ast
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from compact_lightweight_backup import FORMAT, SEEDS, equal_state, load_state, validate_state, verify_archive
from fullhouse_v3_common import ROOT, atomic_json, digest
from train_compact_bounded import load_current


LEARNING_KEYS = ('revision','method','traversals','buffer_capacity','main_steps','batch_size','opening_steps',
    'opening_max_steps','max_opening_tv','root_repetitions_first','root_repetitions','learning_rate','checkpoint_every')


def read(path):return json.loads(Path(path).read_text())


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_frozen(files):
    for name,sha in files.items():
        if digest(name)!=sha:raise ValueError('Frozen source/model/config changed: '+name)


def check_learning_config(before,after):
    if any(before.get(k)!=after.get(k) for k in LEARNING_KEYS):
        raise ValueError('The learning method or sample budget changed')


def validate_config(cfg):
    fixed=dict(seeds=[342,343,344],initial_iteration=240,stages=[300,360],
        evaluation_iterations=[120,240,300,360],variants=['current'],
        evaluation_fields=['public','mixed','noregrets'],evaluation_deals=20000,
        evaluation_deal_seed=2026200000,promotion='none')
    if any(cfg.get(k)!=v for k,v in fixed.items()):
        raise ValueError('The newly authorized fixed-budget study design changed')
    for key,limit in [('max_seconds',14400),('training_seconds_per_stage',5400),('evaluation_seconds_per_cell',2400)]:
        if type(cfg.get(key)) is not int or not 1<=cfg[key]<=limit:raise ValueError('Invalid execution budget')


def verify_stage(campaign,stage,cfg):
    import numpy as np
    import torch
    campaign=Path(campaign);manifest=read(campaign/'source-manifest.json');check_frozen({str(campaign/k):v for k,v in manifest['files'].items()})
    rows=[]
    tree=ast.parse((campaign/'tools/compact_regret.py').read_text())
    array_map=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='ARRAY_MAP' for t in n.targets))
    for seed in cfg['seeds']:
        folder=campaign/f'seed-{seed}';status=read(folder/'status.json');state,path=load_current(campaign,seed)
        if (status.get('status')!='complete' or status.get('iteration')!=stage or status.get('through')!=stage
                or state['config']!=cfg or state['source_manifest']!=manifest or state['iteration']!=stage):
            raise ValueError('Stage did not finish with its declared state')
        validate_state(state,seed,stage)
        if status['quality']['weighted_tv']>cfg['max_opening_tv']:raise ValueError('Fitting quality failed')
        root=state['root_replay'];expected=cfg['root_repetitions_first']+(stage-1)*cfg['root_repetitions']
        if (len(root['features'])!=169 or not np.all(root['count']==expected)
                or not np.allclose(root['mass'],stage*(stage+1)/2,rtol=0,atol=1e-10)):
            raise ValueError('Restored root observations or iteration weights changed')
        policy=folder/f'current_{stage:05d}.npz'
        with np.load(policy,allow_pickle=False) as data:
            for state_key,key in array_map.items():
                for prefix,export_prefix in [('main.',''),('opening.','opening_')]:
                    if not np.array_equal(state['net'][prefix+state_key].numpy(),data[export_prefix+key]):
                        raise ValueError('Model export does not match trained state')
            for key in ('main_scales','opening_scale'):
                if not np.array_equal(state['net'][key].numpy(),data[key]) or np.any(data[key]<=0):
                    raise ValueError('Invalid trained/exported scales')
        if list(folder.glob('policy_*.npz')) or list(folder.glob('mixture_*.json')):
            raise ValueError('Historical policies must not accumulate in this run')
        rows.append(dict(seed=seed,iteration=stage,current_sha256=digest(policy),checkpoint_directory=str(path),
            checkpoint_files={n:digest(path/n) for n in ('state.json','arrays.npz','current.npz')},
            replay_rows=[b['count'] for b in state['buffers']],root_weighting_verified=True,export_parity_verified=True))
        del state
    return dict(complete=True,passed=True,stage=stage,verified_utc=utc(),seeds=rows,
                historical_mixture_available=False)


def snapshot(study,stage):
    study=Path(study).resolve();campaign=study/'campaign';cfg=read(campaign/'config.json')
    proof=read(study/f'stage-{stage}-verification.json')
    if not proof.get('passed') or proof['stage']!=stage:raise ValueError('Stage is not verified')
    exports=ROOT/'artifacts/exports';stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=exports/f'compact-v4-{study.name}-snapshot-{stamp}.tar.gz';locks=[]
    try:
        for seed in SEEDS:
            lock=(campaign/f'seed-{seed}/worker.lock').open('r');locks.append(lock);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix='.bounded-stage-',dir=exports) as temp:
            folder=Path(temp);iterations={};parity={}
            for seed in SEEDS:
                state,path=load_current(campaign,seed)
                if state['iteration']!=stage:raise ValueError('Mixed checkpoint stages')
                target=folder/f'seed-{seed}';shutil.copytree(path,target)
                if not equal_state(state,load_state(target)):raise ValueError('Snapshot state changed')
                iterations[str(seed)]=stage;parity[str(seed)]=dict(all_state_fields_bitwise_equal=True,
                    current_policy_byte_identical=True,replay_rows=[b['count'] for b in state['buffers']])
                del state
            frozen=folder/'frozen';frozen.mkdir()
            for name in ('source','tools'):
                shutil.copytree(campaign/name,frozen/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
            for name in ('config.json','source-manifest.json'):
                shutil.copyfile(campaign/name,frozen/name)
            runtime=frozen/'runtime-scripts';runtime.mkdir()
            for name,sha in read(campaign/'source-manifest.json').get('root_files',{}).items():
                if digest(name)!=sha:raise ValueError('Frozen runtime helper changed before snapshot')
                shutil.copyfile(name,runtime/Path(name).name)
            for name in ('config.json','plan.json','preparation.json',f'stage-{stage}-verification.json'):
                shutil.copyfile(study/name,folder/('study-'+name))
            shutil.copyfile(ROOT/'scripts/compact_lightweight_backup.py',folder/'restore_lightweight.py')
            (folder/'RESTORE.txt').write_text('Latest policy and fixed replay recovery. Run restore_lightweight.py restore ARCHIVE --destination NEW_EMPTY_DIRECTORY.\n'
                'load_state(DIRECTORY/seed-N) returns the network, bounded replay, root statistics and RNG.\n'
                'A separately authorized new training plan is required. This archive does not launch training, restore a historical mixture or support the old --resume worker.\n')
            files={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in folder.rglob('*') if p.is_file()}
            atomic_json(folder/'recovery-manifest.json',dict(format=FORMAT,run=study.name,iterations=iterations,
                replay_capacity_rows_per_seat=100000,latest_policy_available=True,bounded_replay_lossless=True,
                exact_legacy_resume=False,historical_mixture_available=False,parity=parity,files=files))
            with tarfile.open(archive.with_suffix('.partial'),'w:gz',compresslevel=3) as tar:
                for p in sorted(folder.rglob('*')):
                    if p.is_file():tar.add(p,arcname=str(p.relative_to(folder)),recursive=False)
            archive.with_suffix('.partial').replace(archive)
        checked=verify_archive(archive)
        result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),iterations=iterations,
            run=study.name,checkpoint_mode=FORMAT,resumable_selfplay=False,exact_legacy_resume=False,
            historical_mixture_available=False,warm_start_state_verified=True,verified_files=checked['verified_files'],
            external_backup=False,verification=checked,parity=parity)
        atomic_json(archive.with_suffix('.json'),result);atomic_json(study/f'stage-{stage}-backup.json',result)
        return result
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--stage',type=int,required=True);args=parser.parse_args();print(snapshot(args.study,args.stage))
