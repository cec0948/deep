"""Local, lossless rich-family recovery. Never contacts Drive or starts training.

Only the selected immutable checkpoint is packed for each seed. Restore keeps
all numerical state and RNG bytes; optional relocation changes configuration
paths only, then regenerates the atomic checkpoint pointers.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np

from compact_lightweight_backup import equal_state, load_state, save_state, safe_member
from fullhouse_v3_common import ROOT, atomic_json, digest
from rich_regret_model import RichRegretModel
from train_rich_selfplay import validate_checkpoint


FORMAT = 'rich-residual-bounded-replay-v1'
FAMILY = 'rich-regret-residual'
SEEDS = (342, 343, 344)
FILES = ('state.json', 'arrays.npz', 'current.npz')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def teacher_path(campaign, cfg, seed):
    value = Path(cfg['teachers'][str(seed)]['path'])
    return value if value.is_absolute() else Path(campaign)/value


def _source(campaign, cfg):
    path = Path(cfg['source'])
    return path if path.is_absolute() else Path(campaign)/path


def _npz_equal(first, second):
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(
            a[n].dtype == b[n].dtype and np.array_equal(a[n], b[n]) for n in a.files)


def _verify_model(state, checkpoint, teacher, source, cfg, seed):
    dimension = state['model']['feature_dim']
    validate_checkpoint(state, cfg, seed, dimension)
    if (state.get('family') != FAMILY or state.get('average_policy_available') is not False
            or state.get('teacher_knowledge_inherited') is not True
            or dimension != cfg.get('feature_dim', dimension)):
        raise ValueError('Rich recovery family/capability mismatch')
    if digest(teacher) != cfg['teachers'][str(seed)]['sha256']:
        raise ValueError('Frozen teacher hash mismatch')
    model = RichRegretModel(teacher, dimension, source, width=state['model']['width'])
    model.load_state_dict(state['model'])
    with tempfile.TemporaryDirectory(prefix='rich-export-check-') as tmp:
        exported = Path(tmp)/'current.npz'; model.export(exported)
        if not _npz_equal(checkpoint/'current.npz', exported):
            raise ValueError('Rich checkpoint export differs from trained parameters')
    return dict(seed=seed, iteration=state['iteration'], feature_dim=dimension,
                teacher_sha256=digest(teacher), export_arrays_equal=True,
                replay_rows=[int(b['count']) for b in state['buffers']],
                all_rng_states_present=set(state['rng']) == {'python','numpy','generator','torch'})


def verify_stage(campaign, stage, cfg=None):
    """Verify a declared immutable stage even if later workers are running."""
    campaign = Path(campaign).resolve(); cfg = cfg or read(campaign/'config.json')
    if type(stage) is not int or stage not in cfg['stages']:
        raise ValueError('Snapshot must identify a declared rich-family stage')
    if set(cfg['seeds']) != set(SEEDS):
        raise ValueError('All three training seeds are required')
    for name, expected in cfg['frozen_files'].items():
        if digest(name) != expected:
            raise ValueError('Frozen input changed before rich snapshot: '+name)
    proof = []
    for seed in SEEDS:
        checkpoint = campaign/f'seed-{seed}/checkpoints/iter-{stage:05d}'
        if checkpoint.is_symlink() or not checkpoint.is_dir() or any(not (checkpoint/n).is_file() for n in FILES):
            raise ValueError('Incomplete immutable rich checkpoint')
        before = {n:digest(checkpoint/n) for n in FILES}
        state = load_state(checkpoint)
        if state['iteration'] != stage:
            raise ValueError('Mixed rich-family stages')
        warm = read(campaign/f'seed-{seed}/warm-start-verification.json')
        if (warm.get('passed') is not True or warm.get('teacher_probability_byte_identical') is not True
                or warm.get('teacher_sha256') != cfg['teachers'][str(seed)]['sha256']):
            raise ValueError('Initial teacher behavior parity is not verified')
        row = _verify_model(state, checkpoint, teacher_path(campaign,cfg,seed),
                            _source(campaign,cfg), cfg, seed)
        row.update(checkpoint_directory=str(checkpoint), checkpoint_files=before,
                   warm_start_state_verified=True)
        if before != {n:digest(checkpoint/n) for n in FILES}:
            raise ValueError('Rich immutable checkpoint changed during verification')
        proof.append(row); del state
    return dict(passed=True, complete=True, verified_utc=utc(), training_family=FAMILY,
                family_iteration=stage, parent_iteration=360, seeds=proof,
                average_policy_available=False, exact_legacy_resume=False)


def _copy_file(source, destination):
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Recovery input must be a regular file: '+str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _manifest_files(folder):
    return {str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=digest(p))
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'recovery-manifest.json'}


def snapshot(study, stage, output_root=None):
    study = Path(study).resolve(); campaign = study/'campaign'; cfg = read(campaign/'config.json')
    proof = verify_stage(campaign, stage, cfg)
    verification = study/f'stage-{stage}-verification.json'
    if verification.exists():
        supplied = read(verification)
        if supplied.get('passed') is not True or supplied.get('family_iteration',supplied.get('stage')) != stage:
            raise ValueError('Declared stage proof does not match snapshot')
    else:
        atomic_json(verification, proof)
    output_root = Path(output_root or ROOT/'artifacts/exports').resolve()
    output_root.mkdir(parents=True,exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output_root/f'rich-selfplay-{study.name}-stage-{stage:05d}-{stamp}.tar.gz'
    with tempfile.TemporaryDirectory(prefix='.rich-recovery-',dir=output_root) as temp:
        folder=Path(temp); pathmap={}; states={}; checksums={}
        _copy_file(campaign/'config.json',folder/'campaign/config.json')
        source=_source(campaign,cfg)
        for p in sorted(source.rglob('*')):
            if (not p.is_file() or '__pycache__' in p.parts
                    or p.suffix in ('.pyc','.pyo','.nbi','.nbc')):
                continue
            relative=Path('campaign/source')/p.relative_to(source)
            _copy_file(p,folder/relative);pathmap[str(p.resolve())]=str(relative)
        pathmap[str(source.resolve())]='campaign/source'
        for seed in SEEDS:
            teacher=teacher_path(campaign,cfg,seed); relative=Path(f'campaign/teachers/seed-{seed}.npz')
            _copy_file(teacher,folder/relative);pathmap[str(teacher.resolve())]=str(relative)
            src=campaign/f'seed-{seed}/checkpoints/iter-{stage:05d}'
            target=folder/f'campaign/seed-{seed}/checkpoints/iter-{stage:05d}'
            for name in FILES:_copy_file(src/name,target/name)
            state=load_state(src);restored=load_state(target)
            if not equal_state(state,restored):raise ValueError('Rich snapshot numerical state changed')
            states[str(seed)]=dict(all_state_fields_bitwise_equal=True,current_export_byte_identical=True,
                replay_rows=[int(b['count']) for b in state['buffers']])
            checksums[str(seed)]={name:digest(src/name) for name in FILES}
            atomic_json(folder/f'campaign/seed-{seed}/current.json',dict(format=state['format'],
                iteration=stage,directory=target.name,files=checksums[str(seed)],
                teacher_sha256=cfg['teachers'][str(seed)]['sha256'],parent_iteration=360,
                average_policy_available=False))
            _copy_file(campaign/f'seed-{seed}/warm-start-verification.json',
                       folder/f'campaign/seed-{seed}/warm-start-verification.json')
            del state,restored
        # Copy every explicitly frozen dependency. Identical filenames from
        # different directories remain distinct via their path digest.
        for original,expected in cfg['frozen_files'].items():
            p=Path(original).resolve()
            if digest(p)!=expected:raise ValueError('Frozen rich dependency changed')
            if str(p) not in pathmap:
                relative=Path('frozen-inputs')/(hashlib.sha256(str(p).encode()).hexdigest()[:16]+'-'+p.name)
                _copy_file(p,folder/relative);pathmap[str(p)]=str(relative)
        runtime=folder/'runtime-scripts'
        required=['snapshot_rich_selfplay.py','compact_lightweight_backup.py','fullhouse_v3_common.py',
                  'rich_regret_model.py','train_rich_selfplay.py','rich_poker_observation.py',
                  'rich_poker_traversal.py','poker_public_state.py','regret_partition.py']
        for name in required:
            local=Path(__file__).resolve().parent/name
            _copy_file(local,runtime/name)
            pathmap[str(local)]=str(Path('runtime-scripts')/name)
        for name in ['config.json','plan.json',f'stage-{stage}-verification.json']:
            if (study/name).is_file():_copy_file(study/name,folder/'study-metadata'/name)
        (folder/'RESTORE.txt').write_text(
            'This archive starts no training. Install the pinned Python/numpy/torch dependencies.\n'
            'Verify/extract: python runtime-scripts/snapshot_rich_selfplay.py restore ARCHIVE --destination NEW_EMPTY_DIRECTORY\n'
            'Add --relocate to rewrite only path metadata and regenerate checkpoint pointers for the new directory.\n'
            'New-worker resume uses restored/campaign. Never pass it to the old compact worker.\n'
            'Three teacher models, latest residual states, bounded replay and all four RNG states are included.\n')
        manifest=dict(format=FORMAT,training_family=FAMILY,run=study.name,
            family_iterations={str(s):stage for s in SEEDS},parent_iteration=360,
            resumable_selfplay=True,resumable_worker_family=FAMILY,exact_legacy_resume=False,
            historical_mixture_available=False,average_policy_available=False,
            warm_start_state_verified=True,replay_capacity_rows_per_seat=cfg['buffer_capacity'],
            numeric_parity=states,original_to_member=pathmap,files=_manifest_files(folder))
        atomic_json(folder/'recovery-manifest.json',manifest)
        with tarfile.open(archive.with_suffix('.partial'),'w:gz',compresslevel=3) as tar:
            for p in sorted(folder.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(folder)),recursive=False)
        archive.with_suffix('.partial').replace(archive)
    checked=verify_archive(archive)
    for seed in SEEDS:
        src=campaign/f'seed-{seed}/checkpoints/iter-{stage:05d}'
        if checksums[str(seed)]!={name:digest(src/name) for name in FILES}:
            raise ValueError('Source immutable rich stage changed while packing')
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),
        kind='training-checkpoint',checkpoint_mode=FORMAT,training_family=FAMILY,run=study.name,
        family_iterations={str(s):stage for s in SEEDS},parent_iteration=360,
        training_stage_completed=True,resumable_selfplay=True,resumable_worker_family=FAMILY,
        exact_legacy_resume=False,warm_start_state_verified=True,historical_mixture_available=False,
        average_policy_available=False,replay_capacity_rows_per_seat=cfg['buffer_capacity'],
        verified_files=checked['verified_files'],restoration_verified=True,verification=checked,
        external_backup=False,remote_checksum_available=False,remote_sha256_verified=False)
    atomic_json(study/f'stage-{stage}-backup.json',result)
    atomic_json(archive.with_suffix('.json'),result)
    return result


def _relocate(folder,manifest):
    campaign=folder/'campaign';original=read(campaign/'config.json')
    paths={name:str((folder/member).resolve()) for name,member in manifest['original_to_member'].items()}
    def replace(value):
        if isinstance(value,dict):return {paths.get(k,k):replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        return paths.get(value,value) if isinstance(value,str) else value
    relocated=replace(original)
    relocated['source']=str((campaign/'source').resolve())
    for seed in SEEDS:relocated['teachers'][str(seed)]['path']=str((campaign/f'teachers/seed-{seed}.npz').resolve())
    for name,sha in relocated['frozen_files'].items():
        if digest(name)!=sha:raise ValueError('Relocated frozen input differs')
    atomic_json(campaign/'config.original.json',original)
    for seed in SEEDS:
        iteration=manifest['family_iterations'][str(seed)];path=campaign/f'seed-{seed}/checkpoints/iter-{iteration:05d}'
        before=load_state(path);after=dict(before);after['config']=relocated
        # Remove exactly the metadata field before the numerical/RNG comparison.
        if not equal_state({k:v for k,v in before.items() if k!='config'},
                           {k:v for k,v in after.items() if k!='config'}):
            raise ValueError('Relocation changed numerical rich state')
        with tempfile.TemporaryDirectory(prefix='.relocate-',dir=path.parent) as temp:
            encoded=Path(temp)/'state';save_state(after,encoded)
            for name in ['state.json','arrays.npz']:shutil.copyfile(encoded/name,path/name)
        pointer=read(campaign/f'seed-{seed}/current.json');pointer['files']={n:digest(path/n) for n in FILES}
        atomic_json(campaign/f'seed-{seed}/current.json',pointer)
        decoded=load_state(path)
        if not equal_state({k:v for k,v in before.items() if k!='config'},
                           {k:v for k,v in decoded.items() if k!='config'}):
            raise ValueError('Relocated encoding changed numerical/RNG state')
        validate_checkpoint(decoded,relocated,seed,after['model']['feature_dim'])
    atomic_json(campaign/'config.json',relocated)
    atomic_json(folder/'relocation.json',dict(relocated_utc=utc(),path_metadata_only=True,
        original_config_preserved=True,all_numerical_and_rng_state_unchanged=True,
        resume_campaign=str(campaign),automatic_training_started=False))


def verify_archive(archive,destination=None,relocate=False):
    archive=Path(archive)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if (len(names)!=len(set(names)) or len(names)>10000 or any(not m.isfile() or not safe_member(m.name) for m in members)
                or sum(m.size for m in members)>8*1024**3 or names.count('recovery-manifest.json')!=1):
            raise ValueError('Unsafe or duplicate rich recovery archive members')
        manifest=json.load(tar.extractfile('recovery-manifest.json'))
        if (manifest.get('format')!=FORMAT or manifest.get('training_family')!=FAMILY
                or manifest.get('resumable_selfplay') is not True
                or manifest.get('exact_legacy_resume') is not False
                or manifest.get('historical_mixture_available') is not False
                or set(manifest.get('family_iterations',{}))!=set(map(str,SEEDS))
                or set(names)!=set(manifest['files'])|{'recovery-manifest.json'}):
            raise ValueError('Rich recovery manifest coverage/capabilities mismatch')
        def extract(folder):
            for member in members:
                target=folder/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,target.open('wb') as out:shutil.copyfileobj(source,out)
                if member.name!='recovery-manifest.json':
                    row=manifest['files'][member.name]
                    if target.stat().st_size!=row['bytes'] or digest(target)!=row['sha256']:
                        raise ValueError('Rich recovery file size/hash mismatch')
            cfg=read(folder/'campaign/config.json');rows=[]
            for seed in SEEDS:
                iteration=manifest['family_iterations'][str(seed)];checkpoint=folder/f'campaign/seed-{seed}/checkpoints/iter-{iteration:05d}'
                state=load_state(checkpoint)
                if state['iteration']!=iteration:raise ValueError('Recovered rich stage mismatch')
                rows.append(_verify_model(state,checkpoint,folder/f'campaign/teachers/seed-{seed}.npz',folder/'campaign/source',cfg,seed))
                del state
            if relocate:_relocate(folder,manifest)
            return dict(passed=True,verified_files=len(manifest['files']),seeds=rows,
                numerical_state_lossless=True,relocation_performed=bool(relocate),
                resumable_worker_family=FAMILY,exact_legacy_resume=False,automatic_training_started=False)
        if destination is None:
            with tempfile.TemporaryDirectory(prefix='rich-recovery-verify-') as temp:return extract(Path(temp))
        target=Path(destination);target.mkdir(parents=True,exist_ok=False)
        return extract(target)


def prepare_upload(archive,*,max_file_bytes=100*1024**2,part_bytes=64*1024**2):
    """Prepare bounded local pieces; remote IDs/verification are added later.

    This is deliberately not a completed backup record. A split restore
    manifest cannot be called complete until all uploaded IDs, sizes and sole
    owner permissions have been read back by the separate Drive executor.
    """
    archive=Path(archive).resolve()
    if (type(max_file_bytes) is not int or not 0<max_file_bytes<=100*1024**2
            or type(part_bytes) is not int or not 0<part_bytes<=min(max_file_bytes,64*1024**2)):
        raise ValueError('Upload files must be at most100MiB and pieces at most64MiB')
    size=archive.stat().st_size;sha=digest(archive)
    if size<=max_file_bytes:
        return dict(format='single-archive-v1',archive=str(archive),bytes=size,sha256=sha,
                    files=[dict(path=str(archive),name=archive.name,bytes=size,sha256=sha)],
                    external_backup=False,remote_verification_complete=False)
    parts=[]
    with archive.open('rb') as source:
        number=1
        while True:
            chunk=source.read(part_bytes)
            if not chunk:break
            target=archive.with_name(archive.name+f'.part-{number:03d}')
            if target.exists():raise FileExistsError('Upload preparation does not overwrite existing/partial pieces')
            with target.open('xb') as output:output.write(chunk)
            parts.append(dict(order=number,path=str(target),name=target.name,
                bytes=len(chunk),sha256=digest(target),file_id=None,
                verified_remote_size=False,verified_owner_only=False))
            number+=1
    joined=hashlib.sha256()
    for part in parts:
        with Path(part['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):joined.update(chunk)
    if sum(p['bytes'] for p in parts)!=size or joined.hexdigest()!=sha or digest(archive)!=sha:
        raise ValueError('Original and recombined local archive do not match')
    manifest=archive.with_name(archive.name[:-7]+'.restore-manifest.json')
    if manifest.exists():raise FileExistsError('An existing restore manifest must not be overwritten')
    result=dict(format='split-rich-recovery-v1',checkpoint_mode=FORMAT,archive_name=archive.name,
        archive=str(archive),bytes=size,sha256=sha,recombined_local_sha256=joined.hexdigest(),
        part_limit_bytes=part_bytes,parts=parts,manifest_path=str(manifest),
        external_backup=False,remote_verification_complete=False,
        restoration='Download pieces in order, verify each local SHA-256, concatenate, verify archive SHA-256, then restore the inner recovery manifest')
    atomic_json(manifest,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path);parser.add_argument('--stage',type=int)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('operation',nargs='?',choices=['verify','restore'])
    parser.add_argument('archive',nargs='?',type=Path)
    parser.add_argument('--destination',type=Path);parser.add_argument('--relocate',action='store_true')
    args=parser.parse_args()
    if args.operation:
        if args.archive is None or (args.operation=='restore' and args.destination is None):parser.error('Archive and restore destination required')
        result=verify_archive(args.archive,args.destination,args.relocate)
    else:
        if args.study is None or args.stage is None:parser.error('--study and --stage required')
        result=snapshot(args.study,args.stage,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False))
