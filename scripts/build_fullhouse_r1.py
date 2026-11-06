"""Materialize an auditable patched CPU reference from immutable upstream."""
import difflib
from pathlib import Path
import shutil
from fullhouse_runtime import ROOT,SOURCE,validate_source

def build():
    validate_source();dest=ROOT/'.cache/fullhouse-r1';dest.mkdir(exist_ok=True)
    for folder in ('training','bot'):
        shutil.copytree(SOURCE/folder,dest/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
    if not (dest/'data').exists():(dest/'data').symlink_to(SOURCE/'data',target_is_directory=True)
    shutil.copyfile(SOURCE/'LICENSE',dest/'LICENSE.upstream')
    original={name:(SOURCE/'training'/name).read_text() for name in ('train.py','buffer.py')}
    train=original['train.py']
    def replace(old,new):
        nonlocal train
        if train.count(old)!=1:raise RuntimeError('Patch anchor changed: '+old[:70])
        train=train.replace(old,new)
    replace('import argparse','import argparse\nimport random\nfrom . import r1_support as r1\nSTOP_AFTER = None\nAVERAGE_SAMPLER = "importance"')
    start=train.index('        # Per-sample advantage normalization')
    end=train.index('        with torch.amp.autocast(',start)
    train=train[:start]+'        # Regrets are already in fixed BB units; do not center targets.\n        adv_t = adv_t * legal_mask\n\n'+train[end:]
    replace('    _init_numba()','''    if n_workers != 1 or device.type != 'cpu':
        raise ValueError('R1 reference supports CPU workers=1 only')
    if opponent_exploration_start != 0 or opponent_exploration_end != 0 or randomize_stacks:
        raise ValueError('R1 reference requires epsilon=0 and fixed 100BB')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r1.WEIGHT_HORIZON=n_iterations
    r1.AVERAGE_SAMPLER=AVERAGE_SAMPLER
    run_config=dict(revision=r1.REVISION,average_sampler=AVERAGE_SAMPLER,horizon=n_iterations,traversals=n_traversals,
        buffer_capacity=buffer_capacity,strategy_buffer_capacity=strategy_buffer_capacity,
        train_steps=train_steps,strategy_train_steps=strategy_train_steps,batch_size=batch_size,
        lr=lr,seed=seed,checkpoint_every=checkpoint_every)
    if not save_resume_buffers:raise ValueError('R1 checkpoints must include reservoirs')
''')
    replace('        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)', '''        checkpoint = torch.load(resume_file, map_location="cpu", weights_only=False)
        if checkpoint.get('r1_config') != run_config:raise ValueError('Checkpoint config/revision mismatch')''')
    replace('''        print(
            f"[Deep CFR] --resume requested but {resume_file} does not exist; starting from scratch"
        )''','''        raise FileNotFoundError(resume_file)''')
    replace('    avg_loss = 0.0', '''    if resume:
        r1.restore_rng(checkpoint['r1_rng'],rng)
    avg_loss = 0.0''')
    replace('    for iteration in range(resume_iter + 1, n_iterations + 1):','    end_iteration=min(n_iterations,STOP_AFTER or n_iterations)\n    for iteration in range(resume_iter + 1, end_iteration + 1):')
    replace('                strategy_buffer=strategy_buffer,','                strategy_buffer=None,')
    replace('        total_buf_size = sum(len(b) for b in buffers)', '''        collection_stats=r1.collect_average(None,rng,n_traversals,net,strategy_buffer,iteration)
        n_strategy_samples=len(strategy_buffer)-strategy_size_before
        total_buf_size = sum(len(b) for b in buffers)''')
    replace('        iter_time = time.time() - iter_start', """        r1.log_diagnostics(ckpt_path/'diagnostics.jsonl',iteration,collection_stats,strategy_buffer,avg_loss,avg_strategy_loss)
        iter_time = time.time() - iter_start""")
    replace('        if iteration % checkpoint_every == 0 or iteration == n_iterations:', '        if iteration % checkpoint_every == 0 or iteration == end_iteration:')
    replace('            ckpt_data = {', '''            ckpt_data = {
                'r1_config':run_config,'r1_rng':r1.pack_rng(rng),''')
    train=train.replace('torch.save(ckpt_data, save_path)','r1.atomic_save(ckpt_data, save_path)').replace('torch.save(latest_data, resume_file)','r1.atomic_save(latest_data, resume_file)')
    replace('    args = parser.parse_args()', '''    parser.add_argument('--stop-after',type=int)
    parser.add_argument('--average-sampler',choices=['importance','fixed-opponent'],default='importance')
    args = parser.parse_args()
    global STOP_AFTER,AVERAGE_SAMPLER
    STOP_AFTER=args.stop_after
    AVERAGE_SAMPLER=args.average_sampler''')
    buf=original['buffer.py'].replace('import torch','import torch\nfrom . import r1_support as r1')
    buf=buf.replace('        max_w = weights.max().clip(min=1.0)\n        weights = (weights / max_w) ** 1.5','        weights = r1.time_weights(weights)')
    buf=buf.replace('        max_w = weights.max().clamp(min=1.0)\n        weights = (weights / max_w) ** 1.5','        weights = r1.time_weights(weights)')
    patches=[]
    for name,new in [('train.py',train),('buffer.py',buf)]:
        (dest/'training'/name).write_text(new)
        patches.extend(difflib.unified_diff(original[name].splitlines(True),new.splitlines(True),fromfile='a/training/'+name,tofile='b/training/'+name))
    (ROOT/'patches').mkdir(exist_ok=True)
    (ROOT/'patches/fullhouse-r1.patch').write_text(''.join(patches))
    shutil.copyfile(ROOT/'native/fullhouse-r1/support.py',dest/'training/r1_support.py')
    print(dest)
    return dest

if __name__=='__main__':build()
