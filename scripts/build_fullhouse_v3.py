"""Build the fixed-100BB CPU revision without changing historical runs."""
import difflib
from pathlib import Path
import shutil

from build_fullhouse_r1 import build as build_r1
from fullhouse_runtime import ROOT


def replace_once(text,old,new):
    if text.count(old)!=1:
        raise RuntimeError('V3 patch anchor changed: '+old[:80])
    return text.replace(old,new)


def build():
    base=build_r1();dest=ROOT/'.cache/fullhouse-v3';dest.mkdir(exist_ok=True)
    for folder in ('training','bot'):
        shutil.copytree(base/folder,dest/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
    if not (dest/'data').exists():(dest/'data').symlink_to((base/'data').resolve(),target_is_directory=True)
    shutil.copyfile(base/'LICENSE.upstream',dest/'LICENSE.upstream')
    changes={}
    game=(base/'training/game.py').read_text()
    game=replace_once(game,'    deck = rng.permutation(52).astype(np.int8)', '''    if n_players != 6 or (starting_stacks is not None and not np.all(np.asarray(starting_stacks) == STARTING_STACK)):
        raise ValueError('V3 rules contract: six players, equal 100BB starting stacks')
    deck = rng.permutation(52).astype(np.int8)''')
    game=replace_once(game,'    can_raise = my_stack > to_call', '    can_raise = my_stack > to_call and bool(state[S_PENDING + seat])')
    game=replace_once(game,'        raise_size = max(0, state[S_CURBET] - prev_bet)', '''        raise_size = max(0, state[S_CURBET] - prev_bet)
        if raise_size > 0:
            state[S_LASTRAISER] = seat''')
    game=replace_once(game,'            state[S_PENDING + cand] == 1', '            (state[S_PENDING + cand] == 1 or state[S_BETS + cand] < state[S_CURBET])')
    game=replace_once(game,'    if actors == 0:', '    if actors <= 1:')
    game=replace_once(game,'            awarded[w] += split + (rem if idx == 0 else 0)', '            awarded[w] += split + int(idx < rem)')
    # Reusing immutable eval7 cards changes neither randomness nor hand ranking.
    start=game.index('    _CARDS = [');end=game.index('\n\n    board = ',start)
    game=game[:start]+'    from bot.features import _eval7_cards\n    _CARDS = _eval7_cards()'+game[end:]
    changes['training/game.py']=game

    features=(base/'bot/features.py').read_text()
    start=features.index('def _replay_betting_rounds(')
    end=features.index('\ndef _evaluate_hand_strength(',start)
    # One explicit parser supplies contributions and aggression to both paths.
    parser=(ROOT/'native/fullhouse-v3/betting_features.py').read_text()
    features=features[:start]+parser+'\n'+features[end:]
    features=replace_once(features,'    can_raise = stack > owed and min_raise > 0', '    can_raise = stack > owed and min_raise is not None and min_raise > 0')
    changes['bot/features.py']=features

    support=(base/'training/r1_support.py').read_text()+'\n'+(ROOT/'native/fullhouse-v3/strategy_sampling.py').read_text()
    changes['training/r1_support.py']=support
    train=(base/'training/train.py').read_text()
    train=replace_once(train,'revision=r1.REVISION,average_sampler=AVERAGE_SAMPLER,', 'revision=r1.REVISION,strategy_optimizer=r1.STRATEGY_OPTIMIZER,strategy_batch_size=r1.STRATEGY_BATCH_SIZE,average_sampler=AVERAGE_SAMPLER,')
    train=replace_once(train,'        features, targets, weights = strategy_buffer.sample(batch_size)', '        features, targets, weights = sampler.sample(r1.STRATEGY_BATCH_SIZE)')
    anchor='''    ms = sorted(m for m in (milestones or []) if 0 < m <= n_steps)'''
    train=replace_once(train,anchor,'    sampler = r1.StrategySampler(strategy_buffer)\n'+anchor)
    start=train.index('def train_strategy_network(');end=train.index('\ndef train_deep_cfr(',start)
    strategy=train[start:end]
    strategy=replace_once(strategy,'    total_loss = 0.0','    total_loss = 0.0\n    averaged = {}\n    averaged_count = 0')
    strategy=replace_once(strategy,'        optimizer.step()', '''        optimizer.step()
        if step_i >= n_steps // 2:
            averaged_count += 1
            with torch.no_grad():
                for name,parameter in net.named_parameters():
                    if averaged_count == 1:
                        averaged[name] = parameter.detach().clone()
                    else:
                        averaged[name].add_((parameter.detach()-averaged[name])/averaged_count)''')
    strategy=replace_once(strategy,'    net.eval()', '''    with torch.no_grad():
        for name,parameter in net.named_parameters():
            parameter.copy_(averaged[name])
    net.eval()''')
    train=train[:start]+strategy+train[end:]
    for name in ('strategy_w0','trunk_w0'):
        train=replace_once(train,f'        "{name}":',f'        "training_revision": np.asarray(r1.REVISION),\n        "strategy_optimizer": np.asarray(r1.STRATEGY_OPTIMIZER),\n        "{name}":')
    train=replace_once(train,"        r1.log_diagnostics(ckpt_path/'diagnostics.jsonl',iteration,collection_stats,strategy_buffer,avg_loss,avg_strategy_loss)","""        r1.log_diagnostics(ckpt_path/'diagnostics.jsonl',iteration,collection_stats,strategy_buffer,avg_loss,avg_strategy_loss)
        if iteration % checkpoint_every == 0 or iteration == end_iteration:
            r1.write_strategy_audit(ckpt_path/'strategy-audit.jsonl',iteration,strategy_net,strategy_buffer)""")
    train=replace_once(train,'    AVERAGE_SAMPLER=args.average_sampler','''    AVERAGE_SAMPLER=args.average_sampler
    if args.strategy_batch_size < 1: parser.error('positive strategy batch size required')
    r1.STRATEGY_BATCH_SIZE=args.strategy_batch_size''')
    train=replace_once(train,'    args = parser.parse_args()',"    parser.add_argument('--strategy-batch-size',type=int,default=512)\n    args = parser.parse_args()")
    train=replace_once(train,'f"\\n[Deep CFR] Training complete. {n_iterations} iterations', 'f"\\n[Deep CFR] Training segment complete. {end_iteration} iterations')
    changes['training/train.py']=train
    patch=[]
    for name,text in changes.items():
        old=(base/name).read_text();(dest/name).write_text(text)
        patch.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    (ROOT/'patches/fullhouse-v3.patch').write_text(''.join(patch))
    print(dest)
    return dest


if __name__=='__main__':
    build()
