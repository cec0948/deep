"""Decompose public-field paired payoff differences by the first changed action on shared history."""
import json
import math
import os
from pathlib import Path
import subprocess

from build_json_adapter import COMMIT, ROOT, UPSTREAM
from controlled_poker_refit import read, utc
from fullhouse_v3_common import atomic_json, digest
from register_poker_opponent_diversity import STUDY

NATIVE = ROOT/'native/noregrets-json'
SOURCE = NATIVE/'public_divergence.rs'
LABELS = ('specialist480','control504','diverse504')
SEEDS = (342,343,344)
PAIRS = ((2,0),(1,0),(2,1))
CATEGORIES = ('preflop','flop','turn','river','opponent_first','identical')


def binary():
    first = subprocess.check_output(['git','-C',UPSTREAM,'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',UPSTREAM,'status','--porcelain'],text=True).strip()
    if first != COMMIT or dirty:
        raise ValueError('Rules engine changed')
    crate = ROOT/'.cache/noregrets-public-divergence'
    crate.mkdir(exist_ok=True)
    sources = [UPSTREAM/'src'/f'{m}.rs' for m in ('cards','eval','engine')]
    header = [f'#[allow(dead_code)] #[path={json.dumps(str(f))}] mod {f.stem};' for f in sources]
    header += ['#[allow(dead_code)] mod logged { use crate::{cards,engine};',
               f'include!({json.dumps(str(NATIVE/"terminal_audit.rs"))});',
               f'include!({json.dumps(str(SOURCE))});','}',
               'fn main() -> Result<(),Box<dyn std::error::Error>> {logged::aggregate()}']
    name = 'poker-public-divergence'
    for file,text in {'entry.rs':'\n'.join(header),
                      'Cargo.toml':(UPSTREAM/'Cargo.toml').read_text()+f'\n[[bin]]\nname="{name}"\npath="entry.rs"\n',
                      'Cargo.lock':(UPSTREAM/'Cargo.lock').read_text()}.items():
        target=crate/file
        if not target.exists() or target.read_text()!=text:
            target.write_text(text)
    env = dict(os.environ,CARGO_HOME='/workspace/.tools/cargo',RUSTUP_HOME='/workspace/.tools/rustup',
               CARGO_TARGET_DIR=str(UPSTREAM/'target'),CARGO_BUILD_JOBS='1')
    subprocess.run([env['CARGO_HOME']+'/bin/cargo','+1.90.0','build','--offline','--locked','--release','--bin',name],
                   cwd=crate,env=env,check=True)
    return UPSTREAM/'target/release'/name


def main():
    exe=binary()
    panel=STUDY/'population'
    spec=read(panel/'evaluation-plan.json')['fields']['public']
    block_start, n=spec['deal_seed'],spec['deals']
    hands=n*6*len(SEEDS)
    totals=[dict.fromkeys(CATEGORIES,0) for _ in PAIRS]
    sums=[dict.fromkeys(CATEGORIES,0) for _ in PAIRS]
    by_seed={}
    flop_by_seed={}
    commitments={}
    for seed in SEEDS:
        paths=[]
        reported=[]
        for policy in LABELS:
            f=panel/'evaluation'/f'public-seed-{seed}-{policy}'
            result=read(f/'results.json')
            raw=f/'hands.jsonl.gz'
            if (not result['complete'] or result['training_seed']!=seed or result['label']!=policy or
                    result['field']!='public' or result['deals']!=n or result['hands']!=n*6 or
                    result['big_blind_chips']!=100 or result['raw_hands_sha256']!=digest(raw)):
                raise ValueError('Wrong public cell, chip units or raw file')
            paths.append(raw)
            reported.append(result['bb_per_100'])
            commitments[f.name]=digest(f/'results.json')
        result=json.loads(subprocess.check_output([exe,*paths,str(block_start),str(n)]))
        chips=result['policy_chips']
        if not all(math.isclose(total/(n*6),score,abs_tol=1e-9) for total,score in zip(chips,reported,strict=True)):
            raise ValueError('All-hand raw chip totals do not reproduce public scores')
        by_seed[str(seed)]={}
        flop_by_seed[seed]=result['flop_transitions']
        for i,(a,b) in enumerate(PAIRS):
            counts=result['counts'][i]
            gains=result['chip_differences'][i]
            if sum(counts)!=n*6 or sum(gains)!=chips[a]-chips[b] or gains[-1]!=0:
                raise ValueError('Partial or inconsistent matched-hand groups')
            flop=result['flop_transitions'][i]
            if [sum(column) for column in zip(*flop.values(),strict=True)] != [counts[1],gains[1]]:
                raise ValueError('Flop transitions do not reproduce the first-flop group')
            by_seed[str(seed)][LABELS[a]+' minus '+LABELS[b]]=dict(zip(CATEGORIES,gains,strict=True))
            for cat,count,gain in zip(CATEGORIES,counts,gains,strict=True):
                totals[i][cat]+=count
                sums[i][cat]+=gain
    comparisons={}
    for i,(a,b) in enumerate(PAIRS):
        groups={c:dict(hands=totals[i][c],share=totals[i][c]/hands,sum_chip_delta=sums[i][c],
                       contribution_bb_per_100=sums[i][c]/hands) for c in CATEGORIES}
        transitions={}
        categories=set().union(*(flop_by_seed[s][i] for s in SEEDS))
        for key in sorted(categories):
            stratified={}
            for seed in SEEDS:
                count,gain=flop_by_seed[seed][i].get(key,(0,0))
                stratified[str(seed)]=dict(hands=count,sum_chip_delta=gain,contribution_bb_per_100=gain/(n*6))
            count=sum(value['hands'] for value in stratified.values())
            gain=sum(value['sum_chip_delta'] for value in stratified.values())
            transitions[key]=dict(hands=count,sum_chip_delta=gain,contribution_bb_per_100=gain/hands,by_seed=stratified)
        if (sum(x['hands'] for x in transitions.values())!=groups['flop']['hands']
                or sum(x['sum_chip_delta'] for x in transitions.values())!=groups['flop']['sum_chip_delta']):
            raise ValueError('Per-transition impact is not additive')
        comparisons[LABELS[a]+' minus '+LABELS[b]]=dict(
            all_hand_delta_bb_per_100=sum(sums[i].values())/hands,groups=groups,
            flop_transitions_from_reference_to_candidate=transitions)
    receipt=dict(verified_utc=utc(),field='public',original_card_blocks=n,registered_seed_repeats=len(SEEDS),
                 seat_rotations=6,paired_hands_per_contrast=hands,total_logged_hands_replayed=hands*len(LABELS),
                 units='BB/100 across all matched hands, 100 chips per BB',
                 attribution='Post-hoc groups defined by the first distinct logged action on a shared history; each group includes all subsequent play and terminal chip differences.',
                 comparisons=comparisons,cell_result_sha256=commitments,by_seed_sum_chip_delta=by_seed,
                 pinned_rules_commit=COMMIT,terminal_audit_source_sha256=digest(NATIVE/'terminal_audit.rs'),grouping_source_sha256=digest(SOURCE))
    atomic_json(STUDY/'public-divergence-audit.json',receipt)


if __name__=='__main__':
    main()
