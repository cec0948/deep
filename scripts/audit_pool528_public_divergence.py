"""Describe where public-opponent terminal differences first appear on a shared hand history."""
import json
import os
from pathlib import Path
import subprocess
from collections import Counter
from statistics import NormalDist

import numpy as np

from audit_poker_public_divergence import CATEGORIES, COMMIT, NATIVE, PAIRS, ROOT, SOURCE, UPSTREAM
from audit_poker_pool528_evaluation import slot_number
from controlled_poker_refit import read, verify_files
from evaluate_average_baseline import hand_seed
from evaluate_poker_pool528 import OUTPUT, SEEDS, STUDY, TABLES
from fullhouse_v3_common import atomic_json, digest


LABELS = ('current480','mixture504','mixture528')
HERE = NATIVE / 'pool528_public_divergence.rs'
PUBLIC = ROOT/'reports'/STUDY.relative_to(ROOT/'artifacts')


def build_reader():
    head = subprocess.check_output(['git','-C',UPSTREAM,'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',UPSTREAM,'status','--porcelain'],text=True).strip()
    if head != COMMIT or dirty:
        raise ValueError('Unpinned rules checkout')
    crate = ROOT/'.cache/noregrets-pool528-divergence'
    crate.mkdir(exist_ok=True)
    entry = [f'#[allow(dead_code)] #[path={json.dumps(str(UPSTREAM/"src"/(m+".rs")))}] mod {m};'
             for m in ('cards','eval','engine')]
    entry.append('#[allow(dead_code)] mod logged { use crate::{cards,engine};')
    entry.extend(f'include!({json.dumps(str(path))});' for path in (NATIVE/'terminal_audit.rs',SOURCE,HERE))
    entry.append('}\nfn main() -> Result<(), Box<dyn std::error::Error>> {logged::pool528()}')
    name = 'poker-pool528-divergence'
    files = {'entry.rs':'\n'.join(entry), 'Cargo.lock':(UPSTREAM/'Cargo.lock').read_text(),
             'Cargo.toml':(UPSTREAM/'Cargo.toml').read_text()+f'\n[[bin]]\nname="{name}"\npath="entry.rs"\n'}
    for key,value in files.items():
        path=crate/key
        if not path.is_file() or path.read_text()!=value:
            path.write_text(value)
    env=dict(os.environ,CARGO_HOME='/workspace/.tools/cargo',RUSTUP_HOME='/workspace/.tools/rustup',
             CARGO_TARGET_DIR=str(UPSTREAM/'target'),CARGO_BUILD_JOBS='1')
    subprocess.run([env['CARGO_HOME']+'/bin/cargo','+1.90.0','build','--offline','--locked','--release',
                    '--bin',name],env=env,cwd=crate,check=True)
    return UPSTREAM/'target/release'/name


def paired_interval(blocks):
    avg=float(blocks.mean())
    half=NormalDist().inv_cdf(0.975)*float(blocks.std(ddof=1))/np.sqrt(len(blocks))
    return dict(mean=avg,lower=float(avg-half),upper=float(avg+half))


def main():
    exe=build_reader()
    start,n=TABLES['public']
    expected_tape=bytearray()
    for card in range(start,start+n):
        for seat in range(6):
            seed=hand_seed(card,seat)
            expected_tape.extend(seed.to_bytes(8,'big')+bytes([slot_number(seed)]))
    hands=n*6*len(SEEDS)
    cells={}
    plans={p:digest(OUTPUT/p) for p in ('evaluation-plan.json','reference-plan.json')}
    verify_files({OUTPUT/'evaluation-plan.json':read(OUTPUT/'evaluation-bound.json')['sha256']})
    plan=read(OUTPUT/'evaluation-plan.json')
    if plan['reference_plan_sha256']!=plans['reference-plan.json']:
        raise ValueError('Reference plan binding mismatch')
    counts=np.zeros((3,6),dtype=np.int64)
    gains=np.zeros((n,3,6),dtype=np.int64)
    switches=[[(Counter(),Counter()) for _ in range(4)] for _ in PAIRS]
    for seed in SEEDS:
        paths,blocks=[],[]
        for label in LABELS:
            key=f'public-seed-{seed}-{label}'
            folder=OUTPUT/'evaluation'/key
            f=folder/'results.json'
            r=read(f)
            raw=folder/'hands.jsonl.gz'
            if (not r['complete'] or not r['terminal_payoffs_replayed'] or r['training_seed']!=seed
                    or r['field']!='public' or r['label']!=label or r['deal_seed']!=start or r['deals']!=n
                    or r['hands']!=n*6 or r['big_blind_chips']!=100 or r['opponents']!=plan['opponents'][f'public/{seed}']
                    or r['evaluation_plan_sha256']!=plans['reference-plan.json' if key in plan['early_ids'] else 'evaluation-plan.json']
                    or r['candidate_sha256']!=plan['policies'][label][str(seed)]['sha256']):
                raise ValueError(f'Wrong or partial public cell: {key}')
            verify_files({raw:r['raw_hands_sha256']})
            cells[key]=dict(result_sha256=digest(f),raw_sha256=r['raw_hands_sha256'])
            blocks.append(r['deal_blocks'])
            paths.append(raw)
        report=json.loads(subprocess.check_output([exe,*paths,str(start),str(n)],input=expected_tape))
        payoffs=np.asarray([d[0] for d in report['deals']],dtype=np.int64)
        observed=np.asarray([d[1] for d in report['deals']],dtype=np.int64)
        if not np.array_equal(payoffs/6,np.asarray(blocks).T):
            raise ValueError('A replayed card block differs from what the main evaluator reported')
        for i,(a,b) in enumerate(PAIRS):
            if not np.array_equal(observed[:,i].sum(axis=1),payoffs[:,a]-payoffs[:,b]):
                raise ValueError('First-divergence gains do not reproduce paired difference on every card')
            for street,trans in enumerate(report['transitions'][i]):
                c,g=switches[i][street]
                c.update({key:pair[0] for key,pair in trans.items()})
                g.update({key:pair[1] for key,pair in trans.items()})
        counts+=np.asarray(report['counts'],dtype=np.int64)
        gains+=observed
    comparisons={}
    for i,(a,b) in enumerate(PAIRS):
        paired=gains[:,i]/(6*len(SEEDS))
        if (counts[i].sum()!=hands or counts[i][0]!=0 or counts[i][4]!=0 or np.any(gains[:,i,5])):
            raise ValueError('Unaccounted hands, changed common preflop/opponents, or nonzero identical-ledger payoff')
        groups={}
        for j,cat in enumerate(CATEGORIES):
            cnt=int(counts[i,j])
            impact=int(gains[:,i,j].sum())
            row=dict(hands=cnt,share=cnt/hands,sum_chip_delta=impact,
                     contribution_bb_per_100=paired_interval(paired[:,j]))
            if j<4:
                c,g=switches[i][j]
                if sum(c.values())!=cnt or sum(g.values())!=impact:
                    raise ValueError('Action transitions are incomplete')
                row['switches_reference_to_candidate']={key:dict(hands=c[key],sum_chip_delta=g[key],
                        contribution_bb_per_100=g[key]/hands) for key in sorted(c)}
            groups[cat]=row
        comparisons[f'{LABELS[a]} minus {LABELS[b]}']=dict(
            difference_bb_per_100=paired_interval(paired.sum(axis=1)),groups=groups)
    result=dict(kind='post-hoc-public-first-divergence',deal_seed=start,original_card_blocks=n,
                fixed_training_seeds=list(SEEDS),seat_rotations=6,paired_hands_per_contrast=hands,
                logged_hands_replayed=hands*len(LABELS),unit='BB/100 across all matched hands',
                method='Group by first changed logged action on identical history. Each contribution includes all later actions and terminal payoffs; descriptive, not a causal value of changing that action.',
                inference='Pointwise normal 95% intervals across original cards after averaging all six seats and three fixed training seeds. Exploratory, unadjusted across post hoc groups.',
                original_card_ledger_and_each_contrast_reconciled=True,
                comparisons=comparisons,cell_commitments=cells,rules_commit=COMMIT,
                source_sha256={p.name:digest(p) for p in (Path(__file__),HERE,SOURCE,NATIVE/'terminal_audit.rs')})
    for destination in (STUDY,PUBLIC):
        atomic_json(destination/'public-divergence.json',result)


if __name__=='__main__':
    main()
