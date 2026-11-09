"""Verify raw deal blocks and preserve the regression investigation in Git."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def estimate(values):
    mean=statistics.mean(values)
    half=1.96*statistics.stdev(values)/len(values)**.5
    return dict(deals=len(values),bb_per_100=mean,ci95_half_width=half,ci95=[mean-half,mean+half])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    results={};blocks={};total=0
    for path in sorted(args.input.glob('*/results.json')):
        result=json.loads(path.read_text())
        if 'summary' not in result:continue
        if not result['complete']:raise ValueError('Incomplete candidate '+path.parent.name)
        assert hashlib.sha256(Path(result['candidate_path']).read_bytes()).hexdigest()==result['model_hashes']['fullhouse']
        label=path.parent.name
        raw=Path(result['raw_run'])/'hands.jsonl'
        groups={};seen=set()
        with raw.open() as stream,gzip.open(args.output/f'hands-{label}.jsonl.gz','wt',compresslevel=9) as zipped:
            for line in stream:
                row=json.loads(line);key=(row['deal'],row['hero'])
                assert key not in seen and 0<=row['hero']<6
                assert row['seed']==result['deal_seed']+row['deal']
                assert row['chips']==row['utilities'][row['hero']] and sum(row['utilities'])==0
                seen.add(key);groups.setdefault(row['deal'],[]).append(row['chips']);zipped.write(line)
        assert set(groups)==set(range(result['deals_requested'])) and all(len(v)==6 for v in groups.values())
        b={k:statistics.mean(v) for k,v in groups.items()};e=estimate(list(b.values()))
        for k in ('bb_per_100','ci95_half_width'):
            original='deal_ci95_half_width' if k=='ci95_half_width' else k
            assert abs(e[k]-result['summary']['fullhouse'][original])<1e-8
        blocks[label]=b;total+=len(seen)
        results[label]=dict(e,field=result.get('opponent_field','five NoRegrets 1M policies'),
                           deal_seed=result['deal_seed'],model_sha256=result['model_hashes']['fullhouse'],behavior=result['behavior'])
        shutil.copyfile(path,args.output/f'evaluation-{label}.json')
    comparisons=[('medium-600','medium-120'),('medium-300','medium-120'),('medium-600','medium-300'),('medium-600','public'),
                 ('advantage-120','medium-120'),('advantage-300','medium-300'),('advantage-600','medium-600'),
                 ('advantage-600','advantage-120'),('advantage-600','advantage-300'),
                 ('pps-seed7-2048','pps-seed7-128'),('uniform-raw-seed7-128','uniform-raw-seed9-128'),
                 ('public-field-advantage-600','public-field-average-600'),('public-field-advantage-120','public-field-average-600')]
    for seed in (7,8,9):
        comparisons += [(f'pps-seed{seed}-128',f'uniform-raw-seed{seed}-128'),(f'uniform-raw-seed{seed}-128','medium-600')]
    paired={}
    for a,b in comparisons:
        if a not in blocks or b not in blocks:continue
        assert results[a]['deal_seed']==results[b]['deal_seed'] and results[a]['field']==results[b]['field']
        ids=sorted(blocks[a].keys()&blocks[b].keys())
        paired[f'{a} minus {b}']=estimate([blocks[a][i]-blocks[b][i] for i in ids])
    for path in args.input.glob('*.json'):shutil.copyfile(path,args.output/path.name)
    for folder in ('replay','advantage-fitting'):
        path=args.input/folder/'results.json'
        if path.exists():shutil.copyfile(path,args.output/f'{folder}.json')
    report=dict(evaluated_hands=total,candidates=results,paired=paired,
                primary='medium-600 minus medium-120 on 10000 fresh deal blocks',
                limitations=['One self-play training seed; refit seeds vary only final fitting.','Nominal deal-block 95% normal intervals; primary comparison fixed before extended evaluation, remaining ablations exploratory.','Frozen replay refits do not repair historical game-engine errors.','No promotion, no professional-strength claim.'])
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    manifest={str(p.relative_to(args.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('*')) if p.is_file() and p.name!='manifest.json'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'hands':total,'candidates':len(results),'paired':paired},indent=2))


if __name__=='__main__':
    main()
