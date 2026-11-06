"""Summarize reservoir concentration and common-deal learning curve artifacts."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(folder):
    results=json.loads((folder/'results.json').read_text())
    diagnostics={}
    for seed in results['seeds']:
        path=folder/f'seed-{seed}'/'diagnostics.jsonl'
        if not path.exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        if not rows:continue
        diagnostics[str(seed)]={
            'iterations':len(rows),'last_reservoir':rows[-1]['reservoir'],
            'median_collection_ess_fraction':statistics.median(x['collection']['ess_fraction'] for x in rows),
            'minimum_collection_ess_fraction':min(x['collection']['ess_fraction'] for x in rows),
            'maximum_collection_weight':max(x['collection']['maximum'] for x in rows),
            'iterations_with_one_sample_over_half_mass':sum(x['collection']['max_mass_fraction']>.5 for x in rows),
            'checkpoints':{str(x['iteration']):x for x in rows if x['iteration'] in (25,50,75,100)}}
    report=dict(complete=results['complete'],diagnostics=diagnostics,
        evaluation_results=[{k:x[k] for k in ('name','summary','paired_delta_vs_public_bb100','paired_delta_ci95_half_width')} for x in results['evaluations']],
        limitations=['ESS is descriptive weight concentration, not an independent sample count',
        'Common deals and action seeds pair outcomes but policy-dependent trajectories diverge',
        '100 deals and three training seeds do not establish professional-level strength',
        'Candidate selection needs a fresh held-out deal set before promotion',
        'Weighted loss magnitudes are affected by changing replay distribution and are not strength scores'])
    evaluations={x['name']:x for x in results['evaluations']}
    def blocks(item):
        rows=[json.loads(x) for x in (Path(item['directory'])/'hands.jsonl').read_text().splitlines()]
        deals={r['deal'] for r in rows}
        return {d:statistics.mean(r['chips'] for r in rows if r['deal']==d) for d in deals}
    trends={}
    for seed in results['seeds']:
        first=evaluations.get(f'seed-{seed}-iter-25');last=evaluations.get(f'seed-{seed}-iter-100')
        if first and last:
            a=blocks(first);b=blocks(last);diff=[b[d]-a[d] for d in sorted(a)]
            trends[str(seed)]=dict(delta_100_vs_25_bb100=statistics.mean(diff),
                paired_ci95_half_width=1.96*statistics.stdev(diff)/len(diff)**.5)
    report['within_seed_learning_trends']=trends
    (folder/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    summarize(parser.parse_args().folder)
