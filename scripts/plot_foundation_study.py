"""Render exact Leduc learning curves as standalone PNG/SVG artifacts."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/poker-foundation-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(run, output):
    run, output = Path(run), Path(output)
    summary = json.loads((run/'benchmark-summary.json').read_text())
    tabular = json.loads((run/'benchmark/tabular/results.json').read_text())
    neural = [json.loads((run/f'benchmark/deep-{seed}/results.json').read_text()) for seed in (71,72,73)]
    if not all(row['complete'] for row in [tabular]+neural):
        raise ValueError('All reference runs must be complete')
    output.mkdir(parents=True, exist_ok=True)
    colors = ['#2563eb','#059669','#d97706']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(12,4.4), layout='constrained')
    ax = axes[0]
    ax.axhline(summary['initial_uniform_exploitability'], color='#9ca3af', ls=':', label='Uniform baseline')
    ax.plot([r['iteration'] for r in tabular['curve']],
        [r['metrics']['average']['exploitability'] for r in tabular['curve']],
        'o-', color='#111827', lw=2, label='Tabular CFR average')
    for report, color in zip(neural, colors):
        ax.plot([r['iteration'] for r in report['curve']],
            [r['metrics']['realized_average']['exploitability'] for r in report['curve']],
            'o-', color=color, label=f"Deep CFR average, seed {report['seed']}")
    ax.set(yscale='log', xlabel='Iterations (different compute per method)',
           ylabel='Exact exploitability (lower is better)', title='Two-player Leduc: average-strategy learning')
    ax.grid(axis='y', alpha=.18); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]; x = np.arange(3); width=.24
    for i,(name,label,color) in enumerate(zip(
            ['current','realized_average','fitted_average'],
            ['Latest policy','Exact realized average','Fitted average network'],
            ['#d97706','#2563eb','#059669'])):
        values = [r['curve'][-1]['metrics'][name]['exploitability'] for r in neural]
        bars=ax.bar(x+(i-1)*width, values, width=width, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    ax.axhline(summary['initial_uniform_exploitability'], color='#9ca3af', ls=':', label='Uniform baseline')
    ax.set(xticks=x, xticklabels=['Seed71','Seed72','Seed73'], ylim=(0,2.8),
           ylabel='Exact exploitability (lower is better)', title='Deep CFR at120 iterations')
    ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Reference validation only — no claim about six-player Hold’em strength', fontsize=12)
    for extension in ('png','svg'):
        path = output/f'leduc-learning-curves.{extension}'
        fig.savefig(path, dpi=170)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args.run,args.output)
