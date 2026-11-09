"""Instrument the unchanged evaluator; do not modify a candidate or its actions."""
import argparse
from collections import Counter, defaultdict
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import evaluate_fullhouse as evaluator
from fullhouse_runtime import Fullhouse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--deals', type=int, default=10000)
    parser.add_argument('--deal-seed', type=int, default=120261109)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counters = defaultdict(Counter)
    observations = []
    # Independent diagnostic RNG cannot change policy sampling or the deck.
    capture_rng = np.random.default_rng(9871)
    captured_features = []

    class ObservedPolicy(Fullhouse):
        def decide(self, state, seed):
            result = super().decide(state, seed)
            counter = counters[state['street']]
            counter['decisions'] += 1
            counter[result['action']['kind']] += 1
            if result['action']['kind'] == 'raise':
                counter['all_in'] += int(result['action']['to'] == state['raise_bounds'][1])
            counter['facing_bet'] += int(not state['can_check'])
            if not state['can_check']:
                counter['fold_facing_bet'] += int(result['action']['kind'] == 'fold')
            for item in result['distribution']:
                counter['probability_' + item['action']['kind']] += item['probability']
            if capture_rng.random() < .1 and len(captured_features) < 20000:
                from bot.features import encode_state_dict
                x, _ = encode_state_dict(state, self.lookup._equity_tables)
                captured_features.append(x)
                if len(observations) < 300:
                    observations.append(state)
            return result

    evaluator.Fullhouse = ObservedPolicy
    sys.argv = ['evaluate_fullhouse.py', '--deals', str(args.deals), '--deal-seed', str(args.deal_seed),
                '--arms', 'fullhouse', '--timeout', '2400']
    if args.model:
        sys.argv += ['--model', str(args.model)]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        evaluator.main()
    run_path = Path(output.getvalue().strip().splitlines()[-1])
    result = json.loads((run_path / 'results.json').read_text())
    if not result['complete'] or result['summary']['fullhouse']['complete_deals'] != args.deals:
        raise RuntimeError('Incomplete diagnostic evaluation: ' + str(run_path))
    import hashlib
    result['instrumentation_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['raw_run'] = str(run_path)
    result['behavior'] = dict(counters)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'observations.json').write_text(json.dumps(observations) + '\n')
    np.savez_compressed(args.output / 'features.npz', features=np.asarray(captured_features))
    print(json.dumps({'path': str(args.output), 'summary': result['summary'], 'behavior': result['behavior']}), flush=True)


if __name__ == '__main__':
    main()
