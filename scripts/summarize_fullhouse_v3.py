"""Audit all raw deal/seat outcomes before archiving a completed V3 campaign."""
import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics

from fullhouse_v3_common import ROOT, digest, atomic_json


def interval(values):
    if len(values) < 2 or not all(math.isfinite(v) for v in values):
        raise ValueError('At least two finite independent deal blocks required')
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return dict(bb_per_100=mean, ci95_half_width=half,
                ci95=[mean-half, mean+half], deals=len(values))


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(f'Recomputed statistic differs: {a} vs {b}')


def verify_evaluation(folder):
    """Reconstruct blocks from raw hands, without trusting reported summaries."""
    result = json.loads((folder/'results.json').read_text())
    if not result['complete'] or result['hands'] != 6*result['deals']:
        raise ValueError('Incomplete evaluation')
    chips = defaultdict(dict)
    with gzip.open(folder/'hands.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            deal, hero = row['deal'], row['hero']
            if not (isinstance(deal, int) and 0 <= deal < result['deals']
                    and isinstance(hero, int) and 0 <= hero < 6):
                raise ValueError('Invalid deal/hero')
            if hero in chips[deal]:
                raise ValueError('Duplicate deal/hero')
            if row['seed'] != result['deal_seed'] + deal:
                raise ValueError('Wrong deck seed')
            utilities = row['utilities']
            if (len(utilities) != 6 or sum(utilities) != 0
                    or not all(isinstance(v, int) and -10000 <= v <= 50000 for v in utilities)
                    or row['chips'] != utilities[hero] or not 1 <= row['steps'] <= 512):
                raise ValueError('Invalid terminal payoffs/steps')
            chips[deal][hero] = row['chips']
    if len(chips) != result['deals'] or any(len(v) != 6 for v in chips.values()):
        raise ValueError('Missing deal/seat')
    blocks = [statistics.mean(chips[d].values()) for d in range(result['deals'])]
    if len(result['deal_blocks']) != len(blocks):
        raise ValueError('Wrong deal block count')
    for a, b in zip(blocks, result['deal_blocks']):
        close(a, b)
    stats = interval(blocks)
    close(stats['bb_per_100'], result['bb_per_100'])
    close(stats['ci95_half_width'], result['ci95_half_width'])
    if digest(Path(result['candidate'])) != result['candidate_sha256']:
        raise ValueError('Candidate hash mismatch')
    return result, blocks


def paired(a, b):
    if len(a) != len(b):
        raise ValueError('Unpaired deal sets')
    return interval([x-y for x, y in zip(a, b)])


def display(stats):
    return f"{stats['bb_per_100']:+.2f} ±{stats['ci95_half_width']:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', nargs='?', type=Path)
    args = parser.parse_args()
    folder = (args.folder or Path(json.loads(
        (ROOT/'artifacts/fullhouse-v3/current.json').read_text())['folder'])).resolve()
    state = json.loads((folder/'results.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('Campaign has not completed')
    cfg = state['config']
    from train_fullhouse_v3 import verify
    verify(folder, state)
    expected_candidates = {'public', *cfg['baseline_models']}
    for seed in cfg['seeds']:
        expected_candidates |= {f'seed-{seed}-average-{i}' for i in cfg['evaluation_iterations']}
        expected_candidates.add(f'seed-{seed}-advantage-{cfg["through"]}')
    expected_keys = {f'{field}-{name}' for field in cfg['fields'] for name in expected_candidates}
    if set(state['evaluations']) != expected_keys:
        raise ValueError('Missing or unexpected evaluations')
    report = ROOT/'reports/fullhouse-v3'/folder.name
    report.mkdir(parents=True, exist_ok=True)
    results, blocks = {}, {}
    for key in sorted(expected_keys):
        registered = state['evaluations'][key]
        source = folder/registered['path']
        result, values = verify_evaluation(source)
        if result != {k:v for k,v in registered.items() if k != 'path'}:
            raise ValueError('Supervisor/evaluation records differ')
        if result['deals'] != cfg['evaluation_deals'] or result['deal_seed'] != cfg['evaluation_deal_seed']:
            raise ValueError('Evaluation schedule changed')
        field, name = key.split('-', 1)
        if result['field'] != field:
            raise ValueError('Wrong opponent field')
        if name == 'public':
            candidate = ROOT/'.cache/fullhouse-bot/data/deep_cfr_model.npz'
        elif name in cfg['baseline_models']:
            candidate = ROOT/cfg['baseline_models'][name]
        else:
            _, seed, kind, iteration = name.split('-')
            candidate = folder/f'seed-{seed}{"_adv" if kind == "advantage" else ""}'/f'iter_{int(iteration):05d}.npz'
        if Path(result['candidate']).resolve() != candidate.resolve():
            raise ValueError('Wrong candidate model')
        dependencies = {
            'engine_sha256': '.cache/noregrets/target/release/poker-bridge',
            'evaluator_sha256': 'scripts/evaluate_fullhouse_v3.py',
            'legacy_runtime_sha256': 'scripts/fullhouse_runtime.py',
            'v3_runtime_sha256': 'scripts/fullhouse_v3_runtime.py',
        }
        for field, path in dependencies.items():
            if result[field] != state['external_hashes'][path]:
                raise ValueError('Evaluation provenance mismatch: '+field)
        if result['source_features_sha256'] != state['source_hashes']['source/bot/features.py']:
            raise ValueError('V3 features changed')
        registry = json.loads((ROOT/'artifacts/selfplay/current.json').read_text())
        opponent = (state['external_hashes']['.cache/fullhouse-bot/data/deep_cfr_model.npz']
                    if result['field'] == 'public' else registry['sha256'])
        if result['opponent_sha256'] != opponent:
            raise ValueError('Opponent mismatch')
        dest = report/key
        dest.mkdir(exist_ok=True)
        shutil.copyfile(source/'hands.jsonl.gz', dest/'hands.jsonl.gz')
        shutil.copyfile(source/'results.json', dest/'results.json')
        results[key] = {k:v for k,v in result.items() if k != 'deal_blocks'}
        results[key]['raw_sha256'] = digest(dest/'hands.jsonl.gz')
        results[key]['features_used'] = ('V3 frozen features' if 'seed-' in key else
                                        'legacy features; source_features_sha256 is unused by this candidate')
        blocks[key] = values

    differences, groups = {}, {}
    for field in cfg['fields']:
        for seed in cfg['seeds']:
            final_key = f'{field}-seed-{seed}-average-{cfg["through"]}'
            for other in ('public', 'old-120', 'old-600', f'seed-{seed}-average-60',
                          f'seed-{seed}-advantage-{cfg["through"]}'):
                differences[f'{field}/seed-{seed}-final minus {other}'] = paired(blocks[final_key], blocks[f'{field}-{other}'])
        seed_values = [blocks[f'{field}-seed-{seed}-average-{cfg["through"]}'] for seed in cfg['seeds']]
        across_seeds_by_deal = [statistics.mean(v) for v in zip(*seed_values)]
        means = [statistics.mean(v) for v in seed_values]
        groups[field] = dict(
            conditional_fixed_three_seeds=interval(across_seeds_by_deal),
            training_seed_mean_range=[min(means), max(means)],
            training_seed_mean_sample_sd=statistics.stdev(means),
            minus_old_120=paired(across_seeds_by_deal, blocks[f'{field}-old-120']),
            note='Deal-block interval conditional on these three fitted models; not uncertainty over new training seeds.')
    for key, expected in state['paired'].items():
        for name in ('bb_per_100', 'ci95_half_width', 'deals'):
            close(differences[key][name], expected[name])

    audits, timings = {}, {}
    for seed in cfg['seeds']:
        origin = folder/f'seed-{seed}'
        dest = report/f'seed-{seed}'
        dest.mkdir(exist_ok=True)
        for name in ('diagnostics.jsonl', 'strategy-audit.jsonl'):
            shutil.copyfile(origin/name, dest/name)
        rows = [json.loads(line) for line in (origin/'strategy-audit.jsonl').read_text().splitlines()]
        audits[str(seed)] = {str(row['iteration']): row['root'] for row in rows}
        timings[str(seed)] = state['seeds'][str(seed)]['training_seconds']
    summary = dict(run=str(folder.relative_to(ROOT)), revision=cfg['revision'], config=cfg,
                   created_utc=state['created_utc'], completed_utc=state['updated_utc'],
                   evaluations=results, paired=differences, fixed_seed_averages=groups,
                   average_fit_audits=audits, training_seconds_per_seed=timings,
                   total_verified_hands=sum(r['hands'] for r in results.values()),
                   raw_hand_audit_passed=True, source_hashes=state['source_hashes'],
                   external_hashes=state['external_hashes'], promoted_model=None,
                   limitations=['Nominal 95% intervals, not simultaneous multiple-comparison intervals',
                                'Six seats share a deal block; three training seeds also share the same deals',
                                'Fixed opponent fields; no professional-strength or equilibrium claim',
                                'Combined engine/fitting changes do not isolate individual causal contributions'])
    atomic_json(report/'summary.json', summary)
    for name in ('config.json', 'verification-0.json', 'verification-1.json'):
        shutil.copyfile(folder/name, report/name)
    lines = ['# V3 새 버퍼 3-seed 학습 결과', '',
             f"실행 `{folder.name}`. 원시 **{summary['total_verified_hands']:,}핸드**의 좌석·deal·칩 보존·손익을 다시 계산해 확인했다.", '',
             '단위: BB/100 ± 명목 95% deal-block 신뢰구간 반폭. 각 셀은 2,000 deals × 6좌석이다.', '',
             '| 후보 | NoRegrets 1M 5명 상대 | 공개 Fullhouse 5명 상대 |', '| --- | ---: | ---: |']
    names = ['public', 'old-120', 'old-600']
    for seed in cfg['seeds']:
        names += [f'seed-{seed}-average-60', f'seed-{seed}-average-120', f'seed-{seed}-advantage-120']
    for name in names:
        lines.append('| '+name+' | '+' | '.join(display(results[f'{field}-{name}']) for field in cfg['fields'])+' |')
    lines += ['', '## 같은 카드의 paired 차이', '', '| 비교 | BB/100 차이 |', '| --- | ---: |']
    for key, value in differences.items():
        lines.append(f'| {key} | {display(value)} |')
    lines += ['', '세 seed 평균의 구간은 같은 deal 안에서 세 모델을 먼저 평균해 계산했다. '
              '새로운 학습 seed의 불확실성까지 포괄하는 구간이 아니다. 원시 결과와 설정·해시·적합도는 `summary.json` 및 하위 폴더에 보존한다.', '',
              '기본 모델은 자동 승격하지 않았다. 여러 비교는 탐색적이며, 고정 상대의 성적을 프로급이나 균형 수렴으로 해석하지 않는다.', '']
    (report/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(report), verified_hands=summary['total_verified_hands'],
                         final={field:groups[field] for field in cfg['fields']}), indent=2))


if __name__ == '__main__':
    main()
