"""Expose only the independently selected policies for the two-street study."""
import json
import os
from pathlib import Path
import tempfile

from collect_poker_street_progress import REPORT, SEEDS, STUDY
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest


def selected(street):
    folder = STUDY/street
    record = read(folder/'selected.json')
    stage = read(folder/'candidate.json')
    design_hash = digest(STUDY/'design.json')
    dev = read(folder/'dev-score.json')
    if (not record['complete'] or not stage['training_complete']
            or record['design_sha256'] != design_hash or set(record['policies']) != {str(s) for s in SEEDS}):
        raise ValueError(f'Incomplete {street} selection')
    verify_files({folder/'candidate.json': record['candidate_receipt_sha256'],
                  folder/'dev-score.json': record['dev_sha256']})
    confirmation = None
    keep = False
    if dev['eligible_for_confirmation']:
        verify_files({folder/'confirm-score.json': record['confirm_sha256']})
        confirmation = read(folder/'confirm-score.json')
        keep = confirmation['paired_gain']['lower'] > 0
    if (record['selected'] == 'candidate') != keep:
        raise ValueError(f'{street} policy differs from the registered decision rule')
    if keep and record['policies'] != stage['policies']:
        raise ValueError(f'{street} policy differs from the confirmed candidate')
    if street == 'flop' and not keep:
        sources = read(STUDY/'design.json')['source_policies']
        if any(ref['sha256'] != sources[seed]['manifest_sha256'] for seed, ref in record['policies'].items()):
            raise ValueError('Flop fallback differs from its registered parent')
    verify_files({ref['path']: ref['sha256'] for ref in record['policies'].values()})
    return record, dict(selected=record['selected'], selected_stage=record['selected_stage'],
                        dev_gain=dev['paired_gain'], confirm_gain=confirmation['paired_gain'] if confirmation else None,
                        dev_root_deals=dev['unique_original_root_deals'],
                        confirm_root_deals=confirmation['unique_original_root_deals'] if confirmation else 0,
                        selection_sha256=digest(folder/'selected.json'))


def publish():
    flop, flop_summary = selected('flop')
    final, turn_summary = selected('turn')
    if read(STUDY/'turn/dev-score.json')['reach_selection_sha256'] != digest(STUDY/'flop/selected.json'):
        raise ValueError('Turn uses a different first-flop selection')
    if final['selected'] == 'parent' and final['policies'] != flop['policies']:
        raise ValueError('Turn fallback differs from the selected flop policy')
    refs = final['policies']
    destination = STUDY/'play'
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=STUDY) as temp:
            staged = Path(temp)/'play'
            for seed, ref in refs.items():
                entry = staged/f'seed-{seed}'/'policy.json'
                entry.parent.mkdir(parents=True)
                target = os.path.relpath(ref['path'], (destination/f'seed-{seed}'))
                entry.symlink_to(target)
            os.rename(staged, destination)
    for seed, ref in refs.items():
        entry = destination/f'seed-{seed}'/'policy.json'
        if entry.resolve(strict=True) != Path(ref['path']).resolve(strict=True):
            raise ValueError('An existing play entry has a different target')
        verify_files({entry: ref['sha256']})
    result = dict(complete=True, selected_stage=final['selected_stage'],
                  design_sha256=digest(STUDY/'design.json'), flop=flop_summary, turn=turn_summary,
                  policy_sha256={s:v['sha256'] for s,v in refs.items()},
                  entrypoint='python scripts/run_poker.py --backend street-progress --policy '
                             'artifacts/preflop-followup/480-v1/street-progress-20261126/play/seed-{seed}/policy.json',
                  whole_game_strength_verified=False, performance_matches=0)
    paths = [folder/'play-ready.json' for folder in (STUDY, REPORT)]
    previous = [read(path) for path in paths if path.exists()]
    stamp = previous[0]['completed_utc'] if previous else utc()
    result['completed_utc'] = stamp
    if any(record != result for record in previous):
        raise ValueError('Stored play receipt differs from the selected policy')
    for path in paths:
        if not path.exists():
            atomic_json(path, result)
    return result


if __name__ == '__main__':
    print(json.dumps(publish(), indent=2))
