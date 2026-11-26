"""Publish a small source-verified progress record for the November 26 study."""
import json

from controlled_poker_refit import read, utc
from fullhouse_v3_common import atomic_json, digest
from register_poker_pool528 import REPORT, STUDY


PANELS = {'main': (54, 756000), 'crossplay': (27, 162000),
          'training-opponents': (27, 162000)}
FINAL_FILES = ('stage-ready.json', 'training-audit.json', 'main/summary.json',
               'external-audit.json', 'crossplay/summary.json',
               'crossplay/restricted-deviations.json', 'training-opponents/summary.json',
               'backup-verification.json')


def training_progress(seed):
    folder = STUDY/'pool528'/f'seed-{seed}'
    state = read(folder/'status.json')
    log = folder/'diagnostics.jsonl'
    # Only count fully written records when a worker is still appending.
    lines = log.read_bytes().split(b'\n')[:-1] if log.exists() else []
    rows = [json.loads(line) for line in lines]
    return {'state': state['status'], 'last_finished_round': rows[-1]['iteration'] if rows else 504,
            'last_saved_round': state['last_committed_iteration'],
            'new_regret_samples': sum(row['new_regret_rows'] for row in rows)}


def check_final(receipts, progress, panels):
    audit = read(STUDY/'training-audit.json')
    external = read(STUDY/'external-audit.json')
    matrix = read(STUDY/'crossplay/summary.json')
    old = read(STUDY/'training-opponents/summary.json')
    backup = read(STUDY/'backup-verification.json')
    primary = read(STUDY/'main/summary.json')
    restricted = read(STUDY/'crossplay/restricted-deviations.json')
    if not (all(row['last_saved_round'] == row['last_finished_round'] == 528 for row in progress.values())
            and audit['training_complete'] and audit['actual_round_updates'] == 72
            and audit['stage_sha256'] == receipts['stage-ready.json'] == external['training_stage_sha256']
            and external['evaluation_summary_sha256'] == receipts['main/summary.json']
            and external['complete'] and primary['complete'] and matrix['complete'] and old['complete']
            and all(row['completed_cells'] == row['planned_cells'] for row in panels.values())
            and primary['completed_hands'] == external['hands'] == PANELS['main'][1]
            and matrix['terminal_replay_hands'] == old['terminal_replayed_hands'] == 162000
            and restricted['matrix_sha256'] == receipts['crossplay/summary.json']
            and backup['stage_iteration'] == 528 and backup['restoration_verified']
            and backup['external_backup'] and backup['verified_remote_size']
            and backup['verified_owner_only'] and backup['private_index_readback_verified']):
        raise ValueError('Final receipts, seed iterations or evaluation counts disagree')


def summarize():
    progress = {str(seed): training_progress(seed) for seed in (342, 343, 344)}
    receipts = {name: digest(STUDY/name) for name in FINAL_FILES if (STUDY/name).is_file()}
    panels = {
        panel: {'completed_cells': len(list((STUDY/panel/'evaluation').glob('*/results.json'))),
                'planned_cells': cells, 'planned_hands': hands}
        for panel, (cells, hands) in PANELS.items()
    }
    complete = len(receipts) == len(FINAL_FILES)
    if complete:
        check_final(receipts, progress, panels)
    status = dict(observed_utc=utc(), fixed_baseline={'stage': 480, 'policy': 'specialist480'},
                  learning_lineage={'parent': 504, 'from_round': 505, 'through_round': 528,
                                    'source': '2026-11-23 opponent diversity'},
                  training=progress, training_complete='stage-ready.json' in receipts,
                  evaluation=panels, unique_original_cards=6000, planned_evaluation_hands=1080000,
                  verified_receipts_sha256=receipts, complete=complete, model_promoted=False,
                  next_round_authorized=False)
    for folder in (STUDY, REPORT):
        atomic_json(folder/'status.json', status)
    return status


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2))
