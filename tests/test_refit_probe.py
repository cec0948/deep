"""Independent-deck checks and paired uncertainty for the offline probe."""
from pathlib import Path
import statistics
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_compact_refit_probe import check_card_range, summarize


def test_probe_card_range_checks_every_recorded_campaign():
    previous = {'first': [100, 2100], 'extension': [102, 2102]}
    with pytest.raises(ValueError, match='overlap'):
        check_card_range(2101, 2000, previous)
    check_card_range(2102, 2000, previous)


def test_probe_common_cards_are_not_counted_as_independent_training_seeds():
    cfg = dict(seeds=[342, 343, 344], variants=['current', 'mixture', 'refit'],
               evaluation_fields=['public', 'mixed', 'noregrets'])
    changes = [0., 100., 0., 100.]
    rows = {f'{field}-seed-{seed}-{variant}': {
                'deal_blocks': [0.] * 4 if variant == 'current' else changes, 'hands': 24}
            for seed in cfg['seeds'] for field in cfg['evaluation_fields']
            for variant in cfg['variants']}
    result = summarize(rows, cfg)
    expected = 1.96 * statistics.stdev(changes) / 2
    assert result['paired_changes']['public/refit-minus-current']['ci95_half_width'] == pytest.approx(expected)
    assert not result['decisions']['refit']['original_four_conditions_pass']
    assert result['decisions']['refit']['deployment_authorized'] is False
