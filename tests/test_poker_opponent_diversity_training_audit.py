"""The independent seat oracle must reject a double rotation in the production game tree."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_poker_opponent_diversity_training as audit
import poker_opponent_population as population


@pytest.mark.skipif(not (audit.CONTROL/'plan.json').exists(),reason='Requires the original online training source')
def test_live_oracle_detects_wrong_rotation(monkeypatch):
    original=population.np.roll
    monkeypatch.setattr(population.np,'roll',lambda a,shift:original(a,-shift))
    with pytest.raises(ValueError,match='per-tree absolute seat assignment'):
        audit.audit_tree_dispatch(342)
