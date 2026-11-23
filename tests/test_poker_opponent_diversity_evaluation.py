from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

import evaluate_poker_opponent_diversity as evaluator
from fullhouse_v3_common import atomic_json


def test_original_deal_count_and_isolated_new_table():
    design = evaluator.design()
    fields = design['fields']
    ranges = list(design['config']['ranges'].values())
    assert len(design['jobs']) == len({j['id'] for j in design['jobs']}) == 81
    assert sum(fields[j['field']]['deals']*6 for j in design['jobs']) == 1944000
    assert sum(r['deals'] for r in ranges) == 26000
    new = fields['solverlike']
    assert new['role'] == 'holdout'
    assert evaluator.table('solverlike', new['deal_seed']) == ('solverlike',)*5
    assert all(new['deal_seed'] >= r['deal_seed']+r['deals'] for r in ranges if r != design['config']['ranges']['solverlike'])


def test_known_weights_can_be_evaluated_early_without_replaying_them_again(tmp_path, monkeypatch):
    design = evaluator.design()
    reserved = set(design['config']['early_reference_jobs'])
    jobs = design['jobs']
    scheduled = []
    def plan(early=False):
        return dict(frozen_files={},jobs=[j for j in jobs if not early or j['id'] in reserved]), 'bound'
    def submit(folder, commands, workers, phase):
        for argv in commands.values():
            mode,field,seed,label=argv[-4:]
            assigned=[j['id'] for j in jobs if j['field']==field and j['seed']==int(seed)
                      and label in ('all',j['label'])]
            scheduled.extend(assigned)
            if mode=='_early':
                assert len(assigned)==1 and assigned[0] in reserved
            else:
                assert not (set(assigned) & reserved)
        if phase=='early-public':
            atomic_json(folder/'early-public-status.json',dict(state='complete'))
    monkeypatch.setattr(evaluator,'OUTPUT',tmp_path)
    monkeypatch.setattr(evaluator,'get_plan',plan)
    monkeypatch.setattr(evaluator,'verify_files',lambda *args: None)
    monkeypatch.setattr(evaluator.os,'nice',lambda *args: None)
    monkeypatch.setattr(evaluator,'run_jobs',submit)
    monkeypatch.setattr(evaluator,'summarize',lambda: None)
    evaluator.run(True)
    evaluator.run(False)
    assert sorted(scheduled)==sorted(j['id'] for j in jobs)
