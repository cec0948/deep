from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_strength_training import (Budget, component_checks, exact_iteration, game_and_infosets,
    kuhn_features, reach_weighted_average, regret_match, run, independent_average_contributions,
    reference_counterfactual_regret_error)


def test_exact_counterfactual_targets_match_hand_derived_uniform_kuhn():
    game,keys,infosets=game_and_infosets();features=kuhn_features(keys,infosets)
    checks,uniform=component_checks(game,keys,infosets,features)
    regrets,_,_=exact_iteration(game,uniform,infosets)
    for card in range(3):
        assert regrets[str(card)][:2]==pytest.approx([-.125,.125],abs=1e-12)
    assert regrets['0pb'][:2]==pytest.approx([1/12,-1/12],abs=1e-12)
    assert checks['passed'] and checks['full_information_adapter_unique_infosets']==12
    assert checks['nontrivial_positive_scaling_exercised']
    assert checks['independent_reference_first_iteration_regret_max_error']<1e-12


def test_full_information_leduc_collector_matches_independent_first_step_regrets():
    game,keys,infosets=game_and_infosets('leduc_poker');features=kuhn_features(keys,infosets)
    strategies={}
    for k in keys:
        p=np.zeros(5);p[infosets[k]['legal']]=1/len(infosets[k]['legal']);strategies[k]=p
    regrets,_,_=exact_iteration(game,strategies,infosets)
    assert len(keys)==936 and features.shape==(936,51)
    assert reference_counterfactual_regret_error(game,regrets)<1e-10
    assert set(independent_average_contributions(game,strategies))==set(keys)


def test_reach_weighted_average_differs_from_time_only_or_uniform_average():
    strategies=np.array([[.9,.1],[.1,.9]])
    value=reach_weighted_average(strategies,[.2,.8],[1.,3.])
    assert value==pytest.approx([.16153846153846155,.8384615384615385])
    assert not np.allclose(value,np.average(strategies,axis=0,weights=[1.,3.]))
    assert not np.allclose(value,strategies.mean(0))
    with pytest.raises(ValueError):reach_weighted_average(strategies,[0.,0.],[1.,3.])


def test_regret_matching_never_uses_illegal_actions_and_respects_positive_scaling():
    scores=np.array([1.,3.,900.,0.,-1.]);legal=[1,1,0,0,0]
    assert regret_match(scores,legal)==pytest.approx([.25,.75,0,0,0])
    assert regret_match(scores/7.,legal)==pytest.approx(regret_match(scores,legal))
    assert regret_match([-3.,-1.,100.,0.,0.],legal)==pytest.approx([0,1,0,0,0])


@pytest.mark.parametrize('budget',[Budget(iterations=129),Budget(fit_steps=513),Budget(batch_size=0),Budget(max_seconds=121)])
def test_unbounded_component_budgets_rejected(budget):
    with pytest.raises(ValueError):budget.validate()


def test_report_records_component_scope_and_preserves_existing_output(tmp_path):
    output=tmp_path/'check';result=run(output,Budget(iterations=12,fit_steps=96,max_seconds=30.))
    assert result['status']=='complete' and result['component_checks']['passed']
    assert result['production_six_player_path_validated'] is False
    assert result['end_to_end_custom_leduc_proven'] is False
    assert result['next_six_player_training_authorized'] is False
    assert result['six_player_policies_trained']==0
    assert len(list(output.iterdir()))==1 and (output/'report.json').exists()
    original=(output/'report.json').read_bytes()
    with pytest.raises(FileExistsError):run(output,Budget(iterations=1,fit_steps=1))
    assert (output/'report.json').read_bytes()==original
