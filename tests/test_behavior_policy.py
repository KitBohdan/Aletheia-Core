from vct.behavior.policy import BehaviorPolicy, BehaviorInputs

def test_policy_score_bounds():
    p = BehaviorPolicy({"stimulus":0.4,"confidence":0.3,"reward_bias":0.2,"mood":0.1})
    v = p.decide("SIT", BehaviorInputs(1.0,1.0,1.0,1.0))
    assert 0.0 <= v.score <= 1.0 and v.action == "SIT"

def test_policy_threshold_reasonable():
    p = BehaviorPolicy({"stimulus":0.4,"confidence":0.3,"reward_bias":0.2,"mood":0.1})
    v = p.decide("SIT", BehaviorInputs(1.0,0.9,0.7,0.0))
    assert v.score >= 0.6