from vct.behavior.policy import BehaviorInputs, BehaviorPolicy


def test_policy_score_bounds():
    policy = BehaviorPolicy({})
    inputs = BehaviorInputs(
        stimulus=1.0,
        confidence=0.9,
        reward_bias=0.6,
        mood=0.2,
        stress=0.3,
        fatigue=0.4,
        environmental_complexity=0.5,
        social_engagement=0.7,
    )
    vector = policy.decide("SIT", inputs)
    assert 0.0 <= vector.score <= 1.0
    assert vector.action == "SIT"


def test_policy_learns_from_feedback():
    policy = BehaviorPolicy({"seed": 123})
    positive = BehaviorInputs(
        stimulus=1.0,
        confidence=0.95,
        reward_bias=0.9,
        mood=0.8,
        stress=0.05,
        fatigue=0.1,
        environmental_complexity=0.2,
        social_engagement=0.9,
    )
    negative = BehaviorInputs(
        stimulus=0.2,
        confidence=0.2,
        reward_bias=0.1,
        mood=-0.8,
        stress=0.9,
        fatigue=0.85,
        environmental_complexity=0.8,
        social_engagement=0.1,
    )
    baseline_diff = policy.decide("SIT", positive).score - policy.decide("SIT", negative).score
    dataset = [(positive, 0.95), (negative, 0.05)] * 30
    policy.train(dataset, epochs=200)
    trained_diff = policy.decide("SIT", positive).score - policy.decide("SIT", negative).score
    assert trained_diff > baseline_diff
    assert trained_diff > 0.2
