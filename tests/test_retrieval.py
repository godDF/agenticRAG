from app.retrieval import reciprocal_rank_fusion


def test_rrf_deduplicates_and_rewards_multiple_rankings():
    first = [
        {"point_id": "a", "score": 0.80},
        {"point_id": "b", "score": 0.90},
    ]
    second = [
        {"point_id": "a", "score": 0.85},
        {"point_id": "c", "score": 0.95},
    ]
    fused = reciprocal_rank_fusion([first, second])
    assert [item["point_id"] for item in fused] == ["a", "c", "b"]
    assert fused[0]["score"] == 0.85

