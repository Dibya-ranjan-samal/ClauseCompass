from clausecompass.eval.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k


def test_recall_perfect():
    assert recall_at_k({"a"}, ["a", "b"], 5) == 1.0


def test_recall_partial():
    assert recall_at_k({"a", "b"}, ["a", "x"], 5) == 0.5


def test_recall_miss():
    assert recall_at_k({"a"}, ["x", "y"], 5) == 0.0


def test_recall_respects_k():
    assert recall_at_k({"a"}, ["x", "y", "z", "a"], 3) == 0.0


def test_mrr_rank_three():
    assert mrr({"a"}, ["x", "y", "a"]) == 1 / 3


def test_hit_binary():
    assert hit_at_k({"a", "b"}, ["b"], 5) == 1.0


def test_ndcg_rewards_higher_rank():
    """Same gold, same k - ranking it first must score above ranking it third."""
    assert ndcg_at_k({"a"}, ["a", "x", "y"], 5) > ndcg_at_k({"a"}, ["x", "y", "a"], 5)


def test_empty_gold_is_free_score():
    """Documents the footgun: no gold means 1.0, so the runner must filter these."""
    assert recall_at_k(set(), [], 5) == 1.0
    assert hit_at_k(set(), [], 5) == 1.0
    assert mrr(set(), []) == 1.0
