from clausecompass.retrieval.base import Hit
from clausecompass.retrieval.fusion import rrf_fuse


def h(cid):
    return Hit(cid, 0.0, "", {})


def test_agreement_wins():
    """A chunk ranked well by both retrievers beats one ranked #1 by only one."""
    a = [h("x"), h("a"), h("b")]
    b = [h("y"), h("a"), h("c")]
    assert rrf_fuse([a, b], k=60)[0].chunk_id == "a"


def test_dedupes():
    assert len(rrf_fuse([[h("a")], [h("a")]], k=60)) == 1


def test_respects_top_n():
    lst = [h(str(i)) for i in range(30)]
    assert len(rrf_fuse([lst], k=60, top_n=10)) == 10


def test_rrf_k_controls_how_steeply_rank_matters():
    """rrf_k flattens rank differences as it grows.

    Two rank-2 hits always beat one rank-1 hit (2/(k+2) > 1/(k+1) for all
    k >= 0), so k does not change *that* ordering - it changes how much a
    better rank is worth. Rank 1 over rank 5 is worth 3x at k=1 and only
    1.06x at k=60.
    """
    lst = [h("first"), h("b"), h("c"), h("d"), h("fifth")]
    ratio = lambda kk: (lambda o: o["first"] / o["fifth"])(
        {x.chunk_id: x.score for x in rrf_fuse([lst], k=kk)})
    assert ratio(1) > 2.5
    assert ratio(60) < 1.1
    assert ratio(1) > ratio(60)
