from clausecompass.generation.answer import parse_json_lenient, quote_ratio, validate

LABELS = {"1": "7.2 Indemnification. The Supplier shall indemnify the Customer.",
          "2": "12.1 Governing Law. This Agreement is governed by the laws of Delaware."}


def test_keeps_valid_claim():
    kept, stripped, bc, bq = validate({"claims": [
        {"statement": "Delaware law governs.", "citations": ["2"],
         "quote": "governed by the laws of Delaware"}]}, LABELS)
    assert len(kept) == 1 and not stripped and bc == 0 and bq == 0


def test_strips_unknown_citation():
    """A fabricated label must never reach the user."""
    kept, stripped, bc, bq = validate({"claims": [
        {"statement": "x", "citations": ["Governing Law::27"], "quote": ""}]}, LABELS)
    assert not kept and stripped[0]["reason"] == "citation_not_retrieved" and bc == 1


def test_strips_missing_citation():
    kept, stripped, bc, _ = validate({"claims": [
        {"statement": "x", "citations": [], "quote": ""}]}, LABELS)
    assert not kept and bc == 1


def test_strips_invented_quote():
    kept, stripped, _, bq = validate({"claims": [
        {"statement": "Capped at $1m.", "citations": ["2"],
         "quote": "liability shall not exceed one million dollars"}]}, LABELS)
    assert not kept and stripped[0]["reason"] == "quote_unverified" and bq == 1


def test_quote_match_ignores_whitespace():
    assert quote_ratio("governed  by\nthe laws of Delaware", [LABELS["2"]]) == 1.0


def test_parse_strips_fences():
    assert parse_json_lenient('```json\n{"a": 1}\n```') == {"a": 1}
