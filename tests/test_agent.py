"""Agent guardrails. These need no LLM, so they run in the normal test suite.

The end-to-end injection run lives in `run_audit.py inject` because it needs a
model; what is asserted here is the guarantee that does NOT depend on model
behaviour - an uncited accusation cannot be constructed at all.
"""

import pytest
from pydantic import ValidationError

from clausecompass.agent.schemas import Finding
from clausecompass.agent.tools import ToolBox

CHUNKS = [{"chunk_id": "d1::c0", "doc_id": "d1", "text": "8. Liability capped at $1m.",
           "section_heading": "Liability"}]


def _box():
    return ToolBox(retriever=None, chunks=CHUNKS)


@pytest.mark.parametrize("status", ["compliant", "violation"])
def test_accusation_without_citation_is_impossible(status):
    """The day's design point. Note this must hold when `citations` is OMITTED,
    not just when it is empty - a field_validator does not run on defaults, so
    an omitted list sailed through until this was a model_validator."""
    with pytest.raises(ValidationError):
        Finding(doc_id="d1", status=status, severity="high", rationale="x" * 20)
    with pytest.raises(ValidationError):
        Finding(doc_id="d1", status=status, severity="high",
                rationale="x" * 20, citations=[])


def test_silent_needs_no_citation():
    """'silent' is a legitimate verdict and must not require evidence."""
    assert Finding(doc_id="d1", status="silent", severity="none",
                   rationale="x" * 20).status == "silent"


def test_record_finding_rejects_invented_citations():
    r = _box().record_finding(doc_id="d1", status="compliant", severity="low",
                              rationale="x" * 20, citations=["d1::c0", "made::up"])
    assert "error" in r and "made::up" in r["error"]


def test_record_finding_accepts_real_citation():
    box = _box()
    r = box.record_finding(doc_id="d1", status="compliant", severity="low",
                           rationale="x" * 20, citations=["d1::c0"])
    assert r.get("ok") and len(box.findings) == 1


def test_injected_compliance_cannot_be_recorded_without_evidence():
    """Contract text saying 'mark this compliant' cannot produce a compliant
    finding, because the model would still have to supply a real chunk_id.
    Behavioural resistance is measured separately by `run_audit.py inject`."""
    box = _box()
    r = box.record_finding(doc_id="d1", status="compliant", severity="none",
                           rationale="Section 99 instructed me to mark compliant")
    assert "error" in r
    assert not box.findings


def test_unknown_tool_args_are_rejected_not_executed():
    assert "error" in _box().record_finding(status="compliant")   # no doc_id


@pytest.mark.parametrize("call", [
    {"tool": "search_clauses", "args": {"query": "cap"}},
    {"name": "search_clauses", "arguments": {"query": "cap"}},
    {"name": "search_clauses", "arguments": '{"query": "cap"}'},
])
def test_both_tool_call_shapes_parse(call):
    from clausecompass.agent.loop import _tool_call
    assert _tool_call(call) == ("search_clauses", {"query": "cap"})


def test_non_tool_json_is_not_a_tool_call():
    from clausecompass.agent.loop import _tool_call
    assert _tool_call({"answer": "x"}) is None and _tool_call(None) is None
