"""Agent schemas. Structure before behaviour.

The validator on Finding.citations is the design point: a finding that asserts
"compliant" or "violation" without a real citation cannot be constructed. The
agent is physically unable to produce an unsupported accusation.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Status = Literal["compliant", "violation", "silent", "unclear"]


class PolicyPlan(BaseModel):
    """Phase 1 output - inspectable before a single contract is touched."""

    policy_restated: str
    required_elements: list[str] = Field(min_length=1, max_length=8)
    search_queries: list[str] = Field(min_length=1, max_length=8)
    # near-miss clause types that could be mistaken for compliance
    decoy_queries: list[str] = Field(default_factory=list)
    compliance_criteria: str


class Finding(BaseModel):
    doc_id: str
    status: Status
    severity: Literal["none", "low", "medium", "high"]
    rationale: str = Field(min_length=10, max_length=1000)
    citations: list[str] = Field(default_factory=list)
    quote: str | None = None

    # model_validator, not field_validator. A field_validator on `citations`
    # does NOT run when the field is omitted and takes its default, so an
    # accusation with no citations at all sailed straight through - exactly the
    # case this is meant to block.
    @model_validator(mode="after")
    def cites_required_when_asserting(self):
        if self.status in ("compliant", "violation") and not self.citations:
            raise ValueError(
                f"status {self.status!r} requires at least one citation")
        return self


class AuditReport(BaseModel):
    policy: str
    plan: PolicyPlan
    findings: list[Finding]
    n_contracts_examined: int
    duration_s: float
    # 0.0 on the local Ollama path; kept so a hosted provider drops straight in
    total_cost_usd: float = 0.0
    aborted: dict[str, str] = Field(default_factory=dict)

    @property
    def violations(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "violation"]
