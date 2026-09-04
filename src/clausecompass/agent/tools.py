"""Agent tools. Every tool validates its arguments before touching anything.

Three tools, not four. The brief's `list_contracts` is dropped: the loop scopes
one agent to one contract and hands it the doc_id, so nothing ever needs to
enumerate the corpus. A tool the agent cannot use is a prompt the model has to
read past.
"""

from pydantic import BaseModel, Field

from clausecompass.obs.trace import tracer

from .schemas import Finding


class SearchClausesArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    doc_id: str | None = None
    k: int = Field(default=5, ge=1, le=20)


class GetClauseArgs(BaseModel):
    chunk_id: str


class RecordFindingArgs(BaseModel):
    doc_id: str
    status: str
    severity: str = "none"
    rationale: str
    citations: list[str] = Field(default_factory=list)
    quote: str | None = None


class ToolBox:
    SNIPPET = 900          # search results are previews; get_clause has the rest

    def __init__(self, retriever, chunks: list[dict]):
        self.retriever = retriever
        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.findings: list[Finding] = []

    def search_clauses(self, **kw):
        a = SearchClausesArgs(**kw)
        with tracer.span("tool.search_clauses", query=a.query, doc_id=a.doc_id) as s:
            hits = self.retriever.search(a.query, a.k, a.doc_id)
            s["n_results"] = len(hits)
            return [{"chunk_id": h.chunk_id, "score": round(h.score, 3),
                     "section": (self.chunks.get(h.chunk_id) or {}).get("section_heading"),
                     "text": h.text[:self.SNIPPET]} for h in hits]

    def get_clause(self, **kw):
        """Full untruncated text. search_clauses returns previews, so this is
        how the agent reads the tail of a long clause."""
        a = GetClauseArgs(**kw)
        with tracer.span("tool.get_clause", chunk_id=a.chunk_id):
            c = self.chunks.get(a.chunk_id)
            if not c:
                return {"error": f"unknown chunk_id: {a.chunk_id}"}
            return {"chunk_id": a.chunk_id, "text": c["text"],
                    "section": c.get("section_heading")}

    def record_finding(self, **kw):
        with tracer.span("tool.record_finding", doc_id=kw.get("doc_id")) as s:
            try:
                a = RecordFindingArgs(**kw)
            except Exception as e:
                s["rejected"] = "bad_args"
                return {"error": f"invalid arguments: {e}"}
            # Citations must exist in the real chunk store. The agent cannot
            # invent evidence, and this is the check the injection test relies on.
            unknown = [c for c in a.citations if c not in self.chunks]
            if unknown:
                s["rejected"] = "unknown_citations"
                return {"error": f"unknown citations rejected: {unknown}"}
            try:
                f = Finding(**a.model_dump())
            except Exception as e:
                s["rejected"] = "schema"
                return {"error": f"invalid finding: {e}"}
            self.findings.append(f)
            s["status"] = f.status
            return {"ok": True, "recorded": f.doc_id, "status": f.status}


TOOLS = {
    "search_clauses": "Search contract clauses. args: {query, doc_id, k}",
    "get_clause": "Full text of one clause. args: {chunk_id}",
    "record_finding": ("Record the verdict for this contract, once. args: "
                       "{doc_id, status, severity, rationale, citations, quote}. "
                       "citations must be real chunk_ids from search results."),
}
