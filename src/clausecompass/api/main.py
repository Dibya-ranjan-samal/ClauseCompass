"""FastAPI service.

Two things here encode measurements rather than defaults:

1. The cross-encoder runs ONLY when the request names a doc_id. Day 4 measured
   the re-ranker at recall@5 0.762 doc-scoped but 0.331 corpus-wide - worse than
   plain dense (0.363) for 300x the latency. Applying it unconditionally would be
   slower AND less accurate on the common path.
2. /metrics reads evals/baseline.json rather than globbing runs/. sorted() on
   filenames is alphabetical, which on Day 6 silently served a one-contract smoke
   run instead of the real one.
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clausecompass.config import RetrievalConfig
from clausecompass.generation.answer import answer_from_hits

# GROQ_API_KEY and friends may live in .env (gitignored). Before anything reads os.environ.
load_dotenv()

STATE: dict = {}
JOBS: dict = {}
CFG = RetrievalConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE at startup. Loading per request is the classic
    # FastAPI + ML mistake: it turns a 40 ms call into a 10 s one.
    from clausecompass.generation.llm import make_llm
    from clausecompass.retrieval.dense import DenseRetriever
    from clausecompass.retrieval.fusion import HybridRetriever
    from clausecompass.retrieval.rerank import RerankingRetriever
    from clausecompass.retrieval.sparse import BM25Retriever

    dense = DenseRetriever(collection=CFG.collection)
    hybrid = HybridRetriever(dense, BM25Retriever(), rrf_k=CFG.rrf_k,
                             per_retriever_k=CFG.bm25_candidates)
    STATE["dense"] = dense
    STATE["scoped"] = RerankingRetriever(hybrid, candidates=CFG.rerank_candidates,
                                         split_long=CFG.split_long_chunks,
                                         strip_entity=CFG.strip_entity_for_rerank)
    STATE["llm"] = make_llm()
    STATE["chunks"] = json.loads(
        Path("data/interim/chunks_enriched.json").read_text())
    STATE["by_id"] = {c["chunk_id"]: c for c in STATE["chunks"]}
    STATE["metadata"] = json.loads(Path("data/interim/metadata.json").read_text())
    yield
    STATE.clear()


app = FastAPI(title="ClauseCompass",
              description="Contract QA where every citation is verified in code.",
              version="1.0", lifespan=lifespan)

# The Next.js UI (web/) calls this API straight from the browser rather than
# through a Next rewrite: /ask and /audit polls can outlast a proxy timeout.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "UI_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    doc_id: str | None = Field(default=None,
                               description="scope to one contract; enables the cross-encoder")
    top_k: int = Field(default=CFG.top_k_answer, ge=1, le=20)


class Claim(BaseModel):
    statement: str
    citations: list[str] = []
    chunk_ids: list[str] = []
    quote: str | None = None


class Source(BaseModel):
    """One retrieved excerpt, under the label the model saw it as ([1]..[k])."""
    label: str
    chunk_id: str
    doc_id: str
    section_heading: str | None = None
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    answer_found: bool
    claims: list[Claim]
    stripped_claims: list[dict]
    confidence: str
    retrieved_ids: list[str]
    validation: dict
    latency_ms: float
    retriever: str
    llm: str = ""
    cost_usd: float = 0.0
    sources: list[Source] = []


def _ask(question: str, doc_id: str | None, top_k: int) -> AskResponse:
    retriever = STATE["scoped"] if doc_id else STATE["dense"]
    hits = retriever.search(question, top_k, doc_id)
    r = answer_from_hits(question, hits, STATE["llm"])
    return AskResponse(
        answer=r.answer, answer_found=r.answer_found,
        claims=[Claim(**{k: v for k, v in c.items()
                         if k in {"statement", "citations", "chunk_ids", "quote"}})
                for c in r.claims],
        stripped_claims=r.stripped_claims, confidence=r.confidence,
        retrieved_ids=r.retrieved_ids, validation=r.validation,
        latency_ms=r.latency_ms,
        retriever="hybrid+rerank (doc-scoped)" if doc_id else "dense (corpus-wide)",
        llm=STATE["llm"].name, cost_usd=r.cost_usd,
        sources=[Source(label=str(i), chunk_id=h.chunk_id,
                        doc_id=STATE["by_id"].get(h.chunk_id, {}).get("doc_id", ""),
                        section_heading=STATE["by_id"].get(h.chunk_id, {}).get("section_heading"),
                        score=h.score, text=h.text)
                 for i, h in enumerate(hits, 1)])


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if "llm" not in STATE:
        raise HTTPException(503, "models still loading")
    # Retrieval and local inference are blocking CPU/GPU work; off the event loop.
    try:
        return await asyncio.to_thread(_ask, req.question, req.doc_id, req.top_k)
    except Exception as e:
        # An unhandled error becomes a 500 from the outermost middleware, which
        # runs outside CORS: the browser then sees "network error" instead of
        # the cause. HTTPException is answered inside CORS, so the UI can show it.
        raise HTTPException(
            502, f"Pipeline failed ({type(e).__name__}: {e}). "
                 "Check /healthz - Qdrant and the LLM provider must both be up.") from e


class AuditRequest(BaseModel):
    policy: str = Field(min_length=10, max_length=500)
    max_contracts: int = Field(default=12, ge=1, le=100)


def _run_audit_job(job_id: str, policy: str, max_contracts: int):
    from clausecompass.agent.loop import run_audit
    try:
        rep = run_audit(policy, STATE["dense"], STATE["chunks"], STATE["llm"],
                        max_contracts=max_contracts)
        JOBS[job_id] = {"status": "complete", "report": rep.model_dump()}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": f"{type(e).__name__}: {e}"}


@app.post("/audit", status_code=202)
async def start_audit(req: AuditRequest, bg: BackgroundTasks):
    """Audits take minutes (~22 s per contract). Never block the request."""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "report": None}
    bg.add_task(_run_audit_job, job_id, req.policy, req.max_contracts)
    return {"job_id": job_id, "status": "running",
            "poll": f"/audit/{job_id}",
            "eta_s": req.max_contracts * 22}


@app.get("/audit/{job_id}")
async def audit_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job_id")
    return JOBS[job_id]


def _contract_summary(doc_id: str, n_chunks: int) -> dict:
    m = STATE["metadata"].get(doc_id, {})
    return {"doc_id": doc_id, "n_chunks": n_chunks,
            **{k: m.get(k) for k in ("source", "contract_type", "governing_law",
                                     "effective_date", "parties")}}


@app.get("/contracts")
async def contracts():
    counts: dict[str, int] = {}
    for c in STATE["chunks"]:
        counts[c["doc_id"]] = counts.get(c["doc_id"], 0) + 1
    return [_contract_summary(d, n) for d, n in sorted(counts.items())]


@app.get("/contracts/{doc_id}")
async def contract(doc_id: str):
    chunks = [c for c in STATE["chunks"] if c["doc_id"] == doc_id]
    if not chunks:
        raise HTTPException(404, "unknown doc_id")
    keep = ("chunk_id", "section_number", "section_heading", "part", "n_parts", "text")
    return {**_contract_summary(doc_id, len(chunks)),
            "defined_terms": STATE["metadata"].get(doc_id, {}).get("defined_terms", []),
            "chunks": [{k: c.get(k) for k in keep} for c in chunks]}


@app.get("/chunks/{chunk_id:path}")
async def chunk(chunk_id: str):
    c = STATE["by_id"].get(chunk_id)
    if not c:
        raise HTTPException(404, "unknown chunk_id")
    return {k: c.get(k) for k in ("chunk_id", "doc_id", "section_number",
                                   "section_heading", "text")}


@app.get("/traces")
async def traces(limit: int = 50):
    from clausecompass.obs.trace import tracer
    cols = ("trace_id", "name", "started_at", "duration_ms", "status",
            "n_spans", "cost_usd", "question")
    return [dict(zip(cols, row, strict=True)) for row in tracer.traces(limit)]


@app.get("/traces/{trace_id}")
async def trace(trace_id: str):
    from clausecompass.obs.trace import tracer
    # tracer.spans_of drops the tree; the UI's waterfall needs parent and timing.
    rows = tracer.conn.execute(
        "SELECT span_id, parent_id, name, started_at, duration_ms, status, attributes "
        "FROM spans WHERE trace_id=? ORDER BY started_at", (trace_id,)).fetchall()
    if not rows:
        raise HTTPException(404, "unknown trace_id")
    # started_at is written when the span closes, so it is really the END time.
    return [{"span_id": sid, "parent_id": pid, "name": n,
             "start_ms": t * 1000 - d, "duration_ms": d, "status": st,
             "attributes": json.loads(a)}
            for sid, pid, n, t, d, st, a in rows]


@app.get("/healthz")
async def healthz():
    loaded = {"dense", "scoped", "llm"} <= STATE.keys()
    try:
        STATE["dense"].client.get_collections()
        qdrant = True
    except Exception:
        qdrant = False
    # blocking call: off the event loop, same as /ask
    llm = await asyncio.to_thread(STATE["llm"].ping) if loaded else False
    ok = loaded and qdrant and llm
    return {"status": "ok" if ok else "degraded", "models_loaded": loaded,
            "qdrant": qdrant, "llm": llm,
            "llm_provider": STATE["llm"].name if loaded else None}


@app.get("/metrics")
async def metrics():
    """Serve the recorded eval numbers, so quality claims are inspectable at
    runtime rather than only in the README."""
    out = {"retrieval": json.loads(Path("evals/baseline.json").read_text())}
    for name, path in (("agent", "evals/agent_eval.json"),):
        p = Path(path)
        if p.exists():
            out[name] = json.loads(p.read_text())
    gen = sorted(Path("runs").glob("*gen-final.json"), key=lambda p: p.stat().st_mtime)
    if gen:
        out["generation"] = json.loads(gen[-1].read_text())["aggregate"]
    return out
