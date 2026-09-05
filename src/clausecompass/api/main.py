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
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from clausecompass.config import RetrievalConfig
from clausecompass.generation.answer import answer_from_hits

STATE: dict = {}
JOBS: dict = {}
CFG = RetrievalConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE at startup. Loading per request is the classic
    # FastAPI + ML mistake: it turns a 40 ms call into a 10 s one.
    from clausecompass.generation.llm import OllamaProvider
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
    STATE["llm"] = OllamaProvider()
    STATE["chunks"] = json.loads(
        Path("data/interim/chunks_enriched.json").read_text())
    yield
    STATE.clear()


app = FastAPI(title="ClauseCompass",
              description="Contract QA where every citation is verified in code.",
              version="1.0", lifespan=lifespan)


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
        retriever="hybrid+rerank (doc-scoped)" if doc_id else "dense (corpus-wide)")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if "llm" not in STATE:
        raise HTTPException(503, "models still loading")
    # Retrieval and local inference are blocking CPU/GPU work; off the event loop.
    return await asyncio.to_thread(_ask, req.question, req.doc_id, req.top_k)


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


@app.get("/healthz")
async def healthz():
    import httpx
    loaded = {"dense", "scoped", "llm"} <= STATE.keys()
    try:
        STATE["dense"].client.get_collections()
        qdrant = True
    except Exception:
        qdrant = False
    def _ping():
        try:
            return httpx.get("http://localhost:11434/api/tags",
                             timeout=2).status_code == 200
        except Exception:
            return False

    # blocking call: off the event loop, same as /ask
    ollama = await asyncio.to_thread(_ping)
    ok = loaded and qdrant and ollama
    return {"status": "ok" if ok else "degraded", "models_loaded": loaded,
            "qdrant": qdrant, "ollama": ollama}


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
