// Typed client for the FastAPI service in src/clausecompass/api/main.py.
// Called straight from the browser (CORS is enabled there): /ask can run for
// tens of seconds, longer than a Next rewrite proxy is happy to wait.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(
      0,
      `Cannot reach the API at ${API_URL}. Is \`make serve\` running? It accepts requests only once its models have loaded.`,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

// ---- /ask -----------------------------------------------------------------

export type Claim = {
  statement: string;
  citations: string[];
  chunk_ids: string[];
  quote: string | null;
};

export type StrippedClaim = Partial<Claim> & {
  reason: "citation_not_retrieved" | "quote_unverified" | string;
  unknown?: string[];
  match_ratio?: number;
};

export type Source = {
  label: string;
  chunk_id: string;
  doc_id: string;
  section_heading: string | null;
  score: number;
  text: string;
};

export type AskResponse = {
  answer: string;
  answer_found: boolean;
  claims: Claim[];
  stripped_claims: StrippedClaim[];
  confidence: string;
  retrieved_ids: string[];
  validation: {
    n_claims_raw: number;
    n_claims_kept: number;
    n_bad_citation: number;
    n_bad_quote: number;
    parse_failed: boolean;
    truncated: boolean;
  };
  latency_ms: number;
  retriever: string;
  llm: string;
  cost_usd: number;
  sources: Source[];
};

export const ask = (question: string, doc_id: string | null, top_k = 5) =>
  request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify({ question, doc_id, top_k }),
  });

// ---- /audit ---------------------------------------------------------------

export type FindingStatus = "compliant" | "violation" | "silent" | "unclear";

export type Finding = {
  doc_id: string;
  status: FindingStatus;
  severity: "none" | "low" | "medium" | "high";
  rationale: string;
  citations: string[];
  quote: string | null;
};

export type AuditReport = {
  policy: string;
  plan: {
    policy_restated: string;
    required_elements: string[];
    search_queries: string[];
    decoy_queries: string[];
    compliance_criteria: string;
  };
  findings: Finding[];
  n_contracts_examined: number;
  duration_s: number;
  total_cost_usd: number;
  aborted: Record<string, string>;
};

export type AuditJob =
  | { status: "running"; report: null }
  | { status: "complete"; report: AuditReport }
  | { status: "error"; error: string };

export const startAudit = (policy: string, max_contracts: number) =>
  request<{ job_id: string; status: string; eta_s: number }>("/audit", {
    method: "POST",
    body: JSON.stringify({ policy, max_contracts }),
  });

export const getAudit = (jobId: string) => request<AuditJob>(`/audit/${jobId}`);

// ---- contracts & chunks ---------------------------------------------------

export type ContractSummary = {
  doc_id: string;
  n_chunks: number;
  source: string | null;
  contract_type: string | null;
  governing_law: string | null;
  effective_date: string | null;
  parties: string[] | null;
};

export type ContractChunk = {
  chunk_id: string;
  section_number: string | null;
  section_heading: string | null;
  part: number | null;
  n_parts: number | null;
  text: string;
};

export type ContractDetail = ContractSummary & {
  defined_terms: string[];
  chunks: ContractChunk[];
};

export type Chunk = {
  chunk_id: string;
  doc_id: string;
  section_number: string | null;
  section_heading: string | null;
  text: string;
};

// The list is ~530 rows and never changes while the API runs: fetch it once.
let contractsPromise: Promise<ContractSummary[]> | null = null;
export const getContracts = () => {
  contractsPromise ??= request<ContractSummary[]>("/contracts").catch((e) => {
    contractsPromise = null;
    throw e;
  });
  return contractsPromise;
};

export const getContract = (docId: string) =>
  request<ContractDetail>(`/contracts/${encodeURIComponent(docId)}`);

const chunkCache = new Map<string, Promise<Chunk>>();
export const getChunk = (chunkId: string) => {
  if (!chunkCache.has(chunkId)) {
    chunkCache.set(
      chunkId,
      request<Chunk>(`/chunks/${encodeURI(chunkId)}`).catch((e) => {
        chunkCache.delete(chunkId);
        throw e;
      }),
    );
  }
  return chunkCache.get(chunkId)!;
};

// ---- traces, health, metrics ----------------------------------------------

export type TraceSummary = {
  trace_id: string;
  name: string;
  started_at: number;
  duration_ms: number;
  status: string;
  n_spans: number;
  cost_usd: number;
  question: string | null;
};

export type Span = {
  span_id: string;
  parent_id: string | null;
  name: string;
  start_ms: number;
  duration_ms: number;
  status: string;
  attributes: Record<string, unknown>;
};

export const getTraces = (limit = 50) => request<TraceSummary[]>(`/traces?limit=${limit}`);
export const getTrace = (traceId: string) => request<Span[]>(`/traces/${traceId}`);

export type Health = {
  status: "ok" | "degraded";
  models_loaded: boolean;
  qdrant: boolean;
  llm: boolean;
  llm_provider: string | null;
};

/** "groq:openai/gpt-oss-120b" -> { provider: "Groq", model: "openai/gpt-oss-120b" } */
export function describeLlm(name: string | null | undefined) {
  const [p, ...rest] = (name ?? "").split(":");
  const provider = p === "groq" ? "Groq" : p === "ollama" ? "Ollama" : p || "LLM";
  return { provider, model: rest.join(":") };
}
export const getHealth = () => request<Health>("/healthz");

type Aggregate = Record<string, number>;
export type Metrics = {
  retrieval: {
    _note: string;
    day3_dense_baseline: { corpus_wide: Aggregate; doc_scoped: Aggregate };
    corpus_wide: { config: string; retriever: string; git_sha: string; aggregate: Aggregate };
    doc_scoped: { config: string; retriever: string; git_sha: string; aggregate: Aggregate };
  };
  generation?: Aggregate;
  agent?: Record<string, string | number>;
};
export const getMetrics = () => request<Metrics>("/metrics");
