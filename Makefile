.PHONY: help setup ingest index eval eval-gen audit serve ui test lint docker clean
help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n",$$1,$$2}'

setup:            ## deps + Qdrant + pull the local models
	uv sync && docker compose up -d qdrant
	ollama pull qwen2.5:3b && ollama pull qwen2.5:7b

ingest:           ## CUAD + EDGAR -> chunks + metadata  (~90 min, mostly metadata)
	@test -n "$$SEC_CONTACT" || echo "note: SEC_CONTACT unset; only needed to re-run fetch.py"
	uv run python ingest_cuad.py
	uv run python build_corpus.py
	uv run python chunk_structural.py
	uv run python extract_metadata.py

index:            ## embed + index into Qdrant  (~5 min)
	uv run python index.py

eval:             ## retrieval eval, both scopes  (no LLM, ~2 min)
	uv run python -m clausecompass.eval.runner --retriever dense --scope both
	uv run python ablation.py table

eval-gen:         ## generation eval: faithfulness + refusal  (~2 h local)
	uv run python precompute_hits.py rerank
	uv run python -m clausecompass.eval.gen_runner --config final --judge-model qwen2.5:7b

audit:            ## agent audit + injection test + score
	uv run python run_audit.py audit --max-contracts 12
	uv run python run_audit.py inject
	uv run python run_audit.py eval

serve:            ## FastAPI on :8000
	uv run uvicorn clausecompass.api.main:app --reload --port 8000

ui:               ## Streamlit on :8501 (needs `make serve` running)
	uv run streamlit run dashboard.py

test:             ## unit tests, no LLM required
	uv run pytest tests/ -q

lint:
	uv run ruff check src/ tests/ *.py

docker:           ## build + run the whole stack
	docker compose up -d --build

clean:
	rm -f data/interim/bm25.pkl runs/*.json
