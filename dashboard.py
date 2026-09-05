"""ClauseCompass UI: ask questions, and inspect the traces behind them.

One Streamlit app with two tabs rather than the brief's separate app.py and
dashboard.py - same two screens, one process, one command, and the demo can
move between asking a question and reading its trace without switching apps.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(page_title="ClauseCompass", layout="wide")
st.title("ClauseCompass")
st.caption("Contract QA where every citation is verified in code")


@st.cache_data
def load_chunks():
    return {c["chunk_id"]: c
            for c in json.loads(Path("data/interim/chunks_enriched.json").read_text())}


ask_tab, trace_tab = st.tabs(["Ask", "Traces"])

with ask_tab:
    chunks = load_chunks()
    doc_ids = sorted({c["doc_id"] for c in chunks.values()})

    q = st.text_input("Ask about the contract corpus",
                      "What notice period is required to terminate this agreement?")
    scope = st.selectbox(
        "Scope (doc-scoped enables the cross-encoder: recall@5 0.76 vs 0.36)",
        ["whole corpus"] + doc_ids)

    if st.button("Ask", type="primary"):
        payload = {"question": q}
        if scope != "whole corpus":
            payload["doc_id"] = scope
        with st.spinner("Retrieving, generating, verifying citations…"):
            r = requests.post(f"{API}/ask", json=payload, timeout=600).json()

        st.success(r["answer"]) if r["answer_found"] else st.warning(r["answer"])

        st.subheader("Verified claims")
        for c in r["claims"]:
            st.markdown(f"**{c['statement']}**")
            if c.get("quote"):
                st.caption(f'Quoted verbatim: "{c["quote"][:200]}"')
            for cid in c.get("chunk_ids") or []:
                ch = chunks.get(cid, {})
                with st.expander(f"{cid} — {ch.get('section_heading') or 'clause'}"):
                    st.text((ch.get("text") or "")[:2000])

        # The stripped claims are the point of the whole project: these are
        # things the model asserted that could not be verified against source.
        if r["stripped_claims"]:
            st.subheader(f"Rejected by the validator ({len(r['stripped_claims'])})")
            for s in r["stripped_claims"]:
                st.error(f"{s.get('reason')} — cited {s.get('citations')}\n\n"
                         f"{str(s.get('statement'))[:200]}")

        with st.sidebar:
            st.metric("Latency", f"{r['latency_ms']:.0f} ms")
            st.metric("Confidence", r["confidence"])
            st.caption(r["retriever"])
            st.subheader("Validation")
            st.json(r["validation"])

with trace_tab:
    conn = sqlite3.connect("data/traces.db", check_same_thread=False)
    traces = pd.read_sql("""
      SELECT r.trace_id, r.name, r.duration_ms, r.status,
             (SELECT count(*) FROM spans s WHERE s.trace_id=r.trace_id) n_spans,
             json_extract(r.attributes,'$.question') question
      FROM spans r WHERE r.parent_id IS NULL
      ORDER BY r.started_at DESC LIMIT 50""", conn)
    st.dataframe(traces, width="stretch")

    if len(traces):
        tid = st.selectbox("Inspect trace", traces["trace_id"])
        spans = pd.read_sql(
            "SELECT name, duration_ms, status, attributes FROM spans "
            "WHERE trace_id=? ORDER BY started_at", conn, params=(tid,))
        st.dataframe(spans[["name", "duration_ms", "status"]], width="stretch")
        st.subheader("Time by stage")
        st.bar_chart(spans.groupby("name")["duration_ms"].sum())
        for _, s in spans.iterrows():
            with st.expander(f"{s['name']} — {s['duration_ms']:.0f}ms"):
                st.json(json.loads(s["attributes"]))
