"""Streamlit trace viewer. Reads the same one-table span store the agent writes.

  uv run streamlit run dashboard.py
"""

import json
import sqlite3

import pandas as pd
import streamlit as st

conn = sqlite3.connect("data/traces.db", check_same_thread=False)
st.title("ClauseCompass — traces")

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
    st.subheader("Spans")
    st.dataframe(spans[["name", "duration_ms", "status"]], width="stretch")

    st.subheader("Time by stage")
    st.bar_chart(spans.groupby("name")["duration_ms"].sum())

    for _, s in spans.iterrows():
        with st.expander(f"{s['name']} — {s['duration_ms']:.0f}ms"):
            st.json(json.loads(s["attributes"]))
