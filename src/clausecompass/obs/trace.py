"""SQLite span tracing. Built before the agent, because debugging a multi-step
loop without traces is miserable.

One table, not two. The brief's separate `traces` table stores duration, cost
and token totals - all derivable from the spans themselves: a trace's root is
the span with no parent, and its cost is the sum over its spans. Storing both
means two writes per trace that can disagree.
"""

import contextlib
import json
import sqlite3
import time
import uuid
from pathlib import Path

DDL = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT,
    started_at REAL, duration_ms REAL, attributes TEXT, status TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace ON spans(trace_id);
"""


class Tracer:
    def __init__(self, path="data/traces.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(DDL)
        self.conn.commit()
        self._stack: list[str] = []
        self.trace_id: str | None = None

    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        """Callers mutate the yielded dict to attach attributes."""
        # A span with no parent starts a new trace, so there is no separate
        # start_trace/end_trace to forget to call.
        if not self._stack:
            self.trace_id = uuid.uuid4().hex[:16]
        sid = uuid.uuid4().hex[:16]
        parent = self._stack[-1] if self._stack else None
        trace_id = self.trace_id
        self._stack.append(sid)
        bag, status, t0 = dict(attrs), "ok", time.perf_counter()
        try:
            yield bag
        except Exception as e:
            status = f"error: {type(e).__name__}"
            raise
        finally:
            self._stack.pop()
            self.conn.execute(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)",
                (sid, trace_id, parent, name, time.time(),
                 (time.perf_counter() - t0) * 1000, json.dumps(bag, default=str),
                 status))
            self.conn.commit()

    def traces(self, limit=50):
        """Roots, with cost and span count rolled up from the children."""
        return self.conn.execute("""
            SELECT r.trace_id, r.name, r.started_at, r.duration_ms, r.status,
                   (SELECT count(*) FROM spans s WHERE s.trace_id=r.trace_id) AS n_spans,
                   (SELECT coalesce(sum(json_extract(s.attributes,'$.cost_usd')),0)
                      FROM spans s WHERE s.trace_id=r.trace_id) AS cost_usd,
                   json_extract(r.attributes,'$.question') AS question
            FROM spans r WHERE r.parent_id IS NULL
            ORDER BY r.started_at DESC LIMIT ?""", (limit,)).fetchall()

    def spans_of(self, trace_id):
        return self.conn.execute(
            "SELECT name, duration_ms, status, attributes FROM spans "
            "WHERE trace_id=? ORDER BY started_at", (trace_id,)).fetchall()


tracer = Tracer()
