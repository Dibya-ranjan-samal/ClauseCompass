# ClauseCompass web UI

Next.js front end for the FastAPI service in `../src/clausecompass/api`.

```bash
make serve          # from the repo root: API on :8000
cd web && npm install && npm run dev   # UI on :3000
```

| Page | What it does |
|---|---|
| `/` Ask | Question + scope picker. Shows verified claims with clickable citations, the exact quote highlighted in the source excerpt, and every claim the validator removed, with the reason. Recent questions persist locally. |
| `/audit` | Start a policy audit (background job), poll it, then filter findings by status and expand each to read the cited clause. |
| `/contracts` | Browse/filter the 530 contracts; open one to read it section by section and jump to "Ask about this contract". |
| `/traces` | SQLite span traces as a waterfall plus self time by stage. |
| `/evaluation` | The recorded retrieval, generation and agent numbers from `/metrics`. |

`NEXT_PUBLIC_API_URL` overrides the API address (default `http://localhost:8000`).
