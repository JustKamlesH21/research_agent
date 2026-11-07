# Research Assistant (Sessions 2 + 3)

Ask a research question → get a short, cited literature review built from **real** arXiv / OpenAlex / Semantic Scholar records. Runs on free or local models.

## Setup (10 min)

```bash
python -m venv agentlab
source agentlab/bin/activate          # Windows: agentlab\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                  # Windows: copy .env.example .env
# edit .env: paste your API key (Groq or Gemini free tier, or use Ollama)
python test_offline.py                # no key needed -> ALL OFFLINE TESTS PASSED
python tools.py                       # checks the live paper APIs
```

## Run

| What | Command |
|---|---|
| Step 1: plain LLM (watch it invent papers) | `python llm.py` |
| **Session 2**: single agent (tools + loop + memory) | `python agent.py "Recent papers evaluating hallucination in RAG"` |
| E4: 5 validated PaperSummary objects | `python summary.py "your topic"` |
| **Session 3**: six-agent team (CLI) | `python team.py "LLM hallucination in medical QA" --since 2022` |
| **Web UI** (demo this one) | `streamlit run app.py` |
| LangGraph version (pause/resume) | `pip install langgraph` then `python graph.py "question"` |
| MCP server for the tools | `pip install mcp` then `python mcp_server.py` |
| Eval over a test set | `python eval.py test_questions.txt` → `runs/eval.csv` |

Every team run saves `runs/<time>-<question>.md` (report) + `.json` (full state = audit trail).

## Files → slides

| File | Session · slide |
|---|---|
| `llm.py` | S2 · 7 — client by env vars; `call_json` / `call_text` helpers, retry on 429 |
| `tools.py` | S2 · 9–10 + E1, E3 — arXiv, OpenAlex, citations (S2 → OpenAlex fallback), pacing |
| `schemas.py` | S2 · 11 — JSON-schema tool descriptions |
| `agent.py` | S2 · 13, 20, 21 — the loop + guardrails: step limit, token budget, truncation, allowlist, repeat-call block, errors-as-messages, trace, forced first tool call |
| `prompts.py` | S2 · 14 — system prompt (+ recall-first memory rule, E2) |
| `memory.py` | S2 · 16 — Chroma notes (JSON fallback) |
| `summary.py` | S2 · 17 + E4 — `PaperSummary`, validate + retry, API metadata wins |
| `state.py` | S3 · 8 — `ResearchState` |
| `team.py` | S3 · 9–15 — planner, parallel searchers, screener (0.5 rel + 0.2 recency + 0.3 log-citations), reader, writer, critic (code check + LLM judge), orchestrator, human checkpoint |
| `app.py` | S3 · 17, 21 — Streamlit with the shortlist approval step |
| `graph.py` | S3 · 16–17 — LangGraph with `interrupt_before=["read"]` |
| `mcp_server.py` | S3 · 18 |
| `eval.py` | S3 · 19 — citation validity, critic pass, tokens, minutes |

## If something breaks

| Symptom | Fix |
|---|---|
| Agent never calls a tool | Model lacks tool calling → use `llama-3.3-70b-versatile` / `gemini-2.5-flash` |
| `rate limited, retrying` | Normal on free tiers; it waits and retries. Lower `SHORTLIST_K` to cut calls |
| Semantic Scholar HTTP 429 | Already falls back to OpenAlex automatically |
| `chromadb` won't install | Ignore it; memory falls back to `research_memory/notes.json` |
| Windows PowerShell env vars | Not needed; just use `.env` |
