# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Teaching lab code (workshop "Sessions 2 + 3"): a research assistant that turns a question into a short, cited literature review built only from real arXiv / OpenAlex / Semantic Scholar records. Each file maps to lecture slides (see the table in [README.md](README.md)), so keep files small and readable; they are meant to be shown to students. It is a flat set of Python scripts: no package, no build step, no linter config.

**Duplicate tree:** `research_agent/` is currently a byte-for-byte copy of the top-level files. The top level is what `setup.bat` / `run_ui.bat` use. Ask the user before changing or deleting the copy, and don't edit only one of the two without saying so.

## Commands

The venv is named `agentlab` (Windows: `agentlab\Scripts\activate`). `setup.bat` creates it, installs deps, copies `.env.example` → `.env`, and runs both checks. `run_ui.bat` starts the UI.

```bash
pip install -r requirements.txt
python test_offline.py          # the test suite: fake LLM + fake HTTP, no key/internet → "ALL OFFLINE TESTS PASSED"
python tools.py                 # live check of the paper APIs (needs internet, no key)
python agent.py "question"      # Session 2 single agent
python team.py "question" --since 2022 [--yes]   # Session 3 team CLI; --yes skips the approval prompt
streamlit run app.py            # web UI
python graph.py "question"      # needs `pip install langgraph`
python mcp_server.py            # needs `pip install mcp`
python eval.py test_questions.txt   # → runs/eval.csv
```

There is no test framework and you can't run a single test. `test_offline.py` is one script of sequential `assert` sections (tools → agent → summary → team → graph). The graph section is skipped if `langgraph` isn't installed.

## Configuration

Everything goes through env vars loaded from `.env` by `llm.py`: `API_KEY`, `BASE_URL`, `MODEL` (required), plus optional `SMALL_MODEL` (screening, reading), `STRONG_MODEL` (planner, writer), `CRITIC_MODEL` (defaults to STRONG), `TOKEN_BUDGET`, `SHORTLIST_K`, `PRE_SCREEN_K`, `LLM_WORKERS`, `REASONING_EFFORT`, `OPENALEX_EMAIL`, `S2_API_KEY`, `MEMORY_DIR`, `RUNS_DIR`. Any OpenAI-compatible provider works (Groq, Gemini, Ollama). The `openai` SDK is the only LLM client, so don't add provider-specific SDKs.

`SMALL_MODEL` can be a comma-separated list: grading uses the first model, and reading goes round-robin over all of them (`llm.SMALL_POOL`). `REASONING_EFFORT` holds per-model `model=low` pairs, and `chat()` drops it if the provider rejects it. Both settings exist because Groq's free tier limits each model separately (8k tokens/min; qwen models also have a 1k output-tokens/min cap). On that tier, speed comes from spreading roles across models and cutting tokens, not from more threads. Measure with a real run before tuning, and check the `[llm] <model>: RateLimitError` lines.

## Architecture

**LLM layer ([llm.py](llm.py)).** A single module-level `client`, with the SDK's own retries turned off (`max_retries=0`). `chat()` wraps `chat.completions.create` with logged retries on rate limits and connection errors (it honours `retry-after`), and adds to the global `USAGE` dict. `call_json()` tries JSON mode, falls back to plain mode on `BadRequestError`, and strips code fences via `parse_json`. Every LLM call in the project goes through `chat()`. The offline test depends on that: it monkeypatches `llm.client.chat.completions.create`.

**Tools.** The functions in [tools.py](tools.py) (search, citations) and [memory.py](memory.py) (notes) return either a list/dict or `{"error": ...}`. They never raise for expected failures, because errors are fed back to the model as messages. Each module exports a `REGISTRY` of name → function. Tools are described in three separate places that must stay in sync:
- `tools.REGISTRY` / `memory.REGISTRY`: the real code, and the allowlist the agent uses
- `schemas.TOOLS`: the JSON-schema descriptions sent to the model (the descriptions act as prompts)
- `mcp_server.py`: thin wrappers for MCP

arXiv calls are throttled to one every 3 s by a thread-safe `_Pacer`, and Semantic Scholar to one every 1.1 s. `get_citations` falls back to OpenAlex when Semantic Scholar fails or returns 429. Memory uses Chroma and switches to JSON keyword matching at runtime if Chroma fails. It always mirrors notes to `research_memory/notes.json`.

**Session 2: single agent ([agent.py](agent.py)).** A hand-written tool-calling loop with guardrails: step limit, token budget, `tool_choice="required"` on step 0 (dropped if the provider rejects it), a registry allowlist, blocking of repeated identical calls, exceptions returned as tool results, and tool output truncated to 4000 chars. `_assistant_entry` keeps provider extras inside `tool_calls` (Gemini thought signatures need them).

**Session 3: team ([team.py](team.py)).** Six node functions share a `ResearchState` TypedDict ([state.py](state.py)). Each node takes the state and returns **only the keys it changed**. The orchestrator merges them with `state |= node(state)`. The same functions are wired into a LangGraph `StateGraph` in [graph.py](graph.py), so keep node signatures LangGraph-compatible.
- Pipeline: `planner` (at most 4 sub-questions) → `searchers` (every (source, query) job runs in parallel: OpenAlex gets 2 queries per sub-question, arXiv only the first, because arXiv's 3 s spacing dominates) → `screener` (dedupe by normalised title, keyword + recency pre-screen to `PRE_SCREEN_K`, LLM grades those in parallel batches of 25, keyed by paper number; score = 0.5·LLM relevance + 0.2·recency + 0.3·log citations) → **human approval** → `reader` (`read_papers`: parallel `summarise_paper`, with notes saved from the main thread only) → `writer` ⇄ `critic` loop (at most `MAX_WRITER_PASSES` = 3). Writer and critic both get the same compact `numbered()` paper list.
- `find_papers()` and `review_papers()` are the two halves around the human checkpoint, and the CLI (`research()`) uses them. The Streamlit app ([app.py](app.py)) instead calls the node functions one at a time (and `read_papers(on_done=...)`) so it can show live progress, so keep it in sync when the pipeline changes. graph.py pauses with `interrupt_before=["read"]`.
- Anti-hallucination rules in code: `PaperSummary` title/year/url are always overwritten from the API record. The References section is generated by `references()` from the summaries and never written by the model. The critic first checks in code that every `[n]` is within range, then runs an LLM judge.
- Each run saves `runs/<timestamp>-<slug>.md` (the report) and `.json` (the full state plus usage, as an audit trail).

## Gotchas

- **The fake LLM in [test_offline.py](test_offline.py) routes on the system prompt's first words** (`"Split the research"`, `"Grade how relevant"`, `"Summarise ONLY"`, `"Write a literature review"`, `"Strict reviewer"`), and it treats any request that has `tools=` as the agent. If you change the opening of a prompt in `team.py` or `summary.py`, update the test too.
- The test also checks specific behaviour: the critic rejects the first draft's out-of-range `[99]`, so `revisions == 2`; the agent blocks the repeated search call; an unknown tool name returns an error.
- `test_offline.py` sets `MEMORY_DIR` / `RUNS_DIR` to a temp dir before importing anything, because `memory.py` creates its directory at import time.
- `research_memory/`, `runs/`, `.env` and `agentlab/` are gitignored.
