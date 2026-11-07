"""Session 3 · Six specialist agents + a plain-Python orchestrator.

Run:  python team.py "LLM hallucination in medical QA" [--since 2022] [--yes]
"""
import argparse
import datetime as dt
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

from llm import CRITIC, SMALL, STRONG, USAGE, call_json, call_text
from memory import save_note
from state import ResearchState
from summary import summarise_paper
from tools import get_citations, search_arxiv, search_openalex

THIS_YEAR = dt.date.today().year
SHORTLIST_K = int(os.getenv("SHORTLIST_K", "12"))
MAX_WRITER_PASSES = 3                      # = at most 2 revisions


# ---------------------------------------------------------------- 1 · Planner
PLANNER = ("Split the research question into 3-5 sub-questions that together cover it. "
           "For each, give 2-3 keyword queries (3-6 words each). Return JSON: "
           '{"subquestions": [{"q": str, "queries": [str]}]}')


def planner(state):
    out = call_json(PLANNER, state["question"], model=STRONG)
    plan = [s for s in out.get("subquestions", []) if s.get("queries")][:5]
    return {"plan": plan or [{"q": state["question"], "queries": [state["question"]]}]}


# ---------------------------------------------------------------- 2 · Searchers
def search_one(sub: dict, year_from=None) -> list[dict]:
    hits = []
    for q in sub["queries"][:3]:
        for fn in (search_arxiv, search_openalex):
            r = fn(q, max_results=8, year_from=year_from)
            if isinstance(r, list):          # errors are skipped, not fatal
                hits += r
    return hits


def searchers(state):
    with ThreadPoolExecutor(max_workers=3) as pool:   # parallel fan-out
        batches = list(pool.map(lambda s: search_one(s, state.get("year_from")), state["plan"]))
    return {"candidates": [h for b in batches for h in b]}


# ---------------------------------------------------------------- 3 · Screener
def norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())


def recency(year) -> float:
    age = THIS_YEAR - int(year or 0)
    return 1.0 if age <= 3 else max(0.0, 1 - 0.15 * (age - 3))


def _keyword_relevance(p, question) -> float:
    q = set(norm(question).split()) - {"the", "of", "in", "and", "for", "a", "on", "to"}
    text = set(norm(p["title"] + " " + p.get("abstract", "")).split())
    return len(q & text) / max(1, len(q))


def relevance_scores(papers: list[dict], question: str) -> list[float]:
    """LLM grades each paper 1-5 (batched, small model); keyword overlap as fallback."""
    scores = []
    for i in range(0, len(papers), 25):
        chunk = papers[i:i + 25]
        listing = "\n".join(f"{j}. {p['title']} :: {p.get('abstract', '')[:200]}"
                            for j, p in enumerate(chunk))
        try:
            out = call_json("Grade how relevant each paper is to the research question, 1-5. "
                            'Return JSON {"scores": [int, ...]} in the same order.',
                            f"QUESTION: {question}\n\nPAPERS:\n{listing}", model=SMALL)
            s = [(max(1, min(5, int(x))) - 1) / 4 for x in out["scores"]]
            if len(s) != len(chunk):
                raise ValueError("length mismatch")
        except Exception:  # noqa: BLE001
            s = [_keyword_relevance(p, question) for p in chunk]
        scores += s
    return scores


def screener(state, k=SHORTLIST_K):
    uniq: dict[str, dict] = {}
    for p in state["candidates"]:                     # de-duplicate, merge fields
        key = norm(p["title"])
        if key in uniq:
            old = uniq[key]
            if p.get("source") == "arxiv":            # prefer arXiv record + abstract
                p = {**old, **p, "citations": old.get("citations", p.get("citations"))}
            else:
                p = {**p, **old, "citations": p.get("citations", old.get("citations"))}
        uniq[key] = p
    papers = list(uniq.values())
    if not papers:
        return {"shortlist": []}
    for p, r in zip(papers, relevance_scores(papers, state["question"])):
        p["relevance_score"] = r
    # citation lookups only for the top pre-ranked papers that lack a count
    papers.sort(key=lambda p: 0.5 * p["relevance_score"] + 0.2 * recency(p["year"]), reverse=True)
    for p in papers[:2 * k]:
        if p.get("citations") is None:
            c = get_citations(p["title"])
            ok = "error" not in c and norm(c.get("title") or "")[:40] == norm(p["title"])[:40]
            p["citations"] = (c.get("citationCount") or 0) if ok else 0
    max_c = max([p.get("citations") or 0 for p in papers] + [1])
    for p in papers:
        cite = math.log1p(p.get("citations") or 0) / math.log1p(max_c)
        p["score"] = round(0.5 * p["relevance_score"] + 0.2 * recency(p["year"]) + 0.3 * cite, 3)
    best = sorted(papers, key=lambda p: p["score"], reverse=True)
    return {"shortlist": best[:k]}


# ---------------------------------------------------------------- 4 · Reader
def reader(state):
    out = []
    for i, p in enumerate(state["shortlist"], 1):
        print(f"  [reader] {i}/{len(state['shortlist'])} {p['title'][:70]}")
        s = summarise_paper(p, state["question"], model=SMALL)
        save_note(p["id"], f"{s.title} ({s.year}): {s.key_result}")
        out.append(s.model_dump())
    return {"summaries": out}


# ---------------------------------------------------------------- 5 · Writer
WRITER = ("Write a literature review in Markdown. Group findings by THEME, not by paper. "
          "Cite as [n] using ONLY the numbered papers given. Note disagreements and open gaps. "
          "Sections: ## 1. Scope and method, ## 2. Themes (with [n] citations), "
          "## 3. Comparison table (paper [n], method, key result, limitation), "
          "## 4. Gaps and future work. Do NOT write a references list; it is added "
          "automatically. If FIX lists issues, fix every one of them.")


def writer(state):
    papers = "\n".join(f"[{i+1}] {json.dumps(s)}" for i, s in enumerate(state["summaries"]))
    fix = state.get("review", {}).get("issues", [])
    draft = call_text(WRITER, f"Q: {state['question']}\n\nPAPERS:\n{papers}\n\nFIX: {fix}",
                      model=STRONG)
    return {"draft": draft, "revisions": state.get("revisions", 0) + 1}


# ---------------------------------------------------------------- 6 · Critic
CRITIC_PROMPT = ("Strict reviewer: check every claim is supported by its cited [n]; "
                 "flag uncited claims and overstatements. Be concrete and brief. "
                 'Return JSON {"passed": bool, "issues": [str]}.')


def cited_ids(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


def critic(state):
    n_max = len(state["summaries"])
    ids = cited_ids(state["draft"])
    bad = sorted({f"[{n}] is not in the paper list" for n in ids if not 1 <= n <= n_max})
    if not ids:
        bad.append("The draft cites no papers; cite every claim as [n].")
    v = call_json(CRITIC_PROMPT, f"DRAFT:\n{state['draft']}\n\nPAPERS:\n"
                                 f"{json.dumps(state['summaries'])}", model=CRITIC)
    issues = bad + [str(i) for i in v.get("issues", [])]
    return {"review": {"passed": bool(v.get("passed")) and not bad, "issues": issues[:10]}}


def after_critic(state):                               # picks the next edge
    done = state["review"]["passed"] or state["revisions"] >= MAX_WRITER_PASSES
    return "end" if done else "revise"


# ---------------------------------------------------------------- Orchestrator
def references(state) -> str:
    """Reference list generated in code from real records, never by the model."""
    return "\n".join(f"[{i+1}] {s['title']} ({s['year']}). {s['url']}"
                     for i, s in enumerate(state.get("summaries", [])))


def final_report(state) -> str:
    rv = state.get("review", {})
    status = "passed" if rv.get("passed") else f"not passed after {state.get('revisions')} drafts"
    notes = "".join(f"\n- {i}" for i in rv.get("issues", [])) if not rv.get("passed") else ""
    return (f"# {state['question']}\n\n{state.get('draft', '')}\n\n## References\n\n"
            f"{references(state)}\n\n---\n*Critic review: {status}.{notes}*\n")


def find_papers(state: ResearchState) -> ResearchState:
    """Stages 2-4: plan -> search -> screen (prompt chaining + parallelisation)."""
    for step in (planner, searchers, screener):
        print(f"[orchestrator] {step.__name__}")
        state |= step(state)
    return state


def review_papers(state: ResearchState) -> ResearchState:
    """Stages 6-8: read -> (write <-> critique) evaluator-optimiser loop."""
    print("[orchestrator] reader")
    state |= reader(state)
    while True:
        print(f"[orchestrator] writer (pass {state.get('revisions', 0) + 1})")
        state |= writer(state)
        print("[orchestrator] critic")
        state |= critic(state)
        if after_critic(state) == "end":
            return state


def human_approves(state):
    """Checkpoint before the expensive reading step. Returns approved list or None."""
    sl = state["shortlist"]
    print("\nSHORTLIST")
    for i, p in enumerate(sl, 1):
        print(f" {i:>2}. ({p['year']}, {p.get('citations', 0)} cites, score {p['score']}) {p['title']}")
    ans = input("\nEnter = approve all · numbers to drop e.g. 3,7 · q = quit: ").strip().lower()
    if ans == "q":
        return None
    drop = {int(x) for x in re.findall(r"\d+", ans)}
    return [p for i, p in enumerate(sl, 1) if i not in drop]


RUNS_DIR = os.getenv("RUNS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def save_run(state) -> str:
    os.makedirs(RUNS_DIR, exist_ok=True)
    slug = re.sub(r"\W+", "-", state["question"].lower())[:40].strip("-")
    base = os.path.join(RUNS_DIR, f"{dt.datetime.now():%Y%m%d-%H%M%S}-{slug}")
    with open(base + ".json", "w") as f:                # audit trail
        json.dump({**state, "usage": USAGE}, f, indent=1, default=str)
    with open(base + ".md", "w") as f:
        f.write(final_report(state))
    return base + ".md"


def research(question: str, year_from=None, approve=human_approves) -> ResearchState:
    state: ResearchState = {"question": question, "revisions": 0}
    if year_from:
        state["year_from"] = int(year_from)
    state = find_papers(state)
    if not state["shortlist"]:
        print("No papers found. Try a broader question.")
        return state
    approved = approve(state)                           # human checkpoint
    if not approved:
        return state
    state["shortlist"] = approved
    return review_papers(state)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--since", type=int, help="earliest publication year")
    ap.add_argument("--yes", action="store_true", help="auto-approve the shortlist")
    a = ap.parse_args()
    result = research(" ".join(a.question), a.since,
                      approve=(lambda s: s["shortlist"]) if a.yes else human_approves)
    if result.get("draft"):
        path = save_run(result)
        print("\n" + final_report(result))
        print(f"[saved] {path}  ·  [usage] {USAGE['calls']} LLM calls, {USAGE['tokens']} tokens")
