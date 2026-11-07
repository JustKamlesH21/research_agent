"""Offline smoke test: fake LLM + fake arXiv/OpenAlex/S2. No key or internet needed.

Run:  python test_offline.py      (expects: ALL OFFLINE TESTS PASSED)
"""
import json
import os
import re
import tempfile

os.environ["MODEL"] = "mock-model"
_tmp = tempfile.mkdtemp()                         # keep test memory/runs out of the project
os.environ["MEMORY_DIR"] = os.path.join(_tmp, "memory")
os.environ["RUNS_DIR"] = os.path.join(_tmp, "runs")

import llm      # noqa: E402
import memory   # noqa: E402
import tools    # noqa: E402
from openai.types.chat import ChatCompletion  # noqa: E402

memory._notes = None                              # use JSON memory (no embedding download)
tools._arxiv_pace.gap = tools._s2_pace.gap = 0


# ---------------- fake HTTP
class R:
    def __init__(self, code=200, text="", js=None):
        self.status_code, self.text, self._js = code, text, js

    def json(self):
        return self._js

    def raise_for_status(self):
        pass


def entry(i, q):
    return (f"<entry><id>http://arxiv.org/abs/2401.0{i}v1</id><title>{q.title()} study {i}"
            f"</title><published>202{i % 5}-01-01T00:00:00Z</published>"
            f"<summary>We study {q} and find result {i}.</summary>"
            f"<link href='http://arxiv.org/abs/2401.0{i}v1'/></entry>")


def fake_get(url, params=None, **kw):
    params = params or {}
    if "arxiv" in url:
        q = params["search_query"].split(":", 1)[1]
        n = params["max_results"]
        return R(text="<feed xmlns='http://www.w3.org/2005/Atom'>"
                 + "".join(entry(i, q) for i in range(1, n + 1)) + "</feed>")
    if "openalex" in url:
        q = params["search"]
        return R(js={"results": [{"id": f"https://openalex.org/W{i}", "title": f"{q.title()} study {i}",
                                  "publication_year": 2023, "doi": f"https://doi.org/10.1/{i}",
                                  "abstract_inverted_index": {"OpenAlex": [0], "abstract": [1]},
                                  "cited_by_count": 10 * i} for i in range(1, 4)]})
    if "semanticscholar" in url:
        return R(code=429)                        # force the OpenAlex fallback path
    raise AssertionError(url)


tools.requests.get = fake_get


# ---------------- fake LLM
def completion(content=None, tool_calls=None):
    return ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "mock",
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content,
                                 "tool_calls": tool_calls}}]})


def tc(i, name, args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


WRITER_PASSES = {"n": 0}


def fake_create(**kw):
    msgs = kw["messages"]
    sys_ = msgs[0]["content"]
    user = msgs[-1]["content"]
    if "tools" in kw:                                          # Session 2 agent
        n_tool = sum(m["role"] == "tool" for m in msgs)
        if n_tool == 0:
            return completion(tool_calls=[tc(1, "recall_notes", {"query": "rag"}),
                                          tc(2, "search_arxiv", {"query": "rag hallucination"})])
        if n_tool == 2:
            return completion(tool_calls=[tc(3, "get_citations", {"title": "Rag study 1"}),
                                          tc(4, "search_arxiv", {"query": "rag hallucination"}),  # repeat
                                          tc(5, "nonexistent_tool", {})])
        if n_tool == 5:
            return completion(tool_calls=[tc(6, "save_note", {"paper_id": "2401.01v1", "note": "x"})])
        return completion(content="Key papers: [Rag Hallucination study 1](http://arxiv.org/abs/2401.01v1)")
    if sys_.startswith("Split the research"):
        return completion(json.dumps({"subquestions": [
            {"q": "How measured?", "queries": ["hallucination benchmark", "faithfulness metric"]},
            {"q": "Mitigations?", "queries": ["retrieval grounding"]}]}))
    if sys_.startswith("Grade how relevant"):
        n = len(re.findall(r"^\d+\. ", user.split("PAPERS:")[1], re.M))
        return completion("```json\n" + json.dumps({"scores": [5 - i % 5 for i in range(n)]}) + "\n```")
    if sys_.startswith("Summarise ONLY"):
        return completion(json.dumps({"title": "garbled", "year": 1900, "url": "bad",
                                      "problem": "p", "method": "m", "key_result": "k",
                                      "limitation": "not reported", "relevance": 7}))
    if sys_.startswith("Write a literature review"):
        WRITER_PASSES["n"] += 1
        extra = " Also [99]." if WRITER_PASSES["n"] == 1 else ""
        return completion(f"## 1. Scope\nThemes [1] and [2].{extra}")
    if sys_.startswith("Strict reviewer"):
        return completion(json.dumps({"passed": True, "issues": []}))
    raise AssertionError(sys_[:60])


llm.client.chat.completions.create = fake_create

# ---------------- tests
print("== tools")
assert isinstance(tools.search_arxiv("rag", 3), list)
assert all(h["year"] >= 2023 for h in tools.search_arxiv("rag", 5, year_from=2023))
assert tools.search_openalex("rag", 2)[0]["abstract"] == "OpenAlex abstract"
assert tools.get_citations("Rag study 1")["source"] == "openalex"

print("== Session 2 agent")
from agent import run_agent  # noqa: E402
log = []
out = run_agent("RAG hallucination", log=log.append)
print("\n".join(log))
assert out.startswith("Key papers"), out
assert any("-> error" in l and "search_arxiv" in l for l in log)       # repeated call blocked
assert any("nonexistent_tool" in l and "error" in l for l in log)      # unknown tool handled
assert memory.recall_notes("anything") == ["x"]

print("== summary / pydantic")
from summary import summarise_paper  # noqa: E402
s = summarise_paper({"title": "Real Title", "year": "2024", "url": "http://real", "abstract": "a"})
assert (s.title, s.year, s.url, s.relevance) == ("Real Title", 2024, "http://real", 5)

print("== Session 3 team")
from team import final_report, research, save_run  # noqa: E402
st = research("RAG hallucination", year_from=2021, approve=lambda s: s["shortlist"][:4])
assert len(st["plan"]) == 2 and st["candidates"] and len(st["shortlist"]) == 4
assert all(sm["url"].startswith("http") for sm in st["summaries"])
assert st["revisions"] == 2 and st["review"]["passed"], st["review"]   # critic caught [99]
rep = final_report(st)
assert "## References" in rep and "[4]" in rep
path = save_run(st)
assert os.path.exists(path)

print("== graph (LangGraph)")
try:
    import graph  # noqa: E402
    cfg = {"configurable": {"thread_id": "t"}}
    graph.app.invoke({"question": "RAG", "revisions": 0}, cfg)
    assert graph.app.get_state(cfg).next == ("read",)                 # paused at checkpoint
    graph.app.invoke(None, cfg)
    assert graph.app.get_state(cfg).values["review"]["passed"]
except ImportError:
    print("   (langgraph not installed, skipped)")

print(f"\nusage: {llm.USAGE}\nALL OFFLINE TESTS PASSED")
