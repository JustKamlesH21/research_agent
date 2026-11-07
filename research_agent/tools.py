"""Step 2 · Tools: arXiv search, citation lookup, OpenAlex search (exercise E1 + E3)."""
import os
import threading
import time

import feedparser
import requests

HEADERS = {"User-Agent": "research-assistant-lab/1.0"}
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL")  # optional: joins OpenAlex's faster "polite pool"
S2_KEY = os.getenv("S2_API_KEY")              # optional: raises Semantic Scholar quota


class _Pacer:
    """Thread-safe 'one request every N seconds' (arXiv asks for ~3 s)."""
    def __init__(self, gap):
        self.gap, self.last, self.lock = gap, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            delay = self.gap - (time.time() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.time()


_arxiv_pace, _s2_pace = _Pacer(3.0), _Pacer(1.1)


def search_arxiv(query: str, max_results: int = 5, year_from: int | None = None):
    """Search arXiv; return id, title, year, link, abstract."""
    max_results = max(1, min(int(max_results), 20))
    _arxiv_pace.wait()
    try:
        r = requests.get("https://export.arxiv.org/api/query", headers=HEADERS, timeout=30,
                         params={"search_query": f"all:{query}", "sortBy": "relevance",
                                 "max_results": max_results * (3 if year_from else 1)})
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"arXiv unavailable ({e}); try again or use search_openalex"}
    feed = feedparser.parse(r.text)
    hits = [{"id": e.id.split("/abs/")[-1],
             "title": " ".join(e.title.split()),
             "year": int(e.published[:4]),
             "url": e.link,
             "abstract": " ".join(e.summary.split())[:600],
             "source": "arxiv"}
            for e in feed.entries]
    if year_from:                                     # E3: filter in Python
        hits = [h for h in hits if h["year"] >= int(year_from)]
    hits = hits[:max_results]
    return hits or {"error": "Not found: try broader or different keywords"}


def _openalex_abstract(inv: dict | None) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inv:
        return ""
    words = sorted((pos, w) for w, ps in inv.items() for pos in ps)
    return " ".join(w for _, w in words)[:600]


def search_openalex(query: str, max_results: int = 5, year_from: int | None = None):
    """Search OpenAlex (free, no key); same keys as search_arxiv plus citations."""
    params = {"search": query, "per-page": max(1, min(int(max_results), 25))}
    if year_from:
        params["filter"] = f"from_publication_date:{int(year_from)}-01-01"
    if OPENALEX_EMAIL:
        params["mailto"] = OPENALEX_EMAIL
    try:
        r = requests.get("https://api.openalex.org/works", params=params,
                         headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"OpenAlex unavailable ({e}); try search_arxiv"}
    hits = []
    for w in r.json().get("results", []):
        if not w.get("title"):
            continue
        hits.append({"id": w["id"].rsplit("/", 1)[-1],
                     "title": " ".join(w["title"].split()),
                     "year": w.get("publication_year") or 0,
                     "url": w.get("doi") or w["id"],
                     "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
                     "citations": w.get("cited_by_count", 0),
                     "source": "openalex"})
    return hits or {"error": "Not found: try broader or different keywords"}


def get_citations(title: str) -> dict:
    """Look up one paper by title: Semantic Scholar first, OpenAlex as fallback."""
    _s2_pace.wait()
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                         timeout=20, headers={**HEADERS, **({"x-api-key": S2_KEY} if S2_KEY else {})},
                         params={"query": title, "limit": 1,
                                 "fields": "title,year,citationCount,venue"})
        if r.status_code == 200:
            hits = r.json().get("data") or []
            if hits:
                return {**hits[0], "source": "semanticscholar"}
    except requests.RequestException:
        pass
    # fallback: OpenAlex (generous limits) when S2 is rate-limited (HTTP 429) or misses
    oa = search_openalex(title, max_results=1)
    if isinstance(oa, list) and oa:
        p = oa[0]
        return {"title": p["title"], "year": p["year"], "citationCount": p["citations"],
                "venue": None, "source": "openalex"}
    return {"error": "not found on Semantic Scholar or OpenAlex"}


REGISTRY = {"search_arxiv": search_arxiv,
            "search_openalex": search_openalex,
            "get_citations": get_citations}

if __name__ == "__main__":  # test each tool alone before giving it to the model
    import json
    print(json.dumps(search_arxiv("graph neural networks", 2), indent=2))
    print(json.dumps(search_openalex("graph neural networks", 2), indent=2))
    print(json.dumps(get_citations("Attention Is All You Need"), indent=2))
