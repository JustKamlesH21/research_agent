"""Step 6 · Structured output you can trust (Pydantic) + exercise E4 reading list."""
import json
import sys

from pydantic import BaseModel, Field, ValidationError

from llm import SMALL, call_json

READER = ("Summarise ONLY from the text given. If a field is not stated, write "
          "'not reported'. Return JSON matching this schema: {schema}")


class PaperSummary(BaseModel):
    title: str
    year: int
    url: str
    problem: str = Field(description="question the paper tackles")
    method: str
    key_result: str
    limitation: str
    relevance: int = Field(ge=1, le=5, description="relevance to the research question, 1-5")


def summarise_paper(paper: dict, question: str = "", model: str | None = None) -> PaperSummary:
    """Abstract -> validated PaperSummary. Title/year/url always come from the API record."""
    system = READER.format(schema=json.dumps(PaperSummary.model_json_schema()))
    user = (f"Research question: {question or 'general'}\n\n"
            f"Title: {paper['title']}\n\nAbstract: {paper.get('abstract') or 'not available'}")
    meta = {"title": paper["title"], "year": int(paper.get("year") or 0), "url": paper["url"]}
    for attempt in range(2):
        data = call_json(system, user, model=model or SMALL)
        try:
            if isinstance(data.get("relevance"), (int, float)):
                data["relevance"] = max(1, min(5, int(data["relevance"])))
            return PaperSummary(**{**data, **meta})   # metadata from the API wins
        except ValidationError as e:                   # retry once with the error message
            user += f"\n\nYour previous JSON failed validation: {e}. Fix it."
    # last resort: never crash the pipeline on one bad paper
    return PaperSummary(**meta, problem="not reported", method="not reported",
                        key_result="not reported", limitation="not reported", relevance=1)


def reading_list(topic: str, n: int = 5) -> list[dict]:
    """E4: a JSON list of n validated PaperSummary objects for a topic."""
    from tools import search_arxiv
    hits = search_arxiv(topic, max_results=n)
    if isinstance(hits, dict):
        raise SystemExit(hits["error"])
    return [summarise_paper(p, topic).model_dump() for p in hits]


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) or "retrieval augmented generation hallucination"
    print(json.dumps(reading_list(topic), indent=2))
