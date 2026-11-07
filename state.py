"""The shared state: our blackboard. Each node reads it and returns only the keys it changed."""
from typing import TypedDict


class ResearchState(TypedDict, total=False):
    question: str           # from the user
    year_from: int          # stage 1 "clarify": optional earliest year
    plan: list[dict]        # sub-questions + queries
    candidates: list[dict]  # raw search hits
    shortlist: list[dict]   # ranked, human-approved
    summaries: list[dict]   # one PaperSummary each
    draft: str              # Markdown review
    review: dict            # critic: passed + issues
    revisions: int          # writer passes so far
