"""Instructor demo · the same team as a LangGraph graph with a human interrupt.

Run:  pip install langgraph   then   python graph.py "your question"
"""
import sys

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from state import ResearchState
from team import (after_critic, critic, final_report, planner, reader, screener,
                  searchers, writer)

g = StateGraph(ResearchState)
for name, fn in [("plan", planner), ("search", searchers), ("screen", screener),
                 ("read", reader), ("write", writer), ("critique", critic)]:
    g.add_node(name, fn)
g.add_edge(START, "plan");      g.add_edge("plan", "search")
g.add_edge("search", "screen"); g.add_edge("screen", "read")
g.add_edge("read", "write");    g.add_edge("write", "critique")
g.add_conditional_edges("critique", after_critic, {"revise": "write", "end": END})
app = g.compile(checkpointer=MemorySaver(), interrupt_before=["read"])

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "LLM hallucination in medical QA"
    cfg = {"configurable": {"thread_id": "review-001"}}
    app.invoke({"question": q, "revisions": 0}, cfg)        # pauses before "read"
    shortlist = app.get_state(cfg).values["shortlist"]
    for i, p in enumerate(shortlist, 1):
        print(f"{i:>2}. {p['title']} ({p['year']})")
    drop = input("Numbers to drop (Enter = keep all): ")
    drop = {int(x) for x in drop.replace(",", " ").split() if x.isdigit()}
    app.update_state(cfg, {"shortlist": [p for i, p in enumerate(shortlist, 1) if i not in drop]})
    app.invoke(None, cfg)                                   # resumes from the checkpoint
    print(final_report(app.get_state(cfg).values))
    print(app.get_graph().draw_mermaid())
