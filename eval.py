"""Evaluate over a test set (slide 19). Auto-approves the shortlist.

Run:  python eval.py test_questions.txt      -> runs/eval.csv
"""
import csv
import sys
import time

from llm import USAGE
import os

from team import RUNS_DIR, cited_ids, research, save_run


def citation_validity(state) -> float:
    """Share of [n] citations that point at a real paper with a URL. Target: 100%."""
    ids = cited_ids(state.get("draft", ""))
    s = state.get("summaries", [])
    ok = [n for n in ids if 1 <= n <= len(s) and s[n - 1].get("url")]
    return round(100 * len(ok) / len(ids), 1) if ids else 0.0


def main(path):
    questions = [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]
    rows = []
    for q in questions:
        t0, tok0 = time.time(), USAGE["tokens"]
        st = research(q, approve=lambda s: s["shortlist"])
        rows.append({"question": q,
                     "papers": len(st.get("summaries", [])),
                     "citation_validity_%": citation_validity(st),
                     "critic_passed": st.get("review", {}).get("passed"),
                     "drafts": st.get("revisions"),
                     "tokens": USAGE["tokens"] - tok0,
                     "minutes": round((time.time() - t0) / 60, 1),
                     "report": save_run(st) if st.get("draft") else ""})
        print(rows[-1])
    with open(os.path.join(RUNS_DIR, "eval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("saved runs/eval.csv  · now score coverage, faithfulness, usefulness by hand (rubric)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_questions.txt")
