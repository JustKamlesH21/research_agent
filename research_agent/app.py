"""Web UI with the human checkpoint.   Run:  streamlit run app.py"""
import streamlit as st

from llm import USAGE
from team import final_report, find_papers, review_papers, save_run

st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
st.title("📚 Research Assistant")
st.caption("Plan → search → screen → **you approve** → read → write ⇄ critic")

ss = st.session_state
with st.form("ask"):                                       # stage 1 · clarify
    q = st.text_input("Your research question", placeholder="LLM hallucination in medical QA")
    c1, c2 = st.columns(2)
    since = c1.number_input("Earliest year (0 = any)", 0, 2100, 0)
    go = c2.form_submit_button("1 · Find papers")

if go and q:
    with st.status("Planning, searching and screening…", expanded=False):
        ss.state = find_papers({"question": q, "revisions": 0,
                                **({"year_from": int(since)} if since else {})})
    ss.pop("done", None)

state = ss.get("state")
if state and not ss.get("done"):
    with st.expander("Plan (sub-questions)"):
        for s in state.get("plan", []):
            st.markdown(f"- **{s['q']}** · `{'` · `'.join(s['queries'])}`")
    st.subheader(f"Shortlist · {len(state.get('candidates', []))} hits → {len(state['shortlist'])}")
    st.caption("Untick off-topic papers before the (expensive) reading step.")
    keep = []
    for i, p in enumerate(state["shortlist"]):
        label = f"**{p['title']}** · {p['year']} · {p.get('citations', 0)} cites · score {p['score']}"
        if st.checkbox(label, value=True, key=f"p{i}"):
            keep.append(p)
        st.caption(f"[{p['url']}]({p['url']}) — {p.get('abstract', '')[:220]}…")
    if st.button(f"2 · Read {len(keep)} papers & write review", type="primary", disabled=not keep):
        state["shortlist"] = keep
        with st.status("Reading, writing and critiquing…"):
            ss.state = review_papers(state)
        ss.done, ss.path = True, save_run(ss.state)
        st.rerun()

if ss.get("done"):
    report = final_report(ss.state)
    rv = ss.state["review"]
    (st.success if rv["passed"] else st.warning)(
        f"Critic: {'passed' if rv['passed'] else 'not passed'} after {ss.state['revisions']} "
        f"draft(s) · {USAGE['calls']} LLM calls · {USAGE['tokens']:,} tokens · saved {ss.path}")
    st.download_button("Download report (.md)", report, file_name="literature_review.md")
    st.markdown(report)
    with st.expander("Paper summaries (structured)"):
        st.dataframe(ss.state["summaries"])
