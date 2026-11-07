"""Web UI with the human checkpoint.   Run:  streamlit run app.py

Same pipeline as team.py, but the stages are called here one at a time so the page can
show live progress (stepper, skeleton cards, per-paper reading) instead of one spinner.
"""
import html

import streamlit as st

from llm import USAGE
from team import after_critic, critic, final_report, planner, read_papers, save_run, \
    screener, searchers, writer

STAGES = ["Plan", "Search", "Screen", "Approve", "Read", "Write ⇄ Critic"]
e = html.escape

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--bg:#0b1020;--card:#121a2e;--card2:#18223a;--line:#24304d;--text:#e6e9f2;--muted:#8a93a8;
--accent:#7c9cff;--accent2:#b48cff;--good:#34d399;--warn:#fbbf24;--sk1:#172039;--sk2:#25325a}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:2rem;max-width:1100px}
.stApp{font-family:'Inter',system-ui,sans-serif}
.hero{position:relative;overflow:hidden;border-radius:20px;padding:28px 32px;margin-bottom:18px;
 background:linear-gradient(120deg,#1b2550,#2a1f5c,#14305a,#1b2550);background-size:300% 300%;
 animation:drift 14s ease infinite;border:1px solid var(--line)}
.hero h1{margin:0;padding:0;font-size:2rem;font-weight:800;color:#fff;letter-spacing:-.02em}
.hero p{position:relative;margin:.35rem 0 0;color:#c9cfe6;font-size:1rem}
.hero .orb{position:absolute;border-radius:50%;filter:blur(40px);opacity:.45;animation:float 9s ease-in-out infinite}
.hero .o1{width:180px;height:180px;background:var(--accent);right:-40px;top:-60px}
.hero .o2{width:140px;height:140px;background:var(--accent2);right:160px;bottom:-90px;animation-delay:-4s}
@keyframes drift{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(18px)}}
.steps{display:flex;align-items:center;margin:4px 0 20px;overflow-x:auto;padding:6px 2px}
.step{display:flex;flex-direction:column;align-items:center;gap:6px;min-width:78px}
.dot{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-size:.8rem;font-weight:700;
 border:2px solid var(--line);color:var(--muted);background:var(--card);transition:all .3s}
.step.done .dot{background:var(--good);border-color:var(--good);color:#0b1020}
.step.active .dot{border-color:var(--accent);background:var(--accent);color:#0b1020;animation:pulse 1.4s infinite}
.step span{font-size:.75rem;color:var(--muted);white-space:nowrap}
.step.active span,.step.done span{color:var(--text);font-weight:600}
.step.wait .dot{background:var(--warn);border-color:var(--warn);color:#0b1020}
.step.wait span{color:var(--warn);font-weight:700}
.turn{display:flex;align-items:center;gap:12px;background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.35);
 border-radius:14px;padding:12px 16px;margin:6px 0 12px;color:var(--text);font-size:.92rem;animation:rise .4s ease}
.turn b{color:var(--warn)}
.bar{flex:1;min-width:16px;height:2px;background:var(--line);margin-bottom:22px;position:relative;overflow:hidden}
.bar.done{background:var(--good)}
.bar.active::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,var(--accent),transparent);animation:flow 1.2s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(124,156,255,.6)}70%{box-shadow:0 0 0 12px rgba(124,156,255,0)}100%{box-shadow:0 0 0 0 rgba(124,156,255,0)}}
@keyframes flow{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:0 0 10px;transition:border-color .2s,transform .2s}
.card:hover{border-color:#3a4a7a;transform:translateY(-1px)}
.anim{opacity:0;animation:rise .5s ease forwards}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.card a.t{color:var(--text);font-weight:600;text-decoration:none;font-size:.98rem;line-height:1.35}
.card a.t:hover{color:var(--accent)}
.meta{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
.chip{font-size:.72rem;padding:2px 9px;border-radius:999px;background:var(--card2);color:var(--muted);border:1px solid var(--line)}
.chip.arxiv{color:#fca5a5}.chip.openalex{color:#93c5fd}.chip.stars{color:var(--warn)}
.abs{color:var(--muted);font-size:.85rem;line-height:1.5}
.score{height:6px;border-radius:99px;background:var(--card2);overflow:hidden;margin-top:10px}
.score i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2));
 transform-origin:left;animation:grow .9s ease forwards}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.sk{height:12px;border-radius:6px;margin:9px 0;background:linear-gradient(90deg,var(--sk1) 25%,var(--sk2) 37%,var(--sk1) 63%);
 background-size:800px 100%;animation:shimmer 1.3s infinite linear}
.sk.t{height:16px;width:72%}.sk.s{width:45%}.sk.m{width:85%}
.sk.c{display:inline-block;width:62px;height:18px;border-radius:99px;margin:4px 6px 4px 0}
@keyframes shimmer{0%{background-position:-400px 0}100%{background-position:400px 0}}
.log{font-size:.88rem;color:var(--muted);margin:3px 0;animation:rise .35s ease forwards}
.log b{color:var(--text)} .log .ok{color:var(--good);margin-right:6px;font-weight:700}
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--line);border-top-color:var(--accent);
 border-radius:50%;animation:spin .8s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.h{font-size:1.1rem;font-weight:700;color:var(--text);margin:18px 0 4px}
.sub{font-size:.85rem;color:var(--muted);margin:0 0 12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:4px 0 16px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px}
.tile .k{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.tile .v{font-size:1.45rem;font-weight:700;color:var(--text);margin-top:2px}
.tile.good .v{color:var(--good)} .tile.warn .v{color:var(--warn)}
.fields{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px;margin-top:6px}
.fields div{font-size:.85rem;color:var(--text);line-height:1.45}
.fields small{display:block;font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
@media (max-width:640px){.fields{grid-template-columns:1fr}.hero{padding:22px}.hero h1{font-size:1.6rem}}
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"]{background:linear-gradient(90deg,var(--accent),var(--accent2))!important;
 border:0!important;color:#0b1020!important;font-weight:700!important;transition:transform .15s,box-shadow .15s}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(124,156,255,.35)}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.anim,.log{opacity:1}}
</style>"""

HERO = ('<div class="hero"><div class="orb o1"></div><div class="orb o2"></div>'
        '<h1>📚 Research Assistant</h1>'
        '<p>Ask a question → get a cited literature review built only from real arXiv, '
        'OpenAlex and Semantic Scholar records.</p></div>')


# ------------------------------------------------------------------ HTML pieces
def html_md(target, s: str):
    target.markdown(s, unsafe_allow_html=True)


def stepper(active: int, wait: bool = False) -> str:
    """wait=True: the active step is the user's turn (amber, no pulse), not work in progress."""
    out = ['<div class="steps">']
    for i, name in enumerate(STAGES):
        if i:
            bar = "done" if i < active or (wait and i == active) else "active" if i == active else ""
            out.append(f'<div class="bar {bar}"></div>')
        cls = "done" if i < active else ("wait" if wait else "active") if i == active else ""
        dot = "✓" if i < active else "✋" if cls == "wait" else i + 1
        label = "Your turn" if cls == "wait" else name
        out.append(f'<div class="step {cls}"><div class="dot">{dot}</div><span>{label}</span></div>')
    return "".join(out) + "</div>"


def log_lines(done: list[str], doing: str | None = None) -> str:
    s = "".join(f'<div class="log"><span class="ok">✓</span>{d}</div>' for d in done)
    if doing:
        s += f'<div class="log"><span class="spin"></span>{doing}…</div>'
    return s


def _anim(delay):
    return ("card anim", f' style="animation-delay:{delay:.2f}s"') if delay is not None else ("card", "")


def _url(u):
    return e(u) if str(u).startswith("http") else "#"


def paper_card(p: dict, delay=None) -> str:
    cls, style = _anim(delay)
    src = e(p.get("source", ""))
    score = float(p.get("score", 0))
    abstract = (p.get("abstract") or "No abstract available.")[:260]
    return (f'<div class="{cls}"{style}><a class="t" href="{_url(p["url"])}" target="_blank">{e(p["title"])}</a>'
            f'<div class="meta"><span class="chip">{p.get("year") or "n.d."}</span>'
            f'<span class="chip">{p.get("citations") or 0} cites</span>'
            f'<span class="chip {src}">{src}</span><span class="chip">score {score:.2f}</span></div>'
            f'<div class="abs">{e(abstract)}…</div>'
            f'<div class="score"><i style="width:{min(100, int(100 * score))}%"></i></div></div>')


def summary_card(s: dict, n: int, delay=None) -> str:
    cls, style = _anim(delay)
    r = int(s.get("relevance", 1))
    f = "".join(f'<div><small>{label}</small>{e(str(s.get(k, "")))}</div>'
                for label, k in [("Problem", "problem"), ("Method", "method"),
                                 ("Key result", "key_result"), ("Limitation", "limitation")])
    return (f'<div class="{cls}"{style}><a class="t" href="{_url(s["url"])}" target="_blank">'
            f'[{n}] {e(s["title"])}</a><div class="meta"><span class="chip">{s.get("year") or "n.d."}</span>'
            f'<span class="chip stars">{"★" * r}{"☆" * (5 - r)}</span></div><div class="fields">{f}</div></div>')


def skeleton_cards(n: int) -> str:
    one = ('<div class="card"><div class="sk t"></div><div><span class="sk c"></span><span class="sk c"></span>'
           '<span class="sk c"></span></div><div class="sk m"></div><div class="sk s"></div></div>')
    return one * n


def skeleton_report() -> str:
    widths = [40, 95, 88, 92, 60, 0, 35, 97, 90, 84, 70, 0, 45, 93, 80]
    rows = "".join('<div style="height:10px"></div>' if w == 0 else
                   f'<div class="sk{" t" if w < 50 else ""}" style="width:{w}%"></div>' for w in widths)
    return f'<div class="card">{rows}</div>'


def tiles(items: list[tuple[str, str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile {c} anim" style="animation-delay:{i * .07:.2f}s"><div class="k">{k}</div>'
        f'<div class="v">{v}</div></div>' for i, (k, v, c) in enumerate(items)) + "</div>"


def fail(ex: BaseException, *placeholders):
    for p in placeholders:
        p.empty()
    st.error(f"**{type(ex).__name__}**: {ex}")
    st.stop()


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Research Assistant", page_icon="📚", layout="wide")
html_md(st, CSS)
html_md(st, HERO)
ss = st.session_state

with st.form("ask", border=False):                          # stage 1 · clarify
    c1, c2 = st.columns([4, 1])
    q = c1.text_input("Your research question", placeholder="LLM hallucination in medical QA")
    since = c2.number_input("Earliest year (0 = any)", 0, 2100, 0)
    go = st.form_submit_button("🔎  Find papers", type="primary")

steps_ph = st.empty()


def show_steps(i, wait=False):
    html_md(steps_ph, stepper(i, wait))


# ---- phase 1: plan -> search -> screen, with live progress
if go and q.strip():
    for k in ("state", "done", "path", "lines"):
        ss.pop(k, None)
    ss.run = ss.get("run", 0) + 1
    ss.calls0, ss.tok0 = USAGE["calls"], USAGE["tokens"]
    state = {"question": q.strip(), "revisions": 0, **({"year_from": int(since)} if since else {})}
    log, body = st.empty(), st.empty()
    plan = [(planner, "Splitting the question into sub-questions", 3,
             lambda s: f"Planned <b>{len(s['plan'])}</b> sub-questions"),
            (searchers, "Searching arXiv + OpenAlex in parallel", 4,
             lambda s: f"Found <b>{len(s['candidates'])}</b> candidate papers"),
            (screener, "De-duplicating, grading relevance and fetching citations", 6,
             lambda s: f"Shortlisted the top <b>{len(s['shortlist'])}</b> papers")]
    lines = []
    try:
        for i, (fn, doing, n_skel, done) in enumerate(plan):
            show_steps(i)
            html_md(log, log_lines(lines, doing))
            html_md(body, skeleton_cards(n_skel))
            state |= fn(state)
            lines.append(done(state))
    except (Exception, SystemExit) as ex:  # noqa: BLE001  e.g. MODEL not set, provider down
        fail(ex, log, body)
    log.empty()
    body.empty()
    ss.state, ss.lines, ss.fresh = state, lines, True

state = ss.get("state")

if not state:
    show_steps(-1)

# ---- human checkpoint: approve the shortlist
elif not ss.get("done"):
    if not state.get("shortlist"):
        show_steps(3)
        st.warning("No papers found. Try a broader question or an earlier year.")
        st.stop()
    show_steps(3, wait=True)
    top = st.container()        # action bar + live progress, above the (long) list
    listing = st.empty()        # cleared once reading starts
    anim = ss.pop("fresh", False)
    keep = []
    with listing.container():
        html_md(st, log_lines(ss.get("lines", [])))
        with st.expander("Plan · sub-questions and queries"):
            for s in state.get("plan", []):
                st.markdown(f"- **{s['q']}** · `{'` · `'.join(s['queries'])}`")
        html_md(st, f'<div class="h">Shortlist · {len(state.get("candidates", []))} hits → '
                    f'{len(state["shortlist"])} papers</div>')
        for i, p in enumerate(state["shortlist"]):
            c1, c2 = st.columns([1, 24])
            if c1.checkbox("keep", value=True, key=f"k{ss.run}-{i}", label_visibility="collapsed"):
                keep.append(p)
            html_md(c2, paper_card(p, i * 0.06 if anim else None))
    with top:
        html_md(st, '<div class="turn">✋ <span><b>Your turn.</b> Untick off-topic papers below, '
                    'then start the (expensive) reading step.</span></div>')
        go_read = st.button(f"📖  Read {len(keep)} papers & write the review", type="primary",
                            disabled=not keep)

    if go_read:
        listing.empty()
        state["shortlist"] = keep
        show_steps(4)
        with top:
            prog, log, body = st.progress(0.0, text="Reading papers in parallel…"), st.empty(), st.empty()
        html_md(body, skeleton_cards(min(3, len(keep))))

        def on_read(n, out):                                   # called after each paper
            prog.progress(n / len(keep), text=f"Read {n}/{len(keep)} papers")
            html_md(body, "".join(summary_card(s, i + 1) for i, s in enumerate(out) if s)
                    + skeleton_cards(min(2, len(keep) - n)))

        try:
            state["summaries"] = read_papers(keep, state["question"], on_done=on_read)
            prog.empty()
            show_steps(5)
            done = [f"Read and summarised <b>{len(state['summaries'])}</b> papers"]
            while True:                                        # writer <-> critic loop
                html_md(log, log_lines(done, f"Writing draft {state['revisions'] + 1}"))
                html_md(body, skeleton_report())
                state |= writer(state)
                html_md(log, log_lines(done, f"Critic checking draft {state['revisions']}"))
                state |= critic(state)
                if after_critic(state) == "end":
                    break
                done.append(f"Critic found <b>{len(state['review']['issues'])}</b> issues in "
                            f"draft {state['revisions']} → revising")
        except (Exception, SystemExit) as ex:  # noqa: BLE001
            fail(ex, prog, log, body)
        ss.state, ss.done, ss.fresh = state, True, True
        ss.path = save_run(state)
        st.rerun()

# ---- final report
else:
    show_steps(len(STAGES))
    rv = state["review"]
    html_md(st, tiles([
        ("Critic", "Passed" if rv["passed"] else "Not passed", "good" if rv["passed"] else "warn"),
        ("Drafts", str(state["revisions"]), ""),
        ("Papers cited", str(len(state["summaries"])), ""),
        ("LLM calls", str(USAGE["calls"] - ss.get("calls0", 0)), ""),
        ("Tokens", f"{USAGE['tokens'] - ss.get('tok0', 0):,}", ""),
    ]))
    report = final_report(state)
    anim = ss.pop("fresh", False)
    t1, t2, t3 = st.tabs(["📄 Review", "🗂️ Paper summaries", "🧐 Critic notes"])
    with t1:
        c1, c2 = st.columns([1, 1])
        c1.download_button("⬇️  Download report (.md)", report, file_name="literature_review.md")
        if c2.button("✨  New question"):
            for k in ("state", "done", "path", "lines"):
                ss.pop(k, None)
            st.rerun()
        st.caption(f"Saved to `{ss.get('path', '')}`")
        with st.container(border=True):
            st.markdown(report)
    with t2:
        html_md(st, "".join(summary_card(s, n + 1, n * 0.05 if anim else None)
                            for n, s in enumerate(state["summaries"])))
    with t3:
        if rv["issues"]:
            for i in rv["issues"]:
                st.markdown(f"- {i}")
        else:
            st.success("No issues: every claim checked against its cited paper.")
