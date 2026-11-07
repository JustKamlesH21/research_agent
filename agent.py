"""Step 3 · The agent loop (Session 2), with the seven guardrails from slide 20.

Run:  python agent.py "Recent papers evaluating hallucination in RAG"
"""
import json
import os
import sys

from openai import BadRequestError

from llm import MODEL, USAGE, chat
from memory import REGISTRY as MEMORY_TOOLS
from prompts import SYSTEM_PROMPT
from schemas import TOOLS
from tools import REGISTRY as SEARCH_TOOLS

REGISTRY = {**SEARCH_TOOLS, **MEMORY_TOOLS}           # allowlist: name -> real code
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "80000"))
_KEEP = {"role", "content", "tool_calls"}             # portable assistant message


def _assistant_entry(msg) -> dict:
    """Keep only portable fields (but keep provider extras inside tool_calls,
    e.g. Gemini thought signatures, which some models require on the next turn)."""
    d = msg.model_dump(exclude_none=True)
    d = {k: v for k, v in d.items() if k in _KEEP}
    d.setdefault("content", "")
    return d


def run_agent(goal: str, max_steps: int = 8, log=print) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": goal}]
    used, seen, force_first = 0, set(), True
    for step in range(max_steps):
        kw = dict(model=MODEL, messages=messages, tools=TOOLS)
        if step == 0 and force_first:
            kw["tool_choice"] = "required"            # stops "never calls a tool"
        try:
            resp = chat(**kw)
        except BadRequestError:                       # provider lacks tool_choice=required
            kw.pop("tool_choice", None)
            resp = chat(**kw)
        used += resp.usage.total_tokens if resp.usage else 0
        if used > TOKEN_BUDGET:
            return "Stopped: token budget reached."
        msg = resp.choices[0].message
        messages.append(_assistant_entry(msg))
        if not msg.tool_calls:                        # 1. done: final answer
            log(f"[{step}] final answer ({used} tokens)")
            return msg.content
        for call in msg.tool_calls:                   # 2. run each requested tool
            name, raw = call.function.name, call.function.arguments or "{}"
            sig = (name, raw)
            if name not in REGISTRY:
                result = {"error": f"unknown tool {name}; use one of {list(REGISTRY)}"}
            elif sig in seen:
                result = {"error": "already tried this exact call; change the query or answer now"}
            else:
                try:
                    args = json.loads(raw)
                    result = REGISTRY[name](**args)
                except Exception as e:  # noqa: BLE001  errors as messages, not crashes
                    result = {"error": f"{type(e).__name__}: {e}"}
            seen.add(sig)
            n = len(result) if isinstance(result, list) else ("error" if "error" in result else 1)
            log(f"[{step}] {name}({raw}) -> {n}")    # trace
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)[:4000]})
    return "Stopped: step limit reached."             # 3. safety exit


if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]) or "Recent papers evaluating hallucination in RAG"
    print(run_agent(goal))
    print(f"\n[usage] {USAGE['calls']} LLM calls, {USAGE['tokens']} tokens")
