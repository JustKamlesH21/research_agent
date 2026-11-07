"""Step 1 · LLM client. Works with any OpenAI-compatible provider via env vars."""
import json
import os
import re
import time

from openai import APIConnectionError, BadRequestError, OpenAI, RateLimitError

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(base_url=os.getenv("BASE_URL") or None,
                api_key=os.getenv("API_KEY") or "missing",
                max_retries=0)                # chat() retries, and logs it, instead of the SDK
MODEL = os.getenv("MODEL", "")
# many cheap calls (screen, read). A comma-separated list spreads reading across models;
# free tiers rate-limit per model, so two models ≈ twice the throughput.
SMALL_POOL = [m.strip() for m in (os.getenv("SMALL_MODEL") or MODEL).split(",") if m.strip()] or [""]
SMALL = SMALL_POOL[0]
STRONG = os.getenv("STRONG_MODEL") or MODEL   # few judgement calls: plan, write
CRITIC = os.getenv("CRITIC_MODEL") or STRONG  # ideally a different model from the writer

# optional per-model reasoning effort, e.g. "openai/gpt-oss-20b=low": reasoning models spend
# hidden tokens thinking, and on per-minute token quotas fewer tokens means less waiting
EFFORT = dict(x.split("=", 1) for x in os.getenv("REASONING_EFFORT", "").replace(" ", "").split(",")
              if "=" in x)

USAGE = {"calls": 0, "tokens": 0}             # for the cost slide


def chat(**kw):
    """chat.completions.create with retries on rate limits + token accounting."""
    if not kw.get("model"):
        raise SystemExit("MODEL is not set. Copy .env.example to .env and fill it in.")
    if kw["model"] in EFFORT:
        kw.setdefault("reasoning_effort", EFFORT[kw["model"]])
    for attempt in range(8):
        try:
            r = client.chat.completions.create(**kw)
            USAGE["calls"] += 1
            if r.usage:
                USAGE["tokens"] += r.usage.total_tokens or 0
            return r
        except BadRequestError:
            if "reasoning_effort" not in kw:
                raise
            kw.pop("reasoning_effort")        # provider/model doesn't support it: carry on without
        except (RateLimitError, APIConnectionError) as e:
            wait = 5 * (attempt + 1)
            try:  # the provider says how long to wait (often 1-3 s); trust it over our guess
                wait = min(60.0, float(e.response.headers["retry-after"]) + 0.5)
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            print(f"  [llm] {kw['model']}: {type(e).__name__}, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("LLM provider still failing after retries (rate limit / network).")


def parse_json(text: str) -> dict:
    """Parse JSON even if the model wrapped it in ```json fences or added chatter."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_json(system: str, user: str, model: str | None = None) -> dict:
    """JSON mode in, Python dict out. Falls back if the provider rejects JSON mode."""
    model = model or MODEL
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        try:
            r = chat(model=model, messages=msgs, response_format={"type": "json_object"})
        except BadRequestError:  # provider/model without JSON mode, or invalid JSON
            r = chat(model=model, messages=msgs)
        try:
            return parse_json(r.choices[0].message.content)
        except (json.JSONDecodeError, TypeError):
            msgs = msgs + [{"role": "user", "content": "Return ONLY valid JSON, no prose."}]
    raise ValueError("Model did not return valid JSON")


def call_text(system: str, user: str, model: str | None = None) -> str:
    r = chat(model=model or MODEL, messages=[{"role": "system", "content": system},
                                             {"role": "user", "content": user}])
    return r.choices[0].message.content or ""


if __name__ == "__main__":  # Step 1 demo: the model alone answers from memory
    print(call_text("You are a research assistant.", "List 5 recent papers on RAG."))
    print("\n^ Now search for each title. Some may not exist. That is why we add tools.")
