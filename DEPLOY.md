# Hosting the web UI for a class (free, ~10 min)

Students just open a link: no Python, no venv, no API key. Uses **Streamlit Community Cloud** (free).

1. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app → Deploy a public app from GitHub**.
2. Repository: this repo · Branch: the one with this file · Main file path: `app.py`.
3. **Advanced settings** → Python **3.12** → paste into **Secrets**:

   ```toml
   API_KEY = "gsk_..."                      # your provider key (never commit it)
   BASE_URL = "https://api.groq.com/openai/v1"
   MODEL = "openai/gpt-oss-120b"
   SMALL_MODEL = "openai/gpt-oss-20b,openai/gpt-oss-120b"
   CRITIC_MODEL = "openai/gpt-oss-20b"
   REASONING_EFFORT = "openai/gpt-oss-20b=low,openai/gpt-oss-120b=low"
   SHORTLIST_K = "6"
   MEMORY_BACKEND = "json"                  # skip Chroma: the free box has ~1 GB RAM
   ACCESS_CODE = "pick-a-word"              # give this to students; blocks strangers
   DEMO_UNTIL = "2026-12-13"                # app shows "demo has ended" after this date
   ```

4. **Deploy**. First build takes 3–5 min. Share the `https://<name>.streamlit.app` URL + access code.

**During the demo window.** Edit secrets any time (⋮ → Settings → Secrets; the app restarts). An app with no visitors
for a while goes to sleep; the next visitor clicks "wake up" and waits ~1 min.

**Rate limits.** Everyone shares one key. Groq's free tier is ~8k tokens/min per model, so with a whole class
clicking at once runs will queue (`rate limited, retrying`) for minutes. Either demo from one screen, have
students take turns, or put a paid key / Gemini key in the secrets for the demo days.

**After the demo.** ⋮ → **Delete app** on share.streamlit.io, and revoke the API key at your provider.
`DEMO_UNTIL` already stops it being used if you forget.

Saved runs (`runs/`) and notes live on the container's disk and vanish on restart; students use **Download report**.
