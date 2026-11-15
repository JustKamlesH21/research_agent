"""Step 5 · Long-term memory of paper notes (Chroma), with a JSON fallback
so the lab still works if Chroma can't install or download its embedding model."""
import json
import os
import re

MEMORY_DIR = os.getenv("MEMORY_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "research_memory")
NOTES_JSON = os.path.join(MEMORY_DIR, "notes.json")
os.makedirs(MEMORY_DIR, exist_ok=True)

try:
    if os.getenv("MEMORY_BACKEND") == "json":   # hosted demo: skip Chroma's 1 GB+ RAM / model download
        raise RuntimeError("MEMORY_BACKEND=json")
    import chromadb
    from chromadb.config import Settings
    _db = chromadb.PersistentClient(path=MEMORY_DIR,
                                    settings=Settings(anonymized_telemetry=False))
    _notes = _db.get_or_create_collection("paper_notes")
except Exception as e:  # noqa: BLE001
    print(f"[memory] Chroma unavailable ({e}); using JSON keyword memory")
    _notes = None


def _fallback(e):
    global _notes
    print(f"[memory] Chroma error ({e}); switching to JSON keyword memory")
    _notes = None


def _load() -> dict:
    try:
        with open(NOTES_JSON, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_note(paper_id: str, note: str) -> str:
    """Save a short note about one paper (keyed by its id) for future runs."""
    data = _load()                      # always mirror to JSON as a readable audit trail
    data[paper_id] = note
    with open(NOTES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    if _notes is not None:
        try:
            _notes.upsert(ids=[paper_id], documents=[note])
        except Exception as e:  # noqa: BLE001
            _fallback(e)
    return "saved"


def recall_notes(query: str, k: int = 3) -> list[str]:
    """Return up to k saved notes most related to the query."""
    k = max(1, min(int(k), 10))
    if _notes is not None:
        try:
            n = _notes.count()
            if n == 0:
                return ["(no notes saved yet)"]
            return _notes.query(query_texts=[query], n_results=min(k, n))["documents"][0]
        except Exception as e:  # noqa: BLE001
            _fallback(e)
    data = _load()
    if not data:
        return ["(no notes saved yet)"]
    words = set(re.findall(r"\w+", query.lower()))
    ranked = sorted(data.values(), key=lambda t: -len(words & set(re.findall(r"\w+", t.lower()))))
    return ranked[:k]


REGISTRY = {"save_note": save_note, "recall_notes": recall_notes}
