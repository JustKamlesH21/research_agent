"""Step 2 · Describing tools to the model. Descriptions are prompts."""


def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


_search_props = {
    "query": {"type": "string", "description": "3-6 keywords, not a full sentence"},
    "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
    "year_from": {"type": "integer", "description": "only papers from this year onward, e.g. 2023"},
}

TOOLS = [
    _fn("search_arxiv", "Search arXiv for papers. Use 3-6 keywords, not full sentences.",
        _search_props, ["query"]),
    _fn("search_openalex", "Search OpenAlex (all disciplines, includes journals and citation "
        "counts). Use 3-6 keywords. Good when arXiv results are weak or off-field.",
        _search_props, ["query"]),
    _fn("get_citations", "Citation count, venue and year for ONE exact paper title.",
        {"title": {"type": "string"}}, ["title"]),
    _fn("recall_notes", "Recall notes saved in earlier runs. Call this BEFORE searching.",
        {"query": {"type": "string", "description": "topic keywords"},
         "k": {"type": "integer", "minimum": 1, "maximum": 10}}, ["query"]),
    _fn("save_note", "Save a two-line note about one paper you read, for future runs.",
        {"paper_id": {"type": "string", "description": "the paper's id from search results"},
         "note": {"type": "string", "description": "title (year): key finding"}},
        ["paper_id", "note"]),
]
