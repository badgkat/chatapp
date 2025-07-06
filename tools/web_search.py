"""
tools/web_search.py
DuckDuckGo text search. Returns a formatted block of the top hits.
"""
from typing import Dict
from duckduckgo_search import DDGS


def run(input: Dict) -> str:
    query = (input or {}).get("query") or input.get("q") or ""
    if not query:
        raise ValueError("web-search requires 'query' field")

    ddgs = DDGS()
    results = ddgs.text(query, max_results=5)

    lines = []
    for r in results:
        title = r.get("title", "").strip()
        body  = r.get("body", "").strip()
        href  = r.get("href", "").strip()
        lines.append(f"- {title}\n  {body}\n  {href}")

    return "\n\n".join(lines)
