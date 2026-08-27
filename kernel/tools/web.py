"""Core online-search tools: web search, page fetch, arXiv, HuggingFace datasets."""
from __future__ import annotations

import json
import re

import httpx
from langchain_core.tools import tool

TIMEOUT = 30.0
MAX_CHARS = 20_000


@tool
def web_search(query: str, max_results: int = 8) -> str:
    """Search the web. Returns title, URL and snippet per hit."""
    from ddgs import DDGS
    try:
        hits = list(DDGS().text(query, max_results=max_results))
    except Exception as e:  # network/ratelimit
        return f"ERROR: search failed: {type(e).__name__}: {e}"
    if not hits:
        return "(no results)"
    return "\n\n".join(f"{h.get('title','')}\n{h.get('href','')}\n{h.get('body','')[:400]}"
                       for h in hits)


@tool
def fetch_url(url: str, max_chars: int = MAX_CHARS) -> str:
    """Fetch a URL and return its main text content as markdown-ish plain text."""
    try:
        r = httpx.get(url, timeout=TIMEOUT, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (AutoResearcher)"})
        r.raise_for_status()
    except Exception as e:
        return f"ERROR: fetch failed: {type(e).__name__}: {e}"
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype:
        return r.text[:max_chars]
    import trafilatura
    text = trafilatura.extract(r.text, include_links=True, include_tables=True)
    return (text or r.text)[:max_chars]


@tool
def arxiv_search(query: str, max_results: int = 8) -> str:
    """Search arXiv for papers. Returns title, id, date and abstract."""
    import re
    import xml.etree.ElementTree as ET
    try:
        # https, and redirects followed: the http endpoint now answers 301, which
        # httpx does not follow by default, so every search failed on the redirect.
        r = httpx.get("https://export.arxiv.org/api/query", timeout=TIMEOUT,
                      follow_redirects=True,
                      params={"search_query": f"all:{query}", "max_results": max_results,
                              "sortBy": "relevance"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        return f"ERROR: arxiv failed: {type(e).__name__}: {e}"
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        title = re.sub(r"\s+", " ", (e.findtext("a:title", "", ns) or "")).strip()
        summary = re.sub(r"\s+", " ", (e.findtext("a:summary", "", ns) or "")).strip()
        out.append(f"{title}\n{e.findtext('a:id','',ns)}  ({e.findtext('a:published','',ns)[:10]})\n{summary[:700]}")
    return "\n\n".join(out) or "(no results)"


@tool
def hf_search(query: str, kind: str = "dataset", limit: int = 15) -> str:
    """Search HuggingFace for datasets or models. `kind` is 'dataset' or 'model'.

    The Hub matches `search` against repo ids, not meaning, so a description of
    what you want ("egocentric navigation video with camera poses") matches no
    repo name and comes back empty. The phrase is tried first and then its
    individual words, so a natural-language query still returns something.
    """
    path = "datasets" if kind.startswith("data") else "models"
    words = [w for w in re.split(r"[^A-Za-z0-9_.-]+", query) if len(w) > 2]
    attempts = [query] + [w for w in words if w.lower() != query.lower()][:6]
    per = max(3, limit // max(1, len(attempts) - 1)) if len(attempts) > 1 else limit
    groups: list[tuple[str, list[dict]]] = []
    seen: set[str] = set()
    for q in attempts:
        try:
            r = httpx.get(f"https://huggingface.co/api/{path}", timeout=TIMEOUT, follow_redirects=True,
                          params={"search": q, "limit": limit, "full": "false",
                                  "sort": "downloads", "direction": -1})
            r.raise_for_status()
            items = r.json()
        except Exception as e:
            if groups:
                break
            return f"ERROR: hf search failed: {type(e).__name__}: {e}"
        fresh = [it for it in items if it.get("id") not in seen]
        if q == query and fresh:            # the phrase itself matched: nothing to widen
            seen.update(it.get("id") for it in fresh)
            return _hf_rows(fresh[:limit])
        fresh = fresh[:per]
        seen.update(it.get("id") for it in fresh)
        if fresh:
            groups.append((q, fresh))
    if not groups:
        return (f"(no results for {query!r}, nor for any single word in it. "
                f"Hub search matches repo ids, so try one concrete word.)")
    out = [f"(no repo id matches {query!r}. Hub search matches repo ids, not meaning, "
           f"so each word was searched separately.)"]
    for q, items in groups:
        out.append(f"\n# {q}")
        out.append(_hf_rows(items))
    return "\n".join(out)[:MAX_CHARS]


def _hf_rows(items: list[dict]) -> str:
    return "\n".join(
        f"{it.get('id')}  downloads={it.get('downloads', 0)}  likes={it.get('likes', 0)}  "
        f"tags={','.join((it.get('tags') or [])[:6])}" for it in items)


@tool
def hf_info(repo_id: str, kind: str = "dataset") -> str:
    """Fetch the metadata and file listing of a HuggingFace dataset or model repo."""
    path = "datasets" if kind.startswith("data") else "models"
    try:
        r = httpx.get(f"https://huggingface.co/api/{path}/{repo_id}", timeout=TIMEOUT, follow_redirects=True,
                      params={"full": "true"})
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        return f"ERROR: hf info failed: {type(e).__name__}: {e}"
    files = [s.get("rfilename") for s in (d.get("siblings") or [])][:120]
    return json.dumps({"id": d.get("id"), "downloads": d.get("downloads"),
                       "tags": d.get("tags"), "description": (d.get("cardData") or {}).get("pretty_name"),
                       "n_files": len(d.get("siblings") or []), "files": files}, indent=1)[:MAX_CHARS]


TOOLS = [web_search, fetch_url, arxiv_search, hf_search, hf_info]
