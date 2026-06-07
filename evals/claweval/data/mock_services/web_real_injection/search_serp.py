"""
Search SERP — wraps Serper.dev Google Search API.

GET https://google.serper.dev/search
Headers:
    X-API-KEY: <SERP_DEV_KEY / SERPER_API_KEY>
Query params:
    q:      <query>
    num:    <int, 1-10>

Input:  query (str), timeout (int), num (int), start (int, unused)
Output: {"status": <int>, "output": <list[dict]>}
"""

import os
import re
import requests

SERP_API_URL = "https://google.serper.dev/search"
SERP_DEV_KEY = os.getenv("SERP_DEV_KEY") or os.getenv("SERPER_API_KEY", "")


def _detect_language(query: str) -> str:
    """Return hl param for Serper: 'zh' if Chinese chars detected, else 'en'."""
    if re.search(r"[\u4e00-\u9fff]", query):
        return "zh"
    return "en"


def search_serp(
    query: str,
    timeout: int = 20,
    num: int = 10,
    start: int = 1,
    raw_save_path: str | None = None,
) -> dict:
    """Search Google via Serper.dev and return extracted results.

    Args:
        query: Search query string.
        timeout: Request timeout in seconds.
        num: Number of results (1-10).
        start: 1-based result offset (passed as page param to Serper).

    Returns:
        dict with keys:
            status (int): HTTP status code, or -1 on error.
            output (list[dict]): List of result dicts with keys:
                title, link, snippet, date, query.
    """
    if not SERP_DEV_KEY:
        return {"status": 401, "output": [], "error": "No SERP API key set"}

    hl = _detect_language(query)
    params = {
        "q": query,
        "num": min(max(num, 1), 10),
    }
    if start > 1:
        params["page"] = start

    headers = {
        "X-API-KEY": SERP_DEV_KEY,
    }
    if hl == "zh":
        headers["Accept-Language"] = "zh-CN,zh;q=0.9"

    try:
        resp = requests.get(SERP_API_URL, params=params, headers=headers, timeout=timeout)

        if raw_save_path and resp.status_code == 200:
            os.makedirs(os.path.dirname(raw_save_path) or ".", exist_ok=True)
            with open(raw_save_path, "w", encoding="utf-8") as f:
                f.write(resp.text)

        if resp.status_code != 200:
            return {"status": resp.status_code, "output": [], "error": resp.text[:200]}

        data = resp.json()

        # Serper response format: {"organic": [{"title", "link", "snippet", "date"}, ...]}
        organic_results = data.get("organic", [])
        results = [
            {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "date": item.get("date", ""),
                "query": query,
            }
            for item in organic_results
        ]
        return {"status": resp.status_code, "output": results}

    except Exception as e:
        return {"status": -1, "output": [], "error": str(e)}


if __name__ == "__main__":
    import json

    result = search_serp("DCF估值 无风险利率", num=3)
    print(f"status={result['status']}  count={len(result['output'])}")
    print(json.dumps(result["output"], indent=2, ensure_ascii=False)[:2000])
