import asyncio
import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

MAX_CHARS = 5000
HEADERS = {"User-Agent": "Mozilla/5.0"}

# DuckDuckGo rate-limits aggressively, so cap parallel searches
_search_sem = asyncio.Semaphore(2)


def _search_sync(query: str):
    return DDGS().text(query, max_results=5)


async def search(query: str) -> str:
    try:
        async with _search_sem:
            # ddgs is blocking, so run it in a worker thread
            results = await asyncio.to_thread(_search_sync, query)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


async def fetch(url: str, http_client: httpx.AsyncClient) -> str:
    try:
        r = await http_client.get(
            url, timeout=10, headers=HEADERS, follow_redirects=True
        )
        r.raise_for_status()
    except Exception as e:
        return f"Fetch failed: {e}"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:MAX_CHARS]


TOOLS = [
    {
        "name": "search",
        "description": "Search the web. Returns titles, URLs, and snippets.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch",
        "description": "Download a web page and return its text content.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]


async def run_tool(name: str, args: dict, http_client: httpx.AsyncClient) -> str:
    if name == "search":
        return await search(args["query"])
    if name == "fetch":
        return await fetch(args["url"], http_client)
    return f"Unknown tool: {name}"
