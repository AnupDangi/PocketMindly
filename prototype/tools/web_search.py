import asyncio
import aiohttp
import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Dict

class AsyncWebSearchTool:
    """
    Async, privacy-focused web search tool using DuckDuckGo HTML.
    Designed for local LLM summarization.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        self.max_chars_per_page = 2000
        self.fetch_timeout = aiohttp.ClientTimeout(total=4)

    # ---------- SEARCH ----------

    async def search(self, session: aiohttp.ClientSession, query: str, max_results: int = 5):
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        async with session.get(url) as resp:
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        results = []

        for result in soup.find_all("div", class_="result", limit=max_results):
            title = result.find("a", class_="result__a")
            snippet = result.find("a", class_="result__snippet")

            if title and snippet:
                results.append({
                    "title": title.get_text(strip=True),
                    "url": title["href"],
                    "snippet": snippet.get_text(strip=True)
                })

        return results

    # ---------- PAGE FETCH ----------

    async def fetch_page(self, session: aiohttp.ClientSession, result: Dict):
        try:
            async with session.get(result["url"]) as resp:
                if resp.status != 200:
                    return None

                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")

            # Remove junk
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            text = " ".join(text.split())

            if len(text) < 300:
                return None

            return {
                "title": result["title"],
                "url": result["url"],
                "content": text[:self.max_chars_per_page]
            }

        except Exception:
            return None

    # ---------- ORCHESTRATOR ----------

    async def get_context(self, query: str, max_pages: int = 2):
        async with aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.fetch_timeout
        ) as session:

            search_results = await self.search(session, query)
            if not search_results:
                return "No search results found."

            tasks = [
                asyncio.create_task(self.fetch_page(session, r))
                for r in search_results
            ]

            collected = []
            for future in asyncio.as_completed(tasks):
                page = await future
                if page:
                    collected.append(page)
                    if len(collected) >= max_pages:
                        break

            if not collected:
                # fallback to snippets
                return "\n".join(
                    f"{i+1}. {r['title']}: {r['snippet']}"
                    for i, r in enumerate(search_results[:3])
                )

            # Build LLM-friendly context
            context = ""
            for i, page in enumerate(collected):
                context += (
                    f"\nSOURCE {i+1}: {page['title']}\n"
                    f"{page['content']}\n"
                )

            return context.strip()

# ---------- TEST ----------

if __name__ == "__main__":
    tool = AsyncWebSearchTool()
    query = "current price of bitcoin"

    result = asyncio.run(tool.get_context(query))
    print(result)
