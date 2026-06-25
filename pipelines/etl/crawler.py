# pipelines/etl/crawler.py
# NOT OPERATIONAL: This is a placeholder for a future crawler implementation. It is not currently used in the pipeline.
import asyncio
import json
from pathlib import Path
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from pipelines.models import RawDocument
import hashlib, httpx

SKIP_PATTERNS = [
    "reddit.com", "twitter.com", "x.com", "https://community.rationalreminder.ca/", "https://rationalreminder.ca/"   # too much noise
    "youtube.com", "youtu.be",
]

async def _check_robots(url: str) -> bool:
    """Return True if crawling is allowed."""
    from urllib.parse import urlparse
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(base)
            # Simple heuristic: if robots.txt blocks /*, skip
            return "Disallow: /" not in r.text
    except Exception:
        return True  # Allow by default if unreachable

async def crawl_links(
    urls: list[str],
    output_dir: Path,
    max_concurrent: int = 5,
) -> list[RawDocument]:
    # Filter obvious noise
    urls = [u for u in urls if not any(p in u for p in SKIP_PATTERNS)]
    urls = list(set(urls))  # Deduplicate

    semaphore = asyncio.Semaphore(max_concurrent)
    docs = []

    async def _crawl_one(url: str):
        if not await _check_robots(url):
            print(f"[skip] robots.txt blocks: {url}")
            return

        async with semaphore:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(
                    url=url,
                    config=CrawlerRunConfig(output_formats=["markdown"]),
                )
                if result.success and result.markdown:
                    doc = RawDocument(
                        id=hashlib.md5(url.encode()).hexdigest(),
                        source="crawled",
                        url=url,
                        title=result.metadata.get("title", url),
                        content=result.markdown,
                    )
                    docs.append(doc)

    await asyncio.gather(*[_crawl_one(u) for u in urls])

    # Persist to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (output_dir / f"{doc.id}.json").write_text(doc.model_dump_json(indent=2))

    return docs