# scripts/run_etl.py
import asyncio, os
from pathlib import Path
from pipelines.etl.notion_loader import load_from_files, load_from_notion
from pipelines.etl.crawler import crawl_links

# Toggle: "files" uses pre-exported MDs, "notion" fetches live from API
LOAD_MODE = os.getenv("LOAD_MODE")

def main():
    page_name = os.getenv("ETL_PAGE_NAME")

    if LOAD_MODE == "notion":
        print("Fetching from Notion API...")
        if page_name:
            print(f"  Only fetching page filter: {page_name}")
        notion_docs = load_from_notion(
            Path("data/pages.txt"),
            page_filter=page_name,
        )
    else:
        print("Loading from local .md files...")
        notion_docs = load_from_files(Path("data/raw"))

    print(f"  Loaded {len(notion_docs)} Notion pages")

    # TODO: Wait for now until we have a way to crawl the links
    # all_links = list({url for doc in notion_docs for url in doc.child_urls})
    # print(f"  Crawling {len(all_links)} unique outbound links...")
    # asyncio.run(crawl_links(all_links, Path("data/crawled")))
    print("ETL complete.")

if __name__ == "__main__":
    main()