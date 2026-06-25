# pipelines/etl/notion_loader.py
import re
import hashlib
from pathlib import Path
from notion_client import Client
from notion_to_md import NotionToMarkdown
from pipelines.models import RawDocument
from pipelines.etl.image_extractor import download_images_for_doc
from pipelines.etl.pdf_extractor import download_pdfs_for_doc
from pipelines.utils.image_utils import extract_image_urls_from_markdown
from pipelines.utils.pdf_utils import extract_pdf_urls_from_markdown
import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

LINK_RE = re.compile(r'\[.*?\]\((https?://[^\)]+)\)')

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_page_id(url: str) -> str:
    token = url.strip().rstrip('/').split('/')[-1].split('?')[0]
    raw = token.split('-')[-1]
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"

def _safe_filename(name: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\n\r]+', '_', name).strip()
    return safe or "notion_page"

def _doc_from_markdown(name: str, md_str: str, source_url: str = "") -> RawDocument:
    links = LINK_RE.findall(md_str)
    doc_id = hashlib.md5(name.encode()).hexdigest()
    return RawDocument(
        id=doc_id,
        source="notion",
        url=source_url,
        title=name,
        content=md_str,
        child_urls=links,
    )

# ── Option A: load from pre-exported .md files ───────────────────────────────

def load_from_files(md_dir: Path) -> list[RawDocument]:
    """Use this if you already ran your export script manually."""
    if md_dir.name == "raw" and (md_dir / "raw_md").exists():
        md_dir = md_dir / "raw_md"

    docs = []
    for md_file in md_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        docs.append(_doc_from_markdown(md_file.stem, content))
    return docs

# ── Option B: fetch live from Notion via pages.txt ───────────────────────────

def load_from_notion(
    pages_file: Path = Path("data/pages.txt"),
    page_filter: str | None = None,
    documents_dir: Path = Path("data/raw/documents"),
) -> list[RawDocument]:
    """
    Fetch pages directly from Notion, same format as your existing script.
    Requires NOTION_TO_MD_AUTH_TOKEN in environment / .env
    If page_filter is set, only pages whose name or URL contains the filter are fetched.
    """
    if load_dotenv is not None:
        load_dotenv()

    auth_key = os.getenv("NOTION_TO_MD_AUTH_TOKEN")
    if not auth_key:
        raise EnvironmentError(
            "NOTION_TO_MD_AUTH_TOKEN is not set. "
            "Add it to your environment or a .env file in the repo root."
        )

    notion = Client(auth=auth_key)
    n2m = NotionToMarkdown(notion)

    raw_dir = Path("data/raw")
    raw_md_dir = raw_dir / "raw_md"
    images_dir = raw_dir / "images"
    raw_md_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    with open(pages_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, url = [p.strip() for p in line.split("|", 1)]
            if page_filter is not None:
                normalized_filter = page_filter.lower()
                if normalized_filter not in name.lower() and normalized_filter not in url.lower():
                    continue
            page_id = parse_page_id(url)
            print(f"  Fetching: {name}")
            try:
                md_blocks = n2m.page_to_markdown(page_id)
                md_str = n2m.to_markdown_string(md_blocks).get("parent", "")
                file_name = _safe_filename(name)
                file_path = raw_md_dir / f"{file_name}.md"
                file_path.write_text(md_str, encoding="utf-8")
                print(f"  Saved: {file_path}")

                doc = _doc_from_markdown(name, md_str, source_url=url)
                image_urls = extract_image_urls_from_markdown(md_str)
                doc.image_urls = image_urls
                doc.metadata["image_urls"] = image_urls
                if image_urls:
                    downloaded_outputs = download_images_for_doc(doc, images_dir=images_dir)
                    doc.metadata["image_outputs"] = downloaded_outputs
                    downloaded_count = sum(1 for i in downloaded_outputs if i["downloaded"])
                    print(f"  Downloaded {downloaded_count}/{len(downloaded_outputs)} images")
                else:
                    doc.metadata["image_outputs"] = []

                pdf_urls = extract_pdf_urls_from_markdown(md_str)
                doc.pdf_urls = pdf_urls
                doc.metadata["pdf_urls"] = pdf_urls
                if pdf_urls:
                    pdf_outputs = download_pdfs_for_doc(doc, documents_dir=documents_dir)
                    doc.metadata["pdf_outputs"] = pdf_outputs
                    downloaded_count = sum(1 for i in pdf_outputs if i["downloaded"])
                    print(f"  Downloaded {downloaded_count}/{len(pdf_outputs)} PDFs")
                else:
                    doc.metadata["pdf_outputs"] = []

                docs.append(doc)
            except Exception as e:
                print(f"  ✗ Failed ({name}): {e}")
    return docs