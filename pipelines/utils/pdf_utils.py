import re
from urllib.parse import urlparse, unquote

PDF_MARKDOWN_RE = re.compile(
    r'\[[^\]]+\]\((https?://[^\s)]+\.pdf(?:\?[^\s)]*)?)\)',
    re.IGNORECASE,
)
BARE_PDF_URL_RE = re.compile(
    r'^(https?://[^\s]+\.pdf(?:\?[^\s]*)?)$',
    re.IGNORECASE | re.MULTILINE,
)
NOTION_S3_PDF_URL_RE = re.compile(
    r'(https?://prod-files-secure\.s3\.us-west-2\.amazonaws\.com/[^\s)]+\.pdf(?:\?[^\s)]*)?)',
    re.IGNORECASE,
)


def _is_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    path = unquote(parsed.path or "").lower()
    query = unquote(parsed.query or "").lower()
    return path.endswith(".pdf") or query.endswith(".pdf")


def extract_pdf_urls_from_markdown(text: str) -> list[str]:
    urls = []
    urls.extend(PDF_MARKDOWN_RE.findall(text))
    urls.extend(NOTION_S3_PDF_URL_RE.findall(text))
    urls.extend(BARE_PDF_URL_RE.findall(text))
    return [u for u in dict.fromkeys(urls) if _is_pdf_url(u)]


def _clean_blank_lines(text: str) -> str:
    cleaned = re.sub(r"\n\s*\n+", "\n\n", text)
    return cleaned.strip()


def remove_pdf_links(text: str) -> str:
    text = PDF_MARKDOWN_RE.sub("", text)
    text = NOTION_S3_PDF_URL_RE.sub("", text)
    text = BARE_PDF_URL_RE.sub("", text)
    return _clean_blank_lines(text)
