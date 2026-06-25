import re
from pathlib import Path
from urllib.parse import urlparse, unquote

IMAGE_MARKDOWN_RE = re.compile(r'!\[[^\]]*\]\((https?://[^\s)]+)\)')
BARE_IMAGE_URL_RE = re.compile(
    r'^(https?://[^\s]+\.(?:png|jpe?g|gif|webp|bmp|svg|avif))$',
    re.IGNORECASE | re.MULTILINE,
)
NOTION_S3_URL_RE = re.compile(
    r'(https?://prod-files-secure\.s3\.us-west-2\.amazonaws\.com/[^\s)]+)',
    re.IGNORECASE,
)


def _clean_blank_lines(text: str) -> str:
    cleaned = re.sub(r"\n\s*\n+", "\n\n", text)
    return cleaned.strip()


def extract_image_urls_from_markdown(text: str) -> list[str]:
    urls = []
    urls.extend(IMAGE_MARKDOWN_RE.findall(text))
    urls.extend(NOTION_S3_URL_RE.findall(text))
    urls.extend(BARE_IMAGE_URL_RE.findall(text))
    return [u for u in dict.fromkeys(urls) if u]


def remove_image_links(text: str) -> str:
    text = IMAGE_MARKDOWN_RE.sub("", text)
    text = NOTION_S3_URL_RE.sub("", text)
    text = BARE_IMAGE_URL_RE.sub("", text)
    return _clean_blank_lines(text)


def safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\n\r]+', '_', name).strip() or 'file'


def url_to_filename(url: str, default_name: str = "image") -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    basename = Path(path).name
    if basename and "." in basename:
        return safe_filename(basename)
    ext = Path(parsed.path).suffix
    if not ext:
        ext = Path(parsed.query).suffix
    return f"{safe_filename(default_name)}{ext or '.bin'}"
