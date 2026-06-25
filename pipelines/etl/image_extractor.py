import httpx
from pathlib import Path
from tqdm import tqdm
from pipelines.models import RawDocument
from pipelines.utils.image_utils import (
    extract_image_urls_from_markdown,
    remove_image_links,
    safe_filename,
    url_to_filename,
)
from pipelines.utils.pdf_utils import (
    extract_pdf_urls_from_markdown,
    remove_pdf_links,
)


def _download_asset(url: str, target_path: Path, timeout: float = 30.0) -> bool:
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            target_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def _unique_target_path(target_dir: Path, filename: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def download_images_for_doc(
    doc: RawDocument,
    images_dir: Path = Path("data/raw/images"),
) -> list[dict]:
    urls = extract_image_urls_from_markdown(doc.content)
    if not urls:
        return []

    doc_name = safe_filename(doc.title or doc.id)
    target_dir = images_dir / doc_name
    downloaded = []

    for idx, url in enumerate(tqdm(urls, desc=f"Downloading images for {doc.title}", unit="file"), start=1):
        filename = url_to_filename(url, default_name=f"image_{idx}")
        path = _unique_target_path(target_dir, filename)
        success = _download_asset(url, path)
        downloaded.append({
            "url": url,
            "path": str(path),
            "downloaded": success,
        })
    return downloaded


def _remove_media_links(text: str) -> str:
    text = remove_image_links(text)
    text = remove_pdf_links(text)
    return text


def extract_images_from_documents(docs: list[RawDocument]) -> list[RawDocument]:
    for doc in docs:
        image_urls = extract_image_urls_from_markdown(doc.content)
        pdf_urls = extract_pdf_urls_from_markdown(doc.content)

        doc.image_urls = image_urls
        doc.metadata["image_urls"] = image_urls
        doc.pdf_urls = pdf_urls
        doc.metadata["pdf_urls"] = pdf_urls

        doc.content = _remove_media_links(doc.content)

    return docs


def download_images_for_documents(
    docs: list[RawDocument],
    images_dir: Path = Path("data/raw/images"),
) -> list[dict]:
    all_downloads = []
    for doc in docs:
        downloads = download_images_for_doc(doc, images_dir=images_dir)
        all_downloads.extend(downloads)
    return all_downloads
