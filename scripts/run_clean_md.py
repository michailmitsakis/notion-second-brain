from pathlib import Path
from pipelines.models import RawDocument
from pipelines.etl.notion_loader import load_from_files
from pipelines.etl.image_extractor import extract_images_from_documents
from pipelines.utils.image_utils import safe_filename

# Note: PDF/image outputs from `run_marker.py` are handled separately.
# `run_clean.py` only converts raw Notion/exported markdown from
# `data/raw/raw_md` into cleaned markdown in `data/clean/clean_md`.


def save_cleaned_documents(docs: list[RawDocument], output_dir: Path = Path("data/clean/clean_md")):
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        filename = safe_filename(doc.title or doc.id)
        file_path = output_dir / f"{filename}.md"
        file_path.write_text(doc.content, encoding="utf-8")
    print(f"Saved {len(docs)} cleaned documents to {output_dir}")


def main():
    # Load content from initial sources (Notion/Markdown)
    raw_docs = load_from_files(Path("data/raw"))

    # Process initial documents
    cleaned_docs = extract_images_from_documents(raw_docs)

    # NOTE: PDF-derived markdown and images are produced by `run_marker.py`.
    # By request, we do NOT re-process `data/clean/pdfs_md` or
    # `data/clean/images_md` here. `run_clean.py` will only convert
    # raw/exported markdown under `data/raw` into `data/clean/clean_md`.

    # Save all combined documents
    save_cleaned_documents(cleaned_docs)
    print("Cleaning complete.")

if __name__ == "__main__":
    main()