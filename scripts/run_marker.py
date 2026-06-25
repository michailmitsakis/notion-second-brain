# scripts/run_marker.py
"""
Convert PDFs from `data/raw` into markdown under `data/clean`.

- PDFs: convert to markdown using the `marker` library with optional enhancements:
  * GPU acceleration via TORCH_DEVICE=cuda
  * Optional OCR forcing for scanned documents: set MARKER_FORCE_OCR=true
  * Optional LLM improvement (Ollama): set MARKER_USE_LLM=true
  * Workers (parallel conversion): set MARKER_WORKERS=N (default: auto, 5GB VRAM peak per worker)
  * Disable image extraction: set MARKER_DISABLE_IMAGES=true
  
- Image extraction: handled natively by marker and saved alongside markdown
  
VRAM guidance: With 12GB VRAM, use up to 2 workers safely (5GB peak per worker).
  
Usage: python -m scripts.run_marker
  With LLM: MARKER_USE_LLM=true 
  With 2 workers: MARKER_WORKERS=2 
  With force OCR: MARKER_FORCE_OCR=true 
  Disable images: MARKER_DISABLE_IMAGES=false
"""

from pathlib import Path
import os
import shutil
import subprocess
import sys
from tqdm import tqdm
import concurrent.futures
import time

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Options from environment variables
# Parse boolean-like env vars properly so strings like 'false' aren't truthy
FORCE_OCR = os.getenv('MARKER_FORCE_OCR', '').lower() == 'true'
USE_LLM = os.getenv('MARKER_USE_LLM', '').lower() == 'true'
WORKERS = os.getenv('MARKER_WORKERS', '')  # Empty string = auto
DISABLE_IMAGES = os.getenv('MARKER_DISABLE_IMAGES', '').lower() == 'false'

# Which step to run: 'all' (default), 'pdfs', or 'images'
MARKER_STEP = os.getenv('MARKER_STEP', 'images')
# Optional test-subdir filtering for both PDFs and images
TEST_SUBDIR = os.getenv('MARKER_TEST_SUBDIR', '')

from marker.services.ollama import OllamaService

# Verify requested torch device is available; fall back to CPU if not.
requested_torch = os.environ.get('TORCH_DEVICE', 'cuda')
try:
    import torch
    if requested_torch.lower() in ('cuda', 'gpu') and not torch.cuda.is_available():
        print("Warning: TORCH_DEVICE=cuda requested but CUDA is not available in this Python build — falling back to CPU.")
        os.environ['TORCH_DEVICE'] = 'cpu'
except Exception:
    # If torch isn't installed or errors, ensure we use cpu
    if requested_torch.lower() in ('cuda', 'gpu'):
        print("Warning: Torch unavailable or not compiled with CUDA — using CPU device instead.")
        os.environ['TORCH_DEVICE'] = 'cpu'

RAW_DOCS = Path("data/raw/documents")
CLEAN = Path("data/clean")
PDFS_MD = CLEAN / "pdfs_md"
CLEAN_MD = CLEAN / "clean_md"
RAW_IMAGES = Path("data/raw/images")
IMAGES_MD = CLEAN / "images_md"

def _ensure_dirs():
    CLEAN.mkdir(parents=True, exist_ok=True)
    PDFS_MD.mkdir(parents=True, exist_ok=True)
    IMAGES_MD.mkdir(parents=True, exist_ok=True)
    CLEAN_MD.mkdir(parents=True, exist_ok=True)
    
    

def convert_pdfs():
    """Discover PDFs and convert them to markdown, with parallel workers."""
    if not RAW_DOCS.exists():
        print(f"No PDF folder found at {RAW_DOCS}; skipping PDF conversion.")
        return

    # Prefer using Marker Python API when available for better control.
    try:
        from marker.converters.pdf import PdfConverter  # noqa: F401
        from marker.models import create_model_dict  # noqa: F401
        from marker.output import text_from_rendered  # noqa: F401
        marker_api = True
    except Exception:
        marker_api = False

    pdf_files = [p for p in RAW_DOCS.rglob("*.pdf") if p.is_file()]

    # Optional test-subdir filtering (uses global TEST_SUBDIR)
    if TEST_SUBDIR:
        pdf_files = [p for p in pdf_files if TEST_SUBDIR in map(str, p.parts)]

    if not pdf_files:
        print("No PDF files found to process.")
        return

    # Determine number of workers
    max_workers = None
    try:
        if WORKERS:
            max_workers = int(WORKERS)
    except Exception:
        max_workers = None

    # Helper executed in worker processes
    def _process_pdf_file(pdf_path_str: str) -> tuple:
        pdf = Path(pdf_path_str)
        rel = pdf.relative_to(RAW_DOCS)
        out_dir = PDFS_MD / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_md = out_dir / (pdf.stem + ".md")
        if out_md.exists():
            return (str(pdf), 'skipped', 'exists')

        # Try API conversion first
        if marker_api:
            try:
                # Use documented API usage
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict
                from marker.output import text_from_rendered

                converter = PdfConverter(artifact_dict=create_model_dict())
                rendered = converter(str(pdf))
                text, _, images = text_from_rendered(rendered)

                # Write markdown
                out_md.write_text(text or f"# {pdf.stem}\n", encoding="utf-8")

                # Handle images according to DISABLE_IMAGES flag
                if not DISABLE_IMAGES and images:
                    for idx, img in enumerate(images):
                        try:
                            if isinstance(img, dict) and "name" in img and ("bytes" in img or "data" in img):
                                name = img.get("name") or f"{pdf.stem}_img_{idx}.png"
                                data = img.get("bytes") or img.get("data")
                                (out_dir / name).write_bytes(data)
                            elif isinstance(img, (list, tuple)) and len(img) >= 2:
                                name = img[0] or f"{pdf.stem}_img_{idx}.png"
                                data = img[1]
                                (out_dir / name).write_bytes(data)
                            elif isinstance(img, (bytes, bytearray)):
                                name = f"{pdf.stem}_img_{idx}.png"
                                (out_dir / name).write_bytes(img)
                            elif isinstance(img, (str, Path)):
                                src = Path(img)
                                if src.exists():
                                    shutil.copy2(src, out_dir / src.name)
                        except Exception as e:
                            print(f"Warning: failed to write image for {pdf}: {e}")

                flags_used = []
                if USE_LLM:
                    flags_used.append("LLM")
                if FORCE_OCR:
                    flags_used.append("force OCR")
                if DISABLE_IMAGES:
                    flags_used.append("no images")
                flags_str = f" ({', '.join(flags_used)})" if flags_used else ""
                return (str(pdf), 'converted', flags_str)
            except Exception as e:
                # Fall through to CLI fallback
                api_err = str(e)
        else:
            api_err = None

    results = []
    start = time.time()
    if max_workers and max_workers > 1:
        print(f"Starting conversion with up to {max_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(_process_pdf_file, str(p)): p for p in pdf_files}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Converting PDFs"):
                try:
                    res = fut.result()
                    results.append(res)
                    print(res)
                except Exception as e:
                    print(f"Worker failed: {e}")
    else:
        # Single-process (or auto) execution
        print("Starting single-process conversion...")
        for pdf in tqdm(pdf_files, desc="Converting PDFs", unit="pdf"):
            res = _process_pdf_file(str(pdf))
            results.append(res)
            print(res)

    elapsed = time.time() - start
    print(f"Processed {len(results)} files in {elapsed:.1f}s")


def convert_images():
    """Convert images using marker's OCRConverter.
    """
    if not RAW_IMAGES.exists():
        print(f"No images folder found at {RAW_IMAGES}; skipping image conversion.")
        return

    try:
        from marker.converters.ocr import OCRConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        have_ocr = True
    except Exception:
        have_ocr = False

    image_files = [p for p in RAW_IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    # Apply optional test-subdir filter to images as well
    if TEST_SUBDIR:
        image_files = [p for p in image_files if TEST_SUBDIR in map(str, p.parts)]
    if not image_files:
        print("No image files found to process.")
        return

    if have_ocr:
        model_dict = create_model_dict()
        config = {"output_format": "markdown", "keep_chars": False}

        # LLM config (optional): prefer Ollama for local models. Put Ollama
        # settings in the converter config (pdf/ocr code expects service config
        # inside the config dict), and set 'llm_service' to choose Ollama.
        if USE_LLM:
            config['use_llm'] = True
            config['llm_service'] = 'marker.services.ollama.OllamaService'
            ollama_base = os.getenv('OLLAMA_BASE_URL', '')
            ollama_model = os.getenv('OLLAMA_MODEL', '')
            if ollama_base:
                config['ollama_base_url'] = ollama_base
            if ollama_model:
                config['ollama_model'] = ollama_model

        try:
            # If the config specified an explicit llm service class path, pass
            # it as the `llm_service` argument so the converter doesn't fall
            # back to its default (which may be GoogleGeminiService).
            llm_service_arg = None
            if config.get('use_llm') and config.get('llm_service'):
                llm_service_arg = config.get('llm_service')

            if llm_service_arg:
                converter = OCRConverter(artifact_dict=model_dict, config=config, llm_service=llm_service_arg)
            else:
                converter = OCRConverter(artifact_dict=model_dict, config=config)
        except Exception as e:
            # Print the minimal config used for debugging (don't leak secrets)
            debug_cfg = {k: v for k, v in config.items() if 'ollama' in k or k == 'llm_service'}
            print(f"Failed to initialize OCRConverter with config {debug_cfg}: {e}")
            raise

        for img in tqdm(sorted(image_files), desc="Converting images", unit="image"):
            rel = img.relative_to(RAW_IMAGES)
            out_dir = IMAGES_MD / rel.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_md = out_dir / (img.stem + ".md")
            if out_md.exists():
                print(f"Skipping existing image markdown: {out_md}")
                continue
            try:
                rendered = converter(str(img))

                # If OCRJSONOutput is returned, prefer extracting the `html`
                # fragments from each child block and convert to plaintext
                # for Markdown. This avoids saving the full JSON metadata.
                md_text = None
                try:
                    from bs4 import BeautifulSoup

                    pages = getattr(rendered, 'children', None)
                    if pages:
                        lines = []
                        for page in pages:
                            for block in getattr(page, 'children', []) or []:
                                html = getattr(block, 'html', '') or ''
                                text_line = BeautifulSoup(html, 'html.parser').get_text().rstrip('\n')
                                if text_line:
                                    lines.append(text_line)
                        if lines:
                            md_text = '\n\n'.join(lines)
                except Exception as e:
                    print(f"Warning: failed to build markdown from structured render for {img}: {e}")

                # Fallback: try the standard text_from_rendered() (may return JSON)
                if not md_text:
                    try:
                        text, _, images = text_from_rendered(rendered)
                        # If text looks like JSON and we have no html-derived
                        # markdown, try to parse JSON and extract html fields.
                        md_text = text
                    except Exception:
                        md_text = f"# {img.stem}\n"

                out_md.write_text(md_text, encoding="utf-8")
                print(f"Converted image -> MD: {img} -> {out_md}")
            except Exception as e:
                print(f"OCR conversion failed for {img}: {e}; writing markdown referencing original image")
                try:
                    relpath = os.path.relpath(img, start=out_dir)
                except Exception:
                    relpath = str(img)
                out_md.write_text(f"# {img.stem}\n\n![]({relpath})\n", encoding="utf-8")

def main():
    _ensure_dirs()
    # Respect MARKER_STEP to run only PDFs or only images when requested
    if MARKER_STEP == 'pdfs':
        convert_pdfs()
    elif MARKER_STEP == 'images':
        convert_images()
    else:
        convert_pdfs()
        convert_images()
    print("\nMarker conversion step complete.")
    print("\nOptions:")
    print("  - Enable LLM improvements: MARKER_USE_LLM=true python -m scripts.run_marker")
    print("  - Force OCR on all pages: MARKER_FORCE_OCR=true python -m scripts.run_marker")
    print("  - Parallel workers (2-3 safe with 12GB VRAM): MARKER_WORKERS=2 python -m scripts.run_marker")
    print("  - Disable image extraction: MARKER_DISABLE_IMAGES=true python -m scripts.run_marker")
    print("  - Change GPU device: TORCH_DEVICE=cpu python -m scripts.run_marker")
    print("\nExample with multiple options:")
    print("  MARKER_USE_LLM=true MARKER_WORKERS=2 python -m scripts.run_marker")

if __name__ == '__main__':
    main()
