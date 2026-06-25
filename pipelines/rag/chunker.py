# pipelines/rag/chunker.py
import re
import ollama
from pipelines.models import RawDocument, Chunk
import hashlib

# Local LLM used to generate a short context prefix for each chunk
# (Anthropic-style "contextual retrieval"). Disabled by default in
# scripts/run_rag.py because it costs one Ollama generate per chunk.
CONTEXT_PROMPT = """You are a helpful assistant specialized in summarizing documents relative to a given chunk.
<document>
{content}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a succinct context of maximum 150 characters to situate this chunk for retrieval purposes. Reply with ONLY the context string.
"""


# ── Sentence tokenization ───────────────────────────────────────────────────
# nupunkt is a lightweight, no-dep modern sentence boundary detector
# (Cython port inspired by pysbd). Legal-text tuned, which actually helps
# second-brain notes (fewer false breaks on abbreviations like "Inc.", "e.g.").
from nupunkt import sent_tokenize


# ── Markdown structural extractors ──────────────────────────────────────────
# Code fences and tables are extracted as atomic units and re-inserted
# verbatim after chunking, so we never split a code block or table row in
# the middle. Lists and blockquotes are treated as normal prose.

# We build the fence regex from a string variable (not a raw literal) so the
# triple-backtick sequence never confuses the Python parser.
_FENCE_PATTERN = (
    r"(^```.*?\n.*?^```\s*$|^`{3,}.*?`{3,})"
)
_FENCE_RE = re.compile(_FENCE_PATTERN, re.MULTILINE | re.DOTALL)
_TABLE_RE = re.compile(
    r"(^\|.*\|\s*\n\|[-:|\s]+\|\s*\n(?:^\|.*\|\s*\n?)+)",
    re.MULTILINE,
)


def _extract_atomic_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Pull out code fences and tables, replace each with a sentinel.

    Returns the sanitized text (sentinels in place of blocks) and a map from
    sentinel -> original block. Sentence tokenization then runs on the
    sanitized text, so atomic blocks can never be split.
    """
    store: dict[str, str] = {}

    def _stash(match: re.Match) -> str:
        sentinel = f"@@ATOMIC{len(store):04d}@@"
        store[sentinel] = match.group(0)
        return sentinel

    # Order matters: fences first, then tables (tables can contain pipes
    # that confuse the table regex if fences aren't removed first).
    sanitized = _FENCE_RE.sub(_stash, text)
    sanitized = _TABLE_RE.sub(_stash, sanitized)
    return sanitized, store


# ── Section splitter ────────────────────────────────────────────────────────
# Splits markdown by heading lines (H1/H2/H3) and carries heading context
# forward so every chunk can be prefixed with its breadcrumb path
# (e.g. "H1 > H2 > H3"). This makes "find the section about X" queries
# land on the right chunk.

def _split_markdown_sections(md: str) -> list[tuple[str, str]]:
    """Split markdown into sections and carry heading context.

    Returns list of (heading_context, section_text).
    heading_context is a short string like: "H1 > H2 > H3".
    """
    h1 = h2 = h3 = ""
    sections: list[tuple[str, list[str]]] = []
    cur_lines: list[str] = []

    def ctx() -> str:
        parts = [p for p in (h1, h2, h3) if p]
        return " > ".join(parts)

    def flush() -> None:
        nonlocal cur_lines
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append((ctx(), cur_lines))
        cur_lines = []

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            title = s[level:].strip()
            if title:
                flush()
                if level == 1:
                    h1, h2, h3 = title, "", ""
                elif level == 2:
                    h2, h3 = title, ""
                else:
                    h3 = title
                cur_lines = [line]
            continue
        cur_lines.append(line)

    flush()
    return [(c, "\n".join(lines).strip()) for c, lines in sections]


# ── Sentence-aware packing ──────────────────────────────────────────────────
# Greedily pack sentences into chunks targeting `size_words` words, with
# `overlap_sentences` trailing sentences carried into the next chunk for
# context continuity. Sentences that are exactly a sentinel are emitted as
# their own chunk (we never merge a code block into prose). Atomic blocks
# that exceed `size_words` on their own are emitted as a single chunk; if
# they are massive, they get split on their internal blank lines as a
# best-effort fallback.

def _count_words(s: str) -> int:
    return len(s.split())


def _pack_sentences(
    sentences: list[str],
    atomic_store: dict[str, str],
    size_words: int,
    overlap_sentences: int,
) -> list[str]:
    """Pack sentences (and atomic sentinels) into word-budgeted chunks."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    overlap_buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer, buffer_words, overlap_buffer
        if buffer:
            chunks.append("\n\n".join(buffer).strip())
            # Keep last N sentences as overlap for the next chunk.
            overlap_buffer = (
                buffer[-overlap_sentences:] if overlap_sentences > 0 else []
            )
            buffer = []
            buffer_words = 0

    for sent in sentences:
        # An atomic sentinel becomes its own chunk so a code block or
        # table is never merged into surrounding prose. The stored
        # markdown is split on blank lines if it exceeds the budget.
        if sent in atomic_store:
            flush_buffer()
            block = atomic_store[sent]
            if _count_words(block) <= size_words:
                chunks.append(block)
            else:
                # Best-effort split on blank lines for oversized blocks.
                sub_blocks = re.split(r"\n\s*\n+", block)
                for sub in sub_blocks:
                    sub = sub.strip()
                    if sub:
                        chunks.append(sub)
            overlap_buffer = []
            continue

        sent_words = _count_words(sent)

        # If a single sentence is bigger than the budget, emit it alone
        # rather than splitting mid-sentence.
        if sent_words >= size_words:
            flush_buffer()
            chunks.append(sent)
            overlap_buffer = []
            continue

        # Start a new chunk with overlap if the buffer is full.
        if buffer_words + sent_words > size_words and buffer:
            flush_buffer()
            if overlap_buffer:
                buffer.extend(overlap_buffer)
                buffer_words = sum(_count_words(s) for s in buffer)

        buffer.append(sent)
        buffer_words += sent_words

    flush_buffer()
    return [c for c in chunks if c]


def _chunk_section(
    section_text: str,
    size_words: int,
    overlap_sentences: int,
) -> list[str]:
    """Tokenize a section into sentences and pack into chunks."""
    sanitized, atomic_store = _extract_atomic_blocks(section_text)
    # nupunkt needs real text; sentinels contain '@' which is safe for it.
    sentences = sent_tokenize(sanitized)
    return _pack_sentences(sentences, atomic_store, size_words, overlap_sentences)


# ── Public entry point ──────────────────────────────────────────────────────

def chunk_document(
    doc: RawDocument,
    model: str = "qwen3.5:9b",
    add_context: bool = True,
    size_words: int = 512,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Chunk a document into retrieval-ready chunks.

    Strategy:
      1. Split by markdown headings (H1/H2/H3); each chunk is prefixed with
         its breadcrumb heading context.
      2. Within each section, extract code fences + tables as atomic units.
      3. Sentence-tokenize remaining prose with nupunkt.
      4. Greedily pack sentences into ~`size_words` chunks, carrying
         `overlap_sentences` sentences forward for context continuity.
      5. (Optional) Ask the local LLM for a short context prefix per chunk.
    """
    sections = _split_markdown_sections(doc.content)
    if not sections:
        sections = [("", doc.content)]

    chunks: list[Chunk] = []
    chunker_version = 4
    chunk_index = 0

    for heading_ctx, section_text in sections:
        raw_chunks = _chunk_section(section_text, size_words, overlap_sentences)
        for raw in raw_chunks:
            prefix = f"{heading_ctx}\n\n" if heading_ctx else ""
            raw_with_headings = f"{prefix}{raw}"

            context_prefix = ""
            if add_context:
                # Ask local LLM to situate this chunk within the document.
                prompt = CONTEXT_PROMPT.format(
                    content=doc.content[:3000],  # Doc summary window
                    chunk=raw_with_headings,
                )
                resp = ollama.generate(model=model, prompt=prompt)
                context_prefix = resp["response"].strip()

            # Make chunk IDs content-dependent so re-indexing isn't skipped when
            # chunking strategy changes.
            chunk_id = hashlib.md5(
                f"{doc.id}_{chunk_index}_{raw_with_headings}".encode()
            ).hexdigest()
            chunks.append(Chunk(
                id=chunk_id,
                document_id=doc.id,
                content=raw_with_headings,
                context_prefix=context_prefix,
                metadata={
                    **doc.metadata,
                    "source": doc.source,
                    "title": doc.title,
                    "url": doc.url or "",
                    "chunk_index": chunk_index,
                    "chunker_version": chunker_version,
                },
            ))
            chunk_index += 1
    return chunks