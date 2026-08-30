"""
Semantic chunker — replaces the line-based splitter ported from mini_rag.py.

Strategy:
  Pass 1 — code blocks and tables are lifted out whole (unchanged behaviour).
  Pass 2 — the remaining text is divided into heading-delimited sections. Each
           section is split into sentences, the sentences are embedded, and
           chunk boundaries are placed where consecutive sentences are least
           similar to each other (semantic breakpoints).
  Pass 3 — every chunk carries 15-20% of the previous chunk's tail as overlap,
           so a fact that straddles a boundary stays retrievable from both sides.

Why this replaces structure_aware_chunking():
  - The old splitter could only cut BETWEEN lines. PDF text extracted as one
    long line per paragraph became a single enormous chunk (20k chars in, one
    chunk out) which embeds to a meaningless vector.
  - Its size trigger compared characters against `chunk_size * 5`, so a
    configured 400 actually produced ~2000-character chunks.
  - Its overlap was measured in lines and needed >3 lines in the buffer, so
    single-line blocks got no overlap at all.

Sizing contract:
  `chunk_size` caps the NEW content in a chunk. Overlap is added on top, so a
  finished chunk can reach ~1.2x chunk_size. Overlap never crosses a heading,
  table, or code boundary.

Degrades gracefully: with no embedder (or if embedding raises) it falls back to
sentence-packing at the same target size, keeping the overlap behaviour.
"""
import logging
import re
from typing import Optional

import numpy as np

from config import get_settings
from models import Chunk, Document

logger = logging.getLogger("SemanticChunker")
settings = get_settings()

# ─── Pattern definitions ──────────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^#{1,6}\s+\S"),              # Markdown headings
    re.compile(r"^[A-Z][A-Z\s]{3,}:?\s*$"),   # ALL CAPS headings
    re.compile(r"^\d+\.\s+[A-Z].{5,}$"),      # Numbered headings
]

TABLE_ROW_RE   = re.compile(r"^\s*\|.*\|\s*$")
CODE_BLOCK_RE  = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
TABLE_BLOCK_RE = re.compile(
    r"(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)",
    re.MULTILINE,
)

# Sentence boundary: terminal punctuation, optional closing quote/bracket, then space.
SENT_END_RE = re.compile(r'(?<=[.!?])["\')\]]*\s+')

# Trailing tokens that look like a sentence end but are not.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "al",
    "e.g", "i.e", "fig", "no", "inc", "ltd", "co", "approx", "dept", "est",
}


def _is_heading(line: str) -> bool:
    """True when the line opens a new section. Wired to HEADING_PATTERNS so all
    three forms are honoured — the previous version declared numbered headings
    but only ever checked Markdown and ALL CAPS."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in HEADING_PATTERNS)


def _ends_with_abbreviation(fragment: str) -> bool:
    m = re.search(r"([A-Za-z.]+)\.[\"')\]]*\s*$", fragment)
    return bool(m) and m.group(1).lower().rstrip(".") in ABBREVIATIONS


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Break a single over-long sentence on whitespace. This is what makes
    newline-free PDF text splittable at all."""
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    buf: list[str] = []
    size = 0
    for word in text.split(" "):
        # A single token longer than the limit (base64 blob, long URL) is sliced.
        while len(word) > limit:
            if buf:
                out.append(" ".join(buf))
                buf, size = [], 0
            out.append(word[:limit])
            word = word[limit:]
        if buf and size + len(word) + 1 > limit:
            out.append(" ".join(buf))
            buf, size = [], 0
        buf.append(word)
        size += len(word) + 1
    if buf:
        out.append(" ".join(buf))
    return [s for s in out if s]


def split_sentences(text: str, max_len: int) -> list[str]:
    """Split into sentences. Newlines are treated as hard boundaries so bullets
    and short lines stay separate; over-long sentences are wrapped."""
    sentences: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue

        start = 0
        for m in SENT_END_RE.finditer(block):
            fragment = block[start:m.start() + 1]
            if _ends_with_abbreviation(fragment):
                continue
            candidate = block[start:m.start() + 1].strip()
            if candidate:
                sentences.extend(_hard_wrap(candidate, max_len))
            start = m.end()

        tail = block[start:].strip()
        if tail:
            sentences.extend(_hard_wrap(tail, max_len))
    return sentences


class SemanticChunker:
    def __init__(
        self,
        embedder=None,
        chunk_size: int = None,
        min_chunk_chars: int = None,
        overlap_ratio: float = None,
        breakpoint_percentile: int = None,
        buffer_size: int = None,
    ):
        # Pass the shared Embedder singleton — constructing a new one reloads
        # BGE-M3 from disk.
        self.embedder              = embedder
        self.chunk_size            = chunk_size            or settings.chunk_size
        # A minimum that approaches chunk_size makes _merge_small fold every
        # semantic split back together, erasing the boundaries we just found.
        # Cap it at a third of the target regardless of what is configured.
        self.min_chunk_chars       = min(
            min_chunk_chars or settings.min_chunk_chars,
            max(80, int((chunk_size or settings.chunk_size) * 0.33)),
        )
        self.overlap_ratio         = (overlap_ratio if overlap_ratio is not None
                                      else settings.chunk_overlap_ratio)
        self.breakpoint_percentile = (breakpoint_percentile
                                      or settings.semantic_breakpoint_percentile)
        self.buffer_size           = (buffer_size if buffer_size is not None
                                      else settings.semantic_buffer_size)

        # 15-20% band around the configured ratio.
        self.overlap_min = max(0.0, self.overlap_ratio - 0.025)
        self.overlap_max = self.overlap_ratio + 0.025

    # ── Entry point ───────────────────────────────────────────────────────────

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        text = doc.content

        # ── Pass 1: atomic blocks (unchanged) ─────────────────────────────────
        excluded_spans: list[tuple[int, int]] = []

        for m in CODE_BLOCK_RE.finditer(text):
            excluded_spans.append((m.start(), m.end()))
            chunks.append(self._make_chunk(doc, m.group(0).strip(), "code", None))

        for m in TABLE_BLOCK_RE.finditer(text):
            # Skip tables already captured inside a code fence.
            if any(s <= m.start() and m.end() <= e for s, e in excluded_spans):
                continue
            excluded_spans.append((m.start(), m.end()))
            chunks.append(self._make_chunk(doc, m.group(0).strip(), "table", None))

        # Everything outside the excluded spans, with a blank line marking each
        # excision so unrelated text either side is not glued into one sentence.
        excluded_spans.sort()
        remaining_parts: list[str] = []
        cursor = 0
        for start, end in excluded_spans:
            if cursor < start:
                remaining_parts.append(text[cursor:start])
            cursor = max(cursor, end)
        remaining_parts.append(text[cursor:])
        remaining_text = "\n\n".join(remaining_parts)

        # ── Pass 2 + 3: sections → semantic groups → overlap ──────────────────
        for heading, body in self._split_sections(remaining_text):
            for content in self._chunk_section(body):
                chunks.append(self._make_chunk(doc, content, "text", heading))

        return chunks

    # ── Sections ──────────────────────────────────────────────────────────────

    def _split_sections(self, text: str) -> list[tuple[Optional[str], str]]:
        """Cut the document at headings. Each section keeps the heading that
        introduced it; the heading line itself stays out of the body so it is
        not repeated in every chunk."""
        sections: list[tuple[Optional[str], str]] = []
        heading: Optional[str] = None
        buf: list[str] = []

        for line in re.sub(r"\n{3,}", "\n\n", text).split("\n"):
            if _is_heading(line):
                if buf:
                    sections.append((heading, "\n".join(buf)))
                    buf = []
                heading = line.strip()
            else:
                buf.append(line)

        if buf:
            sections.append((heading, "\n".join(buf)))
        return sections

    # ── One section ───────────────────────────────────────────────────────────

    def _chunk_section(self, body: str) -> list[str]:
        sentences = split_sentences(body, self.chunk_size)
        if not sentences:
            return []
        if len(sentences) == 1:
            single = sentences[0].strip()
            return [single] if len(single) > 30 else []

        breakpoints = self._semantic_breakpoints(sentences)
        groups      = self._group(sentences, breakpoints)
        groups      = self._apply_overlap(groups)

        out = []
        for g in groups:
            content = " ".join(g).strip()
            if len(content) > 30:
                out.append(content)
        return out

    def _semantic_breakpoints(self, sentences: list[str]) -> list[int]:
        """Index positions where a new chunk should start, chosen where the
        meaning shifts most. Returns [] when no embedder is available, which
        makes _group fall back to plain size-based packing."""
        if self.embedder is None or len(sentences) < 3:
            return []

        try:
            # Each sentence is embedded together with its neighbours so a single
            # short sentence does not read as a topic change on its own.
            buf = self.buffer_size
            combined = [
                " ".join(sentences[max(0, i - buf): min(len(sentences), i + buf + 1)])
                for i in range(len(sentences))
            ]
            vecs = self.embedder.embed(combined)          # already L2-normalised
            distances = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)
            if distances.size == 0:
                return []
            threshold = float(np.percentile(distances, self.breakpoint_percentile))
            return [i + 1 for i, d in enumerate(distances) if d > threshold]
        except Exception as e:
            logger.warning(f"Semantic breakpoint detection failed ({e}) — falling back to size packing.")
            return []

    def _group(self, sentences: list[str], breakpoints: list[int]) -> list[list[str]]:
        """Cut at the breakpoints, then enforce the size bounds."""
        groups: list[list[str]] = []
        start = 0
        for bp in breakpoints + [len(sentences)]:
            if bp > start:
                groups.append(sentences[start:bp])
                start = bp

        sized: list[list[str]] = []
        for g in groups:
            sized.extend(self._enforce_max(g))
        return self._merge_small(sized)

    def _enforce_max(self, group: list[str]) -> list[list[str]]:
        """Pack sentences until the next one would exceed chunk_size."""
        out: list[list[str]] = []
        buf: list[str] = []
        size = 0
        for s in group:
            if buf and size + len(s) + 1 > self.chunk_size:
                out.append(buf)
                buf, size = [], 0
            buf.append(s)
            size += len(s) + 1
        if buf:
            out.append(buf)
        return out

    def _merge_small(self, groups: list[list[str]]) -> list[list[str]]:
        """Fold undersized groups into a neighbour so a stray sentence does not
        become its own chunk."""
        if len(groups) < 2:
            return groups

        merged: list[list[str]] = []
        for g in groups:
            size = sum(len(s) + 1 for s in g)
            if merged and size < self.min_chunk_chars:
                prev_size = sum(len(s) + 1 for s in merged[-1])
                if prev_size + size <= self.chunk_size:
                    merged[-1].extend(g)
                    continue
            merged.append(g)
        return merged

    def _apply_overlap(self, groups: list[list[str]]) -> list[list[str]]:
        """Prepend whole sentences from the previous chunk's tail so a fact
        sitting on a boundary is retrievable from both sides.

        Overlap is measured against the FINISHED chunk (carried + new content),
        which is the usual convention. Sentences are indivisible, so an exact
        ratio is rarely reachable: we take the sentence count landing closest to
        the target rather than the first one that fits, which is what keeps the
        result inside the 15-20% band instead of undershooting at ~11%.

        Overlap stays inside one section, so it never leaks across a heading,
        table, or code boundary.
        """
        if self.overlap_ratio <= 0 or len(groups) < 2:
            return groups

        out: list[list[str]] = [groups[0]]
        for i in range(1, len(groups)):
            current = groups[i]
            body    = sum(len(s) + 1 for s in current)

            # Never let the carried tail dominate the chunk it is attached to.
            ceiling = body * 0.5

            best: list[str] = []
            best_diff = abs(0.0 - self.overlap_ratio)   # the "carry nothing" option
            in_band: Optional[list[str]] = None
            acc: list[str] = []
            size = 0

            for s in reversed(groups[i - 1]):
                if acc and size + len(s) + 1 > ceiling:
                    break
                acc.insert(0, s)
                size += len(s) + 1

                fraction = size / (body + size)
                # A candidate actually inside the band always wins; whole
                # sentences only sometimes allow one.
                if in_band is None and self.overlap_min <= fraction <= self.overlap_max:
                    in_band = list(acc)
                    break

                diff = abs(fraction - self.overlap_ratio)
                if diff < best_diff:
                    best_diff = diff
                    best = list(acc)
                elif fraction > self.overlap_ratio:
                    break   # past the target and getting worse

            out.append((in_band if in_band is not None else best) + current)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_chunk(self, doc: Document, content: str, chunk_type: str,
                    heading: Optional[str]) -> Chunk:
        return Chunk(
            doc_id     = doc.id,
            source     = doc.source,
            content    = content,
            chunk_type = chunk_type,
            heading    = heading,
            metadata   = {**doc.metadata, "source": doc.source},
        )


# Old name kept so any caller that still imports it keeps working.
StructureAwareChunker = SemanticChunker
