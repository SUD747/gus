"""Split files into retrievable chunks: tree-sitter for code, headings for markdown, pages for PDFs."""

import importlib
import re
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 60  # a chunk bigger than this is split further
PDF_LINES = 25  # prose lines are long; smaller windows keep PDF hits precise
MIN_LINES = 4  # nodes shorter than this (imports, constants) get merged with neighbours

# extension -> (grammar module, language function)
GRAMMARS = {
    ".py": ("tree_sitter_python", "language"),
    ".js": ("tree_sitter_javascript", "language"),
    ".jsx": ("tree_sitter_javascript", "language"),
    ".mjs": ("tree_sitter_javascript", "language"),
    ".cjs": ("tree_sitter_javascript", "language"),
    ".ts": ("tree_sitter_typescript", "language_typescript"),
    ".tsx": ("tree_sitter_typescript", "language_tsx"),
    ".go": ("tree_sitter_go", "language"),
    ".rs": ("tree_sitter_rust", "language"),
    ".java": ("tree_sitter_java", "language"),
    ".c": ("tree_sitter_c", "language"),
    ".h": ("tree_sitter_c", "language"),
    ".cc": ("tree_sitter_cpp", "language"),
    ".cpp": ("tree_sitter_cpp", "language"),
    ".hpp": ("tree_sitter_cpp", "language"),
    ".rb": ("tree_sitter_ruby", "language"),
}
DEFINITION = re.compile(r"function|class|method|struct|impl|interface|trait|enum|module|decorated|type_|namespace")
MD_IMAGE = re.compile(r"!\[[^\]]*\]\(|<img\b", re.I)


@dataclass
class Chunk:
    line_start: int  # 1-based, inclusive; page number for PDFs
    line_end: int
    kind: str
    text: str
    images: int = 0
    context: str = ""  # enclosing signatures, e.g. "class DigestAuth(Auth):"; embedded, not shown


_parsers = {}


def _parser(ext):
    if ext not in _parsers:
        from tree_sitter import Language, Parser

        module, fn = GRAMMARS[ext]
        _parsers[ext] = Parser(Language(getattr(importlib.import_module(module), fn)()))
    return _parsers[ext]


def chunk_file(path: Path, rel: str) -> list[Chunk]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _pdf(path)
    text = path.read_bytes().decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        return []
    root = None
    if ext in GRAMMARS:
        root = _parser(ext).parse(text.encode()).root_node
        spans = _code_spans(root.named_children, lines)
    elif ext in (".md", ".mdx", ".markdown"):
        spans = _heading_spans(lines)
    else:
        spans = _windows(0, len(lines) - 1, "text")
    chunks = []
    for start, end, kind in spans:
        body = "\n".join(lines[start : end + 1])
        if body.strip():
            ctx = _context(root, start, lines) if root else ""
            chunks.append(Chunk(start + 1, end + 1, kind, body, len(MD_IMAGE.findall(body)), ctx))
    return chunks


def _context(root, row, lines):
    """Signature lines of the definitions enclosing `row` (so a split-out method still knows its class)."""
    heads, node = [], root.named_descendant_for_point_range((row, 0), (row, 0))
    while node is not None and node.parent is not None:  # the root (python's "module") isn't a definition
        if node.start_point[0] < row and DEFINITION.search(node.type):
            heads.append(lines[node.start_point[0]].strip())
        node = node.parent
    return " > ".join(reversed(heads))


def _rows(node):
    end = node.end_point[0]
    if node.end_point[1] == 0 and end > node.start_point[0]:
        end -= 1
    return node.start_point[0], end


def _windows(start, end, kind, size=MAX_LINES):
    return [(s, min(s + size - 1, end), kind) for s in range(start, end + 1, size)]


def _code_spans(nodes, lines):
    """Definitions become their own chunk; small glue (imports, constants) merges; oversized nodes recurse."""
    spans, glue, comments = [], None, None  # glue/comments: [start, end]
    prev_end = -1

    def flush():
        nonlocal glue
        if glue:
            spans.extend(_windows(glue[0], glue[1], "block"))
        glue = None

    for node in nodes:
        start, end = _rows(node)
        prev_end, same_line = end, start == prev_end
        if "comment" in node.type:
            if same_line:  # trailing `x = 1  # note`: already inside the previous span
                continue
            # doc comments stick to whatever follows them
            comments = [comments[0] if comments else start, end]
            continue
        if comments and start - comments[1] > 1:  # blank line in between: not a doc comment
            glue = [glue[0] if glue else comments[0], comments[1]]
            comments = None
        if DEFINITION.search(node.type) or end - start + 1 >= MIN_LINES:
            flush()
            if comments:
                start, comments = comments[0], None
            if spans and start <= spans[-1][1]:  # starts on the line the last span ends: one construct, e.g. a
                start = min(start, spans.pop()[0])  # signature's name, parameters and return type before its body
            if end - start + 1 <= MAX_LINES:
                spans.append((start, end, node.type))
            else:
                spans.extend(_split_big(node, start, end, lines))
        else:
            if comments:
                start, comments = comments[0], None
            if not glue and spans and start <= spans[-1][1]:  # same line as the last span: extend it
                spans[-1] = (spans[-1][0], max(spans[-1][1], end), spans[-1][2])
                continue
            # ponytail: a group of small statements is cut into fixed windows, which can split a statement. Closing the
            # group before the statement that overflows it was measured on eval dev: it moved doc-heavy type chunks
            # above implementations and lowered MRR, so the fixed windows stay
            glue = [glue[0] if glue else start, end]
            if glue[1] - glue[0] + 1 >= MAX_LINES:
                flush()
    if comments:
        glue = [glue[0] if glue else comments[0], comments[1]]
    flush()
    return spans


def _split_big(node, start, end, lines):
    inner = _code_spans(node.named_children, lines)  # class body -> methods, impl block -> fns, ...
    if not inner:
        return _windows(start, end, node.type)
    # the header (signature, decorators) goes with the first member, the closing line with the last
    inner[0] = (start, inner[0][1], node.type)
    if len(inner) > 1 and inner[0][1] - start + 1 < MIN_LINES:
        inner[1] = (start, inner[1][1], node.type)
        inner.pop(0)
    inner[-1] = (inner[-1][0], max(inner[-1][1], end), inner[-1][2])
    return inner


def _heading_spans(lines):
    heads, fenced = [], False
    for i, line in enumerate(lines):
        fenced ^= line.lstrip().startswith(("```", "~~~"))
        if line.startswith("#") and not fenced:
            heads.append(i)
    heads = heads or [0]
    if heads[0] != 0:
        heads.insert(0, 0)
    bounds = heads + [len(lines)]
    return [w for a, b in zip(bounds, bounds[1:], strict=False) for w in _windows(a, b - 1, "section")]


def _pdf(path):
    import pdfplumber

    chunks = []
    with pdfplumber.open(path) as pdf:
        for n, page in enumerate(pdf.pages, 1):
            lines = (page.extract_text(x_tolerance=1.5) or "").splitlines()  # default 3 glues tight-kerned words
            if page.images and not "".join(lines).strip():  # a scan or full-page figure: flag it, don't drop it
                lines = [f"[page {n} has no extractable text, only images (a scanned page or a figure)]"]
            for s, e, _ in _windows(0, len(lines) - 1, "pdf_page", PDF_LINES):
                body = "\n".join(lines[s : e + 1])
                if body.strip():
                    chunks.append(Chunk(n, n, "pdf_page", body, len(page.images)))
    return chunks
