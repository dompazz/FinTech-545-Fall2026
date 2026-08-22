#!/usr/bin/env python3
"""Assemble the nine weekly lecture notes into one book.

    cd book && python3 make_book.py && quarto render book.qmd --to pdf

Or from the repo root: `make book`.

The weekly `.qmd` files are the single source of truth. This script never edits
them -- it reads them, transforms a copy in memory, and writes `book.qmd`. Edit a
week and rebuild, and the book picks the change up. Do not edit `book.qmd` by
hand; it is generated and will be overwritten.

What the transform does to each week:

1. Strips the YAML front matter, keeping the `title:` for the chapter heading.
2. Demotes every heading one level, so a week's `#` section becomes a `##`
   inside its chapter. Lines inside `$$ ... $$` are left alone.
3. Namespaces every cross-reference label and reference with the week number,
   so `{#tbl-moments}` becomes `{#tbl-w01-moments}`. The weeks currently have no
   colliding labels, and this keeps it that way as they grow.
4. Rewrites figure paths from `figures/x.png` to `../Week0N/figures/x.png`.

It then checks that every label is unique across the whole book and that every
reference resolves, and refuses to write the file if either fails.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEEKS = [f"{n:02d}" for n in range(1, 10)]
OUT = HERE / "book.qmd"

# Used on a raw week, where a label name has no hyphens yet.
LABEL_RE = re.compile(r"\{#(tbl|fig|sec)-([A-Za-z0-9]+)")
REF_RE = re.compile(r"@(tbl|fig|sec)-([A-Za-z0-9]+)")
# Used on the assembled book, where names carry the wNN- prefix.
LABEL_ANY = re.compile(r"\{#(tbl|fig|sec)-([A-Za-z0-9][A-Za-z0-9-]*)")
REF_ANY = re.compile(r"@(tbl|fig|sec)-([A-Za-z0-9][A-Za-z0-9-]*)")
HEADING_RE = re.compile(r"^(#{1,5}) (?=\S)")
FIGPATH_RE = re.compile(r"\]\(figures/")

BOOK_YAML = r"""---
title: "Quantitative Risk Management"
subtitle: "FinTech 545 --- Lecture Notes"
author: "Dominic Pazzula"
date: today
date-format: "MMMM YYYY"
format:
  pdf:
    output-file: "FinTech 545 - Quantitative Risk Management.pdf"
    documentclass: scrreprt
    pdf-engine: lualatex
    geometry: margin=1in
    fig-pos: 'H'
    include-in-header:
      text: |
        \usepackage{float}
        \usepackage{needspace}
        \usepackage{etoolbox}
        % Tables here are all short. Without this, longtable can place its
        % header at the bottom of a page, break, and repeat the header on the
        % next page with no rows under it. Require room for a few lines first.
        \AtBeginEnvironment{longtable}{\Needspace*{8\baselineskip}}
        % A long file name in a code span has no break points, so TeX runs it
        % into the margin rather than moving it. Make the underscore a legal
        % break so names like model_based_simulation_with_copulas.jl can wrap.
        \newcommand{\breakableunderscore}{\char`\_\allowbreak}
        \let\origtexttt\texttt
        \renewcommand{\texttt}[1]{{\ttfamily\let\_\breakableunderscore #1}}
    number-sections: true
    number-depth: 3
    toc: true
    toc-depth: 2
    lof: false
    lot: false
    colorlinks: true
  html:
    output-file: "FinTech 545 - Quantitative Risk Management.html"
    toc: true
    toc-depth: 2
    html-math-method: katex
---

<!--
GENERATED FILE -- do not edit.
Built by make_book.py from the nine Week0N/week0N.qmd files.
Edit those, then rebuild with `make book` from the repo root.
-->

# Preface {.unnumbered}

These are the lecture notes for FinTech 545, Quantitative Risk Management. The
nine chapters correspond to the nine lectures of the course and are meant to be
read in order. Each one assumes the machinery built in the ones before it.

"""


def read_week(num: str):
    """Return (title, transformed body) for one week."""
    src = ROOT / f"Week{num}" / f"week{num}.qmd"
    if not src.exists():
        sys.exit(f"missing source file: {src}")
    text = src.read_text(encoding="utf-8")

    # --- split off the YAML front matter, keep the title -------------------
    if not text.startswith("---"):
        sys.exit(f"{src} does not start with YAML front matter")
    end = text.index("\n---", 3)
    front, body = text[:end], text[end + 4:]
    m = re.search(r'^title:\s*"(.+)"\s*$', front, re.M)
    if not m:
        sys.exit(f"{src} has no title: in its YAML")
    title = m.group(1)

    # --- demote headings, skipping display math ---------------------------
    out, in_math = [], False
    for line in body.split("\n"):
        if line.strip().startswith("$$"):
            # a line that is exactly $$ toggles the block; $$...$$ on one line
            # opens and closes, so only toggle when the count of $$ is odd
            if line.strip().count("$$") % 2 == 1:
                in_math = not in_math
            out.append(line)
            continue
        if not in_math and HEADING_RE.match(line):
            line = "#" + line
        out.append(line)
    body = "\n".join(out)

    # --- namespace the cross-reference labels and references --------------
    body = LABEL_RE.sub(lambda m: f"{{#{m.group(1)}-w{num}-{m.group(2)}", body)
    body = REF_RE.sub(lambda m: f"@{m.group(1)}-w{num}-{m.group(2)}", body)

    # --- point figure paths back at the week's own folder -----------------
    body = FIGPATH_RE.sub(f"](../Week{num}/figures/", body)

    return title, body.strip("\n")


def check(text: str) -> None:
    """Refuse to ship a book with duplicate labels or dangling references."""
    labels = [f"{a}-{b}" for a, b in LABEL_ANY.findall(text)]
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    refs = {f"{a}-{b}" for a, b in REF_ANY.findall(text)}
    dangling = sorted(refs - set(labels))
    if dupes:
        sys.exit(f"duplicate labels across the book: {dupes}")
    if dangling:
        sys.exit(f"references with no matching label: {dangling}")
    print(f"  {len(set(labels))} labels, {len(refs)} references, all resolve")


def main() -> None:
    parts = [BOOK_YAML]
    for num in WEEKS:
        title, body = read_week(num)
        print(f"Week {num}: {title}")
        parts.append(f"# {title}\n\n{body}\n")
    text = "\n".join(parts)
    check(text)
    OUT.write_text(text, encoding="utf-8")
    words = len(re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S).split())
    print(f"wrote {OUT.name}  ({words:,} words)")


if __name__ == "__main__":
    main()
