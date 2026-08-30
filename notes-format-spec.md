# Lecture Notes Format Specification

The target format is Week09. Weeks 01--08 were machine-converted from Google Docs and
kept the shape of the original slides: flat headings, orphan fragments, uncaptioned
tables, and pandoc residue in the math. This document defines what "Week09 format"
means and how to convert a week to it.

Two files are required reading before drafting: this document and `writing-style.md`
at the repo root.

## Scope of a conversion

In scope:

- Section hierarchy and numbering
- Prose rewrite from note fragments to connected lecture prose, in Dom's voice
- Table captions, labels, and cross-references
- Figure captions, labels, and cross-references
- Cleanup of OMML/pandoc artifacts in the LaTeX
- Cross-references to other weeks
- YAML header standardization

Out of scope, unless Dom asks:

- New empirical results, new tables of computed numbers, new figures from data
- Changes to `week0N.jl` or any other code file
- Worked-example walkthroughs and problem sets. Week09 has them because Week09 was
  written that way. Most weeks do not, and they stay as they are. Weeks that already
  contain problems (Week08) keep them where they are.
- Mathematical content changes. If the conversion turns up an error, note it and ask.
  Do not fix it silently.

## The format

### 1. YAML header

Copy the Week09 block, changing only `title` and the two `output-file` values. The
`needspace` and `etoolbox` lines are not optional -- without them longtable orphans a
table header at the bottom of a page.

```yaml
---
title: "Week NN — <Topic>"
subtitle: "FinTech 545 --- Quantitative Risk Management"
author: "Dominic Pazzula"
format:
  pdf:
    output-file: "Week NN - <Topic>.pdf"
    documentclass: scrartcl
    pdf-engine: lualatex
    geometry: margin=1in
    fig-pos: 'H'
    include-in-header:
      text: |
        \usepackage{float}
        \usepackage{needspace}
        \usepackage{etoolbox}
        \AtBeginEnvironment{longtable}{\Needspace*{8\baselineskip}}
    number-sections: true
    toc: true
    toc-depth: 2
  html:
    output-file: "Week NN - <Topic>.html"
    toc: true
    html-math-method: katex
---
```

The topic name lives in `output-file:`, never in a filename on disk. Plain hyphen in
the filename, em-dash in the title.

### 2. Section hierarchy

Week09 uses three levels. Weeks 01--08 currently use one.

- `#` -- top-level numbered section. Target 5 to 9 per week.
- `##` -- subsection, the level most of the existing `##` headings map to.
- `###` -- used only where a subsection genuinely has parts.

`toc-depth: 2` means only `#` and `##` reach the table of contents. Anything that
must appear there is a `##` or higher.

Every heading is a plain noun phrase. No sentences, no questions, no colons, no
equations. Week02 currently has a display equation as a heading and a full sentence as
a heading. Both are conversion damage.

### 3. Opening section

Week09 opens with `# Why We Need Factor Models`, which motivates the week before any
math appears. Every week gets an equivalent opening -- the problem this week solves,
stated in a few paragraphs, before the first definition.

Weeks that already open with motivation (Week06 on time value of money) need only the
promotion to `#`. Weeks that open cold with a definition (Week01, Week03) need the
motivating section written.

### 4. Tables

Pipe tables with a caption and a label:

```
| Term | Symbol | What it is |
|:--|:--|:--|
| Exposure | $B$ | how much of factor $k$ asset $i$ carries |

: Three distinct objects {#tbl-vocab}
```

Alignment is explicit: `|:--|` for text, `|--:|` for numbers.

Pandoc grid tables (the `+---+---+` and `-----` forms in Weeks 01--08) all convert to
pipe tables. Every table gets a caption and a `{#tbl-}` label, and every label is
referenced at least once in the prose as `@tbl-name`.

### 5. Figures

```
![Caption stating what the reader should see](figures/name.png){#fig-name width=85%}
```

Every figure gets a `{#fig-}` label and at least one `@fig-name` reference in the
prose. Figures live in `Week0N/figures/`. No stray images at the folder root -- Week01
has `pdf.png` and `cdf.png` at the root and the same images again under `figures/`.
Leave the duplicates alone in this pass, but reference only the `figures/` copies.

### 6. Cross-references within a week

Never write a section number as literal text. Sections get renumbered every time one
is added or moved, and a hard-coded "Section 5.4" goes wrong silently. Label the
heading and reference the label:

```
## Student's t Distribution {#sec-t}

... @sec-t covers the distribution where this matters most for us.
```

Quarto resolves `@sec-` to the live number. Only label the headings that are actually
referenced.

### 7. Cross-references between weeks

Week09 refers back to Week03 and Week07 by name and forward where relevant. Each week
should carry two or three of these, in prose, not as links:

> The PSD version from Week 03 does not.

> We cover the covariance and the general N variable case next week.

### 8. Math cleanup

The Google Docs OMML conversion left artifacts in every week. All of these are
mechanical:

| Artifact | Example | Fix |
|:--|:--|:--|
| Escaped spaces | `F_{X}(x)\ = \ \int` | `F_X(x) = \int` |
| Braced accents | `{\widehat{\mu}}_{3}` | `\hat{\mu}_3` |
| Underlined vars | `\underline{X}` | `\bar{X}` if it means a mean |
| Words inside math | `where\ x\ and\ y\ are\ random` | move to prose or `\text{}` |
| Redundant braces | `{{(e}^{\sigma^{2}} - 1)}^{}` | `(e^{\sigma^2} - 1)` |
| `\lbrack \rbrack` | `E\lbrack X\rbrack` | `E[X]` |
| Subscript braces | `\mu_{n}`, `x_{i}` | `\mu_n`, `x_i` |

Fixing these changes rendering, not content. Verify with the counts in the next
section that nothing was lost.

### 9. Voice

`writing-style.md` at the repo root governs. The failure mode is competent generic
explainer prose. Points that matter most for this task:

- Bold is a label, never emphasis. The existing `**Lognormal Distribution** -- is the
  transform of` pattern is correct and should be preserved.
- `--` for a definition or aside, not `---`.
- No contractions, no "moreover", no "furthermore".
- Median sentence around 14 words. Measure it.
- Do not announce structure, and do not close a section with a summary of it.
- Bullets where the content is genuinely a list, prose where it is an argument.

The existing notes are already in Dom's voice, because he wrote them. The rewrite
connects fragments into prose. It does not restyle sentences that are already fine.

Orphan fragments to eliminate: a lone `PDF` or `Notes` or `For reference:` line acting
as a heading, `Where` followed by an unlabeled list, `Moment:` followed by a numbered
list. These become sentences or labeled lists.

## Procedure

1. Read the `.qmd`, the week's `.jl`, and the rendered PDF page count.
2. Draft the new section map -- the `#` sections and what moves under each. Nothing is
   deleted in this step, everything is placed.
3. Mechanical pass: YAML, math artifacts, grid tables to pipe tables, heading levels.
4. Prose pass: write the opening section, connect the fragments, add cross-references.
5. Render and verify.
6. Report: page count before and after, section map, and a list of every claim added
   that was not in the original.

## Verification

Run before reporting. A conversion is not done until all of these pass.

**Math parity.** Count in the original and the rewrite:

```
grep -o '\$\$' f.qmd | wc -l
grep -o '\\sqrt\|\\int\|\\sum\|\\prod\|\\Gamma\|\\frac' f.qmd | wc -l
```

Display-block count may change if an inline equation is promoted, and `\frac` counts
change if a fraction is simplified. Any drop in `\sqrt`, `\int`, `\sum`, `\prod` is a
loss and must be explained.

**Structure.**

- Every heading is a plain noun phrase
- Every `{#tbl-}` and `{#fig-}` label is referenced at least once
- Every `@tbl-` and `@fig-` reference resolves to a label that exists
- No heading contains `$`
- No literal section number appears in the prose -- `grep -n 'Section [0-9]' f.qmd`
  should return nothing before rendering, and the rendered numbers should be checked
  against the heading list after

**Render.** `quarto render Week0N/week0N.qmd --to pdf` completes with no errors, and
no `??` appears in the output PDF text.

**Voice.** Work the checklist at the end of `writing-style.md`. Measure the median
sentence length rather than estimating it.

## Files a conversion touches

Only `Week0N/week0N.qmd` and, if a figure is added, `Week0N/figures/`. The `.jl` files,
the CSVs, and the root `Makefile` are not touched. The root `Makefile` is protected on
the device -- `device_commit_files` refuses it. Edit it through `device_bash` if it
ever needs a change.
