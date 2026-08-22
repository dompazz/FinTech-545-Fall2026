# book

The nine weekly lecture notes assembled into one PDF.

## Building

From the repo root:

```
make book
```

Or from this folder:

```
python3 make_book.py
quarto render book.qmd --to pdf
```

Output: `FinTech 545 - Quantitative Risk Management.pdf`.

## How it works

`make_book.py` reads the nine `Week0N/week0N.qmd` files and writes `book.qmd`.
The weekly files are the single source of truth and are never modified. Edit a
week, rebuild, and the book reflects the change.

**Do not edit `book.qmd`.** It is generated and gets overwritten on every build.

The transform applied to each week:

| Step | What it does |
|:--|:--|
| YAML | Stripped. The `title:` becomes the chapter heading. |
| Headings | Demoted one level, so a week's `#` becomes `##` under its chapter. Lines inside `$$ … $$` are left alone. |
| Labels | Namespaced by week, so `{#tbl-moments}` becomes `{#tbl-w01-moments}`. References are rewritten to match. |
| Figures | `figures/x.png` becomes `../Week0N/figures/x.png`. |

Namespacing exists so two weeks can use the same label name without colliding.
There are no collisions today, and this keeps it that way as the notes grow.

## The build fails on purpose

Before writing `book.qmd`, the script checks that every cross-reference label is
unique across the whole book and that every reference resolves to a label that
exists. If either check fails it prints the offending names and exits without
writing. A book that silently mis-links is worse than one that does not build.

## Adding a week

Edit `WEEKS` in `make_book.py`. Nothing else needs to change.

## Rendering to HTML

```
quarto render book.qmd --to html
```

The `html` format is already configured. Figures resolve through the same
relative paths, so the HTML has to stay in this folder to find them.
