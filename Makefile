# Build the FinTech 545 lecture notes.
#   make          -> all PDFs (named from each .qmd's output-file)
#   make html     -> HTML versions
#   make figures  -> recompile the TikZ figures to SVG and the data-driven PNGs
#   make clean    -> remove rendered output
QMD := $(wildcard Week*/week*.qmd)

all:
	for f in $(QMD); do quarto render $$f --to pdf; done

html:
	for f in $(QMD); do quarto render $$f --to html; done

figures:
	cd Week02/figures && python3 make_figures.py
	cd Week03/figures && python3 make_figures.py
	cd Week04/figures && python3 make_figures.py
	cd Week05/figures && python3 make_figures.py
	cd Week06/figures && for f in tree1 tree2;   do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week06/figures && python3 make_figures.py
	cd Week07/figures && for f in amput divtree; do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week07/figures && python3 make_figures.py
	cd Week09/figures && for f in taxonomy sigma simflow; do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week09/figures && python3 make_figures.py

clean:
	rm -f Week*/*.pdf Week*/*.html
	rm -rf Week*/*_files Week*/figures/*.aux Week*/figures/*.log Week*/figures/*.pdf

.PHONY: all html figures clean
