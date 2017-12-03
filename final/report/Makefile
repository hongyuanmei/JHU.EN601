TITLE=neural-hawkes
BASE=neural-hawkes
TEX_FILENAME=$(TITLE).tex
FLAGS=-halt-on-error
OPEN=open

paper:
	pdflatex $(FLAGS) $(TITLE)
	bibtex $(TITLE)
	pdflatex $(FLAGS) $(TITLE)
	pdflatex $(FLAGS) $(TITLE)

clean:
	rm -rf *.aux *.bbl *.blg *.log *.out build/ *.dvi *.fls *.fdb_latexmk
