XELATEX=xelatex -file-line-error -interaction=nonstopmode
CHART_SRCS=charts/tool-architecture-figure.tex charts/tool-workflow-figure.tex
CHART_PDFS=$(CHART_SRCS:.tex=.pdf)

.PHONY: all
all: charts Thesis.tex
	latexmk -pdf -pdflatex="$(XELATEX)" -use-make Thesis.tex

.PHONY: auto
auto: charts Thesis.tex
	latexmk -pdf -pdflatex="$(XELATEX)" -use-make -pvc Thesis.tex

.PHONY: charts
charts: $(CHART_PDFS)

charts/%-figure.pdf: charts/%-figure.tex
	cd charts && $(XELATEX) -halt-on-error $(notdir $<)

.PHONY: clean
clean:
	rm -f Thesis.pdf *.aux *.bbl *bcf *.blg *.log *.out *.tdo *.toc *.xdv *.synctex.gz *.fdb_latexmk *.fls *.run.xml; \
	rm -f **/*.aux charts/*-figure.log charts/*-figure.xdv charts/*-figure.synctex.gz
