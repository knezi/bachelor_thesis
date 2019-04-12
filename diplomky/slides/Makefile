SHELL=/bin/bash

all: bcintro.pdf

bcintro.pdf: bcintro.tex
	pdflatex bcintro.tex

clean:
	rm -f *~ *.{aux,log,nav,out,pdf,snm,toc,vrb}

upload: bcintro.pdf
	rs bcintro.pdf jw:www/static/vyuka/bc/
