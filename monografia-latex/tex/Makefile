BASE_NAME = texto
CHAPTER_LIST = capitulos.tex
APPENDIX_LIST = apendices.tex

PDFLATEX  = pdflatex
BIBTEX    = bibtex
MAKEINDEX = makeindex

pdf: $(BASE_NAME).pdf

debug:
	$(PDFLATEX) "\def\debugcapaime{1} \input{$(BASE_NAME)}"
	$(BIBTEX) $(BASE_NAME) 
	$(MAKEINDEX) $(BASE_NAME) 
	$(PDFLATEX) "\def\debugcapaime{1} \input{$(BASE_NAME)}"
	$(PDFLATEX) "\def\debugcapaime{1} \input{$(BASE_NAME)}"

$(BASE_NAME).pdf: $(BASE_NAME).tex $(CHAPTER_LIST) $(APPENDIX_LIST)
	$(PDFLATEX) $<
	$(BIBTEX) $(BASE_NAME) 
	$(MAKEINDEX) $(BASE_NAME) 
	$(PDFLATEX) $<
	$(PDFLATEX) $<

clean:
	rm -f $(BASE_NAME)*.ps $(BASE_NAME)*.dvi *.log \
	      *.aux *.blg *.toc *.brf *.ilg *.ind \
	      missfont.log $(BASE_NAME)*.bbl $(BASE_NAME)*.out \
		  $(BASE_NAME)*.lof $(BASE_NAME)*.lot $(BASE_NAME).idx
