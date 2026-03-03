BUILDDIR := /tmp/latex-imo
PDFLATEX := pdflatex -interaction=nonstopmode -output-directory=$(BUILDDIR)

.PHONY: all clean

all: optimisers.pdf off-diag.pdf

optimisers.pdf: optimisers.tex
	mkdir -p $(BUILDDIR)
	$(PDFLATEX) optimisers.tex > $(BUILDDIR)/optimisers-pass1.log 2>&1
	$(PDFLATEX) optimisers.tex > $(BUILDDIR)/optimisers-pass2.log 2>&1
	cp $(BUILDDIR)/optimisers.pdf .

off-diag.pdf: off-diag.tex
	mkdir -p $(BUILDDIR)
	$(PDFLATEX) off-diag.tex > $(BUILDDIR)/off-diag-pass1.log 2>&1
	$(PDFLATEX) off-diag.tex > $(BUILDDIR)/off-diag-pass2.log 2>&1
	cp $(BUILDDIR)/off-diag.pdf .

clean:
	rm -rf $(BUILDDIR)
